from __future__ import annotations

import argparse
import ast
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional visualization dependency
    nx = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PipelineConfig:
    data_path: Path = PROJECT_ROOT / "data"
    viz_path: Path = PROJECT_ROOT / "viz"
    horizon_actions: int = 10
    random_state: int = 42
    n_bootstrap: int = 1000
    recompute_vaep: bool = False
    grid_x_bins: int = 12
    grid_y_bins: int = 8
    max_role_components: int = 24
    role_variance_target: float = 0.90
    min_role_events: int = 80
    min_together_minutes: float = 300.0
    min_replacement_minutes: float = 150.0
    min_actions_condition: int = 30
    min_player_quality_minutes: float = 300.0
    min_mean_replacement_similarity: float = -1.0
    min_within_match_window_minutes: float = 15.0
    min_chain_duo_possessions: int = 30
    graph_team_name: str | None = None

    @property
    def combined_data_path(self) -> Path:
        return self.data_path / "raw" / "combined"

    @property
    def processed_data_path(self) -> Path:
        return self.data_path / "processed"

    @property
    def events_path(self) -> Path:
        return self.combined_data_path / "all_events_combined.parquet"

    @property
    def matches_path(self) -> Path:
        return self.combined_data_path / "all_matches_combined.parquet"


EVENT_COLUMNS = [
    "match_id",
    "period",
    "minute",
    "second",
    "timestamp",
    "index",
    "type",
    "team",
    "team_id",
    "player",
    "player_id",
    "possession",
    "possession_team_id",
    "play_pattern",
    "location_x",
    "location_y",
    "pass_end_location_x",
    "pass_end_location_y",
    "carry_end_location_x",
    "carry_end_location_y",
    "shot_statsbomb_xg",
    "pass_outcome",
    "dribble_outcome",
    "duel_outcome",
    "interception_outcome",
    "shot_outcome",
    "goalkeeper_outcome",
    "ball_receipt_outcome",
    "ball_recovery_recovery_failure",
    "out",
    "substitution_replacement",
    "substitution_replacement_id",
    "tactics",
    "position",
]

NON_ACTION_TYPES = {
    "Starting XI",
    "Half Start",
    "Half End",
    "Tactical Shift",
    "Substitution",
    "Player Off",
    "Player On",
    "Injury Stoppage",
    "Bad Behaviour",
    "Referee Ball-Drop",
    "Own Goal Against",
    "Ball Receipt*",
}

NON_ROLE_EVENT_TYPES = {
    "Starting XI",
    "Half Start",
    "Half End",
    "Tactical Shift",
    "Substitution",
    "Player Off",
    "Player On",
    "Injury Stoppage",
    "Bad Behaviour",
    "Referee Ball-Drop",
}

FAIL_TYPES = {"Dispossessed", "Miscontrol", "Dribbled Past", "Error", "Foul Committed"}
FAIL_GOALKEEPER = {"Fail", "Lost In Play", "Lost Out", "In Play Danger", "No Touch"}
FAIL_SHOTS = {"Blocked", "Off T", "Wayward", "Saved", "Post", "Saved Off Target", "Saved to Post"}

CAT_FEATURES = ["type", "result", "team_id", "play_pattern", "period"]
NUM_FEATURES = [
    "location_x",
    "location_y",
    "pass_end_location_x",
    "pass_end_location_y",
    "carry_end_location_x",
    "carry_end_location_y",
    "shot_statsbomb_xg",
    "event_minute",
]
BASE_FEATURES = CAT_FEATURES + NUM_FEATURES


def _mode_or_first(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return None
    modes = values.astype(str).mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def parse_tactics(value):
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, str) and value:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
    return {}


def lineup_from_tactics(value):
    tactics = parse_tactics(value)
    players, names = set(), {}
    for item in tactics.get("lineup", []):
        player = item.get("player", {})
        pid = player.get("id")
        if pid is not None:
            players.add(int(pid))
            names[int(pid)] = player.get("name")
    return players, names


def load_inputs(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.processed_data_path.mkdir(parents=True, exist_ok=True)
    config.viz_path.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(config.events_path, columns=EVENT_COLUMNS)
    matches = pd.read_parquet(config.matches_path)
    events = events.sort_values(["match_id", "period", "minute", "second", "index"]).reset_index(drop=True)
    events["event_minute"] = events["minute"].astype(float) + events["second"].fillna(0).astype(float) / 60.0
    return events, matches


def true_mask(series: pd.Series) -> pd.Series:
    return series.eq(True) | series.astype("string").str.lower().eq("true")


def infer_result(df: pd.DataFrame) -> np.ndarray:
    failed = np.zeros(len(df), dtype=bool)
    failed |= df["type"].isin(FAIL_TYPES).to_numpy()
    failed |= df["pass_outcome"].notna().to_numpy()
    failed |= df["dribble_outcome"].eq("Incomplete").to_numpy()
    failed |= df["duel_outcome"].astype("string").str.startswith("Lost", na=False).to_numpy()
    failed |= df["interception_outcome"].astype("string").str.startswith("Lost", na=False).to_numpy()
    failed |= df["goalkeeper_outcome"].isin(FAIL_GOALKEEPER).to_numpy()
    failed |= df["ball_receipt_outcome"].eq("Incomplete").to_numpy()
    failed |= true_mask(df["ball_recovery_recovery_failure"]).fillna(False).to_numpy()
    failed |= df["shot_outcome"].isin(FAIL_SHOTS).to_numpy()
    failed |= true_mask(df["out"]).fillna(False).to_numpy()
    return np.where(failed, "fail", "success")


def prepare_actions(events_df: pd.DataFrame) -> pd.DataFrame:
    actions_df = events_df.loc[
        events_df["team_id"].notna() & ~events_df["type"].isin(NON_ACTION_TYPES)
    ].copy()
    actions_df["team_id"] = actions_df["team_id"].astype(int)
    actions_df["result"] = infer_result(actions_df)
    for column in NUM_FEATURES:
        if column in actions_df:
            actions_df[column] = pd.to_numeric(actions_df[column], errors="coerce")
    actions_df["is_goal"] = actions_df["type"].eq("Own Goal For") | (
        actions_df["type"].eq("Shot") & actions_df["shot_outcome"].eq("Goal")
    )
    return actions_df.reset_index(drop=True)


def add_future_goal_labels(actions_df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    actions_df = actions_df.copy()
    scores = np.zeros(len(actions_df), dtype=bool)
    concedes = np.zeros(len(actions_df), dtype=bool)
    teams = actions_df["team_id"].to_numpy()
    goals = actions_df["is_goal"].to_numpy()

    for indices in actions_df.groupby("match_id", sort=False).indices.values():
        idx = np.asarray(indices)
        group_teams = teams[idx]
        group_goals = goals[idx]
        group_scores = np.zeros(len(idx), dtype=bool)
        group_concedes = np.zeros(len(idx), dtype=bool)

        for step in range(horizon):
            if step >= len(idx):
                break
            future_goal = np.zeros(len(idx), dtype=bool)
            future_team = np.full(len(idx), -1, dtype=int)
            future_goal[: len(idx) - step] = group_goals[step:]
            future_team[: len(idx) - step] = group_teams[step:]
            group_scores |= future_goal & (future_team == group_teams)
            group_concedes |= future_goal & (future_team != group_teams)

        scores[idx] = group_scores
        concedes[idx] = group_concedes

    actions_df["label_score"] = scores.astype(int)
    actions_df["label_concede"] = concedes.astype(int)
    return actions_df


def build_state_features(actions_df: pd.DataFrame) -> pd.DataFrame:
    grouped = actions_df.groupby("match_id", sort=False)
    x = pd.DataFrame(index=actions_df.index)

    for lag in range(3):
        shifted = actions_df[BASE_FEATURES] if lag == 0 else grouped[BASE_FEATURES].shift(lag)
        for column in CAT_FEATURES:
            x[f"{column}_lag{lag}"] = shifted[column].astype("string").fillna("missing").astype("category")
        for column in NUM_FEATURES:
            x[f"{column}_lag{lag}"] = pd.to_numeric(shifted[column], errors="coerce").fillna(-1).astype("float32")

    x["same_team_lag1"] = actions_df["team_id"].eq(grouped["team_id"].shift(1)).fillna(False).astype("int8")
    x["same_team_lag2"] = actions_df["team_id"].eq(grouped["team_id"].shift(2)).fillna(False).astype("int8")
    return x


def make_xgb_model(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        enable_categorical=True,
        n_estimators=180,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=25,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=random_state,
    )


def fit_vaep(events_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    actions = prepare_actions(events_df)
    actions = add_future_goal_labels(actions, config.horizon_actions)
    x = build_state_features(actions)
    y_score = actions["label_score"].astype(int)
    y_concede = actions["label_concede"].astype(int)
    groups = actions["match_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.random_state)
    train_idx, valid_idx = next(splitter.split(x, y_score, groups=groups))

    score_model = make_xgb_model(config.random_state)
    concede_model = make_xgb_model(config.random_state)
    score_model.fit(x.iloc[train_idx], y_score.iloc[train_idx])
    concede_model.fit(x.iloc[train_idx], y_concede.iloc[train_idx])

    validation = pd.DataFrame(
        {
            "model": ["P_score", "P_concede"],
            "log_loss": [
                log_loss(y_score.iloc[valid_idx], score_model.predict_proba(x.iloc[valid_idx])[:, 1]),
                log_loss(y_concede.iloc[valid_idx], concede_model.predict_proba(x.iloc[valid_idx])[:, 1]),
            ],
            "positive_rate": [y_score.iloc[valid_idx].mean(), y_concede.iloc[valid_idx].mean()],
        }
    )

    score_model_full = make_xgb_model(config.random_state)
    concede_model_full = make_xgb_model(config.random_state)
    score_model_full.fit(x, y_score)
    concede_model_full.fit(x, y_concede)

    actions["p_score_after"] = score_model_full.predict_proba(x)[:, 1]
    actions["p_concede_after"] = concede_model_full.predict_proba(x)[:, 1]
    actions["prev_team_id"] = actions.groupby("match_id", sort=False)["team_id"].shift(1)
    actions["same_team_as_previous"] = actions["team_id"].eq(actions["prev_team_id"])
    actions["prev_p_score"] = actions.groupby("match_id", sort=False)["p_score_after"].shift(1).fillna(0)
    actions["prev_p_concede"] = actions.groupby("match_id", sort=False)["p_concede_after"].shift(1).fillna(0)
    actions["p_score_before"] = np.where(
        actions["same_team_as_previous"], actions["prev_p_score"], actions["prev_p_concede"]
    )
    actions["p_concede_before"] = np.where(
        actions["same_team_as_previous"], actions["prev_p_concede"], actions["prev_p_score"]
    )
    actions["vaep"] = (
        (actions["p_score_after"] - actions["p_score_before"])
        - (actions["p_concede_after"] - actions["p_concede_before"])
    )

    vaep_columns = [
        "match_id",
        "period",
        "minute",
        "second",
        "event_minute",
        "index",
        "type",
        "team",
        "team_id",
        "player",
        "player_id",
        "p_score_before",
        "p_score_after",
        "p_concede_before",
        "p_concede_after",
        "vaep",
        "label_score",
        "label_concede",
    ]
    actions[vaep_columns].to_parquet(config.processed_data_path / "vaep_actions.parquet", index=False)
    validation.to_parquet(config.processed_data_path / "vaep_validation.parquet", index=False)
    with open(config.processed_data_path / "xgb_score_model.pkl", "wb") as file:
        pickle.dump(score_model_full, file)
    with open(config.processed_data_path / "xgb_concede_model.pkl", "wb") as file:
        pickle.dump(concede_model_full, file)
    return actions[vaep_columns].copy()


def load_or_build_vaep(events_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    vaep_path = config.processed_data_path / "vaep_actions.parquet"
    if vaep_path.exists() and not config.recompute_vaep:
        return pd.read_parquet(vaep_path)
    return fit_vaep(events_df, config)


def extract_goalkeeper_ids(events_df: pd.DataFrame, vaep_actions: pd.DataFrame) -> set[int]:
    gk_ids = set()
    for row in events_df[events_df["type"] == "Starting XI"].itertuples(index=False):
        tactics = parse_tactics(row.tactics)
        for item in tactics.get("lineup", []):
            if item.get("position", {}).get("name") == "Goalkeeper":
                pid = item.get("player", {}).get("id")
                if pid:
                    gk_ids.add(int(pid))
    gk_from_actions = set(
        vaep_actions.loc[vaep_actions["type"].eq("Goal Keeper"), "player_id"].dropna().astype(int).unique()
    )
    return gk_ids | gk_from_actions


def build_player_lookup(events_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    direct = events_df.loc[
        events_df["player_id"].notna(), ["team_id", "team", "player_id", "player"]
    ].dropna(subset=["team_id", "player_id"])
    rows.append(direct)

    replacements = events_df.loc[
        events_df["substitution_replacement_id"].notna(),
        ["team_id", "team", "substitution_replacement_id", "substitution_replacement"],
    ].dropna(subset=["team_id"])
    replacements = replacements.rename(
        columns={"substitution_replacement_id": "player_id", "substitution_replacement": "player"}
    )
    rows.append(replacements)

    lineup_rows = []
    for row in events_df.loc[
        events_df["type"].eq("Starting XI"), ["team_id", "team", "tactics"]
    ].itertuples(index=False):
        players, names = lineup_from_tactics(row.tactics)
        for pid in players:
            lineup_rows.append({"team_id": row.team_id, "team": row.team, "player_id": pid, "player": names.get(pid)})
    rows.append(pd.DataFrame(lineup_rows))

    lookup = pd.concat(rows, ignore_index=True).dropna(subset=["team_id", "player_id", "player"])
    lookup["team_id"] = lookup["team_id"].astype(int)
    lookup["player_id"] = lookup["player_id"].astype(int)
    return lookup.drop_duplicates(["team_id", "player_id"], keep="last")


def build_position_lookup(events_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in events_df.loc[
        events_df["type"].eq("Starting XI"), ["match_id", "team_id", "team", "tactics"]
    ].itertuples(index=False):
        tactics = parse_tactics(row.tactics)
        for item in tactics.get("lineup", []):
            player = item.get("player", {})
            position = item.get("position", {})
            pid = player.get("id")
            if pid is None:
                continue
            rows.append(
                {
                    "team_id": int(row.team_id),
                    "team": row.team,
                    "player_id": int(pid),
                    "lineup_position": position.get("name"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["team_id", "player_id", "lineup_position"])
    positions = pd.DataFrame(rows)
    return (
        positions.groupby(["team_id", "team", "player_id"], as_index=False)
        .agg(lineup_position=("lineup_position", _mode_or_first))
    )


def _norm_position(position: object) -> str:
    if position is None or pd.isna(position):
        return ""
    return str(position).strip().lower()


def _position_side(position: object) -> str | None:
    text = _norm_position(position)
    if text.startswith("left") or " left " in f" {text} ":
        return "left"
    if text.startswith("right") or " right " in f" {text} ":
        return "right"
    return None


def _position_role(position: object) -> str:
    text = _norm_position(position)
    if not text:
        return "unknown"
    if "goalkeeper" in text:
        return "goalkeeper"
    if "center back" in text or "centre back" in text:
        return "center_back"
    if "back" in text:
        return "fullback"
    if "defensive midfield" in text:
        return "defensive_midfield"
    if "attacking midfield" in text:
        return "attacking_midfield"
    if "midfield" in text:
        return "central_midfield"
    if "wing" in text:
        return "winger"
    if "forward" in text or "striker" in text:
        return "forward"
    return "unknown"


def position_compatibility_score(target_position: object, candidate_position: object) -> int:
    target_role = _position_role(target_position)
    candidate_role = _position_role(candidate_position)
    if target_role == "goalkeeper" or candidate_role == "goalkeeper":
        return 0
    if target_role == "unknown" or candidate_role == "unknown":
        return 0
    target_side = _position_side(target_position)
    candidate_side = _position_side(candidate_position)
    if _norm_position(target_position) == _norm_position(candidate_position):
        return 6
    if target_role == "winger":
        if candidate_role == "winger":
            return 5 if target_side and target_side == candidate_side else 4
        if candidate_role == "forward":
            return 3
        if candidate_role == "attacking_midfield":
            return 2
        return 0
    if target_role == "forward":
        if candidate_role == "forward":
            return 5
        if candidate_role == "winger":
            return 4
        if candidate_role == "attacking_midfield":
            return 3
        return 0
    if target_role == "attacking_midfield":
        if candidate_role == "attacking_midfield":
            return 5
        if candidate_role == "central_midfield":
            return 4
        if candidate_role in {"winger", "forward"}:
            return 3
        if candidate_role == "defensive_midfield":
            return 2
        return 0
    if target_role == "central_midfield":
        if candidate_role == "central_midfield":
            return 5
        if candidate_role in {"defensive_midfield", "attacking_midfield"}:
            return 4
        if candidate_role == "winger" and target_side and target_side == candidate_side:
            return 2
        return 0
    if target_role == "defensive_midfield":
        if candidate_role == "defensive_midfield":
            return 5
        if candidate_role == "central_midfield":
            return 4
        if candidate_role == "center_back":
            return 2
        return 0
    if target_role == "fullback":
        if candidate_role == "fullback":
            return 5 if target_side and target_side == candidate_side else 3
        if candidate_role == "winger" and target_side and target_side == candidate_side:
            return 2
        if candidate_role == "center_back":
            return 1
        return 0
    if target_role == "center_back":
        if candidate_role == "center_back":
            return 5
        if candidate_role in {"fullback", "defensive_midfield"}:
            return 2
        return 0
    return 0


def _starting_slots_from_tactics(value) -> dict[int, dict[str, object]]:
    tactics = parse_tactics(value)
    slots = {}
    for slot_id, item in enumerate(tactics.get("lineup", [])):
        player = item.get("player", {})
        position = item.get("position", {})
        pid = player.get("id")
        if pid is None:
            continue
        pid = int(pid)
        slots[int(slot_id)] = {
            "starting_player_id": pid,
            "starting_position": position.get("name"),
            "current_player_id": pid,
            "players": {pid},
        }
    return slots


def _current_slot_players(slots: dict[int, dict[str, object]]) -> tuple[tuple[int, int], ...]:
    pairs = []
    for slot_id, slot in sorted(slots.items()):
        pid = slot.get("current_player_id")
        if pid is not None:
            pairs.append((int(slot_id), int(pid)))
    return tuple(pairs)


def _find_current_slot(slots: dict[int, dict[str, object]], player_id: int) -> int | None:
    for slot_id, slot in slots.items():
        if slot.get("current_player_id") == int(player_id):
            return int(slot_id)
    return None


def _create_extra_slot(slots: dict[int, dict[str, object]], player_id: int | None = None) -> int:
    slot_id = max(slots.keys(), default=-1) + 1
    players = set()
    if player_id is not None:
        players.add(int(player_id))
    slots[slot_id] = {
        "starting_player_id": None,
        "current_player_id": int(player_id) if player_id is not None else None,
        "players": players,
    }
    return slot_id


def _append_presence_segment(
    rows: list[dict[str, object]],
    match_id: int,
    season,
    match_date,
    team_id: int,
    team,
    start_minute: float,
    end_minute: float,
    slots: dict[int, dict[str, object]],
) -> None:
    slot_players = _current_slot_players(slots)
    active = frozenset(player_id for _, player_id in slot_players)
    if not active or end_minute <= start_minute:
        return
    rows.append(
        {
            "match_id": match_id,
            "season": season,
            "match_date": match_date,
            "team_id": team_id,
            "team": team,
            "start_minute": start_minute,
            "end_minute": end_minute,
            "duration": end_minute - start_minute,
            "active": active,
            "slot_players": slot_players,
        }
    )


def build_presence_segments(events_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    segments = []
    match_end = events_df.groupby("match_id")["event_minute"].max().to_dict()
    match_meta = matches_df[["match_id", "season", "match_date"]].drop_duplicates("match_id").set_index("match_id")
    state_events = {"Substitution", "Player Off", "Player On"}

    for (match_id, team_id), group in events_df.dropna(subset=["team_id"]).groupby(
        ["match_id", "team_id"], sort=False
    ):
        team_id = int(team_id)
        start_rows = group.loc[group["type"].eq("Starting XI")]
        if start_rows.empty:
            continue

        slots = _starting_slots_from_tactics(start_rows.iloc[0]["tactics"])
        if not slots:
            continue
        player_to_slot = {
            int(slot["current_player_id"]): int(slot_id)
            for slot_id, slot in slots.items()
            if slot.get("current_player_id") is not None
        }
        temporary_off_slot: dict[int, int] = {}
        last_minute = 0.0
        end_minute = float(match_end.get(match_id, group["event_minute"].max()))
        team = group["team"].dropna().iloc[0] if group["team"].notna().any() else None
        season = match_meta.loc[match_id, "season"] if match_id in match_meta.index else None
        match_date = pd.to_datetime(match_meta.loc[match_id, "match_date"]) if match_id in match_meta.index else pd.NaT
        changes = group.loc[group["type"].isin(state_events)].sort_values(["event_minute", "index"])
        match_segments: list[dict[str, object]] = []

        for row in changes.itertuples(index=False):
            cur = float(row.event_minute)
            _append_presence_segment(
                match_segments,
                match_id,
                season,
                match_date,
                team_id,
                team,
                last_minute,
                cur,
                slots,
            )
            if row.type == "Substitution":
                off_id = int(row.player_id) if pd.notna(row.player_id) else None
                on_id = int(row.substitution_replacement_id) if pd.notna(row.substitution_replacement_id) else None
                slot_id = None
                if off_id is not None:
                    slot_id = _find_current_slot(slots, off_id)
                    if slot_id is None:
                        slot_id = temporary_off_slot.get(off_id)
                    if slot_id is None:
                        slot_id = player_to_slot.get(off_id)
                if slot_id is None:
                    slot_id = _create_extra_slot(slots, on_id)
                    if on_id is not None:
                        player_to_slot[on_id] = int(slot_id)
                else:
                    if off_id is not None:
                        temporary_off_slot.pop(off_id, None)
                    slots[slot_id]["current_player_id"] = on_id
                    if on_id is not None:
                        slots[slot_id]["players"].add(on_id)
                        player_to_slot[on_id] = int(slot_id)
            elif row.type == "Player Off" and pd.notna(row.player_id):
                player_id = int(row.player_id)
                slot_id = _find_current_slot(slots, player_id)
                if slot_id is not None:
                    slots[slot_id]["current_player_id"] = None
                    temporary_off_slot[player_id] = int(slot_id)
            elif row.type == "Player On" and pd.notna(row.player_id):
                player_id = int(row.player_id)
                slot_id = temporary_off_slot.pop(player_id, None)
                if slot_id is None:
                    slot_id = player_to_slot.get(player_id)
                if slot_id is None:
                    slot_id = _create_extra_slot(slots, player_id)
                else:
                    slots[slot_id]["current_player_id"] = player_id
                    slots[slot_id]["players"].add(player_id)
                    player_to_slot[player_id] = int(slot_id)
            last_minute = max(last_minute, cur)

        _append_presence_segment(
            match_segments,
            match_id,
            season,
            match_date,
            team_id,
            team,
            last_minute,
            end_minute,
            slots,
        )

        starter_slots = tuple(
            (int(slot_id), int(slot["starting_player_id"]), slot.get("starting_position"))
            for slot_id, slot in sorted(slots.items())
            if slot.get("starting_player_id") is not None
        )
        match_player_slots = tuple(
            (int(player_id), int(slot_id))
            for slot_id, slot in sorted(slots.items())
            for player_id in sorted(slot["players"])
        )
        match_participants = frozenset(player_id for player_id, _ in match_player_slots)
        for segment in match_segments:
            segment["starter_slots"] = starter_slots
            segment["match_player_slots"] = match_player_slots
            segment["match_participants"] = match_participants
            segments.append(segment)

    return pd.DataFrame(segments)


def build_lineup_slot_segments(segments_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seg in segments_df.itertuples(index=False):
        starter_by_slot = {int(slot_id): int(player_id) for slot_id, player_id, _ in seg.starter_slots}
        starter_position_by_slot = {int(slot_id): position for slot_id, _, position in seg.starter_slots}
        for slot_id, player_id in seg.slot_players:
            rows.append(
                {
                    "match_id": seg.match_id,
                    "season": seg.season,
                    "match_date": seg.match_date,
                    "team_id": int(seg.team_id),
                    "team": seg.team,
                    "slot_id": int(slot_id),
                    "slot_starter_id": starter_by_slot.get(int(slot_id)),
                    "slot_starter_position": starter_position_by_slot.get(int(slot_id)),
                    "player_id": int(player_id),
                    "start_minute": float(seg.start_minute),
                    "end_minute": float(seg.end_minute),
                    "duration": float(seg.duration),
                }
            )
    return pd.DataFrame(rows)


def player_minutes_per_match(segments_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seg in segments_df.itertuples(index=False):
        for pid in seg.active:
            rows.append(
                {
                    "match_id": seg.match_id,
                    "season": seg.season,
                    "match_date": seg.match_date,
                    "team_id": int(seg.team_id),
                    "team": seg.team,
                    "player_id": int(pid),
                    "minutes": seg.duration,
                }
            )
    df = pd.DataFrame(rows)
    return df.groupby(["match_id", "season", "match_date", "team_id", "team", "player_id"], as_index=False)[
        "minutes"
    ].sum()


def build_player_match(
    segments_df: pd.DataFrame, vaep_actions: pd.DataFrame, goalkeeper_ids: set[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vaep_by_player_match = (
        vaep_actions.loc[vaep_actions["player_id"].notna()]
        .assign(team_id=lambda df: df["team_id"].astype(int), player_id=lambda df: df["player_id"].astype(int))
        .groupby(["match_id", "team_id", "player_id"], as_index=False)
        .agg(vaep_sum=("vaep", "sum"), n_actions=("vaep", "count"))
    )
    player_match = player_minutes_per_match(segments_df).merge(
        vaep_by_player_match, on=["match_id", "team_id", "player_id"], how="left"
    )
    player_match["vaep_sum"] = player_match["vaep_sum"].fillna(0.0)
    player_match["n_actions"] = player_match["n_actions"].fillna(0).astype(int)
    player_match["vaep_per90"] = 90 * player_match["vaep_sum"] / player_match["minutes"].replace(0, np.nan)

    player_quality = (
        player_match[~player_match["player_id"].isin(goalkeeper_ids)]
        .groupby("player_id")
        .agg(vaep_total=("vaep_sum", "sum"), min_total=("minutes", "sum"), actions=("n_actions", "sum"))
        .assign(vaep_per90=lambda df: 90 * df["vaep_total"] / df["min_total"].replace(0, np.nan))
    )
    return player_match, player_quality


def build_player_windows(player_match_df: pd.DataFrame, goalkeeper_ids: set[int]) -> pd.DataFrame:
    windows = (
        player_match_df[~player_match_df["player_id"].isin(goalkeeper_ids)]
        .assign(match_date=lambda df: pd.to_datetime(df["match_date"]))
        .groupby(["season", "team_id", "player_id"], as_index=False)
        .agg(first_date=("match_date", "min"), last_date=("match_date", "max"), min_total=("minutes", "sum"))
    )
    windows["team_id"] = windows["team_id"].astype(int)
    windows["player_id"] = windows["player_id"].astype(int)
    return windows


def build_role_profiles(
    events_df: pd.DataFrame,
    player_lookup: pd.DataFrame,
    position_lookup: pd.DataFrame,
    goalkeeper_ids: set[int],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    role_events = events_df.loc[
        events_df["team_id"].notna()
        & events_df["player_id"].notna()
        & events_df["location_x"].notna()
        & events_df["location_y"].notna()
        & ~events_df["type"].isin(NON_ROLE_EVENT_TYPES),
        ["team_id", "team", "player_id", "player", "type", "location_x", "location_y"],
    ].copy()
    role_events["team_id"] = role_events["team_id"].astype(int)
    role_events["player_id"] = role_events["player_id"].astype(int)
    role_events = role_events[~role_events["player_id"].isin(goalkeeper_ids)]
    role_events["location_x"] = pd.to_numeric(role_events["location_x"], errors="coerce")
    role_events["location_y"] = pd.to_numeric(role_events["location_y"], errors="coerce")
    role_events = role_events[
        role_events["location_x"].between(0, 120) & role_events["location_y"].between(0, 80)
    ].copy()

    x_edges = np.linspace(0, 120, config.grid_x_bins + 1)
    y_edges = np.linspace(0, 80, config.grid_y_bins + 1)

    rows = []
    vectors = []
    for (team_id, player_id), group in role_events.groupby(["team_id", "player_id"], sort=True):
        if len(group) < config.min_role_events:
            continue
        hist, _, _ = np.histogram2d(
            group["location_y"].to_numpy(float),
            group["location_x"].to_numpy(float),
            bins=[y_edges, x_edges],
        )
        total = hist.sum()
        if total <= 0:
            continue
        density = (hist / total).astype(float)
        vectors.append(density.reshape(-1))
        rows.append(
            {
                "team_id": int(team_id),
                "player_id": int(player_id),
                "role_events": int(len(group)),
                "mean_x": float(group["location_x"].mean()),
                "mean_y": float(group["location_y"].mean()),
                "std_x": float(group["location_x"].std(ddof=0)),
                "std_y": float(group["location_y"].std(ddof=0)),
                "defensive_third_share": float((group["location_x"] < 40).mean()),
                "middle_third_share": float(group["location_x"].between(40, 80, inclusive="left").mean()),
                "attacking_third_share": float((group["location_x"] >= 80).mean()),
                "left_channel_share": float((group["location_y"] < 26.6667).mean()),
                "central_channel_share": float(group["location_y"].between(26.6667, 53.3333).mean()),
                "right_channel_share": float((group["location_y"] >= 53.3333).mean()),
            }
        )

    if not rows:
        raise RuntimeError("No role profiles were created. Lower min_role_events or check event coordinates.")

    profiles = pd.DataFrame(rows)
    x_matrix = np.vstack(vectors)
    x_mean = x_matrix.mean(axis=0, keepdims=True)
    x_centered = x_matrix - x_mean
    _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    eigvals = (singular_values**2) / max(len(x_matrix) - 1, 1)
    explained = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained)
    k_variance = int(np.searchsorted(cumulative, config.role_variance_target) + 1)
    k_selected = int(min(config.max_role_components, max(2, k_variance), vt.shape[0]))
    scores = x_centered @ vt[:k_selected].T
    scores = (scores - scores.mean(axis=0, keepdims=True)) / (scores.std(axis=0, keepdims=True) + 1e-12)
    norms = np.linalg.norm(scores, axis=1, keepdims=True)
    embeddings = scores / np.maximum(norms, 1e-12)

    pc_columns = pd.DataFrame(
        {f"role_pc_{idx + 1}": scores[:, idx] for idx in range(k_selected)}
    )
    embedding_columns = pd.DataFrame(
        {f"role_emb_{idx + 1}": embeddings[:, idx] for idx in range(k_selected)}
    )
    bin_columns = pd.DataFrame(
        {f"role_bin_{idx:02d}": x_matrix[:, idx] for idx in range(x_matrix.shape[1])}
    )
    profiles = pd.concat([profiles.reset_index(drop=True), pc_columns, embedding_columns, bin_columns], axis=1)

    profiles = profiles.merge(player_lookup, on=["team_id", "player_id"], how="left")
    profiles = profiles.merge(
        position_lookup[["team_id", "player_id", "lineup_position"]],
        on=["team_id", "player_id"],
        how="left",
    )
    profiles["role_components"] = k_selected
    profiles["role_explained_variance"] = float(cumulative[k_selected - 1])
    profiles.to_parquet(config.processed_data_path / "player_role_profiles.parquet", index=False)

    similarity_rows = []
    emb_cols = [f"role_emb_{idx + 1}" for idx in range(k_selected)]
    for team_id, team_profiles in profiles.groupby("team_id", sort=False):
        ids = team_profiles["player_id"].astype(int).to_numpy()
        emb = team_profiles[emb_cols].to_numpy(float)
        if len(ids) < 2:
            continue
        sim = emb @ emb.T
        for i, target_id in enumerate(ids):
            for j, candidate_id in enumerate(ids):
                if i == j:
                    continue
                similarity_rows.append(
                    {
                        "team_id": int(team_id),
                        "target_player_id": int(target_id),
                        "candidate_player_id": int(candidate_id),
                        "role_similarity": float(sim[i, j]),
                    }
                )
    similarities = pd.DataFrame(similarity_rows)
    similarities.to_parquet(config.processed_data_path / "player_role_similarity.parquet", index=False)
    return profiles, similarities


def build_similarity_lookup(similarities: pd.DataFrame) -> dict[tuple[int, int, int], float]:
    return {
        (int(row.team_id), int(row.target_player_id), int(row.candidate_player_id)): float(row.role_similarity)
        for row in similarities.itertuples(index=False)
    }


def _similarity_or_zero(
    similarity_lookup: dict[tuple[int, int, int], float],
    team_id: int,
    target_player_id: int,
    candidate_player_id: int,
) -> float:
    value = similarity_lookup.get((int(team_id), int(target_player_id), int(candidate_player_id)))
    return float(value) if value is not None and np.isfinite(value) else 0.0


def choose_actual_role_proxy(
    seg,
    team_id: int,
    focal_player_id: int,
    absent_partner_id: int,
    active_players: set[int],
    similarity_lookup: dict[tuple[int, int, int], float],
    player_position_lookup: dict[tuple[int, int], object],
    goalkeeper_ids: set[int],
) -> tuple[int | None, int | None, str | None, float]:
    """Choose the real on-pitch player occupying the absent partner's role.

    Priority:
    1. If the absent partner belongs to a real substitution slot in this match,
       use the current occupant of that same slot.
    2. If the absent partner did not play this match, choose the starting slot
       whose starter is most similar to the absent partner, then follow that
       slot's real substitutions.
    """
    current_by_slot = {int(slot_id): int(player_id) for slot_id, player_id in seg.slot_players}
    player_to_slot = {int(player_id): int(slot_id) for player_id, slot_id in seg.match_player_slots}
    starter_by_slot = {int(slot_id): int(player_id) for slot_id, player_id, _ in seg.starter_slots}
    starter_position_by_slot = {int(slot_id): position for slot_id, _, position in seg.starter_slots}

    if int(absent_partner_id) in player_to_slot:
        slot_id = player_to_slot[int(absent_partner_id)]
        proxy_id = current_by_slot.get(slot_id)
        if (
            proxy_id is not None
            and proxy_id != int(focal_player_id)
            and proxy_id in active_players
            and proxy_id not in goalkeeper_ids
        ):
            similarity = _similarity_or_zero(similarity_lookup, team_id, absent_partner_id, proxy_id)
            return int(proxy_id), int(absent_partner_id), "same_substitution_slot", similarity
        return None, None, None, np.nan

    target_position = player_position_lookup.get((int(team_id), int(absent_partner_id)))
    best = None
    for slot_id, starter_id in starter_by_slot.items():
        starter_id = int(starter_id)
        proxy_id = current_by_slot.get(int(slot_id))
        if proxy_id is None:
            continue
        if starter_id in goalkeeper_ids or proxy_id in goalkeeper_ids:
            continue
        if starter_id == int(focal_player_id) or proxy_id == int(focal_player_id):
            continue
        if proxy_id not in active_players:
            continue
        compatibility = position_compatibility_score(target_position, starter_position_by_slot.get(int(slot_id)))
        if compatibility <= 0:
            continue
        similarity = similarity_lookup.get((int(team_id), int(absent_partner_id), starter_id))
        if similarity is None or not np.isfinite(similarity):
            continue
        candidate = (int(compatibility), float(similarity), int(slot_id), starter_id, int(proxy_id))
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is None:
        return None, None, None, np.nan

    _, similarity, _, anchor_player_id, proxy_id = best
    return int(proxy_id), int(anchor_player_id), "starter_similarity_slot", float(similarity)


def aggregate_role_adjusted_samples(
    segments_df: pd.DataFrame,
    vaep_actions: pd.DataFrame,
    windows_df: pd.DataFrame,
    role_profiles: pd.DataFrame,
    similarities: pd.DataFrame,
    position_lookup: pd.DataFrame,
    goalkeeper_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = [
        "minutes_together",
        "focal_vaep_together",
        "partner_vaep_together",
        "duo_vaep_together",
        "duo_pos_vaep_together",
        "duo_actions_together",
        "minutes_proxy",
        "focal_vaep_proxy",
        "proxy_vaep",
        "proxy_pos_vaep",
        "proxy_duo_vaep",
        "proxy_duo_pos_vaep",
        "proxy_duo_actions",
        "proxy_similarity_minutes",
        "proxy_segments",
        "same_slot_proxy_minutes",
        "starter_similarity_proxy_minutes",
    ]
    samples = defaultdict(lambda: {field: 0.0 for field in fields})
    proxy_minutes = defaultdict(float)
    proxy_similarity_minutes = defaultdict(float)
    per_match_proxy_minutes: dict[tuple, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    similarity_lookup = build_similarity_lookup(similarities)
    player_position_lookup = {
        (int(row.team_id), int(row.player_id)): row.lineup_position
        for row in position_lookup[["team_id", "player_id", "lineup_position"]]
        .dropna(subset=["team_id", "player_id"])
        .itertuples(index=False)
    }
    profiled_players = {
        (int(row.team_id), int(row.player_id)) for row in role_profiles[["team_id", "player_id"]].itertuples(index=False)
    }

    action_base = vaep_actions.loc[
        vaep_actions["player_id"].notna(), ["match_id", "team_id", "event_minute", "player_id", "vaep"]
    ].copy()
    action_base["team_id"] = action_base["team_id"].astype(int)
    action_base["player_id"] = action_base["player_id"].astype(int)
    action_base = action_base[~action_base["player_id"].isin(goalkeeper_ids)]
    action_groups = {
        key: grp.sort_values("event_minute")[["event_minute", "player_id", "vaep"]]
        for key, grp in action_base.groupby(["match_id", "team_id"], sort=False)
    }
    window_groups = {
        key: grp[["player_id", "first_date", "last_date"]].copy()
        for key, grp in windows_df.groupby(["season", "team_id"], sort=False)
    }

    for seg in segments_df.itertuples(index=False):
        team_id = int(seg.team_id)
        active_all = {int(pid) for pid in seg.active if int(pid) not in goalkeeper_ids}
        active_profiled = {pid for pid in active_all if (team_id, pid) in profiled_players}
        if len(active_profiled) < 1:
            continue

        windows = window_groups.get((seg.season, team_id))
        if windows is None or windows.empty:
            continue

        match_date = pd.to_datetime(seg.match_date)
        available = set(
            windows.loc[
                windows["first_date"].le(match_date) & windows["last_date"].ge(match_date),
                "player_id",
            ].astype(int)
        ) | active_profiled
        available = {
            int(pid)
            for pid in available
            if int(pid) not in goalkeeper_ids and (team_id, int(pid)) in profiled_players
        }
        absent = available - active_profiled

        action_group = action_groups.get((seg.match_id, team_id))
        if action_group is None:
            seg_actions = pd.DataFrame(columns=["player_id", "vaep"])
        else:
            mask = action_group["event_minute"].ge(seg.start_minute) & action_group["event_minute"].lt(seg.end_minute)
            seg_actions = action_group.loc[mask]
            seg_actions = seg_actions[seg_actions["player_id"].isin(active_all)]

        vaep_map = seg_actions.groupby("player_id")["vaep"].sum().to_dict()
        pos_map = (
            seg_actions.assign(pos_vaep=seg_actions["vaep"].clip(lower=0))
            .groupby("player_id")["pos_vaep"]
            .sum()
            .to_dict()
        )
        actions_map = seg_actions["player_id"].value_counts().to_dict()
        active_sorted = sorted(active_profiled)

        for focal_id in active_sorted:
            focal_vaep = float(vaep_map.get(focal_id, 0.0))
            focal_pos_vaep = float(pos_map.get(focal_id, 0.0))
            focal_actions = int(actions_map.get(focal_id, 0))

            for partner_id in active_sorted:
                if partner_id == focal_id:
                    continue
                key = (team_id, seg.season, int(focal_id), int(partner_id), int(seg.match_id))
                row = samples[key]
                partner_vaep = float(vaep_map.get(partner_id, 0.0))
                partner_pos_vaep = float(pos_map.get(partner_id, 0.0))
                partner_actions = int(actions_map.get(partner_id, 0))
                row["minutes_together"] += float(seg.duration)
                row["focal_vaep_together"] += focal_vaep
                row["partner_vaep_together"] += partner_vaep
                row["duo_vaep_together"] += focal_vaep + partner_vaep
                row["duo_pos_vaep_together"] += focal_pos_vaep + partner_pos_vaep
                row["duo_actions_together"] += focal_actions + partner_actions

            if not absent:
                continue
            for absent_partner_id in absent:
                proxy_id, proxy_anchor_id, proxy_method, similarity = choose_actual_role_proxy(
                    seg,
                    team_id,
                    focal_id,
                    absent_partner_id,
                    active_all,
                    similarity_lookup,
                    player_position_lookup,
                    goalkeeper_ids,
                )
                if proxy_id is None or proxy_anchor_id is None or proxy_method is None or not np.isfinite(similarity):
                    continue
                key = (team_id, seg.season, int(focal_id), int(absent_partner_id), int(seg.match_id))
                row = samples[key]
                proxy_vaep = float(vaep_map.get(proxy_id, 0.0))
                proxy_pos_vaep = float(pos_map.get(proxy_id, 0.0))
                proxy_actions = int(actions_map.get(proxy_id, 0))
                row["minutes_proxy"] += float(seg.duration)
                row["focal_vaep_proxy"] += focal_vaep
                row["proxy_vaep"] += proxy_vaep
                row["proxy_pos_vaep"] += proxy_pos_vaep
                row["proxy_duo_vaep"] += focal_vaep + proxy_vaep
                row["proxy_duo_pos_vaep"] += focal_pos_vaep + proxy_pos_vaep
                row["proxy_duo_actions"] += focal_actions + proxy_actions
                row["proxy_similarity_minutes"] += float(similarity) * float(seg.duration)
                row["proxy_segments"] += 1
                if proxy_method == "same_substitution_slot":
                    row["same_slot_proxy_minutes"] += float(seg.duration)
                elif proxy_method == "starter_similarity_slot":
                    row["starter_similarity_proxy_minutes"] += float(seg.duration)
                choice_key = (
                    team_id,
                    int(focal_id),
                    int(absent_partner_id),
                    int(proxy_id),
                    int(proxy_anchor_id),
                    proxy_method,
                )
                proxy_minutes[choice_key] += float(seg.duration)
                proxy_similarity_minutes[choice_key] += float(similarity) * float(seg.duration)
                per_match_proxy_minutes[
                    (team_id, int(focal_id), int(absent_partner_id), int(seg.match_id))
                ][int(proxy_id)] += float(seg.duration)

    rows = []
    for key, values in samples.items():
        row = {
            "team_id": key[0],
            "season": key[1],
            "focal_player_id": key[2],
            "partner_player_id": key[3],
            "match_id": key[4],
        }
        row.update(values)
        rows.append(row)
    samples_df = pd.DataFrame(rows)

    choice_rows = []
    for key, minutes in proxy_minutes.items():
        sim_minutes = proxy_similarity_minutes[key]
        choice_rows.append(
            {
                "team_id": key[0],
                "focal_player_id": key[1],
                "partner_player_id": key[2],
                "proxy_player_id": key[3],
                "proxy_anchor_player_id": key[4],
                "proxy_method": key[5],
                "proxy_minutes": minutes,
                "mean_role_similarity": sim_minutes / minutes if minutes > 0 else np.nan,
            }
        )
    proxy_choices = pd.DataFrame(choice_rows)

    # Per-match dominant proxy: for each (focal, absent_partner, match), the proxy player
    # that accumulated the most minutes in that match.
    pm_proxy_rows = []
    for (tid, fid, pid, mid), proxy_min_map in per_match_proxy_minutes.items():
        dominant = max(proxy_min_map, key=proxy_min_map.get)
        pm_proxy_rows.append(
            {
                "team_id": tid,
                "focal_id": fid,
                "partner_id": pid,
                "match_id": mid,
                "proxy_id": dominant,
                "proxy_minutes": proxy_min_map[dominant],
            }
        )
    per_match_proxy_df = pd.DataFrame(pm_proxy_rows)

    return samples_df, proxy_choices, per_match_proxy_df


def add_role_adjusted_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    out = samples.copy()
    out["duo_vaep90_together"] = 90 * out["duo_vaep_together"] / out["minutes_together"].replace(0, np.nan)
    out["duo_pos_vaep90_together"] = 90 * out["duo_pos_vaep_together"] / out["minutes_together"].replace(0, np.nan)
    out["focal_vaep90_together"] = 90 * out["focal_vaep_together"] / out["minutes_together"].replace(0, np.nan)
    out["partner_vaep90_together"] = 90 * out["partner_vaep_together"] / out["minutes_together"].replace(0, np.nan)
    out["proxy_duo_vaep90"] = 90 * out["proxy_duo_vaep"] / out["minutes_proxy"].replace(0, np.nan)
    out["proxy_duo_pos_vaep90"] = (
        90 * out["proxy_duo_pos_vaep"] / out["minutes_proxy"].replace(0, np.nan)
    )
    out["focal_vaep90_proxy"] = 90 * out["focal_vaep_proxy"] / out["minutes_proxy"].replace(0, np.nan)
    out["proxy_vaep90"] = 90 * out["proxy_vaep"] / out["minutes_proxy"].replace(0, np.nan)
    out["mean_proxy_similarity"] = out["proxy_similarity_minutes"] / out["minutes_proxy"].replace(0, np.nan)
    out["same_slot_proxy_share"] = out["same_slot_proxy_minutes"] / out["minutes_proxy"].replace(0, np.nan)
    out["starter_similarity_proxy_share"] = (
        out["starter_similarity_proxy_minutes"] / out["minutes_proxy"].replace(0, np.nan)
    )
    out["synergy_score"] = out["duo_vaep90_together"] - out["proxy_duo_vaep90"]
    out["positive_synergy_score"] = out["duo_pos_vaep90_together"] - out["proxy_duo_pos_vaep90"]
    return out


def summarize_directed_synergy(
    samples_df: pd.DataFrame,
    proxy_choices: pd.DataFrame,
    player_lookup: pd.DataFrame,
    player_quality: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    agg = samples_df.groupby(["team_id", "focal_player_id", "partner_player_id"], as_index=False).agg(
        minutes_together=("minutes_together", "sum"),
        focal_vaep_together=("focal_vaep_together", "sum"),
        partner_vaep_together=("partner_vaep_together", "sum"),
        duo_vaep_together=("duo_vaep_together", "sum"),
        duo_pos_vaep_together=("duo_pos_vaep_together", "sum"),
        duo_actions_together=("duo_actions_together", "sum"),
        minutes_proxy=("minutes_proxy", "sum"),
        focal_vaep_proxy=("focal_vaep_proxy", "sum"),
        proxy_vaep=("proxy_vaep", "sum"),
        proxy_pos_vaep=("proxy_pos_vaep", "sum"),
        proxy_duo_vaep=("proxy_duo_vaep", "sum"),
        proxy_duo_pos_vaep=("proxy_duo_pos_vaep", "sum"),
        proxy_duo_actions=("proxy_duo_actions", "sum"),
        proxy_similarity_minutes=("proxy_similarity_minutes", "sum"),
        proxy_segments=("proxy_segments", "sum"),
        same_slot_proxy_minutes=("same_slot_proxy_minutes", "sum"),
        starter_similarity_proxy_minutes=("starter_similarity_proxy_minutes", "sum"),
        matches_together=("minutes_together", lambda x: int((x > 0).sum())),
        matches_proxy=("minutes_proxy", lambda x: int((x > 0).sum())),
    )
    directed = add_role_adjusted_metrics(agg)
    directed = directed.loc[
        directed["minutes_together"].ge(config.min_together_minutes)
        & directed["minutes_proxy"].ge(config.min_replacement_minutes)
        & directed["duo_actions_together"].ge(config.min_actions_condition)
        & directed["proxy_duo_actions"].ge(config.min_actions_condition)
        & directed["mean_proxy_similarity"].ge(config.min_mean_replacement_similarity)
        & directed["duo_vaep90_together"].notna()
        & directed["proxy_duo_vaep90"].notna()
    ].copy()

    top_choice = (
        proxy_choices.sort_values("proxy_minutes", ascending=False)
        .drop_duplicates(["team_id", "focal_player_id", "partner_player_id"])
        .rename(
            columns={
                "proxy_player_id": "main_proxy_player_id",
                "proxy_anchor_player_id": "main_proxy_anchor_player_id",
                "proxy_method": "main_proxy_method",
                "proxy_minutes": "main_proxy_minutes",
                "mean_role_similarity": "main_proxy_similarity",
            }
        )
    )
    directed = directed.merge(
        top_choice[
            [
                "team_id",
                "focal_player_id",
                "partner_player_id",
                "main_proxy_player_id",
                "main_proxy_anchor_player_id",
                "main_proxy_method",
                "main_proxy_minutes",
                "main_proxy_similarity",
            ]
        ],
        on=["team_id", "focal_player_id", "partner_player_id"],
        how="left",
    )

    q_map = player_quality["vaep_per90"].to_dict()
    directed["focal_vaep90_total"] = directed["focal_player_id"].map(q_map)
    directed["partner_vaep90_total"] = directed["partner_player_id"].map(q_map)
    directed["main_proxy_vaep90_total"] = directed["main_proxy_player_id"].map(q_map)

    focal_names = player_lookup.rename(columns={"player_id": "focal_player_id", "player": "focal_player", "team": "team"})[
        ["team_id", "focal_player_id", "focal_player", "team"]
    ]
    partner_names = player_lookup.rename(columns={"player_id": "partner_player_id", "player": "partner_player"})[
        ["team_id", "partner_player_id", "partner_player"]
    ]
    proxy_names = player_lookup.rename(
        columns={"player_id": "main_proxy_player_id", "player": "main_proxy_player"}
    )[["team_id", "main_proxy_player_id", "main_proxy_player"]]
    proxy_anchor_names = player_lookup.rename(
        columns={"player_id": "main_proxy_anchor_player_id", "player": "main_proxy_anchor_player"}
    )[["team_id", "main_proxy_anchor_player_id", "main_proxy_anchor_player"]]

    directed = directed.merge(focal_names, on=["team_id", "focal_player_id"], how="left")
    directed = directed.merge(partner_names, on=["team_id", "partner_player_id"], how="left")
    directed = directed.merge(proxy_names, on=["team_id", "main_proxy_player_id"], how="left")
    directed = directed.merge(proxy_anchor_names, on=["team_id", "main_proxy_anchor_player_id"], how="left")
    return directed.sort_values("synergy_score", ascending=False).reset_index(drop=True)


def score_from_directed_sample(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1)
    minutes_together = totals[:, 0]
    duo_vaep_together = totals[:, 1]
    minutes_proxy = totals[:, 2]
    proxy_duo_vaep = totals[:, 3]
    valid = (minutes_together > 0) & (minutes_proxy > 0)
    scores = np.full(len(totals), np.nan)
    scores[valid] = (
        90 * duo_vaep_together[valid] / minutes_together[valid]
        - 90 * proxy_duo_vaep[valid] / minutes_proxy[valid]
    )
    return scores


def bootstrap_directed_score(
    samples_df: pd.DataFrame,
    directed_df: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0 or directed_df.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    boot_cols = ["minutes_together", "duo_vaep_together", "minutes_proxy", "proxy_duo_vaep"]
    grouped = {
        key: grp[boot_cols].to_numpy(float)
        for key, grp in samples_df.groupby(["team_id", "focal_player_id", "partner_player_id"], sort=False)
    }
    rows = []

    for row in directed_df.itertuples(index=False):
        values = grouped.get((row.team_id, row.focal_player_id, row.partner_player_id))
        if values is None or len(values) == 0:
            continue
        idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        boot_values = values[idx]
        boot = score_from_directed_sample(boot_values)
        boot = boot[np.isfinite(boot)]
        if len(boot) == 0:
            continue
        p_value = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
        rows.append(
            {
                "team_id": row.team_id,
                "focal_player_id": row.focal_player_id,
                "partner_player_id": row.partner_player_id,
                "synergy_score_ci_low": float(np.quantile(boot, 0.025)),
                "synergy_score_ci_high": float(np.quantile(boot, 0.975)),
                "p_value": min(float(p_value), 1.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_pairs(directed: pd.DataFrame) -> pd.DataFrame:
    if directed.empty:
        return pd.DataFrame()
    rows = []
    for row in directed.itertuples(index=False):
        p1, p2 = sorted((int(row.focal_player_id), int(row.partner_player_id)))
        perspective = "score_1_to_2" if int(row.focal_player_id) == p1 else "score_2_to_1"
        rows.append(
            {
                "team_id": int(row.team_id),
                "player_1_id": p1,
                "player_2_id": p2,
                perspective: float(row.synergy_score),
                f"{perspective}_p_value": float(row.p_value) if pd.notna(row.p_value) else np.nan,
                f"{perspective}_minutes_together": float(row.minutes_together),
                f"{perspective}_minutes_proxy": float(row.minutes_proxy),
            }
        )
    wide = pd.DataFrame(rows).groupby(["team_id", "player_1_id", "player_2_id"], as_index=False).first()
    score_cols = [col for col in ["score_1_to_2", "score_2_to_1"] if col in wide]
    wide["synergy_score"] = wide[score_cols].mean(axis=1)
    wide["n_directions"] = wide[score_cols].notna().sum(axis=1)
    wide["min_direction_score"] = wide[score_cols].min(axis=1)
    wide["max_direction_score"] = wide[score_cols].max(axis=1)
    p_cols = [col for col in ["score_1_to_2_p_value", "score_2_to_1_p_value"] if col in wide]
    if p_cols:
        wide["min_p_value"] = wide[p_cols].min(axis=1)
        wide["max_p_value"] = wide[p_cols].max(axis=1)
    return wide.sort_values("synergy_score", ascending=False).reset_index(drop=True)


def add_pair_names(pairs: pd.DataFrame, player_lookup: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    name_1 = player_lookup.rename(columns={"player_id": "player_1_id", "player": "player_1", "team": "team"})[
        ["team_id", "player_1_id", "player_1", "team"]
    ]
    name_2 = player_lookup.rename(columns={"player_id": "player_2_id", "player": "player_2"})[
        ["team_id", "player_2_id", "player_2"]
    ]
    return pairs.merge(name_1, on=["team_id", "player_1_id"], how="left").merge(
        name_2, on=["team_id", "player_2_id"], how="left"
    )


def summarize_player_synergy(directed_significant: pd.DataFrame, directed_all: pd.DataFrame) -> pd.DataFrame:
    base = directed_significant if not directed_significant.empty else directed_all
    if base.empty:
        return pd.DataFrame()
    best_partner = (
        base.sort_values("synergy_score", ascending=False)
        .drop_duplicates("focal_player_id")
        [
            [
                "focal_player_id",
                "partner_player",
                "synergy_score",
                "main_proxy_player",
                "main_proxy_method",
                "main_proxy_similarity",
            ]
        ]
        .rename(
            columns={
                "focal_player_id": "player_id",
                "partner_player": "best_partner",
                "synergy_score": "best_synergy_score",
                "main_proxy_player": "best_actual_proxy_player",
                "main_proxy_method": "best_proxy_method",
                "main_proxy_similarity": "best_proxy_similarity",
            }
        )
    )
    player_synergy = (
        base.groupby(["team_id", "team", "focal_player_id", "focal_player"], as_index=False)
        .agg(
            avg_synergy_score=("synergy_score", "mean"),
            max_synergy_score=("synergy_score", "max"),
            min_synergy_score=("synergy_score", "min"),
            n_partners=("partner_player_id", "count"),
            total_minutes_together=("minutes_together", "sum"),
            total_minutes_proxy=("minutes_proxy", "sum"),
            avg_proxy_similarity=("mean_proxy_similarity", "mean"),
            avg_same_slot_proxy_share=("same_slot_proxy_share", "mean"),
            avg_starter_similarity_proxy_share=("starter_similarity_proxy_share", "mean"),
        )
        .rename(columns={"focal_player_id": "player_id", "focal_player": "player"})
        .merge(best_partner, on="player_id", how="left")
        .sort_values("avg_synergy_score", ascending=False)
        .reset_index(drop=True)
    )
    return player_synergy


def build_chain_synergy(
    events_df: pd.DataFrame,
    vaep_actions: pd.DataFrame,
    per_match_proxy_df: pd.DataFrame,
    player_lookup: pd.DataFrame,
    goalkeeper_ids: set[int],
    config: PipelineConfig,
) -> pd.DataFrame:
    """Proxy-based possession-level (chain) synergy.

    For each pair (A, B) on the same team:
      - With-B possessions: possessions in B-present matches where BOTH A and B acted.
      - With-proxy possessions: possessions in B-absent matches where A and B's designated
        proxy (same substitution slot, or most spatially similar starter) both acted.

    chain_boost(A→B) = vaep_per_action_A_with_B  −  vaep_per_action_A_with_proxy
    chain_synergy(A,B) = (chain_boost(A→B) + chain_boost(B→A)) / 2

    Uses the same proxy selection logic as the segment-level synergy score, now applied
    at possession granularity. Requires per_match_proxy_df from aggregate_role_adjusted_samples.
    """
    min_duo = config.min_chain_duo_possessions

    if per_match_proxy_df.empty:
        return pd.DataFrame()

    # ── 1. Attach possession to vaep_actions ────────────────────────────────
    event_poss = (
        events_df[["match_id", "index", "possession", "possession_team_id"]]
        .dropna(subset=["possession", "possession_team_id"])
        .assign(
            possession=lambda d: d["possession"].astype(int),
            possession_team_id=lambda d: d["possession_team_id"].astype(int),
        )
        .drop_duplicates(["match_id", "index"])
    )
    vaep_p = (
        vaep_actions.loc[vaep_actions["player_id"].notna()]
        .assign(
            player_id=lambda d: d["player_id"].astype(int),
            team_id=lambda d: d["team_id"].astype(int),
        )
        .merge(event_poss, on=["match_id", "index"], how="left")
        .dropna(subset=["possession", "possession_team_id"])
    )
    vaep_p = vaep_p[
        ~vaep_p["player_id"].isin(goalkeeper_ids)
        & vaep_p["team_id"].eq(vaep_p["possession_team_id"].astype(int))
    ].copy()
    vaep_p["possession"] = vaep_p["possession"].astype(int)

    # ── 2. Per (match, possession, player) stats ────────────────────────────
    poss_stats = (
        vaep_p.groupby(["match_id", "possession_team_id", "possession", "player_id"], sort=False)
        .agg(vaep=("vaep", "sum"), n_actions=("vaep", "count"))
        .reset_index()
        .rename(columns={"possession_team_id": "team_id"})
    )

    # ── 3. With-B possessions (B-present matches) ───────────────────────────
    # B can only appear in poss_stats for matches where B actually played,
    # so this naturally restricts to B-present matches.
    poss_focal = poss_stats.rename(columns={"player_id": "focal_id", "vaep": "focal_vaep", "n_actions": "focal_n_actions"})
    poss_partner_presence = poss_stats[["match_id", "team_id", "possession", "player_id"]].rename(
        columns={"player_id": "partner_id"}
    )

    duo = (
        poss_focal.merge(poss_partner_presence, on=["match_id", "team_id", "possession"])
        .pipe(lambda d: d[d["focal_id"].ne(d["partner_id"])])
    )
    duo_agg = (
        duo.groupby(["team_id", "focal_id", "partner_id"], sort=False)
        .agg(
            duo_vaep=("focal_vaep", "sum"),
            duo_n_actions=("focal_n_actions", "sum"),
            n_duo_possessions=("possession", "count"),
            n_duo_matches=("match_id", "nunique"),
        )
        .reset_index()
    )

    # ── 4. With-proxy possessions (B-absent matches) ─────────────────────────
    # per_match_proxy_df: (team_id, focal_id, partner_id, match_id, proxy_id)
    pm = per_match_proxy_df.copy()
    pm["team_id"] = pm["team_id"].astype(int)
    pm["focal_id"] = pm["focal_id"].astype(int)
    pm["partner_id"] = pm["partner_id"].astype(int)
    pm["match_id"] = pm["match_id"].astype(int)
    pm["proxy_id"] = pm["proxy_id"].astype(int)

    # Focal's possession stats in proxy matches
    poss_focal_proxy = poss_stats.rename(columns={
        "player_id": "focal_id", "vaep": "focal_vaep", "n_actions": "focal_n_actions"
    })
    # Proxy's presence in possessions (just need match/possession/proxy_id)
    poss_proxy_presence = poss_stats[["match_id", "team_id", "possession", "player_id"]].rename(
        columns={"player_id": "proxy_id"}
    )

    proxy_poss = (
        pm[["team_id", "focal_id", "partner_id", "match_id", "proxy_id"]]
        .merge(
            poss_focal_proxy[["match_id", "team_id", "focal_id", "possession", "focal_vaep", "focal_n_actions"]],
            on=["match_id", "team_id", "focal_id"],
        )
        .merge(
            poss_proxy_presence,
            on=["match_id", "team_id", "possession", "proxy_id"],
        )
    )
    proxy_agg = (
        proxy_poss.groupby(["team_id", "focal_id", "partner_id"], sort=False)
        .agg(
            proxy_vaep=("focal_vaep", "sum"),
            proxy_n_actions=("focal_n_actions", "sum"),
            n_proxy_possessions=("possession", "count"),
            n_proxy_matches=("match_id", "nunique"),
        )
        .reset_index()
    )

    # ── 5. Combine and compute chain boost ──────────────────────────────────
    result = duo_agg.merge(proxy_agg, on=["team_id", "focal_id", "partner_id"], how="inner")

    result["vaep_per_action_duo"] = result["duo_vaep"] / result["duo_n_actions"].replace(0.0, np.nan)
    result["vaep_per_action_proxy"] = result["proxy_vaep"] / result["proxy_n_actions"].replace(0.0, np.nan)
    result["chain_boost"] = result["vaep_per_action_duo"] - result["vaep_per_action_proxy"]

    # ── 6. Filter minimum data ───────────────────────────────────────────────
    result = result.loc[
        result["n_duo_possessions"].ge(min_duo)
        & result["n_proxy_possessions"].ge(10)
        & result["duo_n_actions"].ge(20)
        & result["proxy_n_actions"].ge(10)
        & result["vaep_per_action_duo"].notna()
        & result["vaep_per_action_proxy"].notna()
    ].copy()

    # ── 7. Symmetric chain synergy ───────────────────────────────────────────
    reverse = result[["team_id", "focal_id", "partner_id", "chain_boost"]].rename(
        columns={"focal_id": "partner_id", "partner_id": "focal_id", "chain_boost": "partner_chain_boost"}
    )
    result = result.merge(reverse, on=["team_id", "focal_id", "partner_id"], how="left")
    result["chain_synergy"] = (result["chain_boost"] + result["partner_chain_boost"]) / 2

    # ── 8. Add names ─────────────────────────────────────────────────────────
    focal_names = player_lookup.rename(
        columns={"player_id": "focal_id", "player": "focal_player", "team": "team"}
    )[["team_id", "focal_id", "focal_player", "team"]]
    partner_names = player_lookup.rename(
        columns={"player_id": "partner_id", "player": "partner_player"}
    )[["team_id", "partner_id", "partner_player"]]

    result = (
        result.merge(focal_names, on=["team_id", "focal_id"], how="left")
        .merge(partner_names, on=["team_id", "partner_id"], how="left")
        .sort_values("chain_synergy", ascending=False)
        .reset_index(drop=True)
    )
    return result


def plot_chain_synergy(chain: pd.DataFrame, config: PipelineConfig) -> None:
    if chain.empty:
        return
    top20 = chain.head(20).copy()
    top20["label"] = (
        top20["focal_player"].fillna("").str.split().str[-1]
        + " + "
        + top20["partner_player"].fillna("").str.split().str[-1]
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    colors = ["#1b9e77" if v >= 0 else "#d95f02" for v in top20["chain_synergy"]]
    axes[0].barh(top20["label"][::-1], top20["chain_synergy"][::-1], color=colors[::-1])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("VAEP/ação (com B) − VAEP/ação (com proxy de B)  [médio das duas direções]")
    axes[0].set_title("Top-20 Chain Synergy (possessions com parceiro vs proxy)")

    top10 = top20.head(10)
    x = np.arange(len(top10))
    width = 0.35
    axes[1].bar(x - width / 2, top10["vaep_per_action_duo"], width, label="A com B na posse", color="#1b9e77")
    axes[1].bar(x + width / 2, top10["vaep_per_action_proxy"], width, label="A com proxy de B na posse", color="#7570b3")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(top10["label"], rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("VAEP por ação")
    axes[1].set_title("Com parceiro real vs. proxy na mesma posse")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(config.viz_path / "chain_synergy_top20.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_within_match_synergy(
    segments_df: pd.DataFrame,
    vaep_actions: pd.DataFrame,
    player_lookup: pd.DataFrame,
    goalkeeper_ids: set[int],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within-match comparison: A's VAEP/90 when B is on pitch vs when B is off pitch.

    Only uses matches where A and B share at least `min_within_match_window_minutes`
    together AND A has at least the same amount of time without B in the same match.
    Returns (match_samples, aggregated_df).
    """
    min_win = config.min_within_match_window_minutes
    action_base = (
        vaep_actions.loc[
            vaep_actions["player_id"].notna(),
            ["match_id", "team_id", "event_minute", "player_id", "vaep"],
        ]
        .copy()
        .assign(
            team_id=lambda df: df["team_id"].astype(int),
            player_id=lambda df: df["player_id"].astype(int),
        )
    )
    action_base = action_base[~action_base["player_id"].isin(goalkeeper_ids)]
    action_groups = {
        key: grp.sort_values("event_minute")[["event_minute", "player_id", "vaep"]]
        for key, grp in action_base.groupby(["match_id", "team_id"], sort=False)
    }

    together: dict[tuple, dict] = defaultdict(lambda: {"minutes": 0.0, "vaep": 0.0})
    focal_only: dict[tuple, dict] = defaultdict(lambda: {"minutes": 0.0, "vaep": 0.0})

    for seg in segments_df.itertuples(index=False):
        team_id = int(seg.team_id)
        active_all = {int(pid) for pid in seg.active if int(pid) not in goalkeeper_ids}
        match_participants_no_gk = {
            int(pid) for pid in seg.match_participants if int(pid) not in goalkeeper_ids
        }
        if not active_all or len(match_participants_no_gk) < 2:
            continue

        action_group = action_groups.get((seg.match_id, team_id))
        if action_group is not None:
            mask = action_group["event_minute"].ge(seg.start_minute) & action_group["event_minute"].lt(
                seg.end_minute
            )
            seg_actions = action_group.loc[mask]
            seg_actions = seg_actions[seg_actions["player_id"].isin(active_all)]
            vaep_map = seg_actions.groupby("player_id")["vaep"].sum().to_dict()
        else:
            vaep_map = {}

        for focal_id in active_all:
            focal_vaep = float(vaep_map.get(focal_id, 0.0))
            for partner_id in match_participants_no_gk:
                if partner_id == focal_id:
                    continue
                key = (team_id, int(seg.match_id), focal_id, partner_id)
                if partner_id in active_all:
                    together[key]["minutes"] += float(seg.duration)
                    together[key]["vaep"] += focal_vaep
                else:
                    focal_only[key]["minutes"] += float(seg.duration)
                    focal_only[key]["vaep"] += focal_vaep

    sample_rows = []
    for key in set(together.keys()) & set(focal_only.keys()):
        team_id, match_id, focal_id, partner_id = key
        t = together[key]
        f = focal_only[key]
        if t["minutes"] < min_win or f["minutes"] < min_win:
            continue
        vaep90_tog = 90.0 * t["vaep"] / t["minutes"]
        vaep90_foc = 90.0 * f["vaep"] / f["minutes"]
        sample_rows.append(
            {
                "team_id": team_id,
                "match_id": match_id,
                "focal_player_id": focal_id,
                "partner_player_id": partner_id,
                "vaep90_with_partner": vaep90_tog,
                "vaep90_without_partner": vaep90_foc,
                "within_match_synergy": vaep90_tog - vaep90_foc,
                "minutes_together": t["minutes"],
                "minutes_without": f["minutes"],
            }
        )

    if not sample_rows:
        return pd.DataFrame(), pd.DataFrame()

    match_samples = pd.DataFrame(sample_rows)

    agg_rows = []
    for (team_id, focal_id, partner_id), grp in match_samples.groupby(
        ["team_id", "focal_player_id", "partner_player_id"], sort=False
    ):
        w = grp["minutes_together"].to_numpy(float)
        w_sum = w.sum()
        if w_sum <= 0:
            continue
        agg_rows.append(
            {
                "team_id": team_id,
                "focal_player_id": focal_id,
                "partner_player_id": partner_id,
                "within_match_synergy": float(np.average(grp["within_match_synergy"], weights=w)),
                "vaep90_with_partner_mean": float(np.average(grp["vaep90_with_partner"], weights=w)),
                "vaep90_without_partner_mean": float(
                    np.average(grp["vaep90_without_partner"], weights=w)
                ),
                "minutes_together_total": float(w_sum),
                "minutes_without_total": float(grp["minutes_without"].sum()),
                "n_match_events": int(len(grp)),
            }
        )

    if not agg_rows:
        return match_samples, pd.DataFrame()

    agg = pd.DataFrame(agg_rows)
    focal_names = player_lookup.rename(
        columns={"player_id": "focal_player_id", "player": "focal_player", "team": "team"}
    )[["team_id", "focal_player_id", "focal_player", "team"]]
    partner_names = player_lookup.rename(
        columns={"player_id": "partner_player_id", "player": "partner_player"}
    )[["team_id", "partner_player_id", "partner_player"]]
    agg = (
        agg.merge(focal_names, on=["team_id", "focal_player_id"], how="left")
        .merge(partner_names, on=["team_id", "partner_player_id"], how="left")
        .sort_values("within_match_synergy", ascending=False)
        .reset_index(drop=True)
    )
    return match_samples, agg


def bootstrap_within_match_synergy(
    match_samples: pd.DataFrame,
    agg_df: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    """Bootstrap CI and p-values for within-match synergy (resampling match events)."""
    if n_bootstrap <= 0 or agg_df.empty or match_samples.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    grouped = {
        key: grp["within_match_synergy"].to_numpy(float)
        for key, grp in match_samples.groupby(
            ["team_id", "focal_player_id", "partner_player_id"], sort=False
        )
    }
    rows = []
    for row in agg_df.itertuples(index=False):
        values = grouped.get((row.team_id, row.focal_player_id, row.partner_player_id))
        if values is None or len(values) < 2:
            continue
        idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        boot = values[idx].mean(axis=1)
        p_value = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
        rows.append(
            {
                "team_id": row.team_id,
                "focal_player_id": row.focal_player_id,
                "partner_player_id": row.partner_player_id,
                "within_match_synergy_ci_low": float(np.quantile(boot, 0.025)),
                "within_match_synergy_ci_high": float(np.quantile(boot, 0.975)),
                "within_match_p_value": min(float(p_value), 1.0),
            }
        )
    return pd.DataFrame(rows)


def plot_focal_baseline_synergy(directed: pd.DataFrame, config: PipelineConfig) -> None:
    """Top-20 pairs by focal_baseline_synergy: A's VAEP/90 with B minus A's overall baseline."""
    col = "focal_baseline_synergy"
    if directed.empty or col not in directed.columns:
        return
    top20 = directed.nlargest(20, col).copy()
    top20["label"] = (
        top20["focal_player"].fillna("").str.split().str[-1]
        + " -> "
        + top20["partner_player"].fillna("").str.split().str[-1]
    )
    top10 = top20.head(10)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    colors = ["#1b9e77" if v >= 0 else "#d95f02" for v in top20[col]]
    axes[0].barh(top20["label"][::-1], top20[col][::-1], color=colors[::-1])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("focal_vaep90_with_B  -  focal_vaep90_baseline")
    axes[0].set_title("Top-20 relacoes direcionais — Focal Baseline Synergy")

    x = np.arange(len(top10))
    width = 0.35
    axes[1].bar(x - width / 2, top10["focal_vaep90_together"], width, label="VAEP90 focal com B", color="#1b9e77")
    axes[1].bar(x + width / 2, top10["focal_vaep90_total"], width, label="Baseline focal (todas partidas)", color="#7570b3")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(top10["label"], rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("VAEP/90 do jogador focal")
    axes[1].set_title("Com B vs baseline")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(config.viz_path / "focal_baseline_synergy_top20.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_within_match_synergy(
    agg: pd.DataFrame, match_samples: pd.DataFrame, config: PipelineConfig
) -> None:
    """Top-20 pairs by within-match synergy with CI bars and scatter for top player."""
    if agg.empty:
        return
    sig = agg[agg.get("within_match_p_value", pd.Series(dtype=float)).lt(0.05)] if "within_match_p_value" in agg.columns else agg
    top20 = (sig if not sig.empty else agg).head(20).copy()
    top20["label"] = (
        top20["focal_player"].fillna("").str.split().str[-1]
        + " -> "
        + top20["partner_player"].fillna("").str.split().str[-1]
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    colors = ["#1b9e77" if v >= 0 else "#d95f02" for v in top20["within_match_synergy"]]
    y_pos = range(len(top20))
    axes[0].barh(list(y_pos), top20["within_match_synergy"][::-1].tolist(), color=colors[::-1])
    if "within_match_synergy_ci_low" in top20.columns:
        xerr_low = (top20["within_match_synergy"] - top20["within_match_synergy_ci_low"]).clip(lower=0)
        xerr_high = (top20["within_match_synergy_ci_high"] - top20["within_match_synergy"]).clip(lower=0)
        axes[0].errorbar(
            top20["within_match_synergy"][::-1].tolist(),
            list(y_pos),
            xerr=[xerr_low[::-1].tolist(), xerr_high[::-1].tolist()],
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1,
        )
    axes[0].set_yticks(list(y_pos))
    axes[0].set_yticklabels(top20["label"][::-1].tolist(), fontsize=8)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("VAEP90_focal_com_B  -  VAEP90_focal_sem_B  (mesma partida)")
    axes[0].set_title("Top-20 Within-Match Synergy")

    if not match_samples.empty and len(top20) > 0:
        best_row = top20.iloc[0]
        pair_data = match_samples[
            match_samples["focal_player_id"].eq(best_row["focal_player_id"])
            & match_samples["partner_player_id"].eq(best_row["partner_player_id"])
            & match_samples["team_id"].eq(best_row["team_id"])
        ]
        axes[1].scatter(
            pair_data["vaep90_without_partner"],
            pair_data["vaep90_with_partner"],
            alpha=0.7,
            color="#1b9e77",
            s=60,
        )
        lim = [
            min(pair_data["vaep90_without_partner"].min(), pair_data["vaep90_with_partner"].min()) - 0.1,
            max(pair_data["vaep90_without_partner"].max(), pair_data["vaep90_with_partner"].max()) + 0.1,
        ]
        axes[1].plot(lim, lim, "k--", linewidth=0.8, label="sem diferenca")
        axes[1].set_xlabel(f"VAEP/90 {best_row.get('focal_player','').split()[-1]} SEM {best_row.get('partner_player','').split()[-1]}")
        axes[1].set_ylabel(f"VAEP/90 {best_row.get('focal_player','').split()[-1]} COM {best_row.get('partner_player','').split()[-1]}")
        axes[1].set_title(f"Por partida: {best_row.get('focal_player','').split()[-1]} -> {best_row.get('partner_player','').split()[-1]}")
        axes[1].legend()
    plt.tight_layout()
    fig.savefig(config.viz_path / "within_match_synergy_top20.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_top_directed_synergies(directed: pd.DataFrame, config: PipelineConfig) -> None:
    if directed.empty:
        return
    top20 = directed.head(20).copy()
    top20["label"] = (
        top20["focal_player"].fillna("").str.split().str[-1]
        + " + "
        + top20["partner_player"].fillna("").str.split().str[-1]
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    colors = ["#1b9e77" if value >= 0 else "#d95f02" for value in top20["synergy_score"]]
    axes[0].barh(top20["label"][::-1], top20["synergy_score"][::-1], color=colors[::-1])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Score de sinergia role-adjusted (VAEP/90)")
    axes[0].set_title("Top-20 relacoes direcionais")

    top10 = top20.head(10)
    x = np.arange(len(top10))
    width = 0.35
    axes[1].bar(x - width / 2, top10["duo_vaep90_together"], width, label="A + B juntos", color="#1b9e77")
    axes[1].bar(
        x + width / 2,
        top10["proxy_duo_vaep90"],
        width,
        label="A + proxy real de B",
        color="#7570b3",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(top10["label"], rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("VAEP/90 da dupla contextual")
    axes[1].set_title("Sem B: slot real ou titular similar")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(config.viz_path / "top20_role_adjusted_synergy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_player_synergy(player_synergy: pd.DataFrame, directed: pd.DataFrame, config: PipelineConfig) -> None:
    if player_synergy.empty or directed.empty:
        return
    top20 = player_synergy.head(20).copy()
    top20["label"] = top20["player"].fillna("").str.split().str[-1] + " (" + top20["team"].fillna("") + ")"
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    colors = ["#1b9e77" if value >= 0 else "#d95f02" for value in top20["avg_synergy_score"]]
    axes[0].barh(top20["label"][::-1], top20["avg_synergy_score"][::-1], color=colors[::-1])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Score medio role-adjusted (VAEP/90)")
    axes[0].set_title("Top-20 jogadores")

    focus_id = int(player_synergy.iloc[0]["player_id"])
    focus_name = str(player_synergy.iloc[0]["player"])
    focus_pairs = directed[directed["focal_player_id"].eq(focus_id)].sort_values("synergy_score", ascending=False).head(15)
    bar_colors = ["#1b9e77" if value >= 0 else "#d95f02" for value in focus_pairs["synergy_score"]]
    axes[1].barh(focus_pairs["partner_player"].fillna("").str.split().str[-1][::-1], focus_pairs["synergy_score"][::-1], color=bar_colors[::-1])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Score direcional (VAEP/90)")
    axes[1].set_title(f"Parceiros de {focus_name.split()[-1]}")
    plt.tight_layout()
    fig.savefig(config.viz_path / "top20_role_adjusted_player_synergy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_synergy_matrix_and_graph(
    directed: pd.DataFrame,
    pairs: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    if directed.empty:
        return
    if config.graph_team_name is None:
        team_scores = (
            directed.groupby(["team_id", "team"], as_index=False)["synergy_score"]
            .apply(lambda x: x.abs().sum())
            .rename(columns={"synergy_score": "abs_score"})
            .sort_values("abs_score", ascending=False)
        )
        team_id = int(team_scores.iloc[0]["team_id"])
        team_name = str(team_scores.iloc[0]["team"])
    else:
        candidates = directed[directed["team"].eq(config.graph_team_name)]
        if candidates.empty:
            return
        team_id = int(candidates.iloc[0]["team_id"])
        team_name = str(config.graph_team_name)

    team_directed = directed[directed["team_id"].eq(team_id)].copy()
    players = sorted(set(team_directed["focal_player"]).union(set(team_directed["partner_player"])))
    matrix = pd.DataFrame(0.0, index=players, columns=players)
    for row in team_directed.itertuples(index=False):
        matrix.loc[row.focal_player, row.partner_player] = row.synergy_score
    matrix.to_csv(config.processed_data_path / f"role_adjusted_synergy_matrix_team_{team_id}.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Score direcional VAEP/90"})
    ax.set_title(f"Matriz role-adjusted - {team_name}")
    ax.set_xlabel("Parceiro")
    ax.set_ylabel("Jogador focal")
    plt.tight_layout()
    fig.savefig(config.viz_path / f"role_adjusted_synergy_matrix_team_{team_id}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if nx is None or pairs.empty:
        return
    team_pairs = pairs[pairs["team_id"].eq(team_id)].head(40)
    if team_pairs.empty:
        return
    graph = nx.Graph(team=team_name)
    for row in team_pairs.itertuples(index=False):
        graph.add_node(row.player_1_id, label=row.player_1)
        graph.add_node(row.player_2_id, label=row.player_2)
        graph.add_edge(row.player_1_id, row.player_2_id, weight=float(row.synergy_score))
    nx.write_gexf(graph, config.processed_data_path / f"role_adjusted_synergy_graph_team_{team_id}.gexf")

    pos = nx.spring_layout(graph, seed=config.random_state, weight="weight")
    edge_weights = np.array([data["weight"] for _, _, data in graph.edges(data=True)])
    widths = 1 + 4 * np.abs(edge_weights) / max(np.abs(edge_weights).max(), 1e-9)
    edge_colors = ["#1b9e77" if weight >= 0 else "#d95f02" for weight in edge_weights]
    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="#e8eef7", edgecolors="#333333", ax=ax)
    nx.draw_networkx_edges(graph, pos, width=widths, edge_color=edge_colors, alpha=0.75, ax=ax)
    labels = {node: data.get("label", str(node)).split()[-1] for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9, ax=ax)
    ax.set_title(f"Grafo role-adjusted - {team_name}")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(config.viz_path / f"role_adjusted_synergy_graph_team_{team_id}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or PipelineConfig()
    events, matches = load_inputs(config)
    vaep_actions = load_or_build_vaep(events, config)
    goalkeeper_ids = extract_goalkeeper_ids(events, vaep_actions)
    player_lookup = build_player_lookup(events)
    position_lookup = build_position_lookup(events)
    segments = build_presence_segments(events, matches)
    lineup_slot_segments = build_lineup_slot_segments(segments)
    lineup_slot_segments.to_parquet(config.processed_data_path / "lineup_slot_segments.parquet", index=False)
    player_match, player_quality = build_player_match(segments, vaep_actions, goalkeeper_ids)
    player_windows = build_player_windows(player_match, goalkeeper_ids)
    role_profiles, similarities = build_role_profiles(
        events, player_lookup, position_lookup, goalkeeper_ids, config
    )
    samples, proxy_choices, per_match_proxy = aggregate_role_adjusted_samples(
        segments, vaep_actions, player_windows, role_profiles, similarities, position_lookup, goalkeeper_ids
    )
    samples.to_parquet(config.processed_data_path / "role_adjusted_context_samples.parquet", index=False)
    proxy_choices.to_parquet(config.processed_data_path / "role_adjusted_proxy_choices.parquet", index=False)
    per_match_proxy.to_parquet(config.processed_data_path / "per_match_proxy.parquet", index=False)

    directed = summarize_directed_synergy(samples, proxy_choices, player_lookup, player_quality, config)
    bootstrap = bootstrap_directed_score(samples, directed, config.n_bootstrap, config.random_state)
    if not bootstrap.empty:
        directed = directed.merge(
            bootstrap, on=["team_id", "focal_player_id", "partner_player_id"], how="left"
        )
    else:
        directed[["synergy_score_ci_low", "synergy_score_ci_high", "p_value"]] = np.nan

    directed = directed.sort_values("synergy_score", ascending=False).reset_index(drop=True)
    directed_significant = directed[directed["p_value"].lt(0.05)].copy()

    # --- Focal Baseline Synergy (derived from existing data, no replacement needed) ---
    # focal_baseline_synergy = VAEP90_A_with_B - VAEP90_A_overall_baseline
    directed["focal_baseline_synergy"] = directed["focal_vaep90_together"] - directed["focal_vaep90_total"]
    directed_significant["focal_baseline_synergy"] = (
        directed_significant["focal_vaep90_together"] - directed_significant["focal_vaep90_total"]
    )

    directed.to_parquet(config.processed_data_path / "directed_synergy_all.parquet", index=False)
    directed_significant.to_parquet(config.processed_data_path / "directed_synergy_significant.parquet", index=False)

    pairs = add_pair_names(summarize_pairs(directed), player_lookup)
    pairs_significant = add_pair_names(summarize_pairs(directed_significant), player_lookup)
    pairs.to_parquet(config.processed_data_path / "pair_synergy_all.parquet", index=False)
    pairs_significant.to_parquet(config.processed_data_path / "pair_synergy_significant.parquet", index=False)

    player_synergy = summarize_player_synergy(directed_significant, directed)
    player_synergy.to_parquet(config.processed_data_path / "player_synergy.parquet", index=False)

    # --- Within-Match Synergy (quasi-experimental: same match, B on vs B off) ---
    within_match_samples, within_match_synergy = build_within_match_synergy(
        segments, vaep_actions, player_lookup, goalkeeper_ids, config
    )
    if not within_match_synergy.empty:
        wm_boot = bootstrap_within_match_synergy(
            within_match_samples, within_match_synergy, config.n_bootstrap, config.random_state
        )
        if not wm_boot.empty:
            within_match_synergy = within_match_synergy.merge(
                wm_boot, on=["team_id", "focal_player_id", "partner_player_id"], how="left"
            )
        within_match_samples.to_parquet(
            config.processed_data_path / "within_match_synergy_samples.parquet", index=False
        )
        within_match_synergy.to_parquet(
            config.processed_data_path / "within_match_synergy.parquet", index=False
        )

    # --- Chain Synergy (proxy-based possession-level) ---
    chain_synergy = build_chain_synergy(
        events, vaep_actions, per_match_proxy, player_lookup, goalkeeper_ids, config
    )
    if not chain_synergy.empty:
        chain_synergy.to_parquet(config.processed_data_path / "chain_synergy.parquet", index=False)

    plot_top_directed_synergies(directed_significant if not directed_significant.empty else directed, config)
    plot_player_synergy(player_synergy, directed_significant if not directed_significant.empty else directed, config)
    plot_synergy_matrix_and_graph(directed_significant if not directed_significant.empty else directed, pairs_significant if not pairs_significant.empty else pairs, config)
    plot_focal_baseline_synergy(directed_significant if not directed_significant.empty else directed, config)
    plot_within_match_synergy(within_match_synergy, within_match_samples, config)
    plot_chain_synergy(chain_synergy, config)

    return {
        "events": events,
        "matches": matches,
        "vaep_actions": vaep_actions,
        "segments": segments,
        "lineup_slot_segments": lineup_slot_segments,
        "player_match": player_match,
        "player_quality": player_quality,
        "role_profiles": role_profiles,
        "role_similarities": similarities,
        "samples": samples,
        "proxy_choices": proxy_choices,
        "directed_synergy": directed,
        "directed_synergy_significant": directed_significant,
        "pair_synergy": pairs,
        "pair_synergy_significant": pairs_significant,
        "player_synergy": player_synergy,
        "within_match_synergy_samples": within_match_samples,
        "within_match_synergy": within_match_synergy,
        "chain_synergy": chain_synergy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Role-adjusted VAEP synergy pipeline.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--recompute-vaep", action="store_true")
    parser.add_argument("--min-together-minutes", type=float, default=300.0)
    parser.add_argument("--min-replacement-minutes", type=float, default=150.0)
    parser.add_argument("--min-role-events", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        n_bootstrap=args.n_bootstrap,
        recompute_vaep=args.recompute_vaep,
        min_together_minutes=args.min_together_minutes,
        min_replacement_minutes=args.min_replacement_minutes,
        min_role_events=args.min_role_events,
    )
    outputs = run_pipeline(config)
    print("Role-adjusted synergy pipeline completed.")
    print(f"Role profiles: {len(outputs['role_profiles'])}")
    print(f"Directed eligible relations: {len(outputs['directed_synergy'])}")
    print(f"Directed significant relations: {len(outputs['directed_synergy_significant'])}")
    if not outputs["directed_synergy"].empty:
        cols = [
            "team",
            "focal_player",
            "partner_player",
            "main_proxy_player",
            "main_proxy_method",
            "synergy_score",
            "duo_vaep90_together",
            "proxy_duo_vaep90",
            "mean_proxy_similarity",
            "p_value",
        ]
        print(outputs["directed_synergy"].head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
