# Synergy VAEP

## Summary

Collects StatsBomb Open Data events from four major European leagues in 2015/2016 (La Liga, Premier League, Serie A, and Ligue 1), values every action with VAEP, and measures the synergy between pairs of teammates:

- **VAEP.** Two XGBoost classifiers estimate the probability of scoring and conceding within the next actions; the difference between states defines the offensive and defensive value of each action.
- **Proxy.** When partner B is off the pitch, the real substitute of the same slot or the most spatially similar starter takes their place.
- **Offensive synergy.** `VAEP90(A+B together) − VAEP90(A+proxy of B)`.
- **Defensive synergy.** Reduction of the opponent's threat with the pair on the pitch, compared with the proxy windows.

## Structure

- `notebooks/00-data_collection.ipynb`: downloads matches and events from StatsBomb Open Data and combines the collection tables.
- `notebooks/01-tactical_synergy_vaep.ipynb`: estimates VAEP, builds presence segments, spatial profiles, and proxies, calculates the synergies, and produces the figures.
- `data/raw/`: matches (`matches/`), events (`events/`), and combined tables (`combined/`).
- `data/processed/`: actions with VAEP, segments, proxies, synergies, and rankings.
- `viz/`: figures produced by notebook 01.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the notebooks in order. The collection reuses files already on disk and can be resumed.

## Outputs

| file or folder | content |
| --- | --- |
| `data/raw/combined/all_matches_combined.parquet` | every match in the scope |
| `data/raw/combined/all_events_combined.parquet` | every event, with split coordinates |
| `data/processed/vaep_actions.parquet` | actions with VAEP probabilities and values |
| `data/processed/directed_synergy_full.parquet` | directed relations with offensive and defensive synergy |
| `data/processed/player_offensive_synergy_ranking.csv` | offensive ranking by player |
| `data/processed/player_defensive_synergy_ranking.csv` | defensive ranking by player |
| `viz/` | pairs, rankings, and VAEP summaries |
