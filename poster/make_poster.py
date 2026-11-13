"""Dark landscape poster: offensive synergy of Messi, Neymar and Munir at Barcelona, 2015/16.

Each teammate is a petal around the central player. The ring is zero.
Petal length = synergy of the pair, outward if positive, inward if negative:
    synergy(A, B) = VAEP90(A + B together) - VAEP90(A + B's replacement).
Each nested line of a petal = 200 minutes the pair played together.

Run from the repository root:  .venv/bin/python poster/make_poster.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, to_rgb

def n2(v):
    """Two decimals, English format, true minus sign."""
    return f"{v:.2f}".replace("-", "−")


def fmt(v):
    return ("+" if v > 0.004 else "−" if v < -0.004 else "") + f"{abs(v):.2f}"

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "poster" / "output"
FONTS = ROOT / "poster" / "fonts"

# ---------------------------------------------------------------- tokens (dark)
BG = "#17181b"
DISK = "#1f2023"
INK = "#efe9dc"
INK_2 = "#bdb6a8"
INK_3 = "#8a8477"
RULE = "#3a3a3d"
SAND = "#a8875f"
IVORY = "#f3ecdc"
PETAL_POS = LinearSegmentedColormap.from_list("pos", [SAND, "#d9c6a3", IVORY])
PETAL_NEG = LinearSegmentedColormap.from_list("neg", ["#6a6357", "#958a76", "#b9ab90"])

for f in FONTS.glob("*.ttf"):
    fm.fontManager.addfont(str(f))
SERIF, SANS = "Libre Baskerville", "Lato"
plt.rcParams.update({"font.family": SANS, "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none"})

PLAYERS = [(217, 5503, "Messi"), (217, 4320, "Neymar"), (217, 6616, "Munir")]
Z = 0.58             # radius of the zero ring
K = 0.58              # radius units per 1 VAEP per 90
MIN_PER_LINE = 200   # minutes together represented by each nested line


MIN_TOGETHER, MIN_PROXY = 300.0, 180.0

SHORT = {
    "Ivan Rakitić": "Rakitić", "Gerard Piqué Bernabéu": "Piqué", "Jérémy Mathieu": "Mathieu",
    "Jordi Alba Ramos": "Jordi Alba", "Sergi Roberto Carnicer": "Sergi Roberto",
    "Javier Alejandro Mascherano": "Mascherano", "Sergio Busquets i Burgos": "Busquets",
    "Daniel Alves da Silva": "Dani Alves", "Luis Alberto Suárez Díaz": "Suárez",
    "Munir El Haddadi Mohamed": "Munir", "Neymar da Silva Santos Junior": "Neymar",
    "Andrés Iniesta Luján": "Iniesta",
    "Lionel Andrés Messi Cuccittini": "Messi", "Rafael Alcântara do Nascimento": "Rafinha",
    "Layvin Kurzawa": "Kurzawa", "Edinson Roberto Cavani Gómez": "Cavani", "Adrien Rabiot": "Rabiot",
    "David Luiz Moreira Marinho": "David Luiz", "Benjamin Stambouli": "Stambouli",
    "Serge Aurier": "Aurier", "Ángel Fabián Di María Hernández": "Di María",
    "Maxwell Scherrer Cabelino Andrade": "Maxwell", "Gregory van der Wiel": "Van der Wiel",
    "Marco Verratti": "Verratti", "Thiago Emiliano da Silva": "Thiago Silva",
    "Blaise Matuidi": "Matuidi", "Marcos Aoás Corrêa": "Marquinhos", "Thiago Motta": "Thiago Motta",
    "Lucas Rodrigues Moura da Silva": "Lucas Moura",
}

# Pitch coordinates (x: left -1 .. right +1, y: own goal -1 .. attack +1), used only to order the ring.
PITCH = {
    "Center Forward": (0.0, 1.0), "Left Center Forward": (-0.25, 1.0), "Right Center Forward": (0.25, 1.0),
    "Left Wing": (-0.85, 0.7), "Right Wing": (0.85, 0.7),
    "Center Attacking Midfield": (0.0, 0.55), "Left Attacking Midfield": (-0.4, 0.55),
    "Right Attacking Midfield": (0.4, 0.55), "Left Midfield": (-0.9, 0.2), "Right Midfield": (0.9, 0.2),
    "Left Center Midfield": (-0.4, 0.15), "Right Center Midfield": (0.4, 0.15), "Center Midfield": (0.0, 0.15),
    "Center Defensive Midfield": (0.0, -0.3), "Left Defensive Midfield": (-0.3, -0.3),
    "Right Defensive Midfield": (0.3, -0.3), "Left Back": (-0.9, -0.55), "Right Back": (0.9, -0.55),
    "Left Wing Back": (-0.9, -0.3), "Right Wing Back": (0.9, -0.3),
    "Left Center Back": (-0.35, -0.95), "Right Center Back": (0.35, -0.95), "Center Back": (0.0, -1.0),
}


# ---------------------------------------------------------------- data
def load():
    s = pd.read_parquet(PROCESSED / "directed_synergy_full.parquet")
    s = s[s.minutes_together.ge(MIN_TOGETHER) & s.minutes_proxy.ge(MIN_PROXY)].copy()
    s["gap"] = s.partner_vaep90_together - s.proxy_vaep90
    s["lift"] = s.focal_impact_p90
    s["synergy"] = s.offensive_synergy_score
    assert np.allclose(s.synergy, s.gap + s.lift)
    a = pd.read_parquet(PROCESSED / "vaep_actions.parquet", columns=["team_id", "player_id", "position"])
    pos = (a.dropna(subset=["position"]).groupby(["team_id", "player_id"]).position
           .agg(lambda x: x.value_counts().index[0]))
    return s, pos


def ring(s, pos, team_id, star_id):
    d = s[s.team_id.eq(team_id) & s.partner_player_id.eq(star_id)].copy()
    d["name"] = d.focal_player.map(SHORT).fillna(d.focal_player)
    d["position"] = [pos.get((team_id, p), "Center Midfield") for p in d.focal_player_id]
    xy = np.array([PITCH.get(p, (0, 0)) for p in d.position])
    d["natural"] = np.degrees(np.arctan2(xy[:, 0], xy[:, 1])) % 360   # clockwise from the top
    d = d.sort_values(["natural", "lift"]).reset_index(drop=True)
    n = len(d)
    step = 360 / n
    offset = np.median(((d.natural - np.arange(n) * step + 180) % 360) - 180)
    d["theta"] = np.radians(90 - (np.arange(n) * step + offset))
    return d


def repel(fig, groups, step=0.04, iters=30):
    """Push a label group outward along its spoke until no two groups overlap."""
    r = fig.canvas.get_renderer()
    for _ in range(iters):
        boxes = [[a.get_window_extent(r).expanded(1.04, 1.1) for a in g["artists"]] for g in groups]
        moved = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(bi.overlaps(bj) for bi in boxes[i] for bj in boxes[j]):
                    k = j if groups[j]["r"] <= groups[i]["r"] else i
                    groups[k]["r"] += step
                    for a_ in groups[k]["artists"]:
                        a_.xy = tuple(polar(groups[k]["r"], groups[k]["th"]))
                    moved = True
        if not moved:
            break


def polar(r, th):
    return np.array([r * np.cos(th), r * np.sin(th)])


def petal_outline(th, signed_len, half_angle, n=60):
    """Leaf from the zero ring: outward when signed_len > 0, inward when < 0."""
    t = np.linspace(0, 1, n)
    r = Z + signed_len * t
    h = half_angle * (1 - t) ** 0.62
    right = np.c_[r * np.cos(th - h), r * np.sin(th - h)]
    left = np.c_[r * np.cos(th + h), r * np.sin(th + h)][::-1]
    a = np.linspace(th + half_angle, th - half_angle, 20)
    base = np.c_[Z * np.cos(a), Z * np.sin(a)]
    return np.vstack([base, right, left])


def petal(ax, th, synergy, minutes, half_angle):
    signed = K * synergy
    cmap = PETAL_POS if synergy >= 0 else PETAL_NEG
    n_lines = max(1, int(round(minutes / MIN_PER_LINE)))
    for k in range(1, n_lines + 1):
        f = k / n_lines
        xy = petal_outline(th, signed * f, half_angle * (0.35 + 0.65 * f))
        ax.plot(xy[:, 0], xy[:, 1], color=cmap(0.15 + 0.85 * f), lw=0.55 if k < n_lines else 0.9,
                alpha=0.55 + 0.45 * f, zorder=3 + f, solid_joinstyle="round")
    return Z + max(signed, 0)


def label(ax, th, r, name, value):
    x, y = polar(r, th)
    c, s_ = np.cos(th), np.sin(th)
    ha = "left" if c > 0.2 else "right" if c < -0.2 else "center"
    ox, oy = 7 * c, 7 * s_
    if s_ > 0.55:
        v = ax.annotate(value, (x, y), xytext=(ox, oy), textcoords="offset points", ha=ha, va="bottom",
                        fontsize=8.6, color=INK_2, zorder=6)
        n = ax.annotate(name, (x, y), xytext=(ox, oy + 11), textcoords="offset points", ha=ha, va="bottom",
                        fontsize=9.4, weight="bold", color=INK, zorder=6)
    else:
        va = "top" if s_ < -0.55 else "center"
        dy = 0 if va == "top" else 5.5
        n = ax.annotate(name, (x, y), xytext=(ox, oy + dy), textcoords="offset points", ha=ha, va=va,
                        fontsize=9.4, weight="bold", color=INK, zorder=6)
        v = ax.annotate(value, (x, y), xytext=(ox, oy + dy - 12), textcoords="offset points", ha=ha, va=va,
                        fontsize=8.6, color=INK_2, zorder=6)
    return [n, v]


def glow(ax, radius, strength=0.15):
    """Soft light around the zero ring, fading to the background."""
    n = 600
    g = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    a = np.clip(1 - (R - Z) / (radius - Z), 0, 1) ** 2.4 * strength
    a[R < Z] = 0
    img = np.zeros((n, n, 4))
    img[..., :3] = to_rgb(IVORY)
    img[..., 3] = a
    ax.imshow(img, extent=(-radius, radius, -radius, radius), origin="lower", zorder=0, interpolation="bicubic")


def sun(fig, rect, d, player):
    ax = fig.add_axes(rect)
    lim = 1.72
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    glow(ax, 1.5)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.fill(Z * np.cos(t), Z * np.sin(t), color=DISK, lw=0, zorder=1)
    ax.plot(Z * np.cos(t), Z * np.sin(t), color=SAND, lw=0.8, zorder=2)
    half = np.pi / len(d) * 0.78
    groups = []
    for row in d.itertuples():
        r_out = petal(ax, row.theta, row.synergy, row.minutes_together, half)
        rl = r_out + 0.06
        groups.append({"artists": label(ax, row.theta, rl, row.name, fmt(row.synergy)), "r": rl, "th": row.theta})
    repel(fig, groups)
    x0, y0, w, h = rect
    cx = x0 + w / 2
    fig.text(cx, y0 + h + 0.006, player, fontsize=20, family=SERIF, weight="bold", color=INK, ha="center",
             va="bottom")
    fig.text(cx, y0 - 0.004, fmt(d.synergy.mean()), fontsize=26, family=SERIF, weight="bold", color=IVORY,
             ha="center", va="top")
    fig.text(cx, y0 - 0.056, f"AVERAGE SYNERGY  ·  {len(d)} TEAMMATES", fontsize=7.8, weight="bold",
             color=INK_3, ha="center", va="top")


def main():
    s, pos = load()
    W, H = 16, 10
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    L, R = 0.045, 0.955

    fig.text(L, 0.935, "A measure of offensive synergy", fontsize=33, family=SERIF, weight="bold",
             color=INK, va="top")
    fig.text(L, 0.862,
             "Messi, Neymar and Munir at Barcelona, La Liga 2015/16, in VAEP per 90 minutes. For each teammate, synergy "
             "compares the pair's output with the player\non the pitch and with his replacement. The ring marks zero: "
             "petals pointing outward show positive synergy, petals pointing inward show negative synergy.",
             fontsize=11.5, family=SERIF, style="italic", color=INK_2, va="top", linespacing=1.6)

    w = 0.27
    h = w * W / H
    for (tid, pid, player), cx in zip(PLAYERS, [0.19, 0.5, 0.81]):
        sun(fig, [cx - w / 2, 0.315, w, h], ring(s, pos, tid, pid), player)

    # bottom strip
    top = 0.205
    fig.add_artist(plt.Line2D([L, R], [top + 0.02, top + 0.02], color=RULE, lw=0.8))

    # reading a petal
    fig.text(L, top, "READING A PETAL", fontsize=8.5, weight="bold", color=INK, va="top")
    ak = fig.add_axes([L, top - 0.175, 0.27, 0.15]); ak.set_xlim(0.05, 2.75); ak.set_ylim(-0.55, 0.55); ak.axis("off")
    th = 0.0
    tip = petal(ak, th, 1.1, 1600, 0.36)
    arc = np.linspace(-0.6, 0.6, 60)
    ak.plot(Z * np.cos(arc), Z * np.sin(arc), color=SAND, lw=0.8, zorder=4)
    ak.annotate("length = synergy of the pair", (tip, 0.0), xytext=(tip + 0.1, 0.0), fontsize=8.6, color=INK_2,
                va="center", ha="left")
    ak.annotate("each line = 200 minutes\nthe pair played together", (Z + K * 0.7, -0.08), xytext=(Z + K * 1.25, -0.42),
                fontsize=8.2, color=INK_3, ha="left", va="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.6, shrinkA=2, shrinkB=1))
    ak.text(Z * np.cos(0.6), Z * np.sin(0.6) + 0.04, "ring = zero", fontsize=7.6, color=INK_3, ha="left",
            va="bottom")

    # one petal, step by step
    ex = s[s.team_id.eq(217) & s.partner_player_id.eq(5503) & s.focal_player.eq("Sergio Busquets i Burgos")].iloc[0]
    a_w, b_w, a_o, b_o = (ex.focal_vaep90_with_partner, ex.partner_vaep90_together,
                          ex.focal_vaep90_with_proxy, ex.proxy_vaep90)
    x_steps = 0.375
    fig.text(x_steps, top, "COMPUTING ONE PETAL:  BUSQUETS AND MESSI, VAEP PER 90 MINUTES", fontsize=8.5,
             weight="bold", color=INK, va="top")
    steps = [
        ("Busquets + Messi", f"{n2(a_w)} + {n2(b_w)} = {n2(a_w + b_w)}"),
        ("Busquets + Munir, the replacement", f"{n2(a_o)} + {n2(max(b_o, 0))} = {n2(a_o + b_o)}"),
        ("Synergy of the pair", f"{n2(a_w + b_w)} − {n2(a_o + b_o)} = {fmt(ex.synergy)}"),
        ("Messi − Munir", fmt(ex.gap)),
        ("Change in Busquets", fmt(ex.lift)),
    ]
    xs = [x_steps, x_steps + 0.128, x_steps + 0.27, x_steps + 0.418, x_steps + 0.5]
    for i, ((lab, val), x) in enumerate(zip(steps, xs)):
        fig.text(x, top - 0.032, f"{i + 1}", fontsize=12, family=SERIF, weight="bold", color=RULE, va="top")
        fig.text(x + 0.014, top - 0.034, lab, fontsize=8.4, color=INK_2, va="top")
        fig.text(x + 0.014, top - 0.058, val, fontsize=14, family=SERIF, weight="bold",
                 color=IVORY if i < 3 else INK_2, va="top")
    fig.text(x_steps + 0.014, top - 0.105,
             "The synergy of the pair splits into steps 4 and 5: the difference between Messi and his replacement, "
             "and the change in Busquets' own output.\nThe average over all teammates is the number below each ring. "
             "Teammates are arranged by position, with forwards at the top.",
             fontsize=8.4, color=INK_3, va="top", linespacing=1.6)

    fig.text(R, 0.035, "Data: StatsBomb", fontsize=8.5, color=INK_3, va="bottom", ha="right")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"synergy_messi_neymar_munir_dark.{ext}", dpi=200, facecolor=BG)
    print("saved")


if __name__ == "__main__":
    main()
