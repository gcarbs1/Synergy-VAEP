"""Dark landscape poster: offensive synergy of Messi, Neymar and Munir at Barcelona, 2015/16.

Each teammate is a petal around the central player. The ring is zero.
Petal length = synergy of the pair, outward if positive, inward if negative:
    synergy(A, B) = VAEP90(A + B together) - VAEP90(A + B's replacement).
Each nested line of a petal = 200 minutes the pair played together.

Reads data/processed/barcelona_star_synergy.csv (the README explains where each column comes from).
Run from the repository root:  .venv/bin/python poster/make_poster.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib import patheffects as pe
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
BG = "#121317"
DISK = "#1B1C21"
INK = "#F1ECE2"
INK_2 = "#B9B3A7"
INK_3 = "#7D786E"
RULE = "#34343A"
SAND = "#B08D5B"
IVORY = "#F6EEDC"
GOLD = "#E6C48A"          # positive values
SLATE = "#93A6C2"         # negative values
PETAL_POS = LinearSegmentedColormap.from_list("pos", ["#7A5F3A", SAND, "#E2CFA8", IVORY])
PETAL_NEG = LinearSegmentedColormap.from_list("neg", ["#2F3A4A", "#5C6F8A", SLATE])
HALO = [pe.withStroke(linewidth=3, foreground=BG)]

for f in FONTS.glob("*.ttf"):
    fm.fontManager.addfont(str(f))
SERIF, SANS = "DM Serif Display", "IBM Plex Sans Condensed"
plt.rcParams.update({"font.family": SANS, "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none"})

PLAYERS = ["Messi", "Neymar", "Munir"]
Z = 0.58             # radius of the zero ring
K = 0.58              # radius units per 1 VAEP per 90
MIN_PER_LINE = 200   # minutes together represented by each nested line


# Pairs need 300 minutes together and 180 with the replacement (filter applied upstream).
DATA = PROCESSED / "barcelona_star_synergy.csv"

# Busquets with Messi, the worked example: VAEP per 90 of Busquets and of his partner, with Messi
# and with Messi's replacement (Munir), as printed by the original pipeline.
EXAMPLE = {"with_star": (0.14, 0.94), "with_replacement": (0.27, 0.00)}


# ---------------------------------------------------------------- data
def load():
    return pd.read_csv(DATA)


def ring(data, star):
    """Teammates of one star; ring_angle is clockwise from the top, ordered by position."""
    d = data[data.star.eq(star)].copy()
    d["name"] = d.teammate
    d["synergy"] = d.offensive_synergy
    minutes = d.minutes_together.fillna(d.minutes_together_rounded_200)
    d["minutes"] = minutes
    d["theta"] = np.radians(90 - d.ring_angle)
    return d.reset_index(drop=True)


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


def petal_profile(t):
    """Half-width along the petal, 0 at the ring and 1 at the tip: a teardrop with a narrow base,
    widest a little past the middle and a rounded tip."""
    width = 0.3 + 0.7 * np.sin(np.pi * np.clip(t, 0, 1) ** 0.85) ** 0.75
    return width * np.sqrt(np.clip(1 - t ** 4, 0, 1))


def petal_outline(th, signed_len, half_angle, n=90):
    """Petal from the zero ring: outward when signed_len > 0, inward when < 0."""
    t = np.linspace(0, 1, n)
    r = Z + signed_len * t
    h = half_angle * petal_profile(t)
    if signed_len < 0:
        # Towards the centre the same angle covers less arc; widen it so the petal keeps its shape.
        h = h * (Z / np.maximum(r, 0.08)) ** 0.85
    right = np.c_[r * np.cos(th - h), r * np.sin(th - h)]
    left = np.c_[r * np.cos(th + h), r * np.sin(th + h)][::-1]
    h0 = half_angle * petal_profile(np.array([0.0]))[0]
    a = np.linspace(th + h0, th - h0, 20)
    base = np.c_[Z * np.cos(a), Z * np.sin(a)]
    return np.vstack([base, right, left])


def petal(ax, th, synergy, minutes, half_angle):
    signed = K * synergy
    positive = synergy >= 0
    cmap = PETAL_POS if positive else PETAL_NEG
    # Inward petals share less room near the centre, so they are drawn narrower.
    half_angle = half_angle if positive else half_angle * 0.55
    n_lines = max(1, int(round(minutes / MIN_PER_LINE)))
    outer = petal_outline(th, signed, half_angle)
    ax.fill(outer[:, 0], outer[:, 1], color=cmap(0.9), alpha=0.07 if positive else 0.12, lw=0,
            zorder=2.5)
    for k in range(1, n_lines + 1):
        f = k / n_lines
        xy = petal_outline(th, signed * f, half_angle * (0.35 + 0.65 * f))
        ax.plot(xy[:, 0], xy[:, 1], color=cmap(0.15 + 0.85 * f), lw=0.5 if k < n_lines else 1.0,
                alpha=0.5 + 0.5 * f, zorder=3 + f, solid_joinstyle="round")
    return Z + max(signed, 0)


def label(ax, th, r, name, value):
    tone = SLATE if value.startswith("−") else GOLD
    x, y = polar(r, th)
    c, s_ = np.cos(th), np.sin(th)
    ha = "left" if c > 0.2 else "right" if c < -0.2 else "center"
    ox, oy = 7 * c, 7 * s_
    if s_ > 0.55:
        v = ax.annotate(value, (x, y), xytext=(ox, oy), textcoords="offset points", ha=ha, va="bottom",
                        fontsize=9, weight="medium", color=tone, zorder=6,
                        path_effects=HALO)
        n = ax.annotate(name, (x, y), xytext=(ox, oy + 11), textcoords="offset points", ha=ha, va="bottom",
                        fontsize=9.8, weight="semibold", color=INK, zorder=6,
                        path_effects=HALO)
    else:
        va = "top" if s_ < -0.55 else "center"
        dy = 0 if va == "top" else 5.5
        n = ax.annotate(name, (x, y), xytext=(ox, oy + dy), textcoords="offset points", ha=ha, va=va,
                        fontsize=9.8, weight="semibold", color=INK, zorder=6,
                        path_effects=HALO)
        v = ax.annotate(value, (x, y), xytext=(ox, oy + dy - 12), textcoords="offset points", ha=ha, va=va,
                        fontsize=9, weight="medium", color=tone, zorder=6,
                        path_effects=HALO)
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
        r_out = petal(ax, row.theta, row.synergy, row.minutes, half)
        rl = r_out + 0.06
        groups.append({"artists": label(ax, row.theta, rl, row.name, fmt(row.synergy)), "r": rl, "th": row.theta})
    repel(fig, groups)
    x0, y0, w, h = rect
    cx = x0 + w / 2
    mean = d.synergy.mean()
    fig.text(cx, y0 + h + 0.006, player, fontsize=24, family=SERIF, color=INK, ha="center",
             va="bottom")
    fig.text(cx, y0 - 0.004, fmt(mean), fontsize=30, family=SERIF, color=GOLD if mean > 0 else SLATE,
             ha="center", va="top")
    fig.text(cx, y0 - 0.062, f"AVERAGE SYNERGY  ·  {len(d)} TEAMMATES", fontsize=8, weight="semibold",
             color=INK_3, ha="center", va="top")


def main():
    data = load()
    W, H = 16, 10
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    L, R = 0.045, 0.955

    fig.text(L, 0.94, "A measure of offensive synergy", fontsize=38, family=SERIF, color=INK, va="top")
    fig.text(L, 0.862,
             "Messi, Neymar and Munir at Barcelona, La Liga 2015/16, in VAEP per 90 minutes. For each teammate, synergy "
             "compares the pair's output with the player\non the pitch and with his replacement. The ring marks zero: "
             "petals pointing outward show positive synergy, petals pointing inward show negative synergy.",
             fontsize=12, color=INK_2, va="top", linespacing=1.55)

    w = 0.27
    h = w * W / H
    for player, cx in zip(PLAYERS, [0.19, 0.5, 0.81]):
        sun(fig, [cx - w / 2, 0.315, w, h], ring(data, player), player)

    # bottom strip
    top = 0.205
    fig.add_artist(plt.Line2D([L, R], [top + 0.02, top + 0.02], color=RULE, lw=0.8))

    # reading a petal
    fig.text(L, top, "READING A PETAL", fontsize=8.8, weight="semibold", color=INK, va="top")
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
    ex = data[data.star.eq("Messi") & data.teammate.eq("Busquets")].iloc[0]
    (a_w, b_w), (a_o, b_o) = EXAMPLE["with_star"], EXAMPLE["with_replacement"]
    synergy, lift = ex.offensive_synergy, ex.focal_impact_p90
    x_steps = 0.375
    fig.text(x_steps, top, "COMPUTING ONE PETAL:  BUSQUETS AND MESSI, VAEP PER 90 MINUTES", fontsize=8.5,
             weight="semibold", color=INK, va="top")
    steps = [
        ("Busquets + Messi", f"{n2(a_w)} + {n2(b_w)} = {n2(a_w + b_w)}"),
        ("Busquets + Munir, the replacement", f"{n2(a_o)} + {n2(max(b_o, 0))} = {n2(a_o + b_o)}"),
        ("Synergy of the pair", f"{n2(a_w + b_w)} − {n2(a_o + b_o)} = {fmt(synergy)}"),
        ("Messi − Munir", fmt(synergy - lift)),
        ("Change in Busquets", fmt(lift)),
    ]
    xs = [x_steps, x_steps + 0.128, x_steps + 0.27, x_steps + 0.418, x_steps + 0.5]
    for i, ((lab, val), x) in enumerate(zip(steps, xs)):
        fig.text(x, top - 0.031, f"{i + 1}", fontsize=13, family=SERIF, color=INK_3, va="top")
        fig.text(x + 0.014, top - 0.034, lab, fontsize=8.8, color=INK_2, va="top")
        fig.text(x + 0.014, top - 0.056, val, fontsize=16, family=SERIF,
                 color=(GOLD if i == 2 else IVORY) if i < 3 else INK_2, va="top")
    fig.text(x_steps + 0.014, top - 0.105,
             "The synergy of the pair splits into steps 4 and 5: the difference between Messi and his replacement, "
             "and the change in Busquets' own output.\nThe average over all teammates is the number below each ring. "
             "Teammates are arranged by position, with forwards at the top.",
             fontsize=8.4, color=INK_3, va="top", linespacing=1.6)

    fig.text(R, 0.035, "Data  StatsBomb Open Data   ·   GABRIEL CARBINATTO", fontsize=8.5, color=INK_3,
             va="bottom", ha="right")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"synergy_messi_neymar_munir_dark.{ext}", dpi=200, facecolor=BG)
    print("saved")


if __name__ == "__main__":
    main()
