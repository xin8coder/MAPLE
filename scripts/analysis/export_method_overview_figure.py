#!/usr/bin/env python3
"""Draw the LiveOpt method-overview figure (release_artifacts/paper_figure_exports/results/method_overview.pdf).

Concept: four state lanes (D/W/A/P), three dashed LLM actions (LPD, TSS
editing, and the Restart Skill), one solid fixed-code path, and the return
loop.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY = "#12335D"
TEAL = "#2A9D9F"
PALE = "#F2FBFA"
GRAY = "#666666"
RED = "#B03030"
LANE_COLORS = ["#F6F8FC", "#F4FAF8", "#FFF9EF", "#F8F5FC"]
LANE_LABELS = ["D  data", "W  program", "A  accepted state", "P  search archives"]

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

FIG_W, FIG_H = 10.4, 3.1
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 15.35)
ax.set_ylim(-3.45, 2.9)
ax.axis("off")


def box(x, y, w, h, *, style, text, sub=None, lw=1.1, fontsize=7.0, subsize=4.6):
    if style == "llm":
        fc, ec, ls, ds = "white", TEAL, (0, (4, 2)), "round,pad=0.06,rounding_size=0.10"
    elif style == "fixed":
        fc, ec, ls, ds = "white", NAVY, "solid", "round,pad=0.06,rounding_size=0.06"
    elif style == "bubble":
        fc, ec, ls, ds = PALE, NAVY, "solid", "round,pad=0.08,rounding_size=0.22"
    elif style == "red":
        fc, ec, ls, ds = "white", RED, "solid", "round,pad=0.05,rounding_size=0.06"
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=ds, fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=3)
    ax.add_patch(patch)
    if sub is None:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=4)
    else:
        ax.text(x, y + 0.13, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x, y - 0.22, sub, ha="center", va="center", fontsize=subsize, zorder=4)


def note(x, y, text, size=5.4, color=GRAY, weight="normal"):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, style="italic" if weight == "normal" else weight, zorder=4)


def arrow(p1, p2, *, dashed=False, color=None, lw=1.2, shrink=1.5, connectionstyle="arc3,rad=0.0"):
    color = color or (TEAL if dashed else NAVY)
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=7,
            lw=lw,
            color=color,
            linestyle=(0, (3, 2)) if dashed else "solid",
            shrinkA=shrink,
            shrinkB=shrink,
            connectionstyle=connectionstyle,
            zorder=2,
        )
    )


def lanes(x0, label):
    ax.text(x0 + 0.1, 1.98, label, ha="left", va="center", fontsize=7.5, fontweight="bold", color=NAVY)
    for i, (color, name) in enumerate(zip(LANE_COLORS, LANE_LABELS)):
        y = 1.335 - i * 0.77
        ax.add_patch(plt.Rectangle((x0, y - 0.285), 2.8, 0.57, fc=color, ec="none", zorder=1))
        ax.text(x0 + 1.4, y, name, ha="center", va="center", fontsize=6.6, fontweight="bold", color="#333333", zorder=2)


# ---------- input bubble ----------
box(1.35, 2.45, 2.5, 0.72, style="bubble", text="user update $u_t$", sub="raise I03's value by 6", fontsize=7.2, subsize=5.2)

# ---------- state lanes ----------
lanes(0.2, "$S_{t-1}$")
lanes(11.8, "$S_t$")

# ---------- LPD ----------
box(4.55, 1.5, 2.35, 0.85, style="llm", text="LPD: identify", sub="what changed", subsize=5.2)
note(4.55, 0.86, "data / decisions / evaluation")

# ---------- TSS ----------
box(6.85, 0.565, 2.55, 0.85, style="llm", text="TSS: edit only", sub="decision / evaluation", subsize=5.2)
note(6.85, -0.12, "fixed: solvers \u2022 search operators \u2022 archive")
box(6.6, -1.02, 2.5, 0.5, style="red", text="$\\times$ free program $\\rightarrow$ constraint errors", fontsize=5.2, lw=1.0)

# ---------- Restart Skill ----------
box(9.95, -0.975, 1.9, 0.85, style="llm", text="Restart Skill:", sub="Warm / Full", subsize=5.2)
note(9.95, -0.10, "reuse risk $\\neq$ amount of table change")

# ---------- fixed-code rail ----------
box(4.55, -2.35, 2.3, 0.62, style="fixed", text="check + repair", fontsize=6.6)
box(7.75, -2.35, 2.5, 0.62, style="fixed", text="optimize (fixed solvers)", fontsize=6.6)
box(11.05, -2.35, 2.5, 0.62, style="fixed", text="save valid state", fontsize=6.6)
arrow((5.7, -2.35), (6.5, -2.35))
arrow((9.0, -2.35), (9.8, -2.35))
note(7.75, -2.95, "fixed procedures; the model never changes them")

# ---------- stored-event / accepted-state chain ----------
box(12.35, 2.45, 1.0, 0.42, style="bubble", text="stored events", fontsize=5.4, lw=0.9)
box(13.85, 2.45, 1.1, 0.42, style="bubble", text="accepted state", fontsize=5.4, lw=0.9)
arrow((12.9, 2.45), (13.3, 2.45), dashed=True, lw=0.9)
arrow((13.2, 2.21), (13.2, 0.15), dashed=True, lw=0.9)

# ---------- flows ----------
arrow((1.6, 2.05), (1.6, 1.75))
arrow((3.0, 1.335), (3.35, 1.4))
arrow((3.0, 0.565), (5.55, 0.565))
arrow((3.0, -0.975), (8.95, -0.975))
arrow((5.1, 1.05), (4.9, -2.0), dashed=True, connectionstyle="arc3,rad=-0.06")
arrow((7.3, 0.1), (5.3, -2.0), dashed=True, connectionstyle="arc3,rad=0.06")
arrow((10.2, -1.4), (9.0, -2.0), dashed=True, connectionstyle="arc3,rad=0.06")
arrow((11.05, -2.04), (12.6, -1.5), connectionstyle="arc3,rad=-0.06")

# ---------- return loop ----------
arrow(
    (14.65, -0.975),
    (1.6, -1.5),
    connectionstyle="arc3,rad=0.0",
    lw=1.2,
    color=NAVY,
)
ax.plot([14.65, 15.05, 15.05, 1.6], [-0.975, -0.975, -3.15, -3.15], color=NAVY, lw=1.2, zorder=1)
ax.add_patch(FancyArrowPatch((1.6, -3.15), (1.6, -1.55), arrowstyle="-|>", mutation_scale=7, lw=1.2, color=NAVY, zorder=2))
note(8.3, -3.33, "next update $u_{t+1}$")

fig.savefig("release_artifacts/paper_figure_exports/results/method_overview.pdf", bbox_inches="tight", pad_inches=0.03)
print("wrote release_artifacts/paper_figure_exports/results/method_overview.pdf")
