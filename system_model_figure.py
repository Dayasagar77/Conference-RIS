#!/usr/bin/env python3
r"""
system_model_figure.py
═══════════════════════════════════════════════════════════════════════════════
Reviewer-1 #5 ("provide a system-model figure illustrating the considered
scenario") and Reviewer-2 #7 ("diagram quality too low"). Produces a clean,
publication-quality schematic of the RIS-assisted UAV-NOMA emergency downlink,
built from the System Model in v24 Section II:

    - post-disaster region Omega: 3000 x 1600 m
    - passive RIS (M = 1024, 32x32) mast-mounted at the perimeter
    - single UAV flying base station at the DQL-learned (437, 584, 108) m
    - 24 survivor IoT devices = 12 near (LoS ring) + 12 far (blocked by 60 dB
      rubble), served as 12 power-domain NOMA pairs on orthogonal sub-bands
    - direct UAV->near LoS links; UAV->far direct path blocked; far survivors
      reached by the two-hop UAV->RIS->far reflected bypass
    - f_c = 3.5 GHz, B = 20 MHz, receiver-side SIC at the near user

Schematic (illustrative device placement for clarity, not a data scatter):
 a top-down map of Omega with the rubble barrier, the blocked direct path, and
 the RIS bypass highlighted; an inset states the key NOMA + RIS parameters.

Output: system_model.png (300 dpi) in ../results/SURVIVOR (or ./results/SURVIVOR).
Pure matplotlib; no project dependency.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
if not os.path.isdir(os.path.dirname(OUT)):
    OUT = os.path.join(HERE, "results", "SURVIVOR")
os.makedirs(OUT, exist_ok=True)

# palette
C_NEAR, C_FAR = "#1f5fa8", "#e8821e"
C_RISLINK, C_DIRECT, C_BLOCK = "#1a9988", "#2ca02c", "#d62728"
C_RUBBLE, C_RIS = "#8a6d57", "#5b6b9a"

fig, ax = plt.subplots(figsize=(11, 6.4))
ax.set_xlim(0, 3000); ax.set_ylim(0, 1600); ax.set_aspect("equal")

# region boundary
ax.add_patch(Rectangle((0, 0), 3000, 1600, fill=False, ec="#444", lw=1.5, ls="-"))
ax.text(40, 1530, r"Post-disaster region $\Omega$ : 3000 m $\times$ 1600 m",
        fontsize=10, style="italic", color="#444")

# ---- illustrative placements (schematic) -------------------------------------
uav = np.array([1350, 1050])              # UAV ground projection
ris = np.array([360, 300])                # RIS at perimeter
np.random.seed(7)
near = uav + np.column_stack([np.random.uniform(-230, 230, 12),
                              np.random.uniform(-200, 210, 12)])
# far survivors beyond the rubble barrier (blocked from the UAV)
far = np.array([2250, 560]) + np.column_stack([np.random.uniform(-230, 260, 12),
                                               np.random.uniform(-180, 220, 12)])

# ---- rubble / debris barrier between UAV and far survivors -------------------
rub_x = np.array([1720, 1980, 2020, 1760])
rub_y = np.array([1180, 1120, 280, 340])
ax.fill(rub_x, rub_y, color=C_RUBBLE, alpha=0.5, hatch="xx", ec=C_RUBBLE, lw=1.2, zorder=2)
ax.text(1870, 980, "rubble\nblockage\n(60 dB)", ha="center", va="center",
        fontsize=8.5, color="#3d2f23", fontweight="bold", zorder=3)

# ---- links -------------------------------------------------------------------
# UAV -> near (direct LoS)
for p in near:
    ax.add_line(Line2D([uav[0], p[0]], [uav[1], p[1]], color=C_DIRECT, lw=0.8,
                       alpha=0.55, zorder=3))
# UAV -> far blocked direct path (crossed)
blk_end = np.array([2050, 620])
ax.add_patch(FancyArrowPatch(uav, blk_end, arrowstyle="-|>", mutation_scale=14,
                             color=C_BLOCK, lw=1.8, ls=(0, (5, 4)), zorder=4))
mid = (uav + blk_end) / 2
ax.plot([mid[0]-45, mid[0]+45], [mid[1]+45, mid[1]-45], color=C_BLOCK, lw=2.4, zorder=5)
ax.plot([mid[0]-45, mid[0]+45], [mid[1]-45, mid[1]+45], color=C_BLOCK, lw=2.4, zorder=5)
ax.text(mid[0]-250, mid[1]+25, "blocked\ndirect path", color=C_BLOCK, fontsize=8.5,
        fontweight="bold", ha="right", zorder=5)
# UAV -> RIS -> far (two-hop bypass)
ax.add_patch(FancyArrowPatch(uav, ris, arrowstyle="-|>", mutation_scale=16,
                             color=C_RISLINK, lw=2.4, zorder=6))
far_centroid = far.mean(0)
ax.add_patch(FancyArrowPatch(ris, far_centroid, arrowstyle="-|>", mutation_scale=16,
                             color=C_RISLINK, lw=2.4, zorder=6))
ax.text(760, 760, "RIS bypass\n(two-hop reflected)\n+7.55 dB", color=C_RISLINK,
        fontsize=9, fontweight="bold", ha="center", zorder=6)

# ---- RIS panel ---------------------------------------------------------------
ax.add_patch(Rectangle((ris[0]-95, ris[1]-60), 190, 120, facecolor=C_RIS,
                       alpha=0.85, ec="black", lw=1.2, hatch="||", zorder=7))
ax.text(ris[0], ris[1]-105, "RIS\nM = 1024 (32$\\times$32)\np = (300, 250, 20) m",
        ha="center", va="top", fontsize=8.5, fontweight="bold", color=C_RIS, zorder=7)

# ---- survivors ---------------------------------------------------------------
ax.scatter(near[:, 0], near[:, 1], s=58, marker="o", c=C_NEAR, ec="white", lw=0.8,
           zorder=8, label="near survivors (12, LoS)")
ax.scatter(far[:, 0], far[:, 1], s=58, marker="s", c=C_FAR, ec="white", lw=0.8,
           zorder=8, label="far survivors (12, blocked)")

# NOMA pairing note (open area, no crossing connector to avoid clutter)
ax.text(1180, 250, "Each near\u2013far survivor pair $\\to$ one power-domain NOMA pair (12 pairs);\n"
        "receiver-side SIC decoded at the near (strong) user.",
        ha="center", va="center", fontsize=8.5, color="#555", zorder=6,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbfb", ec="#bbb", lw=0.8))

# ---- UAV quadcopter glyph ----------------------------------------------------
def quad(ax, c, s=95):
    ax.add_line(Line2D([c[0]-s, c[0]+s], [c[1]-s, c[1]+s], color="#222", lw=2.2, zorder=9))
    ax.add_line(Line2D([c[0]-s, c[0]+s], [c[1]+s, c[1]-s], color="#222", lw=2.2, zorder=9))
    for dx, dy in [(-s, s), (s, s), (-s, -s), (s, -s)]:
        ax.add_patch(Circle((c[0]+dx, c[1]+dy), s*0.42, facecolor="#cfd8e8",
                            ec="#222", lw=1.4, zorder=10))
    ax.add_patch(Circle(c, s*0.5, facecolor="#222", ec="#222", zorder=11))
quad(ax, uav)
ax.text(uav[0], uav[1]+175, "UAV flying base station\n(437, 584, 108) m  |  $H^*$ = 108 m",
        ha="center", fontsize=9, fontweight="bold", zorder=11)

# ---- parameter inset ---------------------------------------------------------
txt = ("$f_c$ = 3.5 GHz   |   $B$ = 20 MHz\n"
       "24 survivors = 12 NOMA pairs\n"
       "$P_T$ = 2 W,  $\\alpha$ = 0.85,  3-bit RIS phase")
ax.add_patch(FancyBboxPatch((2030, 1230), 930, 300, boxstyle="round,pad=12",
                            facecolor="#f4f6fa", ec="#999", lw=1.0, zorder=8,
                            mutation_aspect=1))
ax.text(2495, 1380, txt, ha="center", va="center", fontsize=9, zorder=9)

# ---- legend ------------------------------------------------------------------
handles = [
    Line2D([0], [0], color=C_DIRECT, lw=2, label="UAV$\\to$near direct (LoS)"),
    Line2D([0], [0], color=C_BLOCK, lw=2, ls=(0, (5, 4)), label="UAV$\\to$far direct (blocked)"),
    Line2D([0], [0], color=C_RISLINK, lw=2.4, label="UAV$\\to$RIS$\\to$far (bypass)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NEAR, markersize=9, label="near survivor"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_FAR, markersize=9, label="far survivor"),
]
ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.95,
          ncol=1, bbox_to_anchor=(0.995, 0.02))

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("System model: RIS-assisted UAV-NOMA emergency downlink for blocked survivors",
             fontsize=12, fontweight="bold", pad=12)
ax.grid(True, ls=":", alpha=0.3)
plt.tight_layout()
p = os.path.join(OUT, "system_model.png")
fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
print(f"[saved] {p}")
