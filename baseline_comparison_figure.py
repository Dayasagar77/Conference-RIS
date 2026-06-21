#!/usr/bin/env python3
r"""
baseline_comparison_figure.py
Honest per-seed rendering of the continuous-RL baseline comparison, built from the
WINDOWS-AUTHORITATIVE run (continuous_rl_baseline.py v2, VecNormalize, 60k steps,
3 seeds). Replaces the auto-figure's misleading DDPG mean (142.75 +/- 64.86) with
a per-seed dot plot that exposes the bimodality: DDPG converges to the DQN optimum
in 2/3 seeds and collapses in 1/3 — the instability that motivates the robust DQN.

Per-seed throughput (Mbps), greedy eval N_MC=60, EVAL_SEED=99 (Windows):
  PPO  : 188.88, 188.83, 188.82      -> 188.85 +/- 0.03   (+0.26% vs DQN)
  DDPG : 188.80, 188.43,  51.02      -> 2/3 converged ~188.6, 1 collapsed
  A2C  : 180.63, 181.46, 182.31      -> 181.47 +/- 0.68   (-3.65% vs DQN)
  DQN  : 188.35 (finer-grid optimum, proposed)
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
if not os.path.isdir(os.path.dirname(OUT)):
    OUT = os.path.join(HERE, "results", "SURVIVOR")
os.makedirs(OUT, exist_ok=True)

SEEDVALS = {
    "PPO":  [188.88, 188.83, 188.82],
    "DDPG": [188.80, 188.43, 51.02],
    "A2C":  [180.63, 181.46, 182.31],
}
DQN_GRID = 188.35
COL = {"PPO": "#1f5fa8", "DDPG": "#2ca02c", "A2C": "#9467bd"}
ORDER = ["PPO", "DDPG", "A2C"]

fig, ax = plt.subplots(figsize=(9, 6))

# DQN grid-optimum reference line (the proposed method)
ax.axhline(DQN_GRID, color="#d62728", ls="--", lw=2.0, zorder=2)
ax.text(3.35, DQN_GRID + 1.5, f"DQN (proposed)\ngrid-verified {DQN_GRID:.1f} Mbps",
        color="#d62728", fontsize=9, fontweight="bold", va="bottom", ha="right")

rng = np.random.default_rng(0)
for i, m in enumerate(ORDER):
    vals = np.array(SEEDVALS[m])
    jit = rng.uniform(-0.07, 0.07, len(vals))
    ax.scatter(np.full(len(vals), i) + jit, vals, s=130, color=COL[m],
               ec="black", lw=1.0, zorder=5, alpha=0.9)
    # converged cluster (>150) mean as a short bar
    conv = vals[vals > 150]
    if len(conv):
        ax.hlines(conv.mean(), i - 0.18, i + 0.18, color=COL[m], lw=3, zorder=6)

# annotations
ax.annotate("PPO ties the DQN\n188.85 \u00b1 0.03  (+0.26%)\nall 3 seeds", (0, 188.85),
            xytext=(0, 165), ha="center", fontsize=8.5, color=COL["PPO"],
            arrowprops=dict(arrowstyle="->", color=COL["PPO"], lw=1.2))
ax.annotate("DDPG: 2/3 seeds reach\nthe optimum (\u2248188.6),\n1 seed collapses (51) \u2014\nunstable off-policy",
            (1, 188.6), xytext=(1.0, 120), ha="center", fontsize=8.5, color=COL["DDPG"],
            arrowprops=dict(arrowstyle="->", color=COL["DDPG"], lw=1.2))
ax.annotate("A2C (\u2248 A3C): stable,\n181.47 \u00b1 0.68  (\u22123.65%)", (2, 181.47),
            xytext=(2.0, 160), ha="center", fontsize=8.5, color=COL["A2C"],
            arrowprops=dict(arrowstyle="->", color=COL["A2C"], lw=1.2))

ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, fontsize=11)
ax.set_xlim(-0.6, 3.4); ax.set_ylim(40, 205)
ax.set_ylabel("Converged greedy throughput (Mbps)", fontsize=11)
ax.set_title("Continuous-action RL baselines vs proposed DQN (per-seed, N$_{MC}$=60)\n"
             "same channel model & sum-rate objective; 3 seeds each",
             fontsize=12, fontweight="bold")
ax.grid(True, ls=":", alpha=0.45, axis="y")
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
              markeredgecolor="black", markersize=10, label="per-seed result"),
       Line2D([0], [0], color="#555", lw=3, label="converged-seed mean"),
       Line2D([0], [0], color="#d62728", lw=2, ls="--", label="DQN grid-optimum")]
ax.legend(handles=leg, loc="lower left", fontsize=9, framealpha=0.95)
plt.tight_layout()
p = os.path.join(OUT, "baseline_comparison.png")
fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
print(f"[saved] {p}")
