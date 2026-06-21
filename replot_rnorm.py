#!/usr/bin/env python3
r"""
replot_rnorm.py  — regenerate the R_norm ablation figure from a cached JSON,
WITHOUT re-running the 38-min DQL training.

Requires a rnorm_ablation.json produced by the UPDATED rnorm_ablation.py, which
now stores the per-episode 'curve' array for every run. Panel (a) is rebuilt from
those cached curves; panel (b) from the cached per-seed finals + summary.

Usage:
    python replot_rnorm.py [rnorm_ablation.json] [out.png]
Defaults: ../results/SURVIVOR/rnorm_ablation.json  ->  rnorm_ablation.png (same dir)
"""
import sys, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONV_WINDOW = 30   # must match the moving-average window used during the run

def build(d, out_png):
    rmode = d.get("reward_mode", "sumrate")
    rw, ro = d["runs_with"], d["runs_without"]
    aw, ao = d["with_rnorm"], d["without_rnorm"]
    nseed  = len(d["config"]["seeds"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.7))

    # ---- panel (a): convergence bands from cached per-episode curves ----
    def band(runs, color, label, ls):
        L  = min(len(r["curve"]) for r in runs)
        M  = np.array([r["curve"][:L] for r in runs])
        k  = CONV_WINDOW
        Ms = np.array([np.convolve(m, np.ones(k)/k, mode="valid") for m in M])
        x  = np.arange(Ms.shape[1]); mean, std = Ms.mean(0), Ms.std(0)
        ax1.plot(x, mean, color=color, ls=ls, lw=2.2, label=label)
        ax1.fill_between(x, mean-std, mean+std, color=color, alpha=0.18)
    band(rw, "#1f5fa8", "With R_norm (5-dim)",    "-")
    band(ro, "#d62728", "Without R_norm (4-dim)", "--")
    ax1.set_xlabel(f"Episode ({CONV_WINDOW}-ep moving average)")
    ax1.set_ylabel("Mean step reward (Mbps)" if rmode == "sumrate" else "Mean step reward  J_sla")
    ax1.set_title("(a) Convergence: with vs without R_norm", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9); ax1.grid(True, ls=":", alpha=0.5)

    # ---- panel (b): raw dots dodged LEFT, mean +/- std dodged RIGHT ----
    if rmode == "sumrate":
        wv = [r["agg_mbps"] for r in rw]; ov = [r["agg_mbps"] for r in ro]
        ylab = "Final throughput (Mbps)"
        mw, sw, mo, so = aw["agg_mean"], aw["agg_std"], ao["agg_mean"], ao["agg_std"]
    else:
        wv = [r["sla_rel"]*100 for r in rw]; ov = [r["sla_rel"]*100 for r in ro]
        ylab = "Final far-user SLA reliability (%)"
        mw, sw, mo, so = aw["sla_mean"]*100, aw["sla_std"]*100, ao["sla_mean"]*100, ao["sla_std"]*100
    jit = np.random.default_rng(0)
    for base, vals, col in ((0, wv, "#1f5fa8"), (1, ov, "#d62728")):
        jx = base - 0.16 + jit.uniform(-0.045, 0.045, len(vals))
        ax2.scatter(jx, vals, s=46, c=col, alpha=0.75, edgecolor="white", linewidth=0.6,
                    zorder=3, label=(f"per-seed (n={nseed})" if base == 0 else None))
    ax2.errorbar(0.16, mw, yerr=sw, color="#1f5fa8", capsize=6, lw=2.2, marker="D", ms=7, zorder=4, label="mean \u00b1 std")
    ax2.errorbar(1.16, mo, yerr=so, color="#d62728", capsize=6, lw=2.2, marker="D", ms=7, zorder=4)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["With R_norm", "Without R_norm"])
    ax2.set_xlim(-0.55, 1.55); ax2.set_ylabel(ylab)
    ax2.set_title("(b) Final policy quality & variance", fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5, axis="y"); ax2.legend(loc="center right", fontsize=8.5, framealpha=0.9)

    plt.suptitle(f"R_norm Ablation \u2014 DQL Markov-property check "
                 f"(seed-42 layout, {nseed} seeds, reward={rmode})", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {out_png}")

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_json = os.path.normpath(os.path.join(here, "..", "results", "SURVIVOR", "rnorm_ablation.json"))
    jp = sys.argv[1] if len(sys.argv) > 1 else default_json
    with open(jp) as f:
        d = json.load(f)
    if not d["runs_with"] or "curve" not in d["runs_with"][0]:
        sys.exit("ERROR: this JSON has no cached 'curve' arrays. Re-run the updated "
                 "rnorm_ablation.py ONCE to produce a curve-bearing JSON, then re-plot.")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(jp), "rnorm_ablation.png")
    build(d, out)

if __name__ == "__main__":
    main()
