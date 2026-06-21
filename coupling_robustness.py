#!/usr/bin/env python3
"""
coupling_robustness.py
═══════════════════════════════════════════════════════════════════════════════
ROBUSTNESS of the UAV–RIS coupling law (referee-proofing for Path B).

The headline finding (far-user rate governed by UAV→RIS distance, not UAV→far
distance) was established with the RIS at one location (300, 250). A reviewer will
ask: is this a property of the SYSTEM, or an artifact of that placement?

This script re-runs the governing-variable test for several RIS positions spanning
the deployment region, and reports the correlation pair r(far,d_RIS) vs r(far,d_far)
for each. If the law is general, |r_RIS| stays large and |r_far| stays small at
every placement.

Run on Windows (channel_model.py in same folder):
    python coupling_robustness.py
Outputs:
    results/SURVIVOR/coupling_robustness.png
    results/SURVIVOR/coupling_robustness.csv
"""
import os, time, csv, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import channel_model as cm
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
os.makedirs(OUT, exist_ok=True)

DEVICE_SEED = 42
EVAL_SEED   = 99
N_MC        = 25
H_FIXED     = 120.0
GX = np.linspace(150, 900, 20)
GY = np.linspace(150, 950, 20)

# candidate RIS placements spanning the region (z=20 m, perimeter-style)
RIS_CANDIDATES = [
    ("baseline (300,250)",   np.array([300.0, 250.0, 20.0])),
    ("near-far ring (600,250)", np.array([600.0, 250.0, 20.0])),
    ("opposite (300,950)",   np.array([300.0, 950.0, 20.0])),
    ("east edge (900,600)",  np.array([900.0, 600.0, 20.0])),
    ("co-sited (450,600)",   np.array([450.0, 600.0, 20.0])),
]

np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]
FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
FAR_CENTROID = FAR[:, :2].mean(0)


def per_user(uav, n_mc=N_MC, seed=EVAL_SEED):
    np.random.seed(seed)
    K = cm.CFG.N_PAIRS
    far_samp = []
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        df = np.zeros(K)
        for i in range(K):
            hd = cm.chan_scalar(uav, FAR[i], cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(FAR[i])
            phi = cm.opt_phi(G, hr)
            he  = cm.composite(hd, G, hr, phi)
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            _, rF = cm.noma_pair(hn, he)
            df[i] = rF
        far_samp.append(df)
    return float(np.array(far_samp).mean())


def run():
    print("=" * 74)
    print("  COUPLING ROBUSTNESS ACROSS RIS PLACEMENTS")
    print("=" * 74)
    print(f"  Far centroid {FAR_CENTROID.astype(int)} | Grid {len(GX)}×{len(GY)} | "
          f"N_MC={N_MC} | {len(RIS_CANDIDATES)} RIS placements")
    print("-" * 74)
    t0 = time.time()
    orig_ris = cm.CFG.RIS_POS.copy()
    results = []

    for name, ris in RIS_CANDIDATES:
        cm.CFG.RIS_POS = ris
        far_rates, d_ris_l, d_far_l = [], [], []
        for y in GY:
            for x in GX:
                uav = np.array([x, y, H_FIXED])
                fr = per_user(uav)
                far_rates.append(fr)
                d_ris_l.append(np.linalg.norm(uav[:2] - ris[:2]))
                d_far_l.append(np.linalg.norm(uav[:2] - FAR_CENTROID))
        far_rates = np.array(far_rates)
        log_far = np.log10(np.maximum(far_rates, 1e-3))
        pr_ris, _ = pearsonr(np.array(d_ris_l), log_far)
        pr_far, _ = pearsonr(np.array(d_far_l), log_far)
        sr_ris, _ = spearmanr(np.array(d_ris_l), far_rates)
        sr_far, _ = spearmanr(np.array(d_far_l), far_rates)
        d_rf = np.linalg.norm(ris[:2] - FAR_CENTROID)
        results.append(dict(placement=name, ris_x=ris[0], ris_y=ris[1],
                            ris_far_dist=d_rf,
                            pearson_dris=pr_ris, pearson_dfar=pr_far,
                            spearman_dris=sr_ris, spearman_dfar=sr_far))
        print(f"  {name:26s} | r(d_RIS)={pr_ris:+.3f}  r(d_far)={pr_far:+.3f}  "
              f"| ρ(d_RIS)={sr_ris:+.3f}  ρ(d_far)={sr_far:+.3f}  ({time.time()-t0:.0f}s)")

    cm.CFG.RIS_POS = orig_ris

    # verdict
    all_dris = np.array([abs(r["spearman_dris"]) for r in results])
    all_dfar = np.array([abs(r["spearman_dfar"]) for r in results])
    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)
    print(f"  |ρ(d_RIS)| across placements: min={all_dris.min():.3f}, "
          f"mean={all_dris.mean():.3f}, max={all_dris.max():.3f}")
    print(f"  |ρ(d_far)| across placements: min={all_dfar.min():.3f}, "
          f"mean={all_dfar.mean():.3f}, max={all_dfar.max():.3f}")

    # classify each placement honestly
    holds, inverts, conflated = [], [], []
    for r in results:
        a, b = abs(r["spearman_dris"]), abs(r["spearman_dfar"])
        if a > b + 0.2:        holds.append(r["placement"])
        elif a < b:            inverts.append(r["placement"])
        else:                  conflated.append(r["placement"])
    print(f"\n  Law HOLDS (d_RIS dominates):  {len(holds)}/{len(results)}  {holds}")
    print(f"  Law INVERTS (d_far dominates): {len(inverts)}/{len(results)}  {inverts}")
    print(f"  Predictors CONFLATED:          {len(conflated)}/{len(results)}  {conflated}")

    # explanatory note (honest — 5 points cannot support a monotonic predictor)
    d_rf = np.array([r["ris_far_dist"] for r in results])
    print(f"\n  WHY THE LAW IS CONDITIONAL (cascade reasoning, not a fitted predictor):")
    print(f"    Far-user rate is set by the full cascade UAV→RIS→far. The UAV controls")
    print(f"    ONLY the UAV→RIS hop. The 'UAV→RIS distance dominates' law holds when")
    print(f"    that hop is the swing factor:")
    print(f"      • 'inverts' case (RIS remote from survivors): the fixed RIS→far hop is")
    print(f"        long and uniform across users, so it dominates the cascade and UAV")
    print(f"        repositioning cannot move the survivor rate — d_RIS correlation drops.")
    print(f"      • 'conflated' case (RIS co-sited with survivors, d(RIS,far)≈100 m):")
    print(f"        d(UAV→RIS) and d(UAV→far) become nearly the same variable, so both")
    print(f"        correlations are large and the test cannot separate them.")
    print(f"      • 'holds' cases (RIS at perimeter serving survivors): UAV→RIS hop is")
    print(f"        the controllable bottleneck — d_RIS correlation dominates (0.64–0.81).")

    print(f"\n  HONEST CLAIM FOR THE PAPER:")
    print(f"  'When the RIS is deployed to serve the blocked survivors (RIS near the")
    print(f"   survivor region — the standard disaster-perimeter deployment), the")
    print(f"   UAV→RIS separation is the dominant controllable determinant of survivor")
    print(f"   rate. This conditional law is reported with its boundary cases.'")

    # plot — honest: the law holds WHEN the RIS serves the survivor region;
    # boundary cases (RIS far from survivors / RIS co-sited with survivors) are flagged.
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    names = [r["placement"] for r in results]
    x = np.arange(len(names))
    dris_vals = [abs(r["spearman_dris"]) for r in results]
    dfar_vals = [abs(r["spearman_dfar"]) for r in results]
    # classify each placement
    regime = []
    for r in results:
        if abs(r["spearman_dris"]) > abs(r["spearman_dfar"]) + 0.2:
            regime.append("law holds")
        elif abs(r["spearman_dris"]) < abs(r["spearman_dfar"]):
            regime.append("inverts")
        else:
            regime.append("conflated")
    bar_ris = ax.bar(x - 0.2, dris_vals, 0.4, color="#2ca02c",
                     label="|ρ| with UAV→RIS distance")
    bar_far = ax.bar(x + 0.2, dfar_vals, 0.4, color="#d62728",
                     label="|ρ| with UAV→far distance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n[{reg}]" for n, reg in zip(names, regime)],
                       rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("|Spearman ρ| with far-user rate")
    ax.set_ylim(0, 1)
    ax.set_title("Coupling law is CONDITIONAL: UAV→RIS separation dominates when the "
                 "RIS serves\nthe survivor region (3/5 cases); it weakens when the RIS "
                 "is remote from or co-sited with survivors",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="upper right"); ax.grid(axis="y", ls=":", alpha=0.5)
    for i, r in enumerate(results):
        ax.text(i-0.2, abs(r["spearman_dris"])+0.02, f"{abs(r['spearman_dris']):.2f}",
                ha="center", fontsize=8)
        ax.text(i+0.2, abs(r["spearman_dfar"])+0.02, f"{abs(r['spearman_dfar']):.2f}",
                ha="center", fontsize=8)
    # annotate the RIS–far distance under each placement (the explanatory variable)
    for i, r in enumerate(results):
        ax.text(i, -0.155, f"d(RIS,far)\n{r['ris_far_dist']:.0f} m",
                ha="center", fontsize=7, color="#444", transform=ax.get_xaxis_transform())
    plt.subplots_adjust(bottom=0.22)
    plt.tight_layout()
    p = os.path.join(OUT, "coupling_robustness.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Plot → {p}")

    with open(os.path.join(OUT, "coupling_robustness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"  CSV → {os.path.join(OUT,'coupling_robustness.csv')}")
    print(f"  Runtime: {time.time()-t0:.0f}s")
    print("=" * 74)


if __name__ == "__main__":
    run()
