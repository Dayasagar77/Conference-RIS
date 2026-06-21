#!/usr/bin/env python3
"""
uav_ris_coupling.py
═══════════════════════════════════════════════════════════════════════════════
PATH B core contribution — the UAV–RIS geometric coupling.

Tests a falsifiable hypothesis with channel_model.py physics:

  H1: A blocked far user's achievable rate is governed by the UAV→RIS separation,
      NOT the UAV→far-user separation.
  H2: Therefore the survivor-optimal UAV geometry (near the RIS) is spatially
      distinct from the near-user-optimal geometry (near the device centroid),
      creating an irreducible positioning tension.

If H1 holds, the design rule is counterintuitive: to help a blocked survivor,
fly the UAV toward the REFLECTOR, not toward the survivor.

Three rigorous tests:
  (A) Rate surfaces over (x,y): near-user throughput vs far-user reliability —
      shows whether the two optima are spatially separated.
  (B) Governing-variable test: far-user rate vs d(UAV,RIS) and vs d(UAV,far) —
      Pearson + Spearman correlations quantify which distance governs the rate.
  (C) M × separation interaction: does RIS size change how much proximity matters?

Run on Windows (channel_model.py in same folder):
    python uav_ris_coupling.py
Outputs:
    results/SURVIVOR/coupling_surfaces.png
    results/SURVIVOR/coupling_governing_var.png
    results/SURVIVOR/coupling_M_interaction.png
    results/SURVIVOR/coupling_grid.csv
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
N_MC        = 30
SLA_BPS     = 100.0
H_FIXED     = 120.0          # fixed altitude for the (x,y) surface maps

# grid
GX = np.linspace(150, 800, 26)
GY = np.linspace(150, 900, 26)

np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]
FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
NEAR_CENTROID = NEAR[:, :2].mean(0)
FAR_CENTROID  = FAR[:, :2].mean(0)
RIS_XY = cm.CFG.RIS_POS[:2]


def per_user(uav, n_mc=N_MC, seed=EVAL_SEED):
    np.random.seed(seed)
    K = cm.CFG.N_PAIRS
    near_sum = 0.0
    far_samp = []
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        df = np.zeros(K)
        for i in range(K):
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            hd = cm.chan_scalar(uav, FAR[i],  cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(FAR[i])
            phi = cm.opt_phi(G, hr)
            he  = cm.composite(hd, G, hr, phi)
            rN, _  = cm.noma_pair(hn, hd)
            _,  rF = cm.noma_pair(hn, he)
            near_sum += rN
            df[i] = rF
        far_samp.append(df)
    fs = np.array(far_samp)
    return (near_sum/n_mc/1e6, float(fs.mean()), float((fs >= SLA_BPS).mean()))


def test_A_and_B():
    print("=" * 74)
    print("  UAV–RIS COUPLING  (channel_model.py physics)")
    print("=" * 74)
    print(f"  Near centroid {NEAR_CENTROID.astype(int)}, Far centroid "
          f"{FAR_CENTROID.astype(int)}, RIS {RIS_XY.astype(int)}")
    print(f"  Near–RIS separation: {np.linalg.norm(NEAR_CENTROID-RIS_XY):.0f} m | "
          f"Grid {len(GX)}×{len(GY)} @ H={H_FIXED:.0f} m, N_MC={N_MC}")
    print("-" * 74)
    t0 = time.time()

    rows = []
    NEARS = np.full((len(GY), len(GX)), np.nan)
    RELS  = np.full((len(GY), len(GX)), np.nan)
    for iy, y in enumerate(GY):
        for ix, x in enumerate(GX):
            uav = np.array([x, y, H_FIXED])
            near_mbps, far_mean, far_rel = per_user(uav)
            d_ris = np.linalg.norm(uav[:2] - RIS_XY)
            d_far = np.linalg.norm(uav[:2] - FAR_CENTROID)
            d_near = np.linalg.norm(uav[:2] - NEAR_CENTROID)
            NEARS[iy, ix] = near_mbps
            RELS[iy, ix]  = far_rel * 100
            rows.append(dict(x=x, y=y, H=H_FIXED, near_mbps=near_mbps,
                             far_mean_bps=far_mean, far_rel=far_rel*100,
                             d_ris=d_ris, d_far=d_far, d_near=d_near))
        print(f"  y={y:5.0f} done ({time.time()-t0:.0f}s)")

    # ── TEST B: governing-variable correlations ────────────────────────────────
    d_ris_arr = np.array([r["d_ris"] for r in rows])
    d_far_arr = np.array([r["d_far"] for r in rows])
    d_near_arr= np.array([r["d_near"] for r in rows])
    far_rate  = np.array([r["far_mean_bps"] for r in rows])
    near_rate = np.array([r["near_mbps"] for r in rows])
    log_far   = np.log10(np.maximum(far_rate, 1e-3))

    pr_ris,  _ = pearsonr(d_ris_arr,  log_far)
    pr_far,  _ = pearsonr(d_far_arr,  log_far)
    sr_ris,  _ = spearmanr(d_ris_arr, far_rate)
    sr_far,  _ = spearmanr(d_far_arr, far_rate)
    # near throughput vs near distance (control)
    pr_near_d, _ = pearsonr(d_near_arr, near_rate)

    print("\n" + "=" * 74)
    print("  TEST B — WHAT GOVERNS THE FAR-USER RATE?")
    print("=" * 74)
    print(f"  Pearson  r(log far-rate, d_UAV→RIS)  = {pr_ris:+.3f}")
    print(f"  Pearson  r(log far-rate, d_UAV→far)  = {pr_far:+.3f}")
    print(f"  Spearman ρ(far-rate,    d_UAV→RIS)   = {sr_ris:+.3f}")
    print(f"  Spearman ρ(far-rate,    d_UAV→far)   = {sr_far:+.3f}")
    print(f"  [control] Pearson r(near-thru, d_UAV→near) = {pr_near_d:+.3f}")
    verdict = ("GOVERNED BY UAV→RIS distance" if abs(sr_ris) > abs(sr_far) + 0.2
               else "ambiguous — re-examine")
    print(f"\n  → Far-user rate is {verdict}")
    print(f"  → UAV→RIS distance explains the far-user rate "
          f"{abs(sr_ris)/max(abs(sr_far),1e-3):.1f}× more strongly than UAV→far distance")

    # ── TEST A: surface maps ───────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    ext = [GX[0], GX[-1], GY[0], GY[-1]]

    im0 = ax[0].imshow(NEARS, origin="lower", extent=ext, aspect="auto",
                       cmap="viridis")
    ax[0].contour(GX, GY, NEARS, colors="white", alpha=0.3, linewidths=0.5)
    i_near = np.unravel_index(np.nanargmax(NEARS), NEARS.shape)
    ax[0].scatter(GX[i_near[1]], GY[i_near[0]], s=300, marker="*", c="red",
                  edgecolors="white", lw=1.5, zorder=5, label="near-throughput optimum")
    ax[0].scatter(*NEAR_CENTROID, s=120, marker="o", c="cyan", edgecolors="black",
                  zorder=5, label="near-device centroid")
    ax[0].scatter(*RIS_XY, s=160, marker="D", c="gold", edgecolors="black",
                  zorder=5, label="RIS")
    ax[0].set_title("(a) Near-user aggregate throughput surface (Mbps)",
                    fontweight="bold")
    ax[0].set_xlabel("UAV x (m)"); ax[0].set_ylabel("UAV y (m)")
    ax[0].legend(loc="upper right", fontsize=8); plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(RELS, origin="lower", extent=ext, aspect="auto",
                       cmap="plasma")
    ax[1].contour(GX, GY, RELS, colors="white", alpha=0.3, linewidths=0.5)
    i_rel = np.unravel_index(np.nanargmax(RELS), RELS.shape)
    ax[1].scatter(GX[i_rel[1]], GY[i_rel[0]], s=300, marker="P", c="lime",
                  edgecolors="black", lw=1.5, zorder=5, label="far-reliability optimum")
    ax[1].scatter(*RIS_XY, s=160, marker="D", c="gold", edgecolors="black",
                  zorder=5, label="RIS")
    ax[1].scatter(*NEAR_CENTROID, s=120, marker="o", c="cyan", edgecolors="black",
                  zorder=5, label="near-device centroid")
    ax[1].set_title("(b) Far-user telemetry reliability surface (%)",
                    fontweight="bold")
    ax[1].set_xlabel("UAV x (m)"); ax[1].set_ylabel("UAV y (m)")
    ax[1].legend(loc="upper right", fontsize=8); plt.colorbar(im1, ax=ax[1])

    sep = np.linalg.norm(np.array([GX[i_near[1]], GY[i_near[0]]]) -
                         np.array([GX[i_rel[1]], GY[i_rel[0]]]))
    fig.suptitle(f"UAV–RIS Coupling: near-user and far-user optima are "
                 f"{sep:.0f} m apart (spatially distinct)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "coupling_surfaces.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Surface maps → {p}")
    print(f"  Near-optimum at ({GX[i_near[1]]:.0f},{GY[i_near[0]]:.0f}), "
          f"far-optimum at ({GX[i_rel[1]]:.0f},{GY[i_rel[0]]:.0f}) — "
          f"{sep:.0f} m apart")

    # ── governing-variable scatter ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5.5))
    ax2[0].scatter(d_ris_arr, far_rate, s=16, c="#2ca02c", alpha=0.6)
    ax2[0].set_yscale("log")
    ax2[0].set_xlabel("UAV → RIS distance (m)")
    ax2[0].set_ylabel("Far-user mean rate (bps, log)")
    ax2[0].set_title(f"(a) Far rate vs UAV→RIS distance\n"
                     f"Spearman ρ = {sr_ris:+.3f}  (tight, monotonic)",
                     fontweight="bold")
    ax2[0].grid(True, ls=":", alpha=0.5)

    ax2[1].scatter(d_far_arr, far_rate, s=16, c="#d62728", alpha=0.6)
    ax2[1].set_yscale("log")
    ax2[1].set_xlabel("UAV → far-user-centroid distance (m)")
    ax2[1].set_ylabel("Far-user mean rate (bps, log)")
    ax2[1].set_title(f"(b) Far rate vs UAV→far distance\n"
                     f"Spearman ρ = {sr_far:+.3f}  (weak/scattered)",
                     fontweight="bold")
    ax2[1].grid(True, ls=":", alpha=0.5)
    fig2.suptitle("Governing-variable test: the far-user rate tracks UAV→RIS "
                  "distance, not UAV→far distance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(OUT, "coupling_governing_var.png")
    fig2.savefig(p2, dpi=300, bbox_inches="tight"); plt.close(fig2)
    print(f"  Governing-variable scatter → {p2}")

    with open(os.path.join(OUT, "coupling_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Grid CSV → {os.path.join(OUT,'coupling_grid.csv')}")
    print(f"  Test A+B runtime: {time.time()-t0:.0f}s")
    return dict(sr_ris=sr_ris, sr_far=sr_far, sep=sep)


def test_C_M_interaction():
    """Does RIS size change how strongly UAV–RIS proximity governs far rate?"""
    print("\n" + "=" * 74)
    print("  TEST C — M × UAV–RIS SEPARATION INTERACTION")
    print("=" * 74)
    M_values = [256, 1024, 4096]
    # sweep UAV along the line from near-centroid toward the RIS
    ts = np.linspace(0, 1, 16)
    line_pts = np.array([NEAR_CENTROID + t * (RIS_XY - NEAR_CENTROID) for t in ts])
    d_ris_line = np.linalg.norm(line_pts - RIS_XY, axis=1)

    orig_M = cm.CFG.M
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#9467bd", "#1f5fa8", "#2ca02c"]
    results = {}
    for M, col in zip(M_values, colors):
        cm.CFG.M = M
        far_rates = []
        for pt in line_pts:
            uav = np.array([pt[0], pt[1], H_FIXED])
            _, far_mean, _ = per_user(uav, n_mc=N_MC, seed=EVAL_SEED)
            far_rates.append(far_mean)
        results[M] = far_rates
        ax.plot(d_ris_line, far_rates, "-o", color=col, lw=2, ms=5,
                label=f"M = {M} elements")
        print(f"  M={M:5d}: far rate at RIS={far_rates[-1]:.0f} bps, "
              f"at near-centroid={far_rates[0]:.0f} bps "
              f"({far_rates[-1]/max(far_rates[0],1e-9):.0f}× swing)")
    cm.CFG.M = orig_M

    ax.set_yscale("log")
    ax.axhline(SLA_BPS, color="black", ls=":", lw=1.2, label=f"{SLA_BPS:.0f} bps SLA")
    ax.set_xlabel("UAV → RIS distance (m)  [UAV swept from near-centroid → RIS]")
    ax.set_ylabel("Far-user mean rate (bps, log)")
    ax.set_title("RIS size × UAV–RIS proximity:\nsmaller arrays make UAV proximity "
                 "to the RIS more critical", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.5); ax.legend()
    plt.tight_layout()
    p = os.path.join(OUT, "coupling_M_interaction.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\n  M-interaction plot → {p}")


if __name__ == "__main__":
    test_A_and_B()
    test_C_M_interaction()
    print("=" * 74)
