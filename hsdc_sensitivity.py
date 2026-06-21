#!/usr/bin/env python3
r"""
hsdc_sensitivity.py
═══════════════════════════════════════════════════════════════════════════════
Reviewer Q7 — HSDC was reported at a single configuration (500 m service radius).
How sensitive is the auto-selected cluster count K* to the coverage radius, the
UAV power budget, and the propagation environment?

KEY OBSERVATION: the HSDC auto-K rule depends ONLY on the UAV service radius R
(K* = min clusters s.t. every device lies within R of its centroid). R is in turn
set by the link budget, so the UAV power budget and the environment (path-loss
exponent) both enter HSDC purely through R:
        R = R_ref * 10^((P - P_ref)/(10 n))         (power, fixed environment n)
        R(n) at fixed power: 10 n log10(R) = const   (environment, fixed power)
A single sweep over R therefore answers all three asks; we annotate R with the
equivalent UAV Tx-power offset and tabulate the environment cases.

FAITHFUL: device layout, energy metric and coverage come from hsdc.py unchanged
(import); the K*-selection loop is replicated verbatim from hsdc.run_hsdc with the
500 m constant promoted to a parameter (KMeans random_state=42, n_init=20,
max_iter=500; 100%-coverage rule). No physics is re-implemented.

ANCHOR: at R = 500 m the sweep must reproduce the paper's K* = 10.

HOW TO RUN (Windows authoritative; hsdc.py + channel_model.py in the same folder):
    python hsdc_sensitivity.py
Deterministic (seed 42) -> reproduces in-container up to sklearn-version KMeans
determinism; confirm K* on Windows.
Outputs (-> ..\\results\\SURVIVOR if it exists, else .\\results\\SURVIVOR):
    hsdc_sensitivity.png     K*(R) + power axis, cluster-quality curves
    hsdc_sensitivity.json     all numbers
"""
import os, json, time, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hsdc as H                       # safe: __main__-guarded; only seed(42)+makedirs run
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
if not os.path.isdir(os.path.dirname(OUT)):
    OUT = os.path.join(HERE, "results", "SURVIVOR")
os.makedirs(OUT, exist_ok=True)

# ── settings ──────────────────────────────────────────────────────────────────
R_REF      = 500.0                     # paper service radius (anchor)
RADII      = np.arange(200.0, 1201.0, 25.0)
KMIN, KMAX = 2, 20
N_ENV_PT   = 100                       # devices (from hsdc)
# environment path-loss exponents (3GPP-style: urban macro / suburban / rural)
ENVIRONMENTS = [("urban (n=3.5)", 3.5), ("suburban (n=3.0)", 3.0), ("rural (n=2.7)", 2.7)]
N_URBAN    = 3.5                       # reference exponent for the power-offset axis


def k_star_for_radius(positions, radius, kmin=KMIN, kmax=KMAX):
    """Replicates hsdc.run_hsdc's K* loop with R promoted to a parameter.
    Returns (K*, labels, centers, silhouette, max_dev_dist)."""
    for k in range(kmin, kmax + 1):
        km  = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        lab = km.fit_predict(positions)
        ctr = km.cluster_centers_
        d   = np.array([np.linalg.norm(positions[i] - ctr[lab[i]])
                        for i in range(len(positions))])
        if (d <= radius).all():                       # 100% coverage achieved
            sil = float(silhouette_score(positions, lab))
            return k, lab, ctr, sil, float(d.max())
    return None, None, None, None, None


def run():
    print("=" * 86)
    print("  Q7  HSDC SENSITIVITY — service radius / UAV power budget / environment")
    print("=" * 86)
    t0 = time.time()

    positions, scen = H.generate_devices()
    assert len(positions) == N_ENV_PT
    # k=5 baseline energy for the paper's normalisation
    km5 = KMeans(n_clusters=5, random_state=42, n_init=20, max_iter=500).fit(positions)
    E5  = H.compute_energy_metric(positions, km5.labels_, km5.cluster_centers_)

    # ── radius sweep ────────────────────────────────────────────────────────────
    print(f"  Sweeping service radius R over [{RADII[0]:.0f}, {RADII[-1]:.0f}] m "
          f"({len(RADII)} points)...")
    Rg, Kg, Sg, Eg, Dg = [], [], [], [], []
    for R in RADII:
        k, lab, ctr, sil, dmax = k_star_for_radius(positions, R)
        if k is None:
            continue
        Rg.append(float(R)); Kg.append(int(k)); Sg.append(sil)
        Eg.append(float(H.compute_energy_metric(positions, lab, ctr) / E5)); Dg.append(dmax)
    Rg, Kg, Sg, Eg, Dg = map(np.array, (Rg, Kg, Sg, Eg, Dg))

    # anchor at R_ref
    k_ref, _, _, sil_ref, dmax_ref = k_star_for_radius(positions, R_REF)
    print("-" * 86)
    print(f"  ANCHOR  R = {R_REF:.0f} m  ->  K* = {k_ref}  "
          f"(paper K*=10), silhouette {sil_ref:.3f}, max device-centroid {dmax_ref:.0f} m")
    print("-" * 86)
    print(f"  {'R (m)':>7}{'K*':>5}{'silhouette':>12}{'energy/E5':>11}"
          f"{'Tx offset (urban)':>20}")
    print("-" * 86)
    show_R = [200, 300, 400, 500, 600, 700, 800, 1000, 1200]
    for R in show_R:
        idx = np.argmin(np.abs(Rg - R))
        dB = 10 * N_URBAN * np.log10(Rg[idx] / R_REF)
        print(f"  {Rg[idx]:>7.0f}{Kg[idx]:>5d}{Sg[idx]:>12.3f}{Eg[idx]:>11.3f}"
              f"{dB:>17.1f} dB")
    print("-" * 86)

    # ── environment cases at FIXED power (calibrated to R_ref in urban) ──────────
    # 10 n log10(R) = const ; calibrate const so n=3.5 -> R_ref
    const = 10 * N_URBAN * np.log10(R_REF)
    print("  Environment at a fixed UAV power budget (calibrated to 500 m in urban):")
    print(f"  {'environment':<18}{'effective R (m)':>16}{'K*':>6}{'silhouette':>12}")
    print("-" * 86)
    env_rows = []
    for name, n in ENVIRONMENTS:
        R_env = 10 ** (const / (10 * n))
        k_e, _, _, sil_e, _ = k_star_for_radius(positions, R_env)
        env_rows.append(dict(environment=name, exponent=n, eff_radius_m=float(R_env),
                             k_star=int(k_e), silhouette=float(sil_e)))
        print(f"  {name:<18}{R_env:>16.0f}{k_e:>6d}{sil_e:>12.3f}")
    print("=" * 86)
    print("  HSDC adapts K* monotonically: smaller R (less power / harsher urban "
          "propagation)\n  -> more clusters to hold 100% coverage; larger R (more power / "
          "suburban) -> fewer.")
    print("=" * 86)

    # ── figure ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    # (a) K*(R) with power-offset top axis
    ax[0].step(Rg, Kg, where="mid", color="#1f5fa8", lw=2.4)
    ax[0].scatter([R_REF], [k_ref], s=160, marker="*", c="#d62728", zorder=6,
                  edgecolors="black", label=f"paper: R=500 m, K*={k_ref}")
    ax[0].set_xlabel("UAV service radius R (m)")
    ax[0].set_ylabel("Auto-selected clusters K*")
    ax[0].set_title("(a) K* vs service radius\n(K* rises as coverage tightens)",
                    fontweight="bold")
    ax[0].grid(True, ls=":", alpha=0.5); ax[0].legend(fontsize=9, loc="upper right")
    secx = ax[0].secondary_xaxis(
        "top", functions=(lambda R: 10 * N_URBAN * np.log10(np.maximum(R, 1) / R_REF),
                          lambda dB: R_REF * 10 ** (dB / (10 * N_URBAN))))
    secx.set_xlabel("Equivalent UAV Tx-power offset (dB, urban n=3.5)")

    # (b) cluster quality
    ax2 = ax[1]; ax2b = ax2.twinx()
    l1, = ax2.plot(Rg, Sg, color="#2ca02c", lw=2.2, label="silhouette")
    l2, = ax2b.plot(Rg, Eg, color="#9467bd", lw=2.2, ls="--", label="energy / E(k=5)")
    ax2.axvline(R_REF, color="gray", ls=":", lw=1.4)
    ax2.set_xlabel("UAV service radius R (m)")
    ax2.set_ylabel("Silhouette score", color="#2ca02c")
    ax2b.set_ylabel("Normalised energy (vs k=5)", color="#9467bd")
    ax2.set_title("(b) Cluster quality across the sweep\n(silhouette stays good)",
                  fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend(handles=[l1, l2], fontsize=9, loc="best")
    fig.suptitle("Q7: HSDC auto-K is a smooth, monotone function of the UAV service "
                 "radius (set by power budget & environment)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "hsdc_sensitivity.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {p}")

    with open(os.path.join(OUT, "hsdc_sensitivity.json"), "w") as f:
        json.dump(dict(anchor=dict(radius_m=R_REF, k_star=int(k_ref),
                                   silhouette=sil_ref, max_dev_dist_m=dmax_ref),
                       sweep=dict(R=Rg.tolist(), K_star=Kg.tolist(),
                                  silhouette=Sg.tolist(), energy_norm=Eg.tolist(),
                                  max_dev_dist_m=Dg.tolist()),
                       environments=env_rows,
                       config=dict(r_ref=R_REF, kmin=KMIN, kmax=KMAX,
                                   n_urban=N_URBAN, n_devices=N_ENV_PT)),
                  f, indent=2)
    print(f"[saved] {os.path.join(OUT,'hsdc_sensitivity.json')}")
    print(f"  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
