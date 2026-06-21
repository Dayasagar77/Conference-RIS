#!/usr/bin/env python3
"""
coupling_topology_robustness.py  (STUDY A)
═══════════════════════════════════════════════════════════════════════════════
Multi-topology robustness of the HEADLINE coupling result (Reviewer-2 Q6:
"single-seed robustness of rho=-0.84 and the 533 m separation across topologies").

FAITHFUL EXTENSION of uav_ris_coupling.py: identical 26x26 grid, identical
H=120 m, identical N_MC=30, identical per_user physics and EVAL_SEED=99. The ONLY
addition is an outer loop over DEVICE_SEED that re-draws the survivor/near
topology each iteration, recomputing rho(d_RIS), rho(d_far), the Pearson
log-rate correlation, and the near/far optima separation per topology.

Resumable: each seed's result is appended to results/SURVIVOR/study_A_seeds.json
immediately, so a timeout/interrupt never loses completed seeds. Re-running skips
seeds already present.

USAGE
  python coupling_topology_robustness.py <seed_start> <seed_count> [n_mc]
      -> computes that block of seeds, appends to study_A_seeds.json
  python coupling_topology_robustness.py finalize
      -> reads study_A_seeds.json, prints aggregate stats, writes the figure

WINDOWS (authoritative): channel_model.py + uav_ris_coupling.py in same folder.
  python coupling_topology_robustness.py 100 30          # all 30 seeds (~37 min)
  python coupling_topology_robustness.py finalize
"""
import os, sys, json, time, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import channel_model as cm
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
if not os.path.isdir(os.path.dirname(OUT)):
    OUT = os.path.join(HERE, "results", "SURVIVOR")
os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(OUT, "study_A_seeds.json")

# ── EXACT uav_ris_coupling.py settings ────────────────────────────────────────
EVAL_SEED = 99
N_MC_DEF  = 30
H_FIXED   = 120.0
SLA_BPS   = 100.0
GX = np.linspace(150, 800, 26)
GY = np.linspace(150, 900, 26)
RIS_XY = cm.CFG.RIS_POS[:2].copy()          # baseline (300,250)
ANCHOR_SEED = 42                            # published headline topology


def per_user(uav, NEAR, FAR, n_mc, seed=EVAL_SEED):
    """Identical to uav_ris_coupling.py per_user (near_sum, far_mean, far_rel)."""
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
    return near_sum / n_mc / 1e6, float(fs.mean()), float((fs >= SLA_BPS).mean())


def sweep_one_seed(device_seed, n_mc):
    """Full 26x26 grid sweep for one device topology -> coupling metrics."""
    np.random.seed(device_seed)
    POS, _ = cm.generate_devices()
    NEAR = POS[:cm.CFG.K_NEAR]
    FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
    near_c = NEAR[:, :2].mean(0)
    far_c  = FAR[:, :2].mean(0)

    NEARS = np.full((len(GY), len(GX)), np.nan)
    RELS  = np.full((len(GY), len(GX)), np.nan)
    far_rate, d_ris, d_far = [], [], []
    for iy, y in enumerate(GY):
        for ix, x in enumerate(GX):
            uav = np.array([x, y, H_FIXED])
            nm, fmean, frel = per_user(uav, NEAR, FAR, n_mc)
            NEARS[iy, ix] = nm
            RELS[iy, ix]  = frel * 100
            far_rate.append(fmean)
            d_ris.append(np.linalg.norm(uav[:2] - RIS_XY))
            d_far.append(np.linalg.norm(uav[:2] - far_c))
    far_rate = np.array(far_rate)
    log_far = np.log10(np.maximum(far_rate, 1e-3))
    sr_ris, _ = spearmanr(np.array(d_ris), far_rate)
    sr_far, _ = spearmanr(np.array(d_far), far_rate)
    pr_ris, _ = pearsonr(np.array(d_ris), log_far)
    i_near = np.unravel_index(np.nanargmax(NEARS), NEARS.shape)
    i_rel  = np.unravel_index(np.nanargmax(RELS), RELS.shape)
    near_opt = [float(GX[i_near[1]]), float(GY[i_near[0]])]
    far_opt  = [float(GX[i_rel[1]]),  float(GY[i_rel[0]])]
    sep = float(np.linalg.norm(np.array(near_opt) - np.array(far_opt)))
    return dict(seed=int(device_seed), n_mc=int(n_mc),
                spearman_dris=float(sr_ris), spearman_dfar=float(sr_far),
                pearson_dris=float(pr_ris), near_opt=near_opt, far_opt=far_opt,
                separation_m=sep, far_centroid=[float(far_c[0]), float(far_c[1])])


def load_results():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            return json.load(f)
    return []


def save_result(rec):
    data = load_results()
    data = [d for d in data if d["seed"] != rec["seed"]]   # replace if re-run
    data.append(rec)
    data.sort(key=lambda d: d["seed"])
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)


def run_block(seed_start, seed_count, n_mc):
    done = {d["seed"] for d in load_results()}
    seeds = list(range(seed_start, seed_start + seed_count))
    print(f"[Study A] computing seeds {seeds} (N_MC={n_mc}); "
          f"{len(done)} already done")
    t0 = time.time()
    for s in seeds:
        if s in done:
            print(f"  seed {s}: already present, skipping")
            continue
        ts = time.time()
        rec = sweep_one_seed(s, n_mc)
        save_result(rec)
        print(f"  seed {s}: rho(d_RIS)={rec['spearman_dris']:+.3f} "
              f"rho(d_far)={rec['spearman_dfar']:+.3f} sep={rec['separation_m']:.0f}m "
              f"[{time.time()-ts:.0f}s, total {(time.time()-t0)/60:.1f}min]")


def finalize():
    data = [d for d in load_results() if d["seed"] != ANCHOR_SEED]  # topology seeds
    anchor = [d for d in load_results() if d["seed"] == ANCHOR_SEED]
    if not data:
        print("No topology-seed results yet."); return
    dris = np.array([d["spearman_dris"] for d in data])
    dfar = np.array([d["spearman_dfar"] for d in data])
    sep  = np.array([d["separation_m"] for d in data])
    n = len(data)
    from scipy.stats import t as tdist
    tcrit = float(tdist.ppf(0.975, n - 1)) if n > 1 else 0.0
    def ci(a):
        m, s = a.mean(), a.std(ddof=1) if n > 1 else 0.0
        h = tcrit * s / np.sqrt(n) if n > 1 else 0.0
        return m, s, (m - h, m + h)
    m_dris, s_dris, ci_dris = ci(dris)
    m_dfar, s_dfar, ci_dfar = ci(dfar)
    m_sep,  s_sep,  ci_sep  = ci(sep)
    holds_rate = float((np.abs(dris) > np.abs(dfar)).mean())

    print("=" * 78)
    print(f"  STUDY A — HEADLINE COUPLING ROBUSTNESS across {n} topologies")
    print("=" * 78)
    if anchor:
        a = anchor[0]
        print(f"  Reference (seed 42, published): rho(d_RIS)={a['spearman_dris']:+.3f} "
              f"rho(d_far)={a['spearman_dfar']:+.3f} sep={a['separation_m']:.0f} m")
    print(f"  rho(d_RIS) : mean {m_dris:+.3f} +/- {s_dris:.3f}  95% CI [{ci_dris[0]:+.3f},{ci_dris[1]:+.3f}]")
    print(f"  rho(d_far) : mean {m_dfar:+.3f} +/- {s_dfar:.3f}  95% CI [{ci_dfar[0]:+.3f},{ci_dfar[1]:+.3f}]")
    print(f"  separation : mean {m_sep:.0f} +/- {s_sep:.0f} m   95% CI [{ci_sep[0]:.0f},{ci_sep[1]:.0f}] m")
    print(f"  |rho(d_RIS)| > |rho(d_far)| in {holds_rate*100:.0f}% of topologies")
    print("=" * 78)

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # (a) the two correlation families
    parts = ax1.violinplot([dris, dfar], positions=[0, 1], showmeans=True, widths=0.7)
    for pc, c in zip(parts['bodies'], ["#2ca02c", "#d62728"]):
        pc.set_facecolor(c); pc.set_alpha(0.35)
    ax1.scatter(np.random.normal(0, 0.04, n), dris, c="#2ca02c", s=30, zorder=3, alpha=0.8)
    ax1.scatter(np.random.normal(1, 0.04, n), dfar, c="#d62728", s=30, zorder=3, alpha=0.8)
    if anchor:
        ax1.scatter([0], [anchor[0]["spearman_dris"]], marker="*", s=260,
                    c="gold", edgecolors="black", lw=1.2, zorder=5, label="seed-42 reference")
        ax1.scatter([1], [anchor[0]["spearman_dfar"]], marker="*", s=260,
                    c="gold", edgecolors="black", lw=1.2, zorder=5)
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["ρ(far rate, d→RIS)", "ρ(far rate, d→far)"])
    ax1.set_ylabel("Spearman ρ across topologies")
    ax1.set_title(f"(a) Governing-variable correlation over {n} topologies", fontweight="bold")
    ax1.legend(loc="center right", fontsize=8); ax1.grid(True, axis="y", ls=":", alpha=0.5)

    # (b) separation distribution
    ax2.hist(sep, bins=12, color="#1f5fa8", alpha=0.75, edgecolor="white")
    ax2.axvline(m_sep, color="navy", lw=2, label=f"mean {m_sep:.0f} m")
    if anchor:
        ax2.axvline(anchor[0]["separation_m"], color="gold", lw=2.5, ls="--",
                    label=f"seed-42 ({anchor[0]['separation_m']:.0f} m)")
    ax2.set_xlabel("Near-/far-optimum separation (m)")
    ax2.set_ylabel("Topologies")
    ax2.set_title("(b) Positioning-tension separation distribution", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, axis="y", ls=":", alpha=0.5)

    plt.suptitle(f"UAV–RIS coupling robustness across {n} independent device "
                 f"topologies (26×26 grid, N_MC={data[0]['n_mc']})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "coupling_topology_robustness.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {p}")

    summary = dict(n_topologies=n,
                   rho_dris=dict(mean=m_dris, std=s_dris, ci95=list(ci_dris)),
                   rho_dfar=dict(mean=m_dfar, std=s_dfar, ci95=list(ci_dfar)),
                   separation_m=dict(mean=m_sep, std=s_sep, ci95=list(ci_sep)),
                   holds_rate=holds_rate,
                   anchor_seed42=anchor[0] if anchor else None)
    with open(os.path.join(OUT, "study_A_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {os.path.join(OUT,'study_A_summary.json')}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "finalize":
        finalize()
    elif len(sys.argv) >= 3:
        start = int(sys.argv[1]); count = int(sys.argv[2])
        nmc = int(sys.argv[3]) if len(sys.argv) >= 4 else N_MC_DEF
        run_block(start, count, nmc)
    else:
        # No arguments -> run the FULL default study end-to-end (Windows: no timeout).
        print("No arguments -> running full default study: anchor seed 42 + 20 "
              "topology seeds (100-119) at N_MC=%d, then finalize.\n"
              "(For custom ranges or resuming a partial run, see USAGE in the header.)\n"
              % N_MC_DEF)
        run_block(ANCHOR_SEED, 1, N_MC_DEF)
        run_block(100, 20, N_MC_DEF)
        finalize()
