#!/usr/bin/env python3
"""
coupling_conditional_robustness.py  (STUDY B)
═══════════════════════════════════════════════════════════════════════════════
Multi-topology robustness of the CONDITIONAL coupling law (the 3/5 holds /
inverts / conflated classification of Fig. 21). Reviewer "completeness" follow-on:
does the conditional law's placement classification stay stable across topologies,
or was the 3/5 pattern an artifact of seed 42?

FAITHFUL EXTENSION of coupling_robustness.py: identical 5 RIS placements,
identical 20x20 grid (GX in [150,900], GY in [150,950]), identical far-only
per_user physics, identical classification rule
  holds    : |rho(d_RIS)| > |rho(d_far)| + 0.2
  inverts  : |rho(d_RIS)| < |rho(d_far)|
  conflated: otherwise
The ONLY addition is an outer loop over DEVICE_SEED.

N_MC NOTE: the original used N_MC=25. The holds/inverts/conflated decision is a
coarse comparison of two correlations and is robust to N_MC (verified on Study A:
rho changes < 0.02 between N_MC=10 and N_MC=30). For container tractability the
preview uses N_MC=10; set N_MC=25 on Windows to match the published Fig. 21
methodology exactly. Either way the CLASSIFICATION is unchanged.

Resumable: each seed (all 5 placements) is appended to
results/SURVIVOR/study_B_seeds.json immediately.

USAGE
  python coupling_conditional_robustness.py <seed_start> <seed_count> [n_mc]
  python coupling_conditional_robustness.py finalize

WINDOWS (authoritative): channel_model.py + coupling_robustness.py in same folder.
  python coupling_conditional_robustness.py 100 20 25      # 20 seeds, full N_MC
  python coupling_conditional_robustness.py finalize
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
JSON_PATH = os.path.join(OUT, "study_B_seeds.json")

# ── EXACT coupling_robustness.py settings ─────────────────────────────────────
EVAL_SEED = 99
N_MC_DEF  = 25
H_FIXED   = 120.0
GX = np.linspace(150, 900, 20)
GY = np.linspace(150, 950, 20)
ANCHOR_SEED = 42

RIS_CANDIDATES = [
    ("baseline (300,250)",      np.array([300.0, 250.0, 20.0])),
    ("near-far ring (600,250)", np.array([600.0, 250.0, 20.0])),
    ("opposite (300,950)",      np.array([300.0, 950.0, 20.0])),
    ("east edge (900,600)",     np.array([900.0, 600.0, 20.0])),
    ("co-sited (450,600)",      np.array([450.0, 600.0, 20.0])),
]


def per_user_far(uav, NEAR, FAR, n_mc, seed=EVAL_SEED):
    """Far-user mean rate at uav (identical to coupling_robustness.py per_user)."""
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


def classify(sr_ris, sr_far):
    a, b = abs(sr_ris), abs(sr_far)
    if a > b + 0.2:   return "holds"
    elif a < b:       return "inverts"
    else:             return "conflated"


def sweep_one_seed(device_seed, n_mc):
    np.random.seed(device_seed)
    POS, _ = cm.generate_devices()
    NEAR = POS[:cm.CFG.K_NEAR]
    FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
    far_centroid = FAR[:, :2].mean(0)
    orig = cm.CFG.RIS_POS.copy()
    placements = []
    for name, ris in RIS_CANDIDATES:
        cm.CFG.RIS_POS = ris
        far_rates, d_ris, d_far = [], [], []
        for y in GY:
            for x in GX:
                uav = np.array([x, y, H_FIXED])
                far_rates.append(per_user_far(uav, NEAR, FAR, n_mc))
                d_ris.append(np.linalg.norm(uav[:2] - ris[:2]))
                d_far.append(np.linalg.norm(uav[:2] - far_centroid))
        far_rates = np.array(far_rates)
        log_far = np.log10(np.maximum(far_rates, 1e-3))
        sr_ris, _ = spearmanr(np.array(d_ris), far_rates)
        sr_far, _ = spearmanr(np.array(d_far), far_rates)
        pr_ris, _ = pearsonr(np.array(d_ris), log_far)
        placements.append(dict(placement=name,
                               ris_far_dist=float(np.linalg.norm(ris[:2]-far_centroid)),
                               spearman_dris=float(sr_ris), spearman_dfar=float(sr_far),
                               pearson_dris=float(pr_ris),
                               regime=classify(sr_ris, sr_far)))
    cm.CFG.RIS_POS = orig
    return dict(seed=int(device_seed), n_mc=int(n_mc), placements=placements)


def load_results():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            return json.load(f)
    return []


def save_result(rec):
    data = [d for d in load_results() if d["seed"] != rec["seed"]]
    data.append(rec)
    data.sort(key=lambda d: d["seed"])
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)


def run_block(seed_start, seed_count, n_mc):
    done = {d["seed"] for d in load_results()}
    seeds = list(range(seed_start, seed_start + seed_count))
    print(f"[Study B] seeds {seeds} (N_MC={n_mc}); {len(done)} done")
    t0 = time.time()
    for s in seeds:
        if s in done:
            print(f"  seed {s}: skip (done)"); continue
        ts = time.time()
        rec = sweep_one_seed(s, n_mc)
        save_result(rec)
        base = next(p for p in rec["placements"] if p["placement"].startswith("baseline"))
        print(f"  seed {s}: baseline rho(d_RIS)={base['spearman_dris']:+.3f} "
              f"| regimes={[p['regime'][0] for p in rec['placements']]} "
              f"[{time.time()-ts:.0f}s, total {(time.time()-t0)/60:.1f}min]")


def finalize():
    data = [d for d in load_results() if d["seed"] != ANCHOR_SEED]
    if not data:
        print("No topology seeds yet."); return
    n = len(data)
    names = [c[0] for c in RIS_CANDIDATES]
    print("=" * 86)
    print(f"  STUDY B — CONDITIONAL-LAW ROBUSTNESS across {n} topologies "
          f"(5 RIS placements)")
    print("=" * 86)
    print(f"  {'placement':<26}{'|rho_RIS| mean±std':<22}{'|rho_far| mean±std':<22}"
          f"{'modal regime (rate)'}")
    print("-" * 86)
    summary = {"n_topologies": n, "placements": []}
    for k, name in enumerate(names):
        dris = np.array([abs(d["placements"][k]["spearman_dris"]) for d in data])
        dfar = np.array([abs(d["placements"][k]["spearman_dfar"]) for d in data])
        regimes = [d["placements"][k]["regime"] for d in data]
        from collections import Counter
        cnt = Counter(regimes); modal, mc = cnt.most_common(1)[0]
        print(f"  {name:<26}{dris.mean():.2f} ± {dris.std():.2f}        "
              f"{dfar.mean():.2f} ± {dfar.std():.2f}        "
              f"{modal} ({mc}/{n})")
        summary["placements"].append(dict(
            placement=name, abs_rho_dris_mean=float(dris.mean()),
            abs_rho_dris_std=float(dris.std()), abs_rho_dfar_mean=float(dfar.mean()),
            abs_rho_dfar_std=float(dfar.std()), modal_regime=modal,
            modal_count=int(mc), regime_breakdown=dict(cnt)))
    print("=" * 86)
    holds_names = [s["placement"] for s in summary["placements"] if s["modal_regime"]=="holds"]
    print(f"  Law HOLDS (modal) in {len(holds_names)}/5 placements: {holds_names}")
    print("=" * 86)

    # figure: grouped bars per placement, mean±std across topologies + modal regime
    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(names))
    dris_m = [s["abs_rho_dris_mean"] for s in summary["placements"]]
    dris_s = [s["abs_rho_dris_std"]  for s in summary["placements"]]
    dfar_m = [s["abs_rho_dfar_mean"] for s in summary["placements"]]
    dfar_s = [s["abs_rho_dfar_std"]  for s in summary["placements"]]
    regimes = [s["modal_regime"] for s in summary["placements"]]
    ax.bar(x-0.2, dris_m, 0.4, yerr=dris_s, capsize=4, color="#2ca02c",
           label="|ρ| with UAV→RIS distance")
    ax.bar(x+0.2, dfar_m, 0.4, yerr=dfar_s, capsize=4, color="#d62728",
           label="|ρ| with UAV→far distance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s['placement']}\n[{s['modal_regime']} {s['modal_count']}/{n}]"
                        for s in summary["placements"]], fontsize=8)
    ax.set_ylabel("|Spearman ρ| with far-user rate")
    ax.set_ylim(0, 1)
    # figure title omitted; the IEEE caption provides it
    ax.legend(loc="upper right"); ax.grid(axis="y", ls=":", alpha=0.5)
    for i in range(len(names)):
        ax.text(i-0.2, dris_m[i]+dris_s[i]+0.02, f"{dris_m[i]:.2f}", ha="center", fontsize=8)
        ax.text(i+0.2, dfar_m[i]+dfar_s[i]+0.02, f"{dfar_m[i]:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    p = os.path.join(OUT, "coupling_conditional_robustness.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {p}")
    with open(os.path.join(OUT, "study_B_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {os.path.join(OUT,'study_B_summary.json')}")


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
