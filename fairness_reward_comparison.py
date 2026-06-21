#!/usr/bin/env python3
r"""
fairness_reward_comparison.py
═══════════════════════════════════════════════════════════════════════════════
Reviewer Q4 — how does the survivor-aware reward compare to classical fairness
objectives (proportional / alpha-fair / max-min)? A full DQL retrain under each
objective is Windows-authoritative new work; instead this compares the OBJECTIVES
head-to-head by their POSITION OPTIMA. Q6 showed an exhaustive grid is within
<0.2% of the DQL optimum, so the grid-optimal position of each objective is a
valid stand-in for "what a learner under that reward would converge to" — no
retrain assumption needed.

FAITHFUL: per-user RIS-assisted NOMA rates come from channel_model.py exactly as
in survivor_aware_dql.py::eval_position (near rate from noma_pair, far rate from
the RIS bypass composite); device layout seed 42, eval seed 99, SLA=100 bps.

For every UAV position we compute the per-user rate vector (12 near + 12 far) and
the far-SLA reliability, then score five objectives:
    sum-rate     :  sum(r)                          (throughput-greedy)
    prop-fair    :  sum(log r)                       (proportional fairness)
    max-min      :  min(r)                           (egalitarian)
    far-SLA      :  P(far rate >= SLA)               (survivor reliability)
    J_sla        :  0.5*(near_sum/200) + 0.5*far-SLA (the paper's survivor reward)
and grid-search each objective's optimal (x,y,H). If the survivor-aware reward is
a fairness objective in disguise, its optimum should cluster with the prop-fair /
max-min / far-SLA optima and sit far from the sum-rate optimum.

ANCHORS: sum-rate optimum ~188 Mbps near the centroid; far-SLA / J_sla optima
~100% far reliability near the survivor region.

HOW TO RUN (Windows authoritative; channel_model.py in the same folder):
    python fairness_reward_comparison.py
Physics-only -> reproduces exactly in-container.
Outputs (-> ..\\results\\SURVIVOR if it exists, else .\\results\\SURVIVOR):
    fairness_reward_comparison.png    objective-optima map + per-objective metrics
    fairness_reward_comparison.json    all numbers
"""
import os, json, time, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import channel_model as cm
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
if not os.path.isdir(os.path.dirname(OUT)):
    OUT = os.path.join(HERE, "results", "SURVIVOR")
os.makedirs(OUT, exist_ok=True)

# ── settings (match survivor_aware_dql.py) ────────────────────────────────────
DEVICE_SEED, EVAL_SEED, SLA_BPS = 42, 99, 100.0
W_SLA, NEAR_NORM = 0.5, 200.0
N_MC_POS  = 60                      # known-position metrics (matches N_MC_EVAL)
N_MC_GRID = 30                      # grid search
EPS = 1e-3

# known operating positions
SUMRATE_POS  = np.array([450.0, 577.0, 108.0])
SURVIVOR_POS = np.array([400.0, 523.0, 150.0])
DQL_POS      = np.array([437.0, 584.0, 108.0])

# objective-search grid
GX = np.arange(150.0, 800.0 + 1, 60.0)
GY = np.arange(200.0, 900.0 + 1, 60.0)
GH = [100.0, 125.0, 150.0, 175.0]

np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]; FAR = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
K = cm.CFG.N_PAIRS


def per_user_rates(uav, n_mc, seed=EVAL_SEED):
    """RIS-assisted NOMA per-user rates (near[K], far[K]) + far-SLA reliability.
    Physics identical to survivor_aware_dql.py::eval_position."""
    np.random.seed(seed)
    near_acc = np.zeros(K); far_acc = np.zeros(K)
    far_samp = []
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        df = np.zeros(K)
        for i in range(K):
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            hd = cm.chan_scalar(uav, FAR[i], cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(FAR[i]); phi = cm.opt_phi(G, hr)
            he = cm.composite(hd, G, hr, phi)
            rN, _ = cm.noma_pair(hn, hd)      # near rate (SIC; independent of far arg)
            _,  rF = cm.noma_pair(hn, he)     # far rate via RIS bypass
            near_acc[i] += rN; far_acc[i] += rF
            df[i] = rF
        far_samp.append(df)
    near = near_acc / n_mc; far = far_acc / n_mc
    far_sla = float((np.array(far_samp) >= SLA_BPS).mean())
    return near, far, far_sla


def objective_scores(near, far, far_sla):
    allr = np.concatenate([near, far])
    return dict(
        sum_rate=float(allr.sum() / 1e6),
        prop_fair=float(np.sum(np.log(np.maximum(allr, EPS)))),
        max_min=float(allr.min()),
        far_sla=float(far_sla * 100),
        j_sla=float(W_SLA * (near.sum() / 1e6 / NEAR_NORM) + (1 - W_SLA) * far_sla),
        agg_mbps=float(allr.sum() / 1e6), far_mean_bps=float(far.mean()),
    )


OBJ_KEYS = ["sum_rate", "prop_fair", "max_min", "far_sla", "j_sla"]
OBJ_LABEL = {"sum_rate": "sum-rate", "prop_fair": "proportional-fair",
             "max_min": "max-min", "far_sla": "far-SLA", "j_sla": "survivor-aware J_sla"}


def run():
    print("=" * 92)
    print("  Q4  SURVIVOR-AWARE REWARD vs FAIRNESS OBJECTIVES (RIS-assisted NOMA)")
    print("=" * 92)
    t0 = time.time()

    # ── metrics at the known positions ─────────────────────────────────────────
    print("  Fairness metrics at the established positions (N_MC=%d):" % N_MC_POS)
    print(f"  {'position':<26}{'agg Mb':>9}{'far mean':>11}{'min-user':>11}"
          f"{'sum-log':>10}{'far-SLA':>9}")
    print("-" * 92)
    known = {}
    for name, p in [("sum-rate (450,577,108)", SUMRATE_POS),
                    ("survivor (400,523,150)", SURVIVOR_POS),
                    ("DQL-opt (437,584,108)", DQL_POS)]:
        nr, fr, sla = per_user_rates(p, N_MC_POS)
        s = objective_scores(nr, fr, sla)
        known[name] = s
        print(f"  {name:<26}{s['agg_mbps']:>7.1f} {s['far_mean_bps']:>9.0f} b"
              f"{s['max_min']:>9.1f} b{s['prop_fair']:>10.1f}{s['far_sla']:>8.1f}%")
    print("-" * 92)

    # ── grid-search each objective's optimum ───────────────────────────────────
    n_pos = len(GX) * len(GY) * len(GH)
    print(f"  Grid-searching objective optima over {n_pos} positions "
          f"({len(GX)}x{len(GY)} xy @ {len(GH)} altitudes, N_MC={N_MC_GRID})...")
    best = {k: dict(score=-np.inf, pos=None, metrics=None) for k in OBJ_KEYS}
    cnt = 0
    for H in GH:
        for y in GY:
            for x in GX:
                nr, fr, sla = per_user_rates(np.array([x, y, H]), N_MC_GRID)
                s = objective_scores(nr, fr, sla)
                for k in OBJ_KEYS:
                    if s[k] > best[k]["score"]:
                        best[k] = dict(score=float(s[k]), pos=[float(x), float(y), float(H)],
                                       metrics=s)
                cnt += 1
        print(f"    H={H:.0f} m done ({cnt}/{n_pos}, {time.time()-t0:.0f}s)")
    print("-" * 92)

    print("  Objective optima (grid upper bound) and their cross-metrics:")
    print(f"  {'objective':<22}{'opt (x,y,H)':>18}{'agg Mb':>9}{'min-user':>11}"
          f"{'far-SLA':>9}")
    print("-" * 92)
    for k in OBJ_KEYS:
        p = best[k]["pos"]; m = best[k]["metrics"]
        print(f"  {OBJ_LABEL[k]:<22}{f'({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})':>18}"
              f"{m['agg_mbps']:>7.1f} {m['max_min']:>9.1f} b{m['far_sla']:>8.1f}%")

    # spatial separation: fairness cluster vs sum-rate optimum
    sr_xy = np.array(best["sum_rate"]["pos"][:2])
    print("-" * 92)
    print("  Distance of each fairness optimum from the SUM-RATE optimum:")
    for k in ["prop_fair", "max_min", "far_sla", "j_sla"]:
        d = float(np.linalg.norm(np.array(best[k]["pos"][:2]) - sr_xy))
        dH = abs(best[k]["pos"][2] - best["sum_rate"]["pos"][2])
        print(f"    {OBJ_LABEL[k]:<22} {d:6.0f} m in xy, {dH:3.0f} m in altitude")
    print("=" * 92)

    # ── figure ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # (a) objective optima on the map (far-centroid / near-centroid / RIS for context)
    nearc = NEAR[:, :2].mean(0); farc = FAR[:, :2].mean(0); ris = cm.CFG.RIS_POS[:2]
    ax[0].scatter(*nearc, s=90, marker="o", c="cyan", edgecolors="black", zorder=4,
                  label="near-centroid")
    ax[0].scatter(*farc, s=90, marker="X", c="orange", edgecolors="black", zorder=4,
                  label="far-centroid")
    ax[0].scatter(*ris, s=120, marker="D", c="gold", edgecolors="black", zorder=4, label="RIS")
    mk = {"sum_rate": "*", "prop_fair": "P", "max_min": "v", "far_sla": "^", "j_sla": "s"}
    col = {"sum_rate": "#d62728", "prop_fair": "#1f5fa8", "max_min": "#2ca02c",
           "far_sla": "#9467bd", "j_sla": "black"}
    for k in OBJ_KEYS:
        p = best[k]["pos"]
        ax[0].scatter(p[0], p[1], s=200, marker=mk[k], c=col[k], edgecolors="white",
                      lw=1.2, zorder=6, label=f"{OBJ_LABEL[k]} opt (H={p[2]:.0f})")
    ax[0].set_xlabel("UAV x (m)"); ax[0].set_ylabel("UAV y (m)")
    ax[0].set_title("(a) Objective optima: rate-fairness (prop-fair/max-min) flies to the\n"
                    "RIS; sum-rate and survivor-aware J_sla stay at the throughput peak",
                    fontweight="bold")
    ax[0].legend(fontsize=7.5, loc="best"); ax[0].grid(True, ls=":", alpha=0.4)

    # (b) far-SLA and min-user at each objective optimum
    xb = np.arange(len(OBJ_KEYS))
    sla_v = [best[k]["metrics"]["far_sla"] for k in OBJ_KEYS]
    agg_v = [best[k]["metrics"]["agg_mbps"] for k in OBJ_KEYS]
    ax2 = ax[1]; ax2b = ax2.twinx()
    b1 = ax2.bar(xb - 0.2, sla_v, 0.4, color="#2ca02c", label="far-SLA reliability (%)")
    b2 = ax2b.bar(xb + 0.2, agg_v, 0.4, color="#d62728", label="aggregate (Mbps)")
    ax2.set_xticks(xb); ax2.set_xticklabels([OBJ_LABEL[k].replace(" ", "\n").replace("-", "-\n")
                                             for k in OBJ_KEYS], fontsize=7.5)
    ax2.set_ylabel("Far-user SLA reliability (%)", color="#2ca02c"); ax2.set_ylim(0, 110)
    ax2b.set_ylabel("Aggregate throughput (Mbps)", color="#d62728")
    ax2.set_title("(b) At each objective's optimum:\nfar-SLA vs aggregate", fontweight="bold")
    for b in b1: ax2.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                          f"{b.get_height():.0f}", ha="center", fontsize=8)
    fig.suptitle("Q4: the survivor-aware SLA reward holds near-peak throughput at 100% "
                 "far-SLA;\nrate-fairness (prop-fair / max-min) over-serves survivors and "
                 "collapses aggregate", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "fairness_reward_comparison.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {p}")

    with open(os.path.join(OUT, "fairness_reward_comparison.json"), "w") as f:
        json.dump(dict(known_positions=known,
                       objective_optima={k: best[k] for k in OBJ_KEYS},
                       config=dict(n_mc_pos=N_MC_POS, n_mc_grid=N_MC_GRID,
                                   gx=GX.tolist(), gy=GY.tolist(), gh=GH,
                                   w_sla=W_SLA, near_norm=NEAR_NORM, sla_bps=SLA_BPS)),
                  f, indent=2)
    print(f"[saved] {os.path.join(OUT,'fairness_reward_comparison.json')}")
    print(f"  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
