#!/usr/bin/env python3
r"""
fairness_oma_pareto.py
═══════════════════════════════════════════════════════════════════════════════
Reviewer Q8 — fairness-constrained OMA Pareto frontier, to contextualise the
+54.8% NOMA-over-OMA claim against a FAIRNESS-AWARE OMA rather than only the
throughput-greedy sum-rate OMA.

FAITHFUL EXTENSION of optimized_oma.py: identical physics (cm.oma_pair /
cm.noma_pair, blocked-direct far = no RIS, centroid @ H=125, seed-42 layout,
seed-99 MC, N_MC=100), identical (beta,t) OMA split grid. The ONLY change: each
pair's OMA split is optimised for an ALPHA-FAIR utility, sweeping alpha to trace
the OMA throughput-fairness frontier from sum-rate-optimal (alpha=0) to max-min.
Everything is no-RIS, exactly as optimized_oma.py's baselines — this is the
multiple-access-scheme comparison (the RIS far-user rescue is a separate axis,
covered by the coupling / phase-error studies).

alpha-fair per-pair utility maximised on the grid:
    alpha = 0   -> sum rate            (= optimized_oma.py's sum-rate-OPT OMA)
    alpha = 1   -> proportional fair   (sum log rate)
    alpha -> inf-> max-min             (maximise the weaker user's rate)

ANCHORS: alpha=0 aggregate reproduces optimized_oma.py's 236.84 Mbps; equal-split
OMA reproduces 120.93 Mbps; NOMA reproduces ~187 Mbps. If not, the harness is wrong.

HOW TO RUN (Windows authoritative; channel_model.py in the same folder):
    python fairness_oma_pareto.py
Physics-only -> reproduces exactly in-container.
Outputs (-> ..\\results\\SURVIVOR if it exists, else .\\results\\SURVIVOR):
    fairness_oma_pareto.png    throughput-fairness frontier + far-service bar
    fairness_oma_pareto.json    all numbers
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

# ── EXACT optimized_oma.py setup ──────────────────────────────────────────────
np.random.seed(42)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]; FAR = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
K = cm.CFG.N_PAIRS
BW, P, N0 = cm.CFG.BW_PAIR, cm.CFG.P_PAIR, cm.CFG.NOISE_W
UAV = np.array([cm.CFG.UAV_CX, cm.CFG.UAV_CY, 125.0])
N_MC = 100

betas = np.linspace(0.02, 0.98, 49)
ts    = np.linspace(0.02, 0.98, 49)
BB, TT = np.meshgrid(betas, ts)

ALPHAS = [0.0, 0.5, 1.0, 2.0]    # plus explicit max-min
EPS = 1e-12


def oma_rates_grid(hN2, hF2):
    """Per-pair OMA near/far rate arrays over the (beta,t) split grid — identical
    to optimized_oma.py::oma_opt_pair's rN/rF."""
    rN = BB * BW * np.log2(1 + (TT * P) * hN2 / (BB * N0))
    rF = (1 - BB) * BW * np.log2(1 + ((1 - TT) * P) * hF2 / ((1 - BB) * N0))
    return rN, rF


def alpha_fair_pick(hN2, hF2, alpha):
    rN, rF = oma_rates_grid(hN2, hF2)
    if alpha == 0.0:
        U = rN + rF
    elif alpha == 1.0:
        U = np.log(np.maximum(rN, EPS)) + np.log(np.maximum(rF, EPS))
    else:
        U = (np.maximum(rN, EPS) ** (1 - alpha) +
             np.maximum(rF, EPS) ** (1 - alpha)) / (1 - alpha)
    k = np.unravel_index(np.argmax(U), U.shape)
    return rN[k], rF[k]


def maxmin_pick(hN2, hF2):
    rN, rF = oma_rates_grid(hN2, hF2)
    k = np.unravel_index(np.argmax(np.minimum(rN, rF)), rN.shape)
    return rN[k], rF[k]


def eval_scheme(rate_fn):
    """MC over the fixed layout; identical RNG order to optimized_oma.py so every
    scheme is compared on the same channels. Returns per-user mean near/far rates."""
    np.random.seed(99)
    near_acc = np.zeros(K); far_acc = np.zeros(K)
    for _ in range(N_MC):
        G = cm.G_vector(UAV)                       # match RNG consumption (unused; no RIS)
        for i in range(K):
            hN  = cm.chan_scalar(UAV, NEAR[i], 0.0)
            hFd = cm.chan_scalar(UAV, FAR[i], cm.CFG.BLOCK_DB)
            rN, rF = rate_fn(hN, hFd)
            near_acc[i] += rN; far_acc[i] += rF
    return near_acc / N_MC, far_acc / N_MC


def metrics(near, far):
    allr = np.concatenate([near, far])
    agg  = allr.sum() / 1e6
    jain = float(allr.sum() ** 2 / (len(allr) * np.sum(allr ** 2)))
    return dict(agg_mbps=float(agg), far_mean_bps=float(far.mean()),
                far_total_bps=float(far.sum()), min_user_bps=float(allr.min()),
                jain=jain)


def run():
    print("=" * 90)
    print("  Q8  FAIRNESS-CONSTRAINED OMA PARETO  (no-RIS, centroid @ H=125, N_MC=%d)" % N_MC)
    print("=" * 90)
    t0 = time.time()

    schemes = []
    # OMA alpha-fair frontier
    for a in ALPHAS:
        nr, fr = eval_scheme(lambda hN, hFd, a=a: alpha_fair_pick(abs(hN) ** 2, abs(hFd) ** 2, a))
        m = metrics(nr, fr)
        name = ("OMA sum-rate (alpha=0)" if a == 0 else
                "OMA prop-fair (alpha=1)" if a == 1 else f"OMA alpha-fair (alpha={a})")
        schemes.append(dict(scheme=name, kind="oma_frontier", alpha=a, **m))
    # max-min OMA
    nr, fr = eval_scheme(lambda hN, hFd: maxmin_pick(abs(hN) ** 2, abs(hFd) ** 2))
    schemes.append(dict(scheme="OMA max-min", kind="oma_frontier", alpha=float("inf"),
                        **metrics(nr, fr)))
    # equal-split OMA (naive reference)
    nr, fr = eval_scheme(lambda hN, hFd: cm.oma_pair(hN, hFd))
    eq = metrics(nr, fr); schemes.append(dict(scheme="OMA equal-split", kind="oma_ref", **eq))
    # NOMA reference
    nr, fr = eval_scheme(lambda hN, hFd: cm.noma_pair(hN, hFd))
    nm = metrics(nr, fr); schemes.append(dict(scheme="NOMA (alpha=0.85)", kind="noma", **nm))

    # anchors
    a0 = next(s for s in schemes if s.get("alpha") == 0.0)
    print(f"  ANCHOR alpha=0 OMA  = {a0['agg_mbps']:.2f} Mbps (optimized_oma 236.84) | "
          f"far {a0['far_total_bps']:.0f} bps")
    print(f"  ANCHOR equal-split  = {eq['agg_mbps']:.2f} Mbps (paper 120.93)")
    print(f"  NOMA reference      = {nm['agg_mbps']:.2f} Mbps (paper ~187) | "
          f"far {nm['far_total_bps']:.0f} bps")
    print("-" * 90)
    print(f"  {'scheme':<26}{'aggregate':>11}{'far mean':>12}{'min-user':>12}{'Jain':>8}")
    print("-" * 90)
    for s in schemes:
        print(f"  {s['scheme']:<26}{s['agg_mbps']:>8.2f} Mb{s['far_mean_bps']:>9.1f} b"
              f"{s['min_user_bps']:>9.1f} b{s['jain']:>8.3f}")
    print("=" * 90)
    # honest verdict
    front = [s for s in schemes if s["kind"] == "oma_frontier"]
    best_far_oma = max(front, key=lambda s: s["far_mean_bps"])
    print(f"  OMA can trade throughput for far-service: sum-rate OMA {a0['agg_mbps']:.0f} Mb "
          f"@ {a0['far_mean_bps']:.0f} b/far  ->  max-min OMA "
          f"{best_far_oma['agg_mbps']:.0f} Mb @ {best_far_oma['far_mean_bps']:.0f} b/far")
    print(f"  NOMA: {nm['agg_mbps']:.0f} Mb @ {nm['far_mean_bps']:.0f} b/far, Jain {nm['jain']:.3f}")
    print("=" * 90)

    # ── figure ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    fx = [s["far_mean_bps"] for s in front]
    fy = [s["agg_mbps"] for s in front]
    order = np.argsort(fx)
    ax[0].plot(np.array(fx)[order], np.array(fy)[order], "-o", color="#1f5fa8",
               lw=2, ms=7, label="OMA \u03b1-fair frontier", zorder=3)
    for s in front:
        lab = "max-min" if np.isinf(s["alpha"]) else f"\u03b1={s['alpha']:g}"
        ax[0].annotate(lab, (s["far_mean_bps"], s["agg_mbps"]),
                       textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax[0].scatter(eq["far_mean_bps"], eq["agg_mbps"], s=130, marker="s", c="#9467bd",
                  edgecolors="black", zorder=4, label="OMA equal-split")
    ax[0].scatter(nm["far_mean_bps"], nm["agg_mbps"], s=320, marker="*", c="#d62728",
                  edgecolors="black", lw=1.2, zorder=5, label="NOMA (\u03b1=0.85)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Far-user (survivor) mean rate (bps, log)")
    ax[0].set_ylabel("Aggregate throughput (Mbps)")
    ax[0].set_title("(a) Throughput\u2013fairness frontier: OMA vs NOMA", fontweight="bold")
    ax[0].grid(True, ls=":", alpha=0.5); ax[0].legend(fontsize=8, loc="center left")

    names = [s["scheme"].replace(" (", "\n(") for s in schemes]
    far_b = [s["far_mean_bps"] for s in schemes]
    cols = ["#1f5fa8" if s["kind"] == "oma_frontier" else
            "#9467bd" if s["kind"] == "oma_ref" else "#d62728" for s in schemes]
    ax[1].bar(range(len(schemes)), far_b, color=cols)
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(schemes)))
    ax[1].set_xticklabels(names, fontsize=7, rotation=30, ha="right")
    ax[1].set_ylabel("Far-user mean rate (bps, log)")
    ax[1].set_title("(b) Survivor service per scheme", fontweight="bold")
    ax[1].grid(axis="y", ls=":", alpha=0.5)
    fig.suptitle("Q8: NOMA vs the fairness-aware OMA frontier (no-RIS, multiple-access "
                 "comparison)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "fairness_oma_pareto.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {p}")

    with open(os.path.join(OUT, "fairness_oma_pareto.json"), "w") as f:
        json.dump(dict(schemes=schemes, config=dict(
            uav=UAV.tolist(), n_mc=N_MC, alphas=ALPHAS, grid=len(betas))), f, indent=2)
    print(f"[saved] {os.path.join(OUT,'fairness_oma_pareto.json')}")
    print(f"  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
