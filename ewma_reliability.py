#!/usr/bin/env python3
r"""
ewma_reliability.py
═══════════════════════════════════════════════════════════════════════════════
Reviewer Q3 — can the far-user SLA reliability P[R_far >= R_SLA] be tracked ONLINE
from streaming survivor beacons, without a full Monte-Carlo at deployment? This
designs and validates a lightweight EWMA estimator.

Deployment model: each survivor periodically transmits a telemetry beacon; for
every beacon the UAV observes the far-user achievable rate under the current
(fresh-fading) channel with the RIS reconfigured per beacon (opt_phi on the
instantaneous CSI). The UAV maintains an exponentially-weighted moving average of
the SLA indicator:
        rho_hat[t] = lambda * 1[R_far(t) >= R_SLA] + (1-lambda) * rho_hat[t-1]
This is O(1) per beacon, no Monte-Carlo, no stored history — the online analogue
of the training-time MC reliability.

FAITHFUL: each beacon's far-user rate comes from channel_model.py exactly as in
the MC evaluator (G_vector -> chan_scalar far-direct(blocked) -> H_ris_dev ->
opt_phi -> composite -> noma_pair far branch). The EWMA recursion is the only
addition.

VALIDATION: the EWMA steady state must converge to the MC ground truth at each
position — ~64.6% at the sum-rate position, ~100% at the survivor-aware position.

HOW TO RUN (Windows authoritative; channel_model.py in the same folder):
    python ewma_reliability.py
Physics-only -> reproduces exactly in-container.
Outputs (-> ..\\results\\SURVIVOR if it exists, else .\\results\\SURVIVOR):
    ewma_reliability.png     EWMA convergence vs MC truth
    ewma_reliability.json     all numbers
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

# ── settings ──────────────────────────────────────────────────────────────────
DEVICE_SEED, SLA_BPS = 42, 100.0
TRUTH_MC   = 4000          # channel realisations per far user for the MC ground truth
STREAM_LEN = 1500          # beacons in the online stream
LAMBDAS    = [0.02, 0.05, 0.10]
TRUTH_SEED, STREAM_SEED = 99, 2025
CONV_TOL   = 0.05          # |rho_hat - truth| convergence band

SUMRATE_POS  = np.array([450.0, 577.0, 108.0])   # MC truth ~64.6%
SURVIVOR_POS = np.array([400.0, 523.0, 150.0])   # MC truth ~100%

np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]; FAR = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
K = cm.CFG.N_PAIRS


def far_rate_sample(uav, i):
    """One beacon: instantaneous far-user rate for survivor i (RIS reconfigured on
    the current CSI). Identical physics to the MC evaluator's far branch."""
    G  = cm.G_vector(uav)
    hn = cm.chan_scalar(uav, NEAR[i], 0.0)
    hd = cm.chan_scalar(uav, FAR[i], cm.CFG.BLOCK_DB)
    hr = cm.H_ris_dev(FAR[i]); phi = cm.opt_phi(G, hr)
    he = cm.composite(hd, G, hr, phi)
    _, rF = cm.noma_pair(hn, he)
    return rF


def mc_truth(uav, n_mc=TRUTH_MC, seed=TRUTH_SEED):
    """Ground-truth P[R_far >= SLA] over n_mc realisations x K survivors."""
    np.random.seed(seed)
    hits = 0; tot = 0
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        for i in range(K):
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            hd = cm.chan_scalar(uav, FAR[i], cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(FAR[i]); phi = cm.opt_phi(G, hr)
            he = cm.composite(hd, G, hr, phi)
            _, rF = cm.noma_pair(hn, he)
            hits += (rF >= SLA_BPS); tot += 1
    return hits / tot


def stream_ewma(uav, truth, seed=STREAM_SEED):
    """Stream STREAM_LEN beacons (cycling survivors); EWMA the SLA indicator for
    each lambda. Returns curves + per-lambda steady-state and convergence latency."""
    np.random.seed(seed)
    ind = np.zeros(STREAM_LEN)
    for t in range(STREAM_LEN):
        i = t % K                                  # round-robin survivor beacons
        ind[t] = 1.0 if far_rate_sample(uav, i) >= SLA_BPS else 0.0
    out = {}
    for lam in LAMBDAS:
        rho = np.zeros(STREAM_LEN)
        r = ind[0]
        for t in range(STREAM_LEN):
            r = lam * ind[t] + (1 - lam) * r
            rho[t] = r
        ss = float(rho[-STREAM_LEN // 5:].mean())          # steady-state mean (last 20%)
        ss_std = float(rho[-STREAM_LEN // 5:].std())        # steady-state std (noise floor)
        entered = np.where(np.abs(rho - truth) <= CONV_TOL)[0]
        lat = int(entered[0]) if len(entered) else STREAM_LEN   # first-entry settling time
        out[lam] = dict(curve=rho, steady_state=ss, steady_std=ss_std,
                        abs_err=float(abs(ss - truth)), latency_beacons=lat)
    return ind, out


def run():
    print("=" * 86)
    print("  Q3  ONLINE EWMA FAR-SLA RELIABILITY ESTIMATOR  (no Monte-Carlo at deploy)")
    print("=" * 86)
    print(f"  SLA={SLA_BPS:.0f} bps | truth MC={TRUTH_MC} | stream={STREAM_LEN} beacons | "
          f"lambdas={LAMBDAS}")
    print("-" * 86)
    t0 = time.time()

    cases = [("sum-rate (450,577,108)", SUMRATE_POS, 64.6),
             ("survivor (400,523,150)", SURVIVOR_POS, 100.0)]
    results = {}
    curves = {}
    for name, pos, paper in cases:
        truth = mc_truth(pos)
        _, ew = stream_ewma(pos, truth)
        results[name] = dict(truth_pct=truth * 100, paper_pct=paper, ewma=ew)
        curves[name] = (truth, ew)
        print(f"  {name}: MC truth = {truth*100:.1f}%  (paper ~{paper:.1f}%)  "
              f"[{time.time()-t0:.0f}s]")
        for lam in LAMBDAS:
            e = ew[lam]
            print(f"      lambda={lam:<4}: EWMA steady = {e['steady_state']*100:5.1f}% "
                  f"(std {e['steady_std']*100:4.1f}%)  err {e['abs_err']*100:4.1f} pts  "
                  f"settled in {e['latency_beacons']:>4} beacons")
    print("=" * 86)

    # ── figure ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    cols = {0.02: "#9467bd", 0.05: "#1f5fa8", 0.10: "#2ca02c"}
    for a, (name, pos, paper) in zip(ax, cases):
        truth, ew = curves[name]
        for lam in LAMBDAS:
            a.plot(ew[lam]["curve"] * 100, color=cols[lam], lw=1.4,
                   label=f"EWMA \u03bb={lam}")
        a.axhline(truth * 100, color="black", ls="--", lw=1.6,
                  label=f"MC truth {truth*100:.1f}%")
        a.fill_between(range(STREAM_LEN), (truth - CONV_TOL) * 100, (truth + CONV_TOL) * 100,
                       color="gray", alpha=0.15, label=f"\u00b1{CONV_TOL*100:.0f} pts band")
        a.set_xlabel("Beacon index (online stream)")
        a.set_ylabel("Estimated far-SLA reliability (%)")
        a.set_ylim(0, 105)
        a.set_title(f"{name.split(' ')[0]} position", fontweight="bold")
        a.grid(True, ls=":", alpha=0.5); a.legend(fontsize=8, loc="lower right")
    fig.suptitle("Q3: O(1)-per-beacon EWMA tracks the Monte-Carlo far-SLA reliability "
                 "online (no deployment-time MC)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "ewma_reliability.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {p}")

    dump = {name: dict(truth_pct=results[name]["truth_pct"],
                       paper_pct=results[name]["paper_pct"],
                       ewma={str(l): {k: v for k, v in results[name]["ewma"][l].items()
                                       if k != "curve"} for l in LAMBDAS})
            for name in results}
    with open(os.path.join(OUT, "ewma_reliability.json"), "w") as f:
        json.dump(dict(results=dump, config=dict(sla_bps=SLA_BPS, truth_mc=TRUTH_MC,
                       stream_len=STREAM_LEN, lambdas=LAMBDAS, conv_tol=CONV_TOL)),
                  f, indent=2)
    print(f"[saved] {os.path.join(OUT,'ewma_reliability.json')}")
    print(f"  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
