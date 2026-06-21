#!/usr/bin/env python3
"""
survivor_aware_pareto.py
═══════════════════════════════════════════════════════════════════════════════
EXECUTES Reviewer-2 Critical Flaw A (DQL reward blind to RIS/far-users).

Instead of *acknowledging* that the sum-rate reward ignores blocked survivors,
this script PROVES the consequence with the paper's own physics and then
constructs the survivor-aware objective that fixes it.

Three objectives are evaluated at every candidate UAV position, all using
channel_model.py (identical physics to the 188.36 Mbps headline):

  J_sum   = Σ R_k                     (Mbps)   ← conventional reward (current paper)
  J_pf    = Σ log(R_k)                         ← proportional fairness (survivor-aware)
  J_sla   = w·near_norm + (1-w)·far_SLA_rel    ← telemetry-SLA reward (survivor-aware)

Key questions answered with REAL numbers:
  1. Does the sum-rate-optimal UAV position give *any* gradient toward far users?
  2. Is there a Pareto frontier between aggregate throughput and far-user
     telemetry reliability?
  3. Where does each objective place the UAV, and what is the throughput cost
     of becoming survivor-aware?

Run on Windows (channel_model.py in same folder):
    python survivor_aware_pareto.py
Outputs:
    results/SURVIVOR/pareto_frontier.png
    results/SURVIVOR/pareto_grid.csv
    results/SURVIVOR/operating_points.csv
"""
import os, time, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import channel_model as cm

# ── output dir ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
os.makedirs(OUT, exist_ok=True)

# ── experiment configuration ──────────────────────────────────────────────────
DEVICE_SEED = 42        # same layout as the paper headline
EVAL_SEED   = 99        # same evaluation seed as Module 3
N_MC        = 60        # channel draws per position
SLA_BPS     = 100.0     # far-user telemetry SLA (GPS+heartbeat beacon)
W_SLA       = 0.5       # weight in J_sla (0.5 = equal near/far priority)

# UAV search grid (covers near-centroid ~ (400,600) AND the RIS at (300,250,20))
GX = np.linspace(150, 800, 14)
GY = np.linspace(200, 900, 14)
GH = np.array([50, 80, 108, 150, 200], dtype=float)

# Baselines from the paper (Table VIII, H*=125 m)
OMA_REF = 120.93


def per_user_rates(uav, near, far, n_mc=N_MC, seed=EVAL_SEED):
    """Return (near_rates_bps[N_PAIRS], far_rates_bps[N_PAIRS]) averaged over n_mc draws,
    plus the FULL per-draw far-user sample array for reliability statistics."""
    np.random.seed(seed)
    K = cm.CFG.N_PAIRS
    near_acc = np.zeros(K)
    far_acc  = np.zeros(K)
    far_samples = []                       # every (pair, draw) far-rate sample
    for _ in range(n_mc):
        G = cm.G_vector(uav)               # UAV→RIS (M-vector), re-drawn each MC
        draw_far = np.zeros(K)
        for i in range(K):
            hn = cm.chan_scalar(uav, near[i], 0.0)              # near direct (LoS)
            hd = cm.chan_scalar(uav, far[i],  cm.CFG.BLOCK_DB)  # far direct (blocked)
            hr = cm.H_ris_dev(far[i])                            # RIS→far (M-vector)
            phi = cm.opt_phi(G, hr)                              # closed-form phases
            he  = cm.composite(hd, G, hr, phi)                  # RIS-assisted far channel
            rN, _  = cm.noma_pair(hn, hd)     # near rate (paired with far direct)
            _,  rF = cm.noma_pair(hn, he)     # far rate via RIS composite
            near_acc[i] += rN
            far_acc[i]  += rF
            draw_far[i]  = rF
        far_samples.append(draw_far)
    near_acc /= n_mc
    far_acc  /= n_mc
    return near_acc, far_acc, np.array(far_samples)   # bits/s


def evaluate(uav, near, far):
    """Compute all objectives + diagnostics at one UAV position."""
    nr, fr, fsamp = per_user_rates(uav, near, far)
    near_sum_mbps = nr.sum() / 1e6
    far_sum_mbps  = fr.sum() / 1e6
    agg_mbps      = near_sum_mbps + far_sum_mbps          # J_sum (current reward)

    far_mean_bps  = fr.mean()
    # telemetry reliability = P(far-user rate >= SLA), over all (pair,draw) samples
    rel_100 = float((fsamp >= 100.0).mean())
    rel_50  = float((fsamp >= 50.0).mean())
    rel_500 = float((fsamp >= 500.0).mean())

    # proportional fairness: Σ log(R_k) over all 24 users (rates in bps, floor to avoid -inf)
    all_rates = np.concatenate([nr, fr])
    J_pf = float(np.log(np.maximum(all_rates, 1e-3)).sum())

    # max-min fairness: worst far user mean rate
    min_far_bps = float(fr.min())

    # telemetry-SLA composite reward
    near_norm = near_sum_mbps / 200.0                    # normalise (~0-1)
    J_sla = W_SLA * near_norm + (1.0 - W_SLA) * rel_100

    return dict(
        x=uav[0], y=uav[1], H=uav[2],
        agg_mbps=agg_mbps, near_mbps=near_sum_mbps, far_mbps=far_sum_mbps,
        far_mean_bps=far_mean_bps,
        rel_100=rel_100, rel_50=rel_50, rel_500=rel_500,
        J_sum=agg_mbps, J_pf=J_pf, J_sla=J_sla, min_far_bps=min_far_bps,
    )


def pareto_front(points, xkey, ykey):
    """Return indices on the upper-right Pareto frontier (maximise both x and y)."""
    idx = sorted(range(len(points)), key=lambda i: (points[i][xkey], points[i][ykey]),
                 reverse=True)
    front, best_y = [], -np.inf
    for i in idx:
        if points[i][ykey] >= best_y:
            front.append(i); best_y = points[i][ykey]
    return front


def main():
    print("=" * 74)
    print("  SURVIVOR-AWARE UAV POSITIONING — Pareto frontier (channel_model.py)")
    print("=" * 74)
    np.random.seed(DEVICE_SEED)
    pos, _ = cm.generate_devices()
    near = pos[:cm.CFG.K_NEAR]
    far  = pos[cm.CFG.K_NEAR:cm.CFG.K_SC1]
    print(f"  Devices: {cm.CFG.K_NEAR} near + {len(far)} far (seed {DEVICE_SEED})")
    print(f"  RIS at {tuple(cm.CFG.RIS_POS)}, M={cm.CFG.M}, SLA={SLA_BPS:.0f} bps")
    print(f"  Grid: {len(GX)}×{len(GY)}×{len(GH)} = {len(GX)*len(GY)*len(GH)} positions, "
          f"N_MC={N_MC}")
    print("-" * 74)

    t0 = time.time()
    pts = []
    total = len(GX) * len(GY) * len(GH)
    n = 0
    for H in GH:
        for x in GX:
            for y in GY:
                pts.append(evaluate(np.array([x, y, H]), near, far))
                n += 1
        print(f"  H={H:5.0f} m done ... {n}/{total}  ({time.time()-t0:.0f}s)")

    # ── identify the three operating points ────────────────────────────────────
    i_sum = max(range(len(pts)), key=lambda i: pts[i]["J_sum"])
    i_pf  = max(range(len(pts)), key=lambda i: pts[i]["J_pf"])
    i_sla = max(range(len(pts)), key=lambda i: pts[i]["J_sla"])

    def show(tag, i):
        p = pts[i]
        print(f"  {tag:30s} UAV=({p['x']:.0f},{p['y']:.0f},{p['H']:.0f})  "
              f"agg={p['agg_mbps']:.1f} Mbps  far_mean={p['far_mean_bps']:.0f} bps  "
              f"rel@100={p['rel_100']*100:.1f}%  rel@50={p['rel_50']*100:.1f}%")

    print("\n" + "=" * 74)
    print("  OPERATING POINTS")
    print("=" * 74)
    show("J_sum  (conventional reward)", i_sum)
    show("J_pf   (proportional fairness)", i_pf)
    show("J_sla  (telemetry-SLA reward)",  i_sla)

    psum, ppf = pts[i_sum], pts[i_sla]
    d_thr = 100 * (psum["agg_mbps"] - ppf["agg_mbps"]) / psum["agg_mbps"]
    d_rel = (ppf["rel_100"] - psum["rel_100"]) * 100
    print("\n  ── SURVIVOR-AWARE TRADE-OFF (J_sla vs J_sum) ──")
    print(f"  Aggregate throughput cost : {d_thr:+.1f}%  "
          f"({psum['agg_mbps']:.1f} → {ppf['agg_mbps']:.1f} Mbps)")
    print(f"  Far-user 100-bps gain     : {d_rel:+.1f} pts  "
          f"({psum['rel_100']*100:.1f}% → {ppf['rel_100']*100:.1f}%)")
    print(f"  Far-user mean-rate gain   : "
          f"{psum['far_mean_bps']:.0f} → {ppf['far_mean_bps']:.0f} bps "
          f"({ppf['far_mean_bps']/max(psum['far_mean_bps'],1e-9):.1f}×)")

    # ── Pareto frontier: aggregate throughput vs far-user reliability ──────────
    front = pareto_front(pts, "agg_mbps", "rel_100")
    front = sorted(front, key=lambda i: pts[i]["agg_mbps"])

    # ── plot ───────────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 6.5))
    aggs = [p["agg_mbps"] for p in pts]
    rels = [p["rel_100"] * 100 for p in pts]
    ax.scatter(aggs, rels, s=14, c="#b8c6d8", alpha=0.6, zorder=2,
               label=f"All {len(pts)} UAV positions")
    fx = [pts[i]["agg_mbps"] for i in front]
    fy = [pts[i]["rel_100"] * 100 for i in front]
    ax.plot(fx, fy, "-o", color="#1f5fa8", lw=2, ms=5, zorder=4,
            label="Pareto frontier (throughput ↔ reliability)")

    for tag, i, col, mk in [("Sum-rate optimum\n(current paper)", i_sum, "#d62728", "*"),
                            ("Telemetry-SLA optimum\n(proposed)",  i_sla, "#2ca02c", "P"),
                            ("Proportional-fair optimum",          i_pf,  "#ff7f0e", "D")]:
        p = pts[i]
        ax.scatter(p["agg_mbps"], p["rel_100"] * 100, s=320, marker=mk, c=col,
                   edgecolors="black", lw=1.2, zorder=6)
        ax.annotate(f"{tag}\n({p['x']:.0f},{p['y']:.0f},{p['H']:.0f}) m\n"
                    f"{p['agg_mbps']:.0f} Mbps, {p['rel_100']*100:.0f}%",
                    (p["agg_mbps"], p["rel_100"] * 100),
                    textcoords="offset points", xytext=(12, -8), fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=col, alpha=0.9))

    ax.axvline(OMA_REF, color="#888", ls=":", lw=1.2,
               label=f"OMA baseline = {OMA_REF} Mbps")
    ax.set_xlabel("Aggregate system throughput (Mbps)")
    ax.set_ylabel("Far-user telemetry reliability @ 100 bps (%)")
    ax.set_title("Survivor-Aware UAV Positioning:\nThroughput–Reliability Pareto Frontier "
                 "(channel_model.py physics)", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="center left", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(OUT, "pareto_frontier.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Pareto plot → {fig_path}")

    # ── CSVs ────────────────────────────────────────────────────────────────────
    grid_csv = os.path.join(OUT, "pareto_grid.csv")
    with open(grid_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pts[0].keys()))
        w.writeheader(); w.writerows(pts)
    op_csv = os.path.join(OUT, "operating_points.csv")
    with open(op_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["objective"] + list(pts[0].keys()))
        for tag, i in [("J_sum", i_sum), ("J_pf", i_pf), ("J_sla", i_sla)]:
            row = dict(pts[i]); row["objective"] = tag; w.writerow(row)
    print(f"  Grid CSV → {grid_csv}")
    print(f"  Operating points CSV → {op_csv}")
    print(f"  Runtime: {time.time()-t0:.0f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
