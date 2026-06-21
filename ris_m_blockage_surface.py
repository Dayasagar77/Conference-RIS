#!/usr/bin/env python3
"""
ris_m_blockage_surface.py
═══════════════════════════════════════════════════════════════════════════════
Reviewer Q2 — joint {RIS size M × blockage depth} sensitivity of far-user
(survivor) telemetry reliability. Combines the two 1-D studies already in the
paper (blockage_sensitivity.py's 40-80 dB sweep; the M=1024 RIS-size threshold)
into a single 2-D surface.

FAITHFUL EXTENSION of blockage_sensitivity.py: identical eval_pos physics (every
channel quantity from channel_model.py), identical SLA_BPS=100, N_MC=60,
DEVICE_SEED=42, EVAL_SEED=99, identical survivor / sum-rate operating positions.
The ONLY additions are an outer sweep over cm.CFG.M and a 2-D heatmap. cm.CFG.M
is set then restored; the device layout (seed 42, generated once) is held fixed.

Far-user SLA reliability = P(far-user rate >= SLA_BPS) at a fixed UAV position,
evaluated for each (M, blockage) cell at BOTH:
    • the survivor-aware position (400,523,150)   [the deployed policy]
    • the sum-rate position       (450,577,108)   [survivor-blind baseline]

VALIDATION ANCHOR (M=1024, blockage=60 dB) reproduces the paper's 1-D numbers:
    survivor-pos reliability ~ 100%   (blockage_sensitivity.py)
    sum-rate-pos reliability ~ 64.6%  (blockage_sensitivity.py)
If the anchor cell does not match, the harness is wrong; do not trust the surface.

HOW TO RUN (Windows authoritative; channel_model.py in the same folder):
    python ris_m_blockage_surface.py
Physics-only -> reproduces exactly in-container.
Outputs (-> ..\\results\\SURVIVOR if it exists, else .\\results\\SURVIVOR):
    ris_m_blockage_surface.png    2-panel reliability heatmap
    ris_m_blockage_surface.json   all numbers
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

# ── EXACT blockage_sensitivity.py settings ────────────────────────────────────
DEVICE_SEED, EVAL_SEED, SLA_BPS, N_MC = 42, 99, 100.0, 60
SUMRATE_OPT  = np.array([450.0, 577.0, 108.0])   # survivor-blind baseline
SURVIVOR_OPT = np.array([400.0, 523.0, 150.0])   # deployed survivor-aware position

# ── sweep axes ────────────────────────────────────────────────────────────────
M_VALUES        = [128, 256, 512, 1024, 2048, 4096]
BLOCK_DB_VALUES = [40, 50, 60, 70, 80]
ANCHOR_M, ANCHOR_B = 1024, 60

np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]
FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
K = cm.CFG.N_PAIRS


def eval_pos(uav, n_mc=N_MC, seed=EVAL_SEED):
    """Far-user mean rate + SLA reliability at uav. Identical to
    blockage_sensitivity.py::eval_pos (RIS-assisted far branch); all physics
    from channel_model.py."""
    np.random.seed(seed)
    far = []
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        df = np.zeros(K)
        for i in range(K):
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            hd = cm.chan_scalar(uav, FAR[i], cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(FAR[i]); phi = cm.opt_phi(G, hr)
            he = cm.composite(hd, G, hr, phi)
            _, rF = cm.noma_pair(hn, he)
            df[i] = rF
        far.append(df)
    far = np.array(far)
    return float(far.mean()), float((far >= SLA_BPS).mean())


def run():
    print("=" * 88)
    print("  Q2  {RIS size M x blockage depth}  far-user SLA reliability surface")
    print("=" * 88)
    print(f"  SLA={SLA_BPS:.0f} bps | N_MC={N_MC} | M={M_VALUES} | blockage={BLOCK_DB_VALUES} dB")
    print(f"  survivor-pos {tuple(SURVIVOR_OPT.astype(int))} | "
          f"sum-rate-pos {tuple(SUMRATE_OPT.astype(int))}")
    print("-" * 88)
    t0 = time.time()
    orig_M, orig_B = cm.CFG.M, cm.CFG.BLOCK_DB

    nM, nB = len(M_VALUES), len(BLOCK_DB_VALUES)
    REL_SV = np.full((nM, nB), np.nan)   # survivor-pos reliability (%)
    REL_SR = np.full((nM, nB), np.nan)   # sum-rate-pos reliability (%)
    FAR_SV = np.full((nM, nB), np.nan)   # survivor-pos far mean rate (bps)
    for iM, M in enumerate(M_VALUES):
        cm.CFG.M = M
        cells = []
        for iB, b in enumerate(BLOCK_DB_VALUES):
            cm.CFG.BLOCK_DB = float(b)
            fsv, rsv = eval_pos(SURVIVOR_OPT)
            _,   rsr = eval_pos(SUMRATE_OPT)
            REL_SV[iM, iB] = rsv * 100
            REL_SR[iM, iB] = rsr * 100
            FAR_SV[iM, iB] = fsv
            cells.append(f"b{b}:sv{rsv*100:3.0f}%/sr{rsr*100:3.0f}%")
        print(f"  M={M:>4} | " + " | ".join(cells) + f"  [{time.time()-t0:.0f}s]")
    cm.CFG.M, cm.CFG.BLOCK_DB = orig_M, orig_B

    iMa, iBa = M_VALUES.index(ANCHOR_M), BLOCK_DB_VALUES.index(ANCHOR_B)
    print("-" * 88)
    print(f"  ANCHOR (M={ANCHOR_M}, {ANCHOR_B} dB): "
          f"survivor={REL_SV[iMa,iBa]:.1f}% (paper ~100%)  |  "
          f"sum-rate={REL_SR[iMa,iBa]:.1f}% (paper ~64.6%)")
    print("=" * 88)

    # ── 2-panel reliability heatmap ─────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    for a, Z, title in ((ax[0], REL_SV, "(a) Survivor-aware position (deployed)"),
                        (ax[1], REL_SR, "(b) Sum-rate position (survivor-blind)")):
        im = a.imshow(Z, origin="lower", cmap="viridis", vmin=0, vmax=100, aspect="auto")
        a.set_xticks(range(nB)); a.set_xticklabels([f"{b}" for b in BLOCK_DB_VALUES])
        a.set_yticks(range(nM)); a.set_yticklabels([f"{M}" for M in M_VALUES])
        a.set_xlabel("Blockage depth (dB)"); a.set_ylabel("RIS size M (elements)")
        a.set_title(title, fontweight="bold")
        for ii in range(nM):
            for jj in range(nB):
                if not np.isnan(Z[ii, jj]):
                    a.text(jj, ii, f"{Z[ii,jj]:.0f}", ha="center", va="center",
                           fontsize=8, color="white" if Z[ii, jj] < 55 else "black")
        plt.colorbar(im, ax=a, label="far-user SLA reliability (%)")
    fig.suptitle("Q2: far-user telemetry reliability over {RIS size x blockage}\n"
                 "the survivor-aware position sustains reliability where the "
                 "sum-rate position cannot", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "ris_m_blockage_surface.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {p}")

    out = dict(
        M_values=M_VALUES, block_db=BLOCK_DB_VALUES,
        reliability_survivor_pct=REL_SV.tolist(),
        reliability_sumrate_pct=REL_SR.tolist(),
        far_mean_bps_survivor=FAR_SV.tolist(),
        anchor=dict(M=ANCHOR_M, block_db=ANCHOR_B,
                    survivor_pct=float(REL_SV[iMa, iBa]),
                    sumrate_pct=float(REL_SR[iMa, iBa])),
        config=dict(sla_bps=SLA_BPS, n_mc=N_MC, device_seed=DEVICE_SEED,
                    eval_seed=EVAL_SEED, survivor_pos=SURVIVOR_OPT.tolist(),
                    sumrate_pos=SUMRATE_OPT.tolist()),
    )
    with open(os.path.join(OUT, "ris_m_blockage_surface.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {os.path.join(OUT,'ris_m_blockage_surface.json')}")
    print(f"  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
