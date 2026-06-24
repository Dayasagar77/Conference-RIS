#!/usr/bin/env python3
"""
phase_error_sweep.py
═══════════════════════════════════════════════════════════════════════════════
Reviewer Q5 — imperfect CSI: does the +7.55 dB far-user RIS gain degrade under
RIS phase-configuration error?  The paper already reports the 3-bit quantization
loss (continuous +8.37 dB -> 3-bit +8.15 dB, 0.22 dB). This adds the missing
CASCADED phase-estimation-error sweep.

MODEL (standard RIS phase-noise): the RIS applies its 3-bit-quantized ideal phase
cm.opt_phi(G, h_r), plus a residual per-element Gaussian error e ~ N(0, sigma^2)
representing cascaded CSI-estimation error:
        phi_applied = opt_phi(G, h_r) + e ,   e_m ~ N(0, sigma_phi^2)
The error is drawn from an INDEPENDENT RNG so every sigma is evaluated on the
SAME channel realisations (seed 99) — the curve isolates the phase-error effect,
not Monte-Carlo reshuffling. sigma_phi = 0 reproduces the clean baseline exactly.

FAITHFUL: every channel quantity and the phase/quantization come from
channel_model.py (G_vector, chan_scalar, H_ris_dev, opt_phi, composite,
noma_pair); the only addition is the additive phase error and the sweep.

Two metrics over sigma_phi in {0,5,10,15,20,30,45,60} deg:
  (a) mean far-user RIS gain (dB) at the canonical gain position (400,600,100)
      — the quantity the +7.55/+8.15 dB headline refers to
  (b) far-user SLA reliability (%) at the deployed survivor-aware position
      (400,523,150) — the practical impact on the delivered service

VALIDATION ANCHOR (sigma_phi = 0): the gain must match the paper's reported RIS
gain at that position, and survivor-position reliability must be ~100%. If not,
the harness is wrong.

HOW TO RUN (Windows authoritative; channel_model.py in the same folder):
    python phase_error_sweep.py
Physics-only -> reproduces exactly in-container.
Outputs (-> ..\\results\\SURVIVOR if it exists, else .\\results\\SURVIVOR):
    phase_error_sweep.png     2-panel degradation figure
    phase_error_sweep.json    all numbers
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
DEVICE_SEED, EVAL_SEED, SLA_BPS = 42, 99, 100.0
ERR_SEED       = 7                       # independent RNG for the phase error
N_MC_GAIN      = 500                     # matches channel_model.verify_ris_gain
N_MC_REL       = 60                      # matches eval_pos / blockage_sensitivity
GAIN_POS       = np.array([400.0, 600.0, 100.0])   # canonical RIS-gain position
SURV_POS       = np.array([400.0, 523.0, 150.0])   # deployed survivor-aware position
SIGMA_PHI_DEG  = [0, 5, 10, 15, 20, 30, 45, 60]

np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]
FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
K = cm.CFG.N_PAIRS


def far_ris_gain_db(uav, sigma_phi_deg, n_mc=N_MC_GAIN, seed=EVAL_SEED, err_seed=ERR_SEED):
    """Mean far-user RIS gain (dB) = 10log10(|h_eff|^2/|h_d|^2). Identical loop to
    channel_model.verify_ris_gain's far branch, plus additive phase error."""
    np.random.seed(seed)                       # channel RNG (same draws for every sigma)
    perr = np.random.default_rng(err_seed)     # independent phase-error RNG
    sigma = np.deg2rad(sigma_phi_deg)
    gains = []
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        for f in FAR:
            hd = cm.chan_scalar(uav, f, cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(f)
            phi = cm.opt_phi(G, hr)
            if sigma > 0:
                phi = phi + perr.normal(0.0, sigma, size=phi.shape)
            he = cm.composite(hd, G, hr, phi)
            gains.append(10 * np.log10(abs(he) ** 2 / max(abs(hd) ** 2, 1e-40)))
    return float(np.mean(gains))


def far_reliability(uav, sigma_phi_deg, n_mc=N_MC_REL, seed=EVAL_SEED, err_seed=ERR_SEED):
    """(far_mean_bps, SLA reliability) at uav under phase error. NOMA far branch
    identical to eval_position; physics from channel_model.py."""
    np.random.seed(seed)
    perr = np.random.default_rng(err_seed)
    sigma = np.deg2rad(sigma_phi_deg)
    far = []
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        df = np.zeros(K)
        for i in range(K):
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            hd = cm.chan_scalar(uav, FAR[i], cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(FAR[i])
            phi = cm.opt_phi(G, hr)
            if sigma > 0:
                phi = phi + perr.normal(0.0, sigma, size=phi.shape)
            he = cm.composite(hd, G, hr, phi)
            _, rF = cm.noma_pair(hn, he)
            df[i] = rF
        far.append(df)
    far = np.array(far)
    return float(far.mean()), float((far >= SLA_BPS).mean())


def run():
    print("=" * 88)
    print("  Q5  RIS PHASE-ERROR / IMPERFECT-CSI SWEEP")
    print("=" * 88)
    print(f"  gain pos {tuple(GAIN_POS.astype(int))} (N_MC={N_MC_GAIN}) | "
          f"survivor pos {tuple(SURV_POS.astype(int))} (N_MC={N_MC_REL}) | "
          f"SLA={SLA_BPS:.0f} bps | 3-bit RIS")
    print(f"  phase-error std sweep (deg): {SIGMA_PHI_DEG}")
    print("-" * 88)
    t0 = time.time()

    gain_db, rel_pct, far_bps = [], [], []
    for s in SIGMA_PHI_DEG:
        g = far_ris_gain_db(GAIN_POS, s)
        fmean, rel = far_reliability(SURV_POS, s)
        gain_db.append(g); rel_pct.append(rel * 100); far_bps.append(fmean)
        print(f"  sigma_phi={s:>2}deg | far RIS gain={g:+6.2f} dB | "
              f"survivor reliability={rel*100:5.1f}% | far mean={fmean:7.0f} bps "
              f"[{time.time()-t0:.0f}s]")
    cm_gain0 = gain_db[0]
    drop3 = cm_gain0 - np.interp(15, SIGMA_PHI_DEG, gain_db)   # loss at 15 deg
    print("-" * 88)
    print(f"  ANCHOR (sigma=0): far RIS gain = {cm_gain0:+.2f} dB "
          f"(paper: 3-bit +8.15 dB / headline +7.55 dB) | "
          f"survivor reliability = {rel_pct[0]:.1f}% (paper ~100%)")
    print(f"  Degradation: gain loss at 15 deg phase error = {drop3:.2f} dB; "
          f"survivor reliability stays {min(rel_pct[:SIGMA_PHI_DEG.index(20)+1]):.0f}%+ "
          f"up to 20 deg")
    print("=" * 88)

    # ── 2-panel figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(SIGMA_PHI_DEG, gain_db, "-o", color="#1f5fa8", lw=2.2, ms=7)
    ax[0].axhline(0, color="k", ls="--", lw=1, label="0 dB (no RIS benefit)")
    ax[0].axhline(gain_db[0], color="#2ca02c", ls=":", lw=1.5,
                  label=f"sigma=0 baseline ({gain_db[0]:+.2f} dB)")
    ax[0].set_xlabel("RIS phase-error std  $\\sigma_\\phi$  (degrees)")
    ax[0].set_ylabel("Mean far-user RIS gain (dB)")
    ax[0].set_title("(a) RIS gain vs phase-estimation error", fontweight="bold")
    ax[0].grid(True, ls=":", alpha=0.5); ax[0].legend(fontsize=9)

    ax[1].plot(SIGMA_PHI_DEG, rel_pct, "-s", color="#d62728", lw=2.2, ms=7)
    ax[1].axhline(95, color="gray", ls="--", lw=1, label="95% SLA reliability")
    ax[1].set_xlabel("RIS phase-error std  $\\sigma_\\phi$  (degrees)")
    ax[1].set_ylabel("Far-user SLA reliability (%)")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("(b) Survivor reliability vs phase error\n(deployed survivor-aware position)",
                    fontweight="bold")
    ax[1].grid(True, ls=":", alpha=0.5); ax[1].legend(fontsize=9)
    # figure title omitted; the IEEE caption provides it
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "phase_error_sweep.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {p}")

    out = dict(sigma_phi_deg=SIGMA_PHI_DEG, far_ris_gain_db=gain_db,
               survivor_reliability_pct=rel_pct, far_mean_bps=far_bps,
               anchor=dict(gain0_db=cm_gain0, reliability0_pct=rel_pct[0]),
               config=dict(gain_pos=GAIN_POS.tolist(), surv_pos=SURV_POS.tolist(),
                           n_mc_gain=N_MC_GAIN, n_mc_rel=N_MC_REL,
                           device_seed=DEVICE_SEED, eval_seed=EVAL_SEED,
                           err_seed=ERR_SEED, sla_bps=SLA_BPS, b_phase=cm.CFG.B_PHASE))
    with open(os.path.join(OUT, "phase_error_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {os.path.join(OUT,'phase_error_sweep.json')}")
    print(f"  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
