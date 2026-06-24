#!/usr/bin/env python3
"""
outage_analysis.py  —  Outage Probability: Far Users With vs Without RIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Paper  : RIS-Assisted Hybrid NOMA with receiver-side SIC for UAV-Enabled
         Emergency Communications in 6G Heterogeneous Networks
Journal: IEEE Access  |  Manuscript ID: Access-2026-21197
Author : Dayasagar G + Dr. Deepa Nivethika S  |  VIT Chennai

Experiment 4 of 4 — generates Figure 16 for the revised submission.

Reviewer concern addressed
  "How does the RIS affect outage performance for the cell-edge
   (far) users that face severe blockage?"

Definition
  Outage event: per-user throughput R < R_min (threshold)
  P_out = fraction of (K × N_MC) realizations in outage

Computed for
  • Near users — with RIS and without RIS (should be negligible difference)
  • Far  users — with RIS and without RIS (dramatic difference expected)
  At: UAV = (437, 584, 108) m, device seed = 42, eval seed = 99

Figure content
  Left  — P_out vs R_min threshold (0 → max) for near and far users
  Right — Bar chart at three R_min values (0.05, 0.10, 0.50 Mbps)

Expected result
  Far no-RIS  : P_out ≈ 1.0 for all practical R_min  (60 dB blockage)
  Far with-RIS: P_out << 1.0 — significant outage reduction
  Near (both) : P_out ≈ 0 for R_min ≤ a few Mbps

Run on Windows:
  cd D:\\DAYA PHD\\PHD WORK\\RIS\\channel_model
  python outage_analysis.py

Output:
  results\\hq_figures\\fig_outage_analysis.png   (300 DPI, 2-panel)
  results\\hq_figures\\outage_results.txt
"""

import os, time, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'results', 'hq_figures'))
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM PARAMETERS  (identical to channel_model.py and per_user_cdf.py)
# ─────────────────────────────────────────────────────────────────────────────
FC       = 3.5e9
C_LIGHT  = 3.0e8
LAMBDA   = C_LIGHT / FC

BW_TOTAL = 20.0e6
N_PAIRS  = 12
BW_PAIR  = BW_TOTAL / N_PAIRS
P_TOTAL  = 2.0
P_PAIR   = P_TOTAL / N_PAIRS
ALPHA    = 0.85

N0_W_HZ  = 10.0 ** ((-167.0 - 30.0) / 10.0)
NOISE_W  = N0_W_HZ * BW_PAIR

AL_A = 9.61;  AL_B = 0.16
ETA_LOS  = 1.0
ETA_NLOS = 10.0 ** (20.0 / 10.0)   # = 100
K_RICE   = 10.0

BLOCKAGE_LIN = 10.0 ** (60.0 / 10.0)

M_RIS    = 1024
RIS_POS  = np.array([300., 250., 20.])
NBITS    = 3
N_LEVELS = 2 ** NBITS
PHI_STEP = 2.0 * np.pi / N_LEVELS

UAV_POS  = np.array([437., 584., 108.])

CENTROID  = np.array([400., 600., 0.])
NEAR_R_MIN, NEAR_R_MAX = 50.,  250.
FAR_R_MIN,  FAR_R_MAX  = 450., 800.
K_NEAR = K_FAR = 12

DEVICE_SEED = 42
EVAL_SEED   = 99
N_MC        = 50

# Outage thresholds
R_MIN_MARK  = 0.10   # Primary threshold highlighted in figure (Mbps)
R_MIN_BARS  = [0.05, 0.10, 0.50]   # Thresholds shown in bar chart


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHANNEL MODEL  (same as per_user_cdf.py)
# ─────────────────────────────────────────────────────────────────────────────

def fspl_gain(a, b):
    d = max(float(np.linalg.norm(np.asarray(a) - np.asarray(b))), 0.01)
    return (LAMBDA / (4.0 * np.pi * d)) ** 2

def p_los(uav, dev):
    dx   = float(uav[0] - dev[0])
    dy   = float(uav[1] - dev[1])
    d2d  = max(np.hypot(dx, dy), 1e-3)
    elev = np.degrees(np.arctan(float(uav[2]) / d2d))
    return 1.0 / (1.0 + AL_A * np.exp(-AL_B * (elev - AL_A)))

def quantize_phase(theta):
    return (np.round(theta / PHI_STEP).astype(int) % N_LEVELS) * PHI_STEP


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIMULATION  (returns all per-user, per-MC rate samples)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_rates():
    """
    Run full MC simulation and return per-user rate matrices (K, N_MC).
    Reuses the exact same channel generation as per_user_cdf.py.
    """
    # ── Device placement (seed = 42) ────────────────────────────────────
    np.random.seed(DEVICE_SEED)
    r_n  = np.random.uniform(NEAR_R_MIN, NEAR_R_MAX, K_NEAR)
    th_n = np.random.uniform(0.0, 2.0*np.pi, K_NEAR)
    near = np.column_stack([CENTROID[0]+r_n*np.cos(th_n),
                            CENTROID[1]+r_n*np.sin(th_n),
                            np.zeros(K_NEAR)])
    r_f  = np.random.uniform(FAR_R_MIN, FAR_R_MAX, K_FAR)
    th_f = np.random.uniform(0.0, 2.0*np.pi, K_FAR)
    far  = np.column_stack([CENTROID[0]+r_f*np.cos(th_f),
                            CENTROID[1]+r_f*np.sin(th_f),
                            np.zeros(K_FAR)])

    plos_n = np.array([p_los(UAV_POS, near[k]) for k in range(K_NEAR)])
    plos_f = np.array([p_los(UAV_POS, far[k])  for k in range(K_FAR)])
    g_n    = np.array([fspl_gain(UAV_POS, near[k]) for k in range(K_NEAR)])
    g_f_b  = np.array([fspl_gain(UAV_POS, far[k])  for k in range(K_FAR)]) / BLOCKAGE_LIN
    g_ur   = fspl_gain(UAV_POS, RIS_POS)
    g_rf   = np.array([fspl_gain(RIS_POS, far[k]) for k in range(K_FAR)])

    # ── Monte Carlo (seed = 99) ─────────────────────────────────────────
    np.random.seed(EVAL_SEED)

    # Near channels (K_NEAR, N_MC)
    is_los_n = np.random.random((K_NEAR, N_MC)) < plos_n[:, None]
    Ω_ls_n   = g_n[:, None] / ETA_LOS
    h_los_n  = (np.sqrt(K_RICE/(K_RICE+1)*Ω_ls_n) * np.exp(1j*np.random.uniform(0,2*np.pi,(K_NEAR,N_MC)))
                + np.sqrt(Ω_ls_n/(K_RICE+1)/2)*(np.random.standard_normal((K_NEAR,N_MC))
                   +1j*np.random.standard_normal((K_NEAR,N_MC))))
    h_nls_n  = (np.sqrt(g_n[:,None]/ETA_NLOS/2) *
                (np.random.standard_normal((K_NEAR,N_MC))+1j*np.random.standard_normal((K_NEAR,N_MC))))
    H_near   = np.where(is_los_n, h_los_n, h_nls_n)

    # Far direct channels (K_FAR, N_MC) — blocked
    is_los_f = np.random.random((K_FAR, N_MC)) < plos_f[:, None]
    Ω_ls_f   = g_f_b[:, None] / ETA_LOS
    h_los_f  = (np.sqrt(K_RICE/(K_RICE+1)*Ω_ls_f) * np.exp(1j*np.random.uniform(0,2*np.pi,(K_FAR,N_MC)))
                + np.sqrt(Ω_ls_f/(K_RICE+1)/2)*(np.random.standard_normal((K_FAR,N_MC))
                   +1j*np.random.standard_normal((K_FAR,N_MC))))
    h_nls_f  = (np.sqrt(g_f_b[:,None]/ETA_NLOS/2) *
                (np.random.standard_normal((K_FAR,N_MC))+1j*np.random.standard_normal((K_FAR,N_MC))))
    H_fd     = np.where(is_los_f, h_los_f, h_nls_f)

    # UAV→RIS (M, N_MC), RIS→far (M, K_FAR, N_MC)
    H_r = np.sqrt(g_ur/2)*(np.random.standard_normal((M_RIS,N_MC))
                            +1j*np.random.standard_normal((M_RIS,N_MC)))
    G   = (np.sqrt(g_rf/2)[None,:,None] *
           (np.random.standard_normal((M_RIS,K_FAR,N_MC))
            +1j*np.random.standard_normal((M_RIS,K_FAR,N_MC))))

    theta   = np.angle(H_r[:,None,:]) - np.angle(G)
    Phi     = np.exp(1j*quantize_phase(theta))
    H_ris   = np.sum(np.conj(H_r[:,None,:])*Phi*G, axis=0)   # (K_FAR, N_MC)
    H_eff   = H_fd + H_ris

    # ── Rates (Mbps) ────────────────────────────────────────────────────
    # Near (same ±RIS by design)
    sinr_n    = (1-ALPHA)*P_PAIR*np.abs(H_near)**2 / NOISE_W
    R_near    = BW_PAIR*np.log2(1+sinr_n)*1e-6          # (K_NEAR, N_MC)

    # Far without RIS
    pw_d      = np.abs(H_fd)**2
    sinr_fnr  = ALPHA*P_PAIR*pw_d/((1-ALPHA)*P_PAIR*pw_d+NOISE_W)
    R_far_nr  = BW_PAIR*np.log2(1+sinr_fnr)*1e-6        # (K_FAR, N_MC)

    # Far with RIS
    pw_e      = np.abs(H_eff)**2
    sinr_fr   = ALPHA*P_PAIR*pw_e/((1-ALPHA)*P_PAIR*pw_e+NOISE_W)
    R_far_r   = BW_PAIR*np.log2(1+sinr_fr)*1e-6         # (K_FAR, N_MC)

    return R_near, R_far_nr, R_far_r


# ─────────────────────────────────────────────────────────────────────────────
# 4. OUTAGE CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def outage_prob(rates, r_min):
    """Fraction of (K × N_MC) samples below r_min."""
    return float((rates < r_min).mean())


def outage_vs_threshold(rates, thresholds):
    """Outage probability curve over a range of thresholds."""
    return np.array([outage_prob(rates, t) for t in thresholds])


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    t0 = time.time()
    print()
    print("=" * 66)
    print("  Experiment 4: Outage Probability Analysis")
    print("  IEEE Access-2026-21197")
    print("=" * 66)
    print(f"  UAV       : ({UAV_POS[0]:.0f}, {UAV_POS[1]:.0f}, {UAV_POS[2]:.0f}) m  (DQL-optimal, fixed)")
    print(f"  R_min     : primary threshold = {R_MIN_MARK} Mbps")
    print(f"  Devices   : seed={DEVICE_SEED}, MC seed={EVAL_SEED}, N_MC={N_MC}")
    print("-" * 66)

    print("  Simulating channels ...", end=' ', flush=True)
    R_near, R_far_nr, R_far_r = simulate_rates()
    print("done.")

    # ── Outage at primary threshold ────────────────────────────────────
    Pout = {
        'near_nr' : outage_prob(R_near,   R_MIN_MARK),
        'near_r'  : outage_prob(R_near,   R_MIN_MARK),   # same by design
        'far_nr'  : outage_prob(R_far_nr, R_MIN_MARK),
        'far_r'   : outage_prob(R_far_r,  R_MIN_MARK),
    }

    print()
    print(f"  ── Outage Probability at R_min = {R_MIN_MARK} Mbps ──────")
    print(f"  Near users (no RIS)  : P_out = {Pout['near_nr']:.4f}  "
          f"({Pout['near_nr']*100:.1f}%)")
    print(f"  Near users (w/  RIS) : P_out = {Pout['near_r']:.4f}  "
          f"({Pout['near_r']*100:.1f}%)")
    print(f"  Far  users (no RIS)  : P_out = {Pout['far_nr']:.4f}  "
          f"({Pout['far_nr']*100:.1f}%)")
    print(f"  Far  users (w/  RIS) : P_out = {Pout['far_r']:.4f}  "
          f"({Pout['far_r']*100:.1f}%)")

    delta = Pout['far_nr'] - Pout['far_r']
    print(f"\n  RIS outage reduction  : "
          f"Δ = {delta:.4f}  ({delta*100:.1f} percentage points)")

    # ── Outage at multiple thresholds ─────────────────────────────────
    print()
    print(f"  ── Outage at Multiple Thresholds ──────────────────────")
    print(f"  {'R_min (Mbps)':>14}  {'Far no-RIS':>12}  {'Far w/ RIS':>12}  "
          f"{'Δ pp':>8}")
    for rm in R_MIN_BARS:
        pn = outage_prob(R_far_nr, rm)
        pr = outage_prob(R_far_r,  rm)
        print(f"  {rm:>14.2f}  {pn:>11.4f}  {pr:>11.4f}  "
              f"{(pn-pr)*100:>7.1f}%")

    # ── Rate statistics recap ──────────────────────────────────────────
    print()
    print(f"  ── Rate Statistics ────────────────────────────────────")
    print(f"  Near  (no/w RIS, same) : mean={R_near.mean():.3f} Mbps  "
          f"min={R_near.min():.3f}  max={R_near.max():.3f}")
    print(f"  Far   no RIS           : mean={R_far_nr.mean():.5f} Mbps  "
          f"min={R_far_nr.min():.5f}")
    print(f"  Far   w/  RIS          : mean={R_far_r.mean():.4f} Mbps  "
          f"max={R_far_r.max():.4f}")

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.2f} s")
    print("-" * 66)

    # ── Save log ──────────────────────────────────────────────────────
    log_path = os.path.join(OUT_DIR, 'outage_results.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("OUTAGE PROBABILITY ANALYSIS — IEEE Access-2026-21197\n")
        f.write(f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"UAV={tuple(UAV_POS)}, seed_d={DEVICE_SEED}, "
                f"seed_e={EVAL_SEED}, N_MC={N_MC}\n\n")
        f.write(f"PRIMARY THRESHOLD: R_min = {R_MIN_MARK} Mbps\n")
        f.write(f"  Near no-RIS : {Pout['near_nr']:.6f}\n")
        f.write(f"  Near w/ RIS : {Pout['near_r']:.6f}\n")
        f.write(f"  Far  no-RIS : {Pout['far_nr']:.6f}\n")
        f.write(f"  Far  w/ RIS : {Pout['far_r']:.6f}\n")
        f.write(f"  Delta (far) : {delta:.6f} ({delta*100:.2f} pp)\n\n")
        f.write(f"MULTIPLE THRESHOLDS (far users):\n")
        f.write(f"{'R_min':>10}  {'No-RIS':>10}  {'w/-RIS':>10}  {'Delta_pp':>10}\n")
        for rm in R_MIN_BARS:
            pn = outage_prob(R_far_nr, rm)
            pr = outage_prob(R_far_r,  rm)
            f.write(f"{rm:>10.3f}  {pn:>10.6f}  {pr:>10.6f}  "
                    f"{(pn-pr)*100:>10.2f}\n")
    print(f"  Log → {log_path}")

    return R_near, R_far_nr, R_far_r, Pout, delta


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_outage(R_near, R_far_nr, R_far_r, Pout, delta, out_dir):
    """
    2-panel figure:
      Left  — Outage probability vs rate threshold (curves)
      Right — Bar chart at R_min = 0.05, 0.10, 0.50 Mbps
    """
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    fig.subplots_adjust(wspace=0.30)

    C_NR_NEAR = '#5FA2DD'   # blue    — near, no RIS (same as w/ RIS)
    C_NR_FAR  = '#d62728'   # red     — far, no RIS
    C_R_FAR   = '#2ca02c'   # green   — far, with RIS
    LW = 2.0

    # Build sweep range
    r_near_max = float(R_near.max())
    r_far_max  = float(R_far_r.max())
    x_max_near = min(r_near_max * 1.05, 25.0)
    x_max_far  = min(r_far_max  * 1.15, max(r_far_max * 1.5, 2.0))

    # ── Left panel: Outage vs threshold ──────────────────────────────────
    ax = axes[0]

    # Near users — outage curve
    thres_near = np.linspace(0, x_max_near, 300)
    ax.plot(thres_near,
            outage_vs_threshold(R_near, thres_near),
            color=C_NR_NEAR, lw=LW, label='Near users (±RIS same)')

    # Far users — no RIS
    thres_far = np.linspace(0, x_max_far * 0.5, 300)
    ax.plot(thres_far,
            outage_vs_threshold(R_far_nr, thres_far),
            color=C_NR_FAR, lw=LW, linestyle='--',
            label='Far users — no RIS  (60 dB block)')

    # Far users — with RIS
    ax.plot(thres_far,
            outage_vs_threshold(R_far_r, thres_far),
            color=C_R_FAR, lw=LW, linestyle='-.',
            label=f'Far users — with RIS  (M={M_RIS})')

    # Vertical marker at R_min
    ax.axvline(R_MIN_MARK, color='#888', linewidth=1.2, linestyle=':',
               label=f'R_min = {R_MIN_MARK} Mbps')

    # Annotate outage values at R_min
    ax.annotate(f"P_out={Pout['far_nr']:.3f}",
                xy=(R_MIN_MARK, Pout['far_nr']),
                xytext=(R_MIN_MARK * 2.5, Pout['far_nr'] - 0.08),
                fontsize=9, color=C_NR_FAR,
                arrowprops=dict(arrowstyle='->', color=C_NR_FAR, lw=1.2))
    ax.annotate(f"P_out={Pout['far_r']:.3f}",
                xy=(R_MIN_MARK, Pout['far_r']),
                xytext=(R_MIN_MARK * 2.5, Pout['far_r'] + 0.05),
                fontsize=9, color=C_R_FAR,
                arrowprops=dict(arrowstyle='->', color=C_R_FAR, lw=1.2))

    ax.set_xlabel('Rate Threshold R_min (Mbps)')
    ax.set_ylabel('Outage Probability  P(R < R_min)')
    ax.set_title('Outage Probability vs Rate Threshold', fontweight='bold')
    ax.set_xlim([0, thres_far[-1]])
    ax.set_ylim([-0.03, 1.08])
    ax.legend(loc='lower right', fontsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.45)

    # Delta annotation
    ax.annotate('',
                xy=(R_MIN_MARK * 0.92, Pout['far_r']),
                xytext=(R_MIN_MARK * 0.92, Pout['far_nr']),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    ax.text(R_MIN_MARK * 0.6, (Pout['far_nr']+Pout['far_r'])/2,
            f'Δ={delta*100:.0f} pp', ha='center', fontsize=9,
            color='purple', fontweight='bold')

    # ── Right panel: Bar chart at 3 R_min values ─────────────────────────
    ax2 = axes[1]

    r_labels = [f'{rm:.2f}' for rm in R_MIN_BARS]
    x        = np.arange(len(R_MIN_BARS))
    w        = 0.28

    pout_far_nr = [outage_prob(R_far_nr, rm) for rm in R_MIN_BARS]
    pout_far_r  = [outage_prob(R_far_r,  rm) for rm in R_MIN_BARS]
    pout_near   = [outage_prob(R_near,   rm) for rm in R_MIN_BARS]

    bars_fnr = ax2.bar(x - w, pout_far_nr, w, color=C_NR_FAR,
                       edgecolor='white', linewidth=0.7, label='Far — no RIS')
    bars_fr  = ax2.bar(x,     pout_far_r,  w, color=C_R_FAR,
                       edgecolor='white', linewidth=0.7, label='Far — with RIS')
    bars_n   = ax2.bar(x + w, pout_near,   w, color=C_NR_NEAR,
                       edgecolor='white', linewidth=0.7, label='Near (±RIS same)')

    # Value labels on bars
    for bars in [bars_fnr, bars_fr, bars_n]:
        for bar in bars:
            h = bar.get_height()
            txt = f'{h:.2f}' if h >= 0.01 else f'{h:.3f}'
            ax2.text(bar.get_x()+bar.get_width()/2, h+0.012,
                     txt, ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'R_min = {rm} Mbps' for rm in r_labels], fontsize=9.5)
    ax2.set_ylabel('Outage Probability  P(R < R_min)')
    ax2.set_title('Outage Probability at Key R_min Values', fontweight='bold')
    ax2.set_ylim([0, 1.18])
    ax2.legend(loc='upper right', fontsize=9.5)
    ax2.grid(axis='y', linestyle=':', alpha=0.45)

    # ── Suptitle ──────────────────────────────────────────────────────────
    # figure title omitted; the IEEE caption provides it

    fig_path = os.path.join(out_dir, 'fig_outage_analysis.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig → {fig_path}")
    return fig_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_wall = time.time()

    R_near, R_far_nr, R_far_r, Pout, delta = run()
    plot_outage(R_near, R_far_nr, R_far_r, Pout, delta, OUT_DIR)

    total = time.time() - t_wall
    print()
    print(f"  ✓ Experiment 4 complete — {total:.2f} s total")
    print()
    print("  KEY FINDINGS FOR PAPER (Section V-B):")
    print(f"    At R_min = {R_MIN_MARK} Mbps:")
    print(f"      Far  no-RIS  : P_out = {Pout['far_nr']:.3f} "
          f"({Pout['far_nr']*100:.0f}% in outage)")
    print(f"      Far  w/ RIS  : P_out = {Pout['far_r']:.3f} "
          f"({Pout['far_r']*100:.0f}% in outage)")
    print(f"      Reduction    : {delta*100:.0f} percentage points")
    print(f"    Near users: P_out ≈ {Pout['near_nr']:.3f} "
          f"(negligible — strong direct path)")
    print()
    print("  ── ALL 4 EXPERIMENTS COMPLETE ──────────────────────────────")
    print()
    print("  Generated figures (300 DPI, ready for submission):")
    print("    Fig 13: fig_per_user_cdf.png      (per_user_cdf.py)")
    print("    Fig 14: fig_multi_geometry.png    (multi_geometry_eval.py)")
    print("    Fig 15: fig_hsdc_sc1.png          (hsdc_sc1_only.py)")
    print("    Fig 16: fig_outage_analysis.png   (outage_analysis.py)")
    print()
    print("  NEXT: Insert Figs 13–16 into Daya_IEEE_Access_Revised_v12_FINAL.docx")
    print("        Then start Response to Reviewers letter")
    print("=" * 66)
    print()
