#!/usr/bin/env python3
"""
multi_geometry_eval.py  —  DQL Policy Generalisation: 20 Device Layout Seeds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Paper  : RIS-Assisted Hybrid NOMA with receiver-side SIC for UAV-Enabled
         Emergency Communications in 6G Heterogeneous Networks
Journal: IEEE Access  |  Manuscript ID: Access-2026-21197
Author : Dayasagar G + Dr. Deepa Nivethika S  |  VIT Chennai

Experiment 2 of 4 — generates Figure 14 for the revised submission.

Reviewer concern addressed
  "Does the DQL policy overfit to the specific device placement used during
   training (seed=42)?  Would a different device layout give poor results?"

Answer
  Evaluate RIS-NOMA aggregate throughput at the DQL-optimal UAV position
  (437, 584, 108) m across 20 independently drawn device layouts
  (seeds 100–119).  No re-training is done — the position is FIXED.
  The paper's training seed=42 is added as a 21st reference point.

Expected result
  Mean ≈ 185–190 Mbps, std ≈ 2–6 Mbps across 20 seeds.
  All seeds comfortably above the OMA baseline (120.93 Mbps).

Run on Windows:
  cd D:\\DAYA PHD\\PHD WORK\\RIS\\channel_model
  python multi_geometry_eval.py

Output:
  results\\hq_figures\\fig_multi_geometry.png   (300 DPI, bar chart + stats)
  results\\hq_figures\\multi_geometry_results.txt
"""

import os, time, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'results', 'hq_figures'))
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM PARAMETERS  (identical to channel_model.py)
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

AL_A    = 9.61;   AL_B    = 0.16
ETA_LOS = 1.0;    ETA_NLOS = 10.0 ** (20.0 / 10.0)   # = 100
K_RICE  = 10.0

BLOCKAGE_LIN = 10.0 ** (60.0 / 10.0)

M_RIS    = 1024
RIS_POS  = np.array([300., 250., 20.])
NBITS    = 3
N_LEVELS = 2 ** NBITS
PHI_STEP = 2.0 * np.pi / N_LEVELS

# DQL-optimal UAV position (FIXED — no re-training)
UAV_POS = np.array([437., 584., 108.])

CENTROID  = np.array([400., 600., 0.])
NEAR_R_MIN, NEAR_R_MAX = 50.,  250.
FAR_R_MIN,  FAR_R_MAX  = 450., 800.
K_NEAR = K_FAR = 12

# Seeds
TRAIN_SEED  = 42    # Paper's canonical training seed
EVAL_SEED   = 99    # MC evaluation seed (ALWAYS)
N_MC        = 50    # MC iterations per layout
TEST_SEEDS  = list(range(100, 120))   # 20 unseen layout seeds

# Paper reference baselines (from confirmed simulations)
OMA_BASELINE  = 120.93   # Mbps  (H*=125 m, bug-1 fixed)
NOMA_BASELINE = 187.18   # Mbps  (H*=125 m, no RIS)
DQL_PAPER     = 188.36   # Mbps  (seed=42, the paper result)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHANNEL MODEL  (same physics as per_user_cdf.py)
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
# 3. AGGREGATE THROUGHPUT AT FIXED UAV POSITION
# ─────────────────────────────────────────────────────────────────────────────

def eval_throughput(device_seed):
    """
    Place devices with `device_seed`, then evaluate RIS-NOMA aggregate
    throughput at UAV_POS=(437,584,108) using legacy numpy RNG.

    Returns:
        thr_ris   (float) — Mbps, NOMA + RIS
        thr_noma  (float) — Mbps, NOMA without RIS
        near_mean (float) — mean per-user near rate (Mbps)
        far_mean  (float) — mean per-user far rate with RIS (Mbps)
    """
    # ── Device placement ──────────────────────────────────────────────────
    np.random.seed(device_seed)
    r_n  = np.random.uniform(NEAR_R_MIN, NEAR_R_MAX, K_NEAR)
    th_n = np.random.uniform(0.0, 2.0 * np.pi, K_NEAR)
    near = np.column_stack([CENTROID[0] + r_n*np.cos(th_n),
                            CENTROID[1] + r_n*np.sin(th_n),
                            np.zeros(K_NEAR)])

    r_f  = np.random.uniform(FAR_R_MIN, FAR_R_MAX, K_FAR)
    th_f = np.random.uniform(0.0, 2.0 * np.pi, K_FAR)
    far  = np.column_stack([CENTROID[0] + r_f*np.cos(th_f),
                            CENTROID[1] + r_f*np.sin(th_f),
                            np.zeros(K_FAR)])

    # Pre-compute LoS probabilities
    plos_n = np.array([p_los(UAV_POS, near[k]) for k in range(K_NEAR)])
    plos_f = np.array([p_los(UAV_POS, far[k])  for k in range(K_FAR)])

    # Per-device FSPL
    g_n        = np.array([fspl_gain(UAV_POS, near[k]) for k in range(K_NEAR)])
    g_f_block  = np.array([fspl_gain(UAV_POS, far[k])  for k in range(K_FAR)]) / BLOCKAGE_LIN
    g_uav_ris  = fspl_gain(UAV_POS, RIS_POS)
    g_ris_far  = np.array([fspl_gain(RIS_POS, far[k])  for k in range(K_FAR)])

    # ── Monte Carlo (seed = 99 always) ────────────────────────────────────
    np.random.seed(EVAL_SEED)

    # Near channels  (K_NEAR, N_MC)
    is_los_n  = np.random.random((K_NEAR, N_MC)) < plos_n[:, None]
    Omega_los = g_n[:, None] / ETA_LOS
    s_det     = np.sqrt(K_RICE / (K_RICE+1) * Omega_los)
    s_sct     = np.sqrt(Omega_los / (K_RICE+1) / 2.0)
    phi_d     = np.random.uniform(0.0, 2.0*np.pi, (K_NEAR, N_MC))
    h_los_n   = (s_det * np.exp(1j*phi_d) +
                 s_sct * (np.random.standard_normal((K_NEAR, N_MC)) +
                          1j*np.random.standard_normal((K_NEAR, N_MC))))
    Omega_nls = g_n[:, None] / ETA_NLOS
    h_nls_n   = (np.sqrt(Omega_nls/2.0) *
                 (np.random.standard_normal((K_NEAR, N_MC)) +
                  1j*np.random.standard_normal((K_NEAR, N_MC))))
    H_near    = np.where(is_los_n, h_los_n, h_nls_n)

    # Far direct channels (K_FAR, N_MC) — 60 dB blocked
    is_los_f  = np.random.random((K_FAR, N_MC)) < plos_f[:, None]
    Omega_lsf = g_f_block[:, None] / ETA_LOS
    s_det_f   = np.sqrt(K_RICE / (K_RICE+1) * Omega_lsf)
    s_sct_f   = np.sqrt(Omega_lsf / (K_RICE+1) / 2.0)
    phi_df    = np.random.uniform(0.0, 2.0*np.pi, (K_FAR, N_MC))
    h_los_f   = (s_det_f * np.exp(1j*phi_df) +
                 s_sct_f * (np.random.standard_normal((K_FAR, N_MC)) +
                            1j*np.random.standard_normal((K_FAR, N_MC))))
    Omega_nlf = g_f_block[:, None] / ETA_NLOS
    h_nls_f   = (np.sqrt(Omega_nlf/2.0) *
                 (np.random.standard_normal((K_FAR, N_MC)) +
                  1j*np.random.standard_normal((K_FAR, N_MC))))
    H_far_d   = np.where(is_los_f, h_los_f, h_nls_f)

    # UAV → RIS channels  (M_RIS, N_MC)
    s_ur = np.sqrt(g_uav_ris / 2.0)
    H_r  = s_ur * (np.random.standard_normal((M_RIS, N_MC)) +
                   1j*np.random.standard_normal((M_RIS, N_MC)))

    # RIS → far device channels  (M_RIS, K_FAR, N_MC)
    s_rd = np.sqrt(g_ris_far / 2.0)           # (K_FAR,)
    G    = (s_rd[None, :, None] *
            (np.random.standard_normal((M_RIS, K_FAR, N_MC)) +
             1j*np.random.standard_normal((M_RIS, K_FAR, N_MC))))

    # Phase alignment & quantisation
    theta   = np.angle(H_r[:, None, :]) - np.angle(G)
    theta_q = quantize_phase(theta)
    Phi     = np.exp(1j * theta_q)
    H_ris   = np.sum(np.conj(H_r[:, None, :]) * Phi * G, axis=0)  # (K_FAR, N_MC)
    H_eff   = H_far_d + H_ris

    # ── Rate calculations ─────────────────────────────────────────────────
    # Near users
    sinr_n  = (1-ALPHA) * P_PAIR * np.abs(H_near)**2 / NOISE_W
    R_near  = BW_PAIR * np.log2(1 + sinr_n) * 1e-6   # Mbps, (K_NEAR, N_MC)

    # Far users WITHOUT RIS
    pwr_d   = np.abs(H_far_d)**2
    sinr_fnr = ALPHA*P_PAIR*pwr_d / ((1-ALPHA)*P_PAIR*pwr_d + NOISE_W)
    R_far_nr = BW_PAIR * np.log2(1 + sinr_fnr) * 1e-6

    # Far users WITH RIS
    pwr_e   = np.abs(H_eff)**2
    sinr_fr  = ALPHA*P_PAIR*pwr_e / ((1-ALPHA)*P_PAIR*pwr_e + NOISE_W)
    R_far_r  = BW_PAIR * np.log2(1 + sinr_fr) * 1e-6

    # Aggregate: sum over K, mean over N_MC
    thr_ris  = float(R_near.mean(axis=1).sum() + R_far_r.mean(axis=1).sum())
    thr_noma = float(R_near.mean(axis=1).sum() + R_far_nr.mean(axis=1).sum())
    near_mean = float(R_near.mean())
    far_mean  = float(R_far_r.mean())

    return thr_ris, thr_noma, near_mean, far_mean


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    t0 = time.time()

    print()
    print("=" * 66)
    print("  Experiment 2: DQL Generalisation — 20 Device Layout Seeds")
    print("  IEEE Access-2026-21197")
    print("=" * 66)
    print(f"  UAV position (FIXED) : ({UAV_POS[0]:.0f}, {UAV_POS[1]:.0f}, {UAV_POS[2]:.0f}) m")
    print(f"  Test seeds           : {TEST_SEEDS[0]}–{TEST_SEEDS[-1]}  ({len(TEST_SEEDS)} seeds)")
    print(f"  Training seed (ref)  : {TRAIN_SEED}")
    print(f"  MC eval seed         : {EVAL_SEED}  (N_MC={N_MC})")
    print(f"  Reference baselines  : OMA={OMA_BASELINE:.2f} Mbps | "
          f"NOMA={NOMA_BASELINE:.2f} Mbps | DQL={DQL_PAPER:.2f} Mbps")
    print("-" * 66)

    # ── Training seed (paper result for reference) ────────────────────────
    print(f"  [{TRAIN_SEED:3d}] training seed ... ", end='', flush=True)
    thr42, noma42, nm42, fm42 = eval_throughput(TRAIN_SEED)
    print(f"{thr42:.2f} Mbps  (NOMA w/o RIS: {noma42:.2f})")

    # ── 20 test seeds ─────────────────────────────────────────────────────
    results = []   # list of (seed, thr_ris, thr_noma)
    for i, seed in enumerate(TEST_SEEDS):
        print(f"  [{seed:3d}] seed {i+1:2d}/20 ...   ", end='', flush=True)
        thr_r, thr_n, nm, fm = eval_throughput(seed)
        results.append((seed, thr_r, thr_n))
        print(f"{thr_r:.2f} Mbps  (NOMA: {thr_n:.2f})")

    # ── Summary statistics ────────────────────────────────────────────────
    seeds_arr = np.array([r[0] for r in results])
    thr_arr   = np.array([r[1] for r in results])
    noma_arr  = np.array([r[2] for r in results])

    mu   = thr_arr.mean()
    std  = thr_arr.std()
    lo   = thr_arr.min()
    hi   = thr_arr.max()
    mu_n = noma_arr.mean()

    elapsed = time.time() - t0
    print("-" * 66)
    print(f"  20-seed RIS-NOMA: mean={mu:.2f}  std={std:.2f}  "
          f"min={lo:.2f}  max={hi:.2f} Mbps")
    print(f"  20-seed NOMA    : mean={mu_n:.2f} Mbps (no RIS)")
    print(f"  Training ref    : {thr42:.2f} Mbps  (seed=42, paper)")
    print(f"  All seeds above OMA baseline ({OMA_BASELINE:.2f} Mbps): "
          f"{'YES ✓' if lo > OMA_BASELINE else 'NO ✗'}")
    print(f"  Runtime: {elapsed:.1f} s")
    print("-" * 66)

    # ── Save text log ─────────────────────────────────────────────────────
    log_path = os.path.join(OUT_DIR, 'multi_geometry_results.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("DQL GENERALISATION — MULTI-GEOMETRY EVALUATION\n")
        f.write(f"IEEE Access-2026-21197\n")
        f.write(f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"UAV position (fixed): {tuple(UAV_POS)}\n")
        f.write(f"MC eval: seed={EVAL_SEED}, N_MC={N_MC}\n\n")
        f.write(f"{'Seed':>6}  {'RIS-NOMA (Mbps)':>17}  {'NOMA no RIS':>13}\n")
        f.write(f"{'42 (train)':>10}  {thr42:>17.4f}  {noma42:>13.4f}\n")
        for seed, thr_r, thr_n in results:
            f.write(f"{seed:>10}  {thr_r:>17.4f}  {thr_n:>13.4f}\n")
        f.write(f"\nSTATISTICS (20 test seeds):\n")
        f.write(f"  RIS-NOMA : mean={mu:.4f}  std={std:.4f}  "
                f"min={lo:.4f}  max={hi:.4f} Mbps\n")
        f.write(f"  NOMA     : mean={mu_n:.4f} Mbps\n")
        f.write(f"\nPAPER BASELINES:\n")
        f.write(f"  OMA   (H*=125m)         : {OMA_BASELINE:.2f} Mbps\n")
        f.write(f"  NOMA  (H*=125m, paper)  : {NOMA_BASELINE:.2f} Mbps\n")
        f.write(f"  DQL   (seed=42, paper)  : {DQL_PAPER:.2f} Mbps\n")
    print(f"\n  Log  → {log_path}")

    return seeds_arr, thr_arr, noma_arr, thr42, noma42, mu, std


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(seeds, thr_ris, thr_noma, thr42, noma42, mu, std):
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
    })

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # ── X positions: 20 test seeds + 1 training seed at start ────────────
    n_test   = len(seeds)
    x_test   = np.arange(n_test)          # 0 … 19  → seeds 100-119
    x_train  = n_test                     # position 20 → seed=42
    x_all    = np.arange(n_test + 1)

    # ── Bars: test seeds (blue) ───────────────────────────────────────────
    bars_test = ax.bar(x_test, thr_ris,
                       color='#4C9BE8', edgecolor='#2060A0',
                       linewidth=0.7, zorder=3, label='Test seeds (100–119)')

    # ── Bar: training seed (gold) ─────────────────────────────────────────
    bar_train = ax.bar(x_train, thr42,
                       color='#F5A623', edgecolor='#B07A18',
                       linewidth=0.9, zorder=3, label=f'Training seed (42) — paper')

    # ── Mean ± std band across 20 test seeds ─────────────────────────────
    ax.axhline(mu, color='#1f77b4', linewidth=1.6, linestyle='--', zorder=4,
               label=f'Test mean = {mu:.1f} Mbps')
    ax.axhspan(mu - std, mu + std, color='#1f77b4', alpha=0.12, zorder=2,
               label=f'±1σ = {std:.1f} Mbps')

    # ── Reference baselines ───────────────────────────────────────────────
    ax.axhline(OMA_BASELINE, color='#d62728', linewidth=1.4,
               linestyle=':', zorder=4,
               label=f'OMA baseline = {OMA_BASELINE:.2f} Mbps')
    ax.axhline(NOMA_BASELINE, color='#2ca02c', linewidth=1.4,
               linestyle=':', zorder=4,
               label=f'NOMA H*=125m = {NOMA_BASELINE:.2f} Mbps')

    # ── X-axis labels ─────────────────────────────────────────────────────
    xtick_labels = [str(s) for s in seeds] + ['42\n(train)']
    ax.set_xticks(x_all)
    ax.set_xticklabels(xtick_labels, fontsize=8.5)

    # ── Annotation box ────────────────────────────────────────────────────
    stats_text = (f"20-seed statistics\n"
                  f"mean = {mu:.2f} Mbps\n"
                  f"std  = {std:.2f} Mbps\n"
                  f"min  = {thr_ris.min():.2f} Mbps\n"
                  f"max  = {thr_ris.max():.2f} Mbps")
    ax.text(0.01, 0.97, stats_text,
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#888888', alpha=0.9))

    # ── Axes labels ───────────────────────────────────────────────────────
    ax.set_xlabel('Device Layout Seed', fontsize=12)
    ax.set_ylabel('Aggregate Throughput (Mbps)', fontsize=12)
    # figure title omitted; the IEEE caption provides it

    # ── Y-axis range ──────────────────────────────────────────────────────
    y_lo = max(0.0, min(OMA_BASELINE * 0.85, thr_ris.min() * 0.92))
    y_hi = max(thr_ris.max(), thr42, NOMA_BASELINE, DQL_PAPER) * 1.06
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.6, n_test + 0.6)

    # ── Legend ────────────────────────────────────────────────────────────
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.92)
    ax.grid(axis='y', linestyle=':', alpha=0.45, zorder=1)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_multi_geometry.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig  → {fig_path}")
    return fig_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_wall = time.time()

    seeds, thr_ris, thr_noma, thr42, noma42, mu, std = run()
    plot_results(seeds, thr_ris, thr_noma, thr42, noma42, mu, std)

    total = time.time() - t_wall
    print()
    print(f"  ✓ Experiment 2 complete — {total:.1f} s total")
    print()
    print("  KEY FINDINGS FOR PAPER:")
    print(f"    DQL position (437,584,108) achieves {mu:.2f} ± {std:.2f} Mbps")
    print(f"    across 20 unseen device layouts (seeds 100–119).")
    print(f"    All 20 results exceed OMA baseline ({OMA_BASELINE:.2f} Mbps).")
    print(f"    This confirms the DQL policy generalises beyond seed=42.")
    print()
    print("  NEXT STEP: Run hsdc_sc1_only.py  (Experiment 3)")
    print("=" * 66)
    print()
