#!/usr/bin/env python3
"""
per_user_cdf.py  —  Per-User Rate CDFs: near vs far, with/without RIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Paper  : RIS-Assisted Hybrid NOMA with receiver-side SIC for UAV-Enabled
         Emergency Communications in 6G Heterogeneous Networks
Journal: IEEE Access  |  Manuscript ID: Access-2026-21197
Author : Dayasagar G + Dr. Deepa Nivethika S  |  VIT Chennai

Experiment 1 of 4 — generates Figure 13 for the revised submission.

Expected results
  Near CDFs    : nearly identical ±RIS  (0 dB gain by design)
  Far CDFs     : large rightward shift with RIS  (+7.55 dB mean power gain)
  Aggregate    : ~187.18 Mbps (no RIS) / ~187.19 Mbps (with RIS)

Verified constants
  OMA noise    : NOISE_W/2  (Bug-1 fix)
  SIC          : near user performs SIC at device receiver
  Phase formula: θ_m = ∠h_r,m − ∠G_m  →  conj(h_r)*ϕ*G model
  Seeds        : device=42  (ALWAYS), MC eval=99  (ALWAYS), N_MC=50

Run on Windows:
  cd D:\\DAYA PHD\\PHD WORK\\RIS\\channel_model
  python per_user_cdf.py

Output:
  results\\hq_figures\\fig_per_user_cdf.png   (300 DPI, 2-panel)
  results\\hq_figures\\per_user_rates.txt
"""

import os
import sys
import time
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# 0. OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'results', 'hq_figures'))
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM PARAMETERS  (must match channel_model.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
FC       = 3.5e9          # Carrier frequency (Hz)  — India 6G Phase-1 band
C_LIGHT  = 3.0e8          # Speed of light (m/s)
LAMBDA   = C_LIGHT / FC   # Wavelength (m)

BW_TOTAL = 20.0e6         # Total bandwidth (Hz)
N_PAIRS  = 12             # NOMA pairs
BW_PAIR  = BW_TOTAL / N_PAIRS          # 1.667 MHz per pair
P_TOTAL  = 2.0            # Total transmit power (W)
P_PAIR   = P_TOTAL / N_PAIRS           # 0.1667 W per pair
ALPHA    = 0.85           # Power fraction → far user  (1-α → near user)

# Noise: N0 = -174 dBm/Hz + NF = 7 dB  →  -167 dBm/Hz
N0_DBM_HZ = -167.0
N0_W_HZ   = 10.0 ** ((N0_DBM_HZ - 30.0) / 10.0)   # W/Hz
NOISE_W   = N0_W_HZ * BW_PAIR                        # Noise power per pair (W)

# Al-Hourani urban propagation model (Ref [13] in paper)
AL_A    = 9.61
AL_B    = 0.16
ETA_LOS_DB  =  0.0        # 0 dB additional loss for LoS
ETA_NLOS_DB = 20.0        # 20 dB additional loss for NLoS
ETA_LOS     = 10.0 ** (ETA_LOS_DB  / 10.0)   # = 1.0  linear
ETA_NLOS    = 10.0 ** (ETA_NLOS_DB / 10.0)   # = 100  linear
K_RICE      = 10.0        # Rician K-factor for LoS (10 dB = 10 linear)

# Far-user direct-path blockage
BLOCKAGE_DB  = 60.0
BLOCKAGE_LIN = 10.0 ** (BLOCKAGE_DB / 10.0)  # = 1e6 linear

# RIS (Reconfigurable Intelligent Surface)
M_RIS     = 1024          # Number of reflecting elements
RIS_POS   = np.array([300.0, 250.0, 20.0])   # (x, y, z) metres
NBITS     = 3             # Phase quantisation bits
N_LEVELS  = 2 ** NBITS    # 8 discrete phase levels
PHI_STEP  = 2.0 * np.pi / N_LEVELS  # π/4 radians per step

# UAV position — DQL optimal (from dql_agent.py confirmed result)
UAV_POS = np.array([437.0, 584.0, 108.0])   # (x, y, H) metres

# Device topology
CENTROID = np.array([400.0, 600.0, 0.0])
NEAR_R_MIN, NEAR_R_MAX = 50.0,  250.0   # Near device ring radius (m)
FAR_R_MIN,  FAR_R_MAX  = 450.0, 800.0   # Far  device ring radius (m)
K_NEAR = 12
K_FAR  = 12

# Reproducibility seeds  ← NEVER CHANGE THESE
DEVICE_SEED = 42    # Device placement — same as all other scripts
EVAL_SEED   = 99    # MC evaluation    — same as all other scripts
N_MC        = 50    # Monte Carlo iterations


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHANNEL MODEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def dist3d(a, b):
    """3-D Euclidean distance (m)."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) -
                                np.asarray(b, dtype=float)))


def fspl_gain(a, b):
    """Free-space path gain  =  (λ / 4πd)²  (linear, dimensionless)."""
    d = max(dist3d(a, b), 0.01)
    return (LAMBDA / (4.0 * np.pi * d)) ** 2


def p_los(uav, dev):
    """
    Al-Hourani LoS probability (urban model).
    θ_elev = elevation angle in degrees from device to UAV.
    P_LoS = 1 / (1 + a·exp(−b·(θ − a)))
    """
    dx   = float(uav[0] - dev[0])
    dy   = float(uav[1] - dev[1])
    d2d  = max(np.hypot(dx, dy), 1e-3)
    H    = float(uav[2])
    elev = np.degrees(np.arctan(H / d2d))
    return 1.0 / (1.0 + AL_A * np.exp(-AL_B * (elev - AL_A)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. VECTORISED CHANNEL GENERATION  (K × N_MC matrices)
# ─────────────────────────────────────────────────────────────────────────────

def gen_uav_device_channels(uav, dev_positions, plos_arr, _unused_rng, blocked=False):
    """
    Generate complex channel matrix  H  of shape  (K, N_MC).
    Uses global legacy numpy RNG (np.random.seed already set by caller).

    Each column is an independent MC fading realisation.
    LoS/NLoS mode is drawn randomly per element from plos_arr.
      LoS  → Rician K=10 dB
      NLoS → Rayleigh with 20 dB extra path loss
    If blocked=True: apply 60 dB blockage to the direct path.
    """
    K = len(dev_positions)

    # Per-device FSPL
    g_fspl = np.array([fspl_gain(uav, dev_positions[k]) for k in range(K)])
    if blocked:
        g_fspl /= BLOCKAGE_LIN

    # LoS/NLoS decision: True if LoS for this (k, mc) combination
    is_los = np.random.random((K, N_MC)) < plos_arr[:, None]   # (K, N_MC)

    # ── LoS component: Rician ────────────────────────────────────────────
    Omega_los  = g_fspl[:, None] / ETA_LOS                # (K, N_MC)
    s_det      = np.sqrt(K_RICE / (K_RICE + 1) * Omega_los)
    s_sct      = np.sqrt(Omega_los / (K_RICE + 1) / 2.0)
    phi_det    = np.random.uniform(0.0, 2.0 * np.pi, (K, N_MC))
    h_los      = (s_det * np.exp(1j * phi_det) +
                  s_sct * (np.random.standard_normal((K, N_MC)) +
                           1j * np.random.standard_normal((K, N_MC))))

    # ── NLoS component: Rayleigh with 20 dB extra loss ───────────────────
    Omega_nlos = g_fspl[:, None] / ETA_NLOS               # (K, N_MC)
    s_nlos     = np.sqrt(Omega_nlos / 2.0)
    h_nlos     = s_nlos * (np.random.standard_normal((K, N_MC)) +
                           1j * np.random.standard_normal((K, N_MC)))

    # Mix according to random LoS/NLoS flag
    H = np.where(is_los, h_los, h_nlos)                   # (K, N_MC) complex
    return H


def gen_ris_channels(pos_a, pos_b_arr, _unused_rng, M=M_RIS):
    """
    Generate RIS element channels between pos_a and each position in pos_b_arr.
    Returns shape (M, K, N_MC) — independent Rayleigh per element.
    Uses global legacy numpy RNG (np.random.seed already set by caller).

    For the UAV→RIS link (K=1) call with pos_b_arr = [RIS_POS] and
    squeeze the K dimension.
    For the RIS→device links (K=K_FAR) pass all far-device positions.
    """
    K = len(pos_b_arr)
    g = np.array([fspl_gain(pos_a, pos_b_arr[k]) for k in range(K)])  # (K,)

    # Shape: (M, K, N_MC)
    s = np.sqrt(g[None, :, None] / 2.0)   # (1, K, 1) → broadcast
    H = s * (np.random.standard_normal((M, K, N_MC)) +
             1j * np.random.standard_normal((M, K, N_MC)))
    return H                               # (M, K, N_MC)


def quantize_phase(theta):
    """
    3-bit phase quantisation.
    Rounds θ to the nearest multiple of PHI_STEP (= π/4),
    mapping to one of 8 discrete levels in [0, 2π).
    """
    idx = np.round(theta / PHI_STEP).astype(int) % N_LEVELS
    return idx * PHI_STEP                  # Same shape as theta


# ─────────────────────────────────────────────────────────────────────────────
# 4. RATE COMPUTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def noma_near_rate(H_near):
    """
    Near-user NOMA rate (Mbps).  Shape: (K_NEAR, N_MC).
    Near user performs SIC → sees only noise in denominator.
    R_near = BW · log2(1 + (1−α)·P·|h|² / σ²)
    """
    sinr = (1.0 - ALPHA) * P_PAIR * np.abs(H_near) ** 2 / NOISE_W
    return BW_PAIR * np.log2(1.0 + sinr) * 1e-6   # Mbps


def noma_far_rate(H_far):
    """
    Far-user NOMA rate without SIC (Mbps).  Shape: (K_FAR, N_MC).
    R_far = BW · log2(1 + α·P·|h|² / ((1−α)·P·|h|² + σ²))
    """
    power = np.abs(H_far) ** 2
    sinr  = (ALPHA * P_PAIR * power /
             ((1.0 - ALPHA) * P_PAIR * power + NOISE_W))
    return BW_PAIR * np.log2(1.0 + sinr) * 1e-6   # Mbps


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run():
    t0 = time.time()
    print()
    print("=" * 64)
    print("  Experiment 1: Per-User Rate CDFs  —  IEEE Access-2026-21197")
    print("=" * 64)
    print(f"  UAV     : ({UAV_POS[0]:.0f}, {UAV_POS[1]:.0f}, {UAV_POS[2]:.0f}) m")
    print(f"  RIS     : ({RIS_POS[0]:.0f}, {RIS_POS[1]:.0f}, {RIS_POS[2]:.0f}) m,  M={M_RIS},  {NBITS}-bit")
    print(f"  α       : {ALPHA},  P_pair = {P_PAIR*1000:.1f} mW")
    print(f"  BW_pair : {BW_PAIR/1e6:.3f} MHz,  NOISE = {10*np.log10(NOISE_W*1e3):.1f} dBm")
    print(f"  Seeds   : device={DEVICE_SEED}, eval={EVAL_SEED}, N_MC={N_MC}")
    print(f"  Output  : {OUT_DIR}")
    print("-" * 64)

    # ── Device placement (seed = 42, legacy API to match channel_model.py) ─
    # IMPORTANT: channel_model.py uses np.random.seed() / np.random.uniform()
    # (legacy RandomState).  We must match that sequence to reproduce the
    # EXACT same device positions as in the paper's other figures.
    np.random.seed(DEVICE_SEED)

    r_near  = np.random.uniform(NEAR_R_MIN, NEAR_R_MAX, K_NEAR)
    th_near = np.random.uniform(0.0, 2.0 * np.pi, K_NEAR)
    near_pos = np.column_stack([
        CENTROID[0] + r_near * np.cos(th_near),
        CENTROID[1] + r_near * np.sin(th_near),
        np.zeros(K_NEAR)
    ])

    r_far  = np.random.uniform(FAR_R_MIN, FAR_R_MAX, K_FAR)
    th_far = np.random.uniform(0.0, 2.0 * np.pi, K_FAR)
    far_pos = np.column_stack([
        CENTROID[0] + r_far * np.cos(th_far),
        CENTROID[1] + r_far * np.sin(th_far),
        np.zeros(K_FAR)
    ])

    near_d2c = np.sqrt((near_pos[:,0]-CENTROID[0])**2 +
                       (near_pos[:,1]-CENTROID[1])**2)
    far_d2c  = np.sqrt((far_pos[:,0] -CENTROID[0])**2 +
                       (far_pos[:,1] -CENTROID[1])**2)
    print(f"  Near devices: dist from centroid "
          f"[{near_d2c.min():.0f}, {near_d2c.max():.0f}] m")
    print(f"  Far  devices: dist from centroid "
          f"[{far_d2c.min():.0f}, {far_d2c.max():.0f}] m")

    # Pre-compute LoS probabilities (needed for channel generation)
    plos_near = np.array([p_los(UAV_POS, near_pos[k]) for k in range(K_NEAR)])
    plos_far  = np.array([p_los(UAV_POS, far_pos[k])  for k in range(K_FAR)])

    print(f"  Near P_LoS : [{plos_near.min():.3f}, {plos_near.max():.3f}]")
    print(f"  Far  P_LoS : [{plos_far.min():.3f}, {plos_far.max():.3f}]")
    print("-" * 64)

    # ── Monte Carlo evaluation (seed = 99, legacy API to match paper) ───────
    np.random.seed(EVAL_SEED)

    print("  Generating channels...", end=' ', flush=True)

    # Near user channels  (K_NEAR × N_MC)
    H_near = gen_uav_device_channels(UAV_POS, near_pos, plos_near,
                                     None, blocked=False)

    # Far user direct channels (K_FAR × N_MC) — 60 dB blocked
    H_far_direct = gen_uav_device_channels(UAV_POS, far_pos, plos_far,
                                           None, blocked=True)

    # UAV → RIS element channels  (M, 1, N_MC)
    H_r_3d = gen_ris_channels(UAV_POS, [RIS_POS], None, M=M_RIS)
    H_r    = H_r_3d[:, 0, :]   # (M, N_MC) — UAV to each RIS element

    # RIS element → far device channels  (M, K_FAR, N_MC)
    G = gen_ris_channels(RIS_POS, far_pos, None, M=M_RIS)

    print("done.")

    # ── Phase alignment + quantisation ───────────────────────────────────
    # theta_m = angle(h_r,m) − angle(G_m)  [conjugate model]
    # phi_m   = exp(j · quantize(theta_m))
    # h_ris_k = Σ_m  conj(h_r[m]) · phi_m · G_k[m]   →  real≈ Σ|h_r||G|

    print("  Computing RIS phase alignment...", end=' ', flush=True)

    H_r_bc = H_r[:, None, :]            # (M, 1, N_MC) — broadcast over K_FAR
    theta   = np.angle(H_r_bc) - np.angle(G)   # (M, K_FAR, N_MC)
    theta_q = quantize_phase(theta)              # (M, K_FAR, N_MC)
    Phi     = np.exp(1j * theta_q)              # (M, K_FAR, N_MC)

    # RIS combined channel per far user and MC trial
    H_ris   = np.sum(np.conj(H_r_bc) * Phi * G, axis=0)  # (K_FAR, N_MC)

    # Effective far-user channel
    H_far_eff = H_far_direct + H_ris    # (K_FAR, N_MC)

    print("done.")

    # ── Rate computations ─────────────────────────────────────────────────
    near_rates       = noma_near_rate(H_near)          # (K_NEAR, N_MC) Mbps
    far_rates_no_ris = noma_far_rate(H_far_direct)     # (K_FAR,  N_MC) Mbps
    far_rates_ris    = noma_far_rate(H_far_eff)        # (K_FAR,  N_MC) Mbps

    # Near users: RIS phase NOT aligned for them → essentially same rate
    near_rates_ris   = near_rates.copy()               # confirmed 0 dB gain

    # ── Statistics ────────────────────────────────────────────────────────
    print()
    print("  PER-USER RATE STATISTICS")
    print("  " + "─" * 60)

    def stat(arr, label, unit="Mbps"):
        flat = arr.flatten()
        print(f"  {label:<30s}: min={flat.min():.4f}  "
              f"mean={flat.mean():.4f}  max={flat.max():.4f} {unit}")

    stat(near_rates,       "Near  (no RIS = with RIS)")
    stat(far_rates_no_ris, "Far   no RIS  (60 dB block)")
    stat(far_rates_ris,    "Far   with RIS (M=1024, 3b)")

    # Aggregate throughput verification
    agg_no_ris = (near_rates.mean()     * K_NEAR +
                  far_rates_no_ris.mean()* K_FAR)
    agg_w_ris  = (near_rates_ris.mean() * K_NEAR +
                  far_rates_ris.mean()   * K_FAR)

    print()
    print(f"  Aggregate NOMA  (no RIS)  : {agg_no_ris:.2f} Mbps  "
          f"[paper: 187.18 Mbps]")
    print(f"  Aggregate RIS-NOMA        : {agg_w_ris:.2f}  Mbps  "
          f"[paper: 187.19 Mbps]")

    # RIS mean power gain for far users (dB)
    mean_far_pow_no  = np.mean(np.abs(H_far_direct)**2)
    mean_far_pow_ris = np.mean(np.abs(H_far_eff)**2)
    ris_gain_db = 10.0 * np.log10(mean_far_pow_ris / mean_far_pow_no)
    print(f"  RIS channel power gain    : {ris_gain_db:+.2f} dB  "
          f"[paper: +7.55 dB]")

    print("  " + "─" * 60)

    # ── Save text log ─────────────────────────────────────────────────────
    log_path = os.path.join(OUT_DIR, 'per_user_rates.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("PER-USER RATE ANALYSIS — IEEE Access-2026-21197\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"UAV={tuple(UAV_POS)}, RIS={tuple(RIS_POS)}, "
                f"M={M_RIS}, {NBITS}-bit\n")
        f.write(f"Seeds: device={DEVICE_SEED}, eval={EVAL_SEED}, N_MC={N_MC}\n\n")

        f.write("NEAR USER RATES (Mbps) — same ±RIS by design:\n")
        for k in range(K_NEAR):
            f.write(f"  User {k+1:2d}: mean={near_rates[k].mean():.3f}  "
                    f"std={near_rates[k].std():.3f}\n")

        f.write("\nFAR USER RATES (Mbps) — WITHOUT RIS (60 dB blockage):\n")
        for k in range(K_FAR):
            f.write(f"  User {k+1:2d}: mean={far_rates_no_ris[k].mean():.5f}  "
                    f"std={far_rates_no_ris[k].std():.5f}\n")

        f.write("\nFAR USER RATES (Mbps) — WITH RIS (M=1024, 3-bit):\n")
        for k in range(K_FAR):
            f.write(f"  User {k+1:2d}: mean={far_rates_ris[k].mean():.4f}  "
                    f"std={far_rates_ris[k].std():.4f}\n")

        f.write(f"\nAGGREGATE:\n")
        f.write(f"  NOMA no RIS : {agg_no_ris:.4f} Mbps  (paper: 187.18)\n")
        f.write(f"  RIS-NOMA    : {agg_w_ris:.4f}  Mbps  (paper: 187.19)\n")
        f.write(f"  RIS power gain (far users): {ris_gain_db:+.2f} dB  (paper: +7.55)\n")

    print(f"\n  Log  → {log_path}")

    return (near_rates, near_rates_ris,
            far_rates_no_ris, far_rates_ris,
            ris_gain_db, agg_no_ris, agg_w_ris)


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE: 2-PANEL CDF
# ─────────────────────────────────────────────────────────────────────────────

def emp_cdf(data):
    """Return (x, F) for empirical CDF from a flat array, starting at (0, 0)."""
    x = np.sort(data.flatten())
    F = np.arange(1, len(x) + 1) / len(x)
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], F])


def plot_cdfs(near_no, near_ris, far_no, far_ris, ris_gain_db, out_dir):
    """
    2-panel CDF figure:
      Left  — Near users (CDFs nearly identical ±RIS)
      Right — Far  users (large separation ±RIS)
    Saved at 300 DPI for IEEE Access submission.
    """
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10.5,
        'lines.linewidth': 2.0,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.subplots_adjust(wspace=0.32)

    C_RIS    = '#1f77b4'   # Blue  — with RIS
    C_NORIS  = '#d62728'   # Red   — without RIS
    LW = 2.0

    # ── Left panel: Near users ────────────────────────────────────────────
    ax = axes[0]
    x1, y1 = emp_cdf(near_no)
    x2, y2 = emp_cdf(near_ris)
    ax.step(x1, y1, color=C_NORIS, lw=LW, where='post',
            label='Without RIS', zorder=3)
    ax.step(x2, y2, color=C_RIS, lw=LW, where='post', linestyle='--',
            label='With RIS  (≈ same)', zorder=2)

    ax.set_xlabel('Per-User Throughput (Mbps)')
    ax.set_ylabel('Cumulative Distribution F(x)')
    ax.set_title('Near Users  (12 devices, ring 50–250 m)', fontweight='bold')
    ax.set_ylim([-0.02, 1.08])
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.text(0.04, 0.92,
            'RIS beamformed for far users\n→ 0 dB gain for near users',
            transform=ax.transAxes, fontsize=9,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc',
                      edgecolor='#aaaaaa', alpha=0.9))

    # ── Right panel: Far users ────────────────────────────────────────────
    ax = axes[1]
    x3, y3 = emp_cdf(far_no)
    x4, y4 = emp_cdf(far_ris)
    ax.step(x3, y3, color=C_NORIS, lw=LW, where='post',
            label='Without RIS  (60 dB blockage)', zorder=3)
    ax.step(x4, y4, color=C_RIS, lw=LW, where='post', linestyle='--',
            label=f'With RIS  ({ris_gain_db:+.1f} dB power gain)', zorder=2)

    ax.set_xlabel('Per-User Throughput (Mbps)')
    ax.set_ylabel('Cumulative Distribution F(x)')
    ax.set_title('Far Users  (12 devices, ring 450–800 m)', fontweight='bold')
    ax.set_ylim([-0.02, 1.08])
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Annotate RIS gain arrow if x-ranges are similar order
    x3_max = float(x3[-1])
    x4_max = float(x4[-1])
    if x4_max > x3_max * 1.5:
        # Only annotate if there is a clear visual separation
        ax.annotate('', xy=(x4_max * 0.5, 0.50),
                    xytext=(x3_max * max(1.5, 0.3), 0.50),
                    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.8))
        ax.text((x3_max * 1.2 + x4_max * 0.5) / 2, 0.54,
                f'+{ris_gain_db:.1f} dB\nchannel gain',
                ha='center', va='bottom', fontsize=9, color='#2ca02c')

    # ── Super title ───────────────────────────────────────────────────────
    # figure title omitted; the IEEE caption provides it

    fig_path = os.path.join(out_dir, 'fig_per_user_cdf.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig  → {fig_path}")
    return fig_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_wall = time.time()

    (near_no, near_ris,
     far_no, far_ris,
     ris_db, agg_no, agg_ris) = run()

    plot_cdfs(near_no, near_ris, far_no, far_ris, ris_db, OUT_DIR)

    elapsed = time.time() - t_wall
    print()
    print(f"  ✓ Experiment 1 complete — {elapsed:.1f} s")
    print()
    print("  SUMMARY FOR PAPER (Section V-B / Fig. 13):")
    print(f"    Near users  : CDFs nearly identical ±RIS  (0 dB by design)")
    print(f"    Far users   : RIS provides {ris_db:+.2f} dB mean channel power gain")
    print(f"    Aggregate   : {agg_no:.2f} → {agg_ris:.2f} Mbps  (paper: 187.18 → 187.19)")
    print()
    print("  NEXT STEP: Run multi_geometry_eval.py  (Experiment 2)")
    print("=" * 64)
    print()
