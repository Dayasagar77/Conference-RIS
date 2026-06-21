#!/usr/bin/env python3
"""
fig13_per_user_cdf_FAITHFUL.py  —  Per-user CDF using channel_model.py itself
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Imports the MAIN simulation module and reuses its exact channel functions
(chan_scalar, G_vector, H_ris_dev, opt_phi, composite, noma_pair) so the
per-user rates are on the SAME scale and use the SAME physics as the
paper's Table IV / Fig 3 headline numbers (NOMA 187, RIS 187).

Device positions come from channel_model.generate_devices() (seed 42) —
the real Scenario-1 layout, not a re-derivation.

UAV held at the DQL-optimised position (437, 584, 108) m (Figs 10/11).

Run on Windows (channel_model.py must be in the same folder):
  cd D:\\DAYA PHD\\PHD WORK\\RIS\\channel_model
  python fig13_per_user_cdf_FAITHFUL.py
"""
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import channel_model as cm

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'results', 'FIGS_13_16'))
os.makedirs(OUT, exist_ok=True)

UAV   = np.array([437.0, 584.0, 108.0])   # DQL-optimal position
N_MC  = 300
EVAL_SEED = 99

# --- real SC1 device layout (seed 42, exactly as the paper) ---------------
np.random.seed(42)
positions, labels = cm.generate_devices()
near = positions[:cm.CFG.K_NEAR]
far  = positions[cm.CFG.K_NEAR:cm.CFG.K_SC1]
KN, KF = cm.CFG.K_NEAR, cm.CFG.K_FAR

# --- Monte-Carlo per-user rates using channel_model's own functions -------
np.random.seed(EVAL_SEED)
R_near, R_far_nr, R_far_r = [], [], []
gains_far = []   # channel_model's RIS-gain definition (mean of per-realisation dB)

for _ in range(N_MC):
    G_v = cm.G_vector(UAV)
    for i in range(KN):
        h_near = cm.chan_scalar(UAV, near[i], 0.0)
        h_far_d = cm.chan_scalar(UAV, far[i], cm.CFG.BLOCK_DB)
        # near rate (independent of RIS — SIC removes far signal)
        rN, _ = cm.noma_pair(h_near, h_far_d)
        R_near.append(rN/1e6)
        # far without RIS
        _, rF_nr = cm.noma_pair(h_near, h_far_d)
        R_far_nr.append(rF_nr/1e6)
        # far with RIS (channel_model composite + opt_phi)
        h_r   = cm.H_ris_dev(far[i])
        phi   = cm.opt_phi(G_v, h_r)
        h_far_e = cm.composite(h_far_d, G_v, h_r, phi)
        _, rF_r = cm.noma_pair(h_near, h_far_e)
        R_far_r.append(rF_r/1e6)
        gains_far.append(10*np.log10(abs(h_far_e)**2/max(abs(h_far_d)**2,1e-40)))

R_near   = np.array(R_near)
R_far_nr = np.array(R_far_nr)
R_far_r  = np.array(R_far_r)
ris_gain_far_dB = float(np.mean(gains_far))   # should match Fig 8 (+7.55 dB)

print("Per-user rates from channel_model.py (same physics as headline):")
print(f"  Near        : mean={R_near.mean():.3f}  min={R_near.min():.3f}  max={R_near.max():.3f} Mbps")
print(f"  Far  no-RIS : mean={R_far_nr.mean():.5f} Mbps")
print(f"  Far  w/ RIS : mean={R_far_r.mean():.4f} Mbps")
print(f"  Far RIS gain (channel_model definition) = {ris_gain_far_dB:+.2f} dB   [Fig 8 = +7.55 dB]")
print(f"  Aggregate NOMA(no RIS) = {R_near.mean()*KN + R_far_nr.mean()*KF:.2f} Mbps")
print(f"  Aggregate RIS-NOMA     = {R_near.mean()*KN + R_far_r.mean()*KF:.2f} Mbps")

def ecdf(d):
    x = np.sort(d); y = np.arange(1, len(x)+1)/len(x)
    return np.r_[0, x], np.r_[0, y]

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8)); fig.subplots_adjust(wspace=0.32)
cb, cr = '#1f77b4', '#d62728'

x1, y1 = ecdf(R_near)
ax[0].step(x1, y1, color=cr, lw=2, where='post', label='Without RIS')
ax[0].step(x1, y1, color=cb, lw=2, where='post', ls='--', label='With RIS (identical)')
ax[0].set_xlabel('Per-User Throughput (Mbps)'); ax[0].set_ylabel('CDF')
ax[0].set_title('Near Users', fontweight='bold')
ax[0].set_ylim(-0.02, 1.05); ax[0].legend(loc='lower right'); ax[0].grid(True, ls=':', alpha=0.5)
ax[0].text(0.04, 0.92, 'RIS optimised for far users\n\u2192 0 dB near gain by design',
           transform=ax[0].transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc', alpha=0.9))

x3, y3 = ecdf(R_far_nr); x4, y4 = ecdf(R_far_r)
ax[1].step(x3, y3, color=cr, lw=2, where='post', label='Without RIS (60 dB block)')
ax[1].step(x4, y4, color=cb, lw=2, where='post', ls='--',
           label=f'With RIS (M={cm.CFG.M}, {cm.CFG.B_PHASE}-bit)')
ax[1].set_xlabel('Per-User Throughput (Mbps)'); ax[1].set_ylabel('CDF')
ax[1].set_title('Far Users', fontweight='bold')
ax[1].set_ylim(-0.02, 1.05); ax[1].legend(loc='lower right'); ax[1].grid(True, ls=':', alpha=0.5)

# Figure title intentionally omitted; the IEEE caption provides it.
out = os.path.join(OUT, 'fig13_per_user_cdf_FAITHFUL.png')
fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
print(f"  saved -> {out}")
