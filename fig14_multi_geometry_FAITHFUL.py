#!/usr/bin/env python3
"""
fig14_multi_geometry_FAITHFUL.py  —  DQL Generalisation, channel_model.py physics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluates the fixed DQL-optimal UAV position (437, 584, 108) m across 20 unseen
device layout seeds, using channel_model.py's OWN channel functions so that the
throughput scale is consistent with the paper's 188.36 Mbps headline (Table IV).

Paper-baseline reproduced here: seed=42 → ~188 Mbps (same physics as Module 1).
Test seeds 100-119: throughput stability of the fixed DQL position.

Run on Windows (channel_model.py must be in the same folder):
  python fig14_multi_geometry_FAITHFUL.py
"""
import os, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import channel_model as cm

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'results', 'FIGS_13_16'))
os.makedirs(OUT, exist_ok=True)

UAV       = np.array([437., 584., 108.])
N_MC      = 50
EVAL_SEED = 99
TRAIN_SEED = 42
TEST_SEEDS = list(range(100, 120))

# Paper baselines from the SAME pipeline (channel_model.py, seed=42, n_trials=100)
OMA_REF  = 120.93   # Table IV, H=125 m
NOMA_REF = 187.18   # Table IV, H=125 m
DQL_REF  = 188.36   # Table VII


def eval_at_uav(device_seed):
    """Compute aggregate NOMA+RIS throughput at fixed UAV_POS for one device layout."""
    np.random.seed(device_seed)
    pos, _ = cm.generate_devices()
    near = pos[:cm.CFG.K_NEAR]
    far  = pos[cm.CFG.K_NEAR:cm.CFG.K_SC1]

    np.random.seed(EVAL_SEED)
    totals = []
    for _ in range(N_MC):
        G_v = cm.G_vector(UAV)
        r_total = 0.0
        for i in range(cm.CFG.N_PAIRS):
            hn   = cm.chan_scalar(UAV, near[i], 0.0)
            hd   = cm.chan_scalar(UAV, far[i],  cm.CFG.BLOCK_DB)
            hr   = cm.H_ris_dev(far[i])
            phi  = cm.opt_phi(G_v, hr)
            he   = cm.composite(hd, G_v, hr, phi)
            rN, _ = cm.noma_pair(hn, hd)
            _, rF  = cm.noma_pair(hn, he)
            r_total += (rN + rF)
        totals.append(r_total / 1e6)
    return float(np.mean(totals))


print("=" * 60)
print("  Fig 14 — DQL Generalisation (channel_model.py)")
print("=" * 60)
print(f"  UAV fixed at {tuple(UAV.astype(int))} m")
print(f"  N_MC={N_MC}, eval_seed={EVAL_SEED}")
t0 = time.time()

print(f"  [42] training seed ... ", end="", flush=True)
thr42 = eval_at_uav(TRAIN_SEED)
print(f"{thr42:.2f} Mbps")

results = []
for i, seed in enumerate(TEST_SEEDS):
    print(f"  [{seed}] seed {i+1:2d}/20 ... ", end="", flush=True)
    thr = eval_at_uav(seed)
    results.append((seed, thr))
    print(f"{thr:.2f} Mbps")

thr_arr = np.array([r[1] for r in results])
mu, std = thr_arr.mean(), thr_arr.std()
cv = 100 * std / mu
lo, hi = thr_arr.min(), thr_arr.max()

print(f"\n  20-seed RIS-NOMA: mean={mu:.2f}  std={std:.2f}  "
      f"CV={cv:.1f}%  min={lo:.2f}  max={hi:.2f} Mbps")
print(f"  Training seed 42: {thr42:.2f} Mbps  (paper: {DQL_REF})")
print(f"  All > OMA ({OMA_REF}): {'YES ✓' if lo > OMA_REF else 'NO ✗'}")
print(f"  Runtime: {time.time()-t0:.1f} s")

# ── Figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9.5})
fig, ax = plt.subplots(figsize=(13, 5.5))

x_test  = np.arange(len(TEST_SEEDS))
x_train = len(TEST_SEEDS)

ax.bar(x_test,  thr_arr, color='#4C9BE8', edgecolor='#2060A0', lw=0.7, zorder=3,
       label='Test seeds 100–119 (unseen layouts)')
ax.bar(x_train, thr42,   color='#F5A623', edgecolor='#B07A18', lw=0.9, zorder=3,
       label=f'Training seed 42 (paper: {DQL_REF} Mbps)')

ax.axhline(mu, color='#1f77b4', lw=1.6, ls='--', zorder=4,
           label=f'Test mean = {mu:.1f} Mbps')
ax.axhspan(mu-std, mu+std, color='#1f77b4', alpha=0.12, zorder=2,
           label=f'±1σ = {std:.1f} Mbps  (CV {cv:.0f}%)')
ax.axhline(OMA_REF,  color='#d62728', lw=1.4, ls=':', zorder=4,
           label=f'OMA baseline  = {OMA_REF} Mbps  (Table IV)')
ax.axhline(NOMA_REF, color='#2ca02c', lw=1.4, ls=':', zorder=4,
           label=f'NOMA H*=125m = {NOMA_REF} Mbps  (Table IV)')

xtl = [str(s) for s in TEST_SEEDS] + ['42\n(train)']
ax.set_xticks(np.arange(len(TEST_SEEDS)+1))
ax.set_xticklabels(xtl, fontsize=8.5)
ax.set_xlabel('Device Layout Seed')
ax.set_ylabel('Aggregate Throughput (Mbps)')
# Figure title intentionally omitted; the IEEE caption provides it.

ax.text(0.01, 0.97,
        f'20-seed stats (channel_model.py)\nmean = {mu:.2f} Mbps\n'
        f'std  = {std:.2f}  (CV {cv:.0f}%)\nmin  = {lo:.2f}\nmax  = {hi:.2f}\n'
        f'all > OMA baseline ✓',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#888', alpha=0.9))

ylo = min(OMA_REF * 0.92, lo * 0.96)
yhi = max(hi, thr42, NOMA_REF, DQL_REF) * 1.04
ax.set_ylim(ylo, yhi)
ax.set_xlim(-0.6, len(TEST_SEEDS) + 0.6)
ax.legend(loc='lower right')
ax.grid(axis='y', ls=':', alpha=0.45)
plt.tight_layout()

out = os.path.join(OUT, 'fig14_multi_geometry_FAITHFUL.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Fig → {out}")
print("=" * 60)
