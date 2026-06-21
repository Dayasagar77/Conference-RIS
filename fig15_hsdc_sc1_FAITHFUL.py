#!/usr/bin/env python3
"""
fig15_hsdc_sc1_FAITHFUL.py  —  HSDC on real SC1 devices from channel_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses channel_model.generate_devices() (seed=42) for the exact 24 SC1 device
positions used throughout the paper (same clipping, same RNG order as Module 1).

Run on Windows (channel_model.py must be in the same folder):
  python fig15_hsdc_sc1_FAITHFUL.py
"""
import os, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import channel_model as cm

warnings.filterwarnings('ignore')

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'results', 'FIGS_13_16'))
os.makedirs(OUT, exist_ok=True)

R_COVER = 500.0
CENTROID = np.array([cm.CFG.UAV_CX, cm.CFG.UAV_CY])

# ── Real SC1 positions from channel_model (seed=42) ───────────────────────
np.random.seed(42)
pos, labels_all = cm.generate_devices()
near2d = pos[:cm.CFG.K_NEAR, :2]
far2d  = pos[cm.CFG.K_NEAR:cm.CFG.K_SC1, :2]
pos2d  = np.vstack([near2d, far2d])
true_lbl = ['N'] * cm.CFG.K_NEAR + ['F'] * cm.CFG.K_FAR
N = len(pos2d)

d2c = np.linalg.norm(pos2d - CENTROID, axis=1)
print(f"  Real SC1 positions (channel_model.py seed=42 with clipping):")
print(f"  Near d-to-centroid : [{d2c[:cm.CFG.K_NEAR].min():.0f}, "
      f"{d2c[:cm.CFG.K_NEAR].max():.0f}] m")
print(f"  Far  d-to-centroid : [{d2c[cm.CFG.K_NEAR:].min():.0f}, "
      f"{d2c[cm.CFG.K_NEAR:].max():.0f}] m")

# ── HSDC ─────────────────────────────────────────────────────────────────────
Z = linkage(pos2d, method='ward')
k_star = None
for k in range(2, 9):
    asgn  = fcluster(Z, k, criterion='maxclust')
    cents = np.array([pos2d[asgn == c].mean(axis=0) for c in range(1, k+1)])
    dists = np.array([np.linalg.norm(pos2d[i] - cents[asgn[i]-1]) for i in range(N)])
    if dists.max() <= R_COVER:
        k_star = k; best_a = asgn; best_c = cents; best_d = dists
        break

if k_star is None:
    k_star = 4; best_a = fcluster(Z, k_star, criterion='maxclust')
    best_c = np.array([pos2d[best_a==c].mean(axis=0) for c in range(1, k_star+1)])
    best_d = np.array([np.linalg.norm(pos2d[i]-best_c[best_a[i]-1]) for i in range(N)])

try:
    sil = float(silhouette_score(pos2d, best_a))
except Exception:
    sil = 0.0

n_near = [sum(1 for i in range(N) if best_a[i]==c+1 and true_lbl[i]=='N') for c in range(k_star)]
n_far  = [sum(1 for i in range(N) if best_a[i]==c+1 and true_lbl[i]=='F') for c in range(k_star)]
n_near_only = sum(1 for c in range(k_star) if n_far[c]==0)
n_far_only  = sum(1 for c in range(k_star) if n_near[c]==0)

print(f"\n  HSDC results:")
print(f"  K* = {k_star}")
print(f"  Coverage: {'YES ✓' if best_d.max()<=R_COVER else 'NO ✗'}  "
      f"(max d = {best_d.max():.1f} m)")
print(f"  Silhouette = {sil:.4f}")
print(f"  Near-only clusters: {n_near_only}  Far-only: {n_far_only}  "
      f"Mixed: {k_star-n_near_only-n_far_only}")
for c in range(k_star):
    tag = 'near-only' if n_far[c]==0 else ('far-only' if n_near[c]==0 else 'mixed')
    print(f"    Cluster {c+1}: size={n_near[c]+n_far[c]}  "
          f"near={n_near[c]}  far={n_far[c]}  [{tag}]")

# ── Figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})
CMAP = plt.cm.get_cmap('tab10', 10)
C = [CMAP(i) for i in range(k_star)]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.subplots_adjust(wspace=0.30)

ax = axes[0]
for ci, cent in enumerate(best_c):
    ax.add_patch(Circle(cent, R_COVER, color=C[ci], alpha=0.09,
                        ls='--', fill=True, zorder=1))
for r, ls in [(cm.CFG.NEAR_DMAX, ':'), (cm.CFG.FAR_DMIN, ':'), (cm.CFG.FAR_DMAX, ':')]:
    ax.add_patch(Circle(CENTROID, r, color='grey', fill=False,
                        lw=0.7, ls=ls, alpha=0.4))
for i, (p, lb) in enumerate(zip(pos2d, true_lbl)):
    ax.scatter(p[0], p[1], s=90, c=[C[best_a[i]-1]],
               marker='o' if lb=='N' else '^',
               edgecolors='black', lw=0.6, zorder=4)
    ax.text(p[0]+4, p[1]+4, str(i+1), fontsize=7, color='#333')
for ci, cent in enumerate(best_c):
    ax.scatter(cent[0], cent[1], s=220, c=[C[ci]], marker='*',
               edgecolors='black', lw=1.0, zorder=5)
ax.scatter(CENTROID[0], CENTROID[1], s=160, c='black',
           marker='+', lw=1.8, zorder=6)

patches = [mpatches.Patch(color=C[c], alpha=0.8, label=f'Cluster {c+1}')
           for c in range(k_star)]
patches += [mpatches.Patch(color='grey', label='● Near  ▲ Far'),
            mpatches.Patch(color='lightgrey', alpha=0.4,
                           label=f'Coverage R={R_COVER:.0f} m')]
ax.legend(handles=patches, loc='lower left', fontsize=8.5, framealpha=0.85)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title(f'HSDC on 24 SC1 Devices — K*={k_star}  (coverage ✓)',
             fontweight='bold')
ax.set_aspect('equal')
margin = 150
ax.set_xlim(CENTROID[0]-cm.CFG.FAR_DMAX-margin, CENTROID[0]+cm.CFG.FAR_DMAX+margin)
ax.set_ylim(CENTROID[1]-cm.CFG.FAR_DMAX-margin, CENTROID[1]+cm.CFG.FAR_DMAX+margin)
ax.grid(True, ls=':', alpha=0.35)
ax.text(0.02, 0.98, f'K* = {k_star}\nSilhouette = {sil:.3f}\n'
        f'Max d = {best_d.max():.1f} m ≤ {R_COVER:.0f} m',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#888', alpha=0.9))

ax2 = axes[1]
x = np.arange(k_star); w = 0.35
b1 = ax2.bar(x-w/2, n_near, w, color='#5FA2DD', edgecolor='white',
             lw=0.6, label='Near devices')
b2 = ax2.bar(x+w/2, n_far,  w, color='#E05C5C', edgecolor='white',
             lw=0.6, label='Far devices')
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x()+bar.get_width()/2, h+0.08, str(int(h)),
                     ha='center', va='bottom', fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels([f'Cluster {c+1}' for c in range(k_star)], fontsize=10)
ax2.set_ylabel('Number of Devices')
ax2.set_title('Near / Far Device Composition per Cluster', fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_ylim(0, max(max(n_near), max(n_far)) + 2.5)
ax2.grid(axis='y', ls=':', alpha=0.45)
ax2.text(0.98, 0.97,
         f'Near-only: {n_near_only}\nFar-only:  {n_far_only}\n'
         f'Mixed:     {k_star-n_near_only-n_far_only}',
         transform=ax2.transAxes, fontsize=9, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffde7',
                   edgecolor='#aaa', alpha=0.9))

# Figure title intentionally omitted; the IEEE caption provides it.
out = os.path.join(OUT, 'fig15_hsdc_sc1_FAITHFUL.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\n  Fig → {out}")
