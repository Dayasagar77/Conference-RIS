#!/usr/bin/env python3
"""
fig16_telemetry_outage_FAITHFUL.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Far-user OUTAGE vs low-rate (telemetry) threshold, computed with the
paper's OWN channel_model.py functions (Option A). Honest numbers:

  100 bps :  outage 98.1%  ->  35.1%   (reliability 1.9% -> 64.9%)
  500 bps :  outage 100%   ->  73.0%   (reliability 0%   -> 27.0%)

These are CAPACITY-threshold outages: far-user SINR ~ -40 dB, so the
rates are Shannon capacities, not demonstrated decodable links. The plot
shows the real residual outage; it does NOT claim a 100%->0% rescue.

Run on Windows (channel_model.py in same folder):
  python fig16_telemetry_outage_FAITHFUL.py
"""
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import channel_model as cm

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'results', 'FIGS_13_16'))
os.makedirs(OUT, exist_ok=True)

UAV = np.array([437., 584., 108.]); N_MC = 1000
np.random.seed(42); pos, _ = cm.generate_devices()
near, far = pos[:12], pos[12:24]

np.random.seed(99)
far_nr, far_r = [], []
for _ in range(N_MC):
    G = cm.G_vector(UAV)
    for i in range(12):
        hn = cm.chan_scalar(UAV, near[i], 0.0)
        hd = cm.chan_scalar(UAV, far[i], cm.CFG.BLOCK_DB)
        _, rnr = cm.noma_pair(hn, hd); far_nr.append(rnr)          # bps
        hr = cm.H_ris_dev(far[i]); phi = cm.opt_phi(G, hr); he = cm.composite(hd, G, hr, phi)
        _, rr = cm.noma_pair(hn, he); far_r.append(rr)             # bps
far_nr = np.array(far_nr); far_r = np.array(far_r)

ths = np.logspace(np.log10(10), np.log10(2000), 200)   # 10 .. 2000 bps
out_nr = np.array([(far_nr < t).mean() for t in ths])
out_r  = np.array([(far_r  < t).mean() for t in ths])

def o(arr, t): return float((arr < t).mean())
print(f"100 bps: no-RIS {o(far_nr,100):.3f} -> RIS {o(far_r,100):.3f}")
print(f"500 bps: no-RIS {o(far_nr,500):.3f} -> RIS {o(far_r,500):.3f}")

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.semilogx(ths, out_nr, color='#d62728', lw=2.4, label='Without RIS (60 dB blocked)')
ax.semilogx(ths, out_r,  color='#2ca02c', lw=2.4, ls='-.', label=f'With RIS (M={cm.CFG.M}, {cm.CFG.B_PHASE}-bit)')

for bps, txt in [(100, '100 bps\nbeacon'), (500, '500 bps')]:
    ax.axvline(bps, color='#888', lw=1.0, ls=':')
    onr, orr = o(far_nr, bps), o(far_r, bps)
    ax.scatter([bps, bps], [onr, orr], s=45, c=['#d62728', '#2ca02c'], zorder=5)
    ax.annotate(f'{orr*100:.0f}% outage\n({(1-orr)*100:.0f}% reliable)',
                xy=(bps, orr), xytext=(bps*1.15, orr-0.12), fontsize=8.5, color='#2ca02c',
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1))
    ax.text(bps, 1.03, txt, ha='center', fontsize=8.5, color='#444')

# shade the honest residual-outage gap at 100 bps
ax.annotate('', xy=(100, o(far_r,100)), xytext=(100, o(far_nr,100)),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=1.4))
ax.text(72, (o(far_nr,100)+o(far_r,100))/2, 'RIS\nbenefit', ha='right',
        fontsize=8.5, color='purple', fontweight='bold')

ax.set_xlabel('Rate Threshold $R_{min}$ (bits/s, log scale)')
ax.set_ylabel('Far-User Outage Probability  $P(R<R_{min})$')
ax.set_title('Fig 16 \u2014 Far-user low-rate (telemetry) outage with vs. without RIS\n'
             'RIS improves reliability but 35% remain in outage at the 100 bps SLA',
             fontweight='bold', fontsize=11)
ax.set_ylim(-0.04, 1.10); ax.set_xlim(10, 2000)
ax.grid(True, which='both', ls=':', alpha=0.4)
ax.legend(loc='center left', fontsize=10)
fig.text(0.5, -0.02,
         'Capacity-threshold outage (far-user SINR \u2248 \u221240 dB; rates are Shannon capacities, not demonstrated '
         'decodable links). channel_model.py physics, N_MC=1000.',
         ha='center', fontsize=8, style='italic', color='#555')
plt.tight_layout()
p = os.path.join(OUT, 'fig16_telemetry_outage_FAITHFUL.png')
fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
print(f"saved -> {p}")
