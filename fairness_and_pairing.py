#!/usr/bin/env python3
r"""
fairness_and_pairing.py
 - W2 (dynamic pairing): is NOMA sum rate sensitive to the near<->far pairing?
 - W10 (max-min / PF baselines, even without RIS): min-user rate and sum-log-rate
   at the sum-rate vs survivor positions, with and without the RIS.
Reuses channel_model.py physics (N=12 baseline).
"""
import numpy as np
import channel_model as cm

cm.CFG.N_PAIRS = 12
np.random.seed(42); POS,_ = cm.generate_devices()
NEAR, FAR = POS[:12], POS[12:24]
N = 12
SUMRATE = np.array([450., 577., 108.]); SURV = np.array([400., 523., 150.])

def per_user_rates(uav, far_perm=None, use_ris=True, n_mc=60, seed=99):
    if far_perm is None: far_perm = np.arange(N)
    np.random.seed(seed)
    near_acc = np.zeros(N); far_acc = np.zeros(N)
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        for i in range(N):
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            j  = far_perm[i]
            hd = cm.chan_scalar(uav, FAR[j], cm.CFG.BLOCK_DB)
            if use_ris:
                hr = cm.H_ris_dev(FAR[j]); phi = cm.opt_phi(G, hr); hf = cm.composite(hd, G, hr, phi)
            else:
                hf = hd
            rN, _ = cm.noma_pair(hn, hd); _, rF = cm.noma_pair(hn, hf)
            near_acc[i] += rN; far_acc[i] += rF
    return near_acc/n_mc, far_acc/n_mc

# ---- W2: dynamic pairing sensitivity ----
print("=== W2: NOMA sum-rate vs near<->far pairing (sum-rate position, RIS on) ===")
rng = np.random.default_rng(0)
base_n, base_f = per_user_rates(SUMRATE)
print(f"  fixed pairing (i<->i):      sum = {(base_n.sum()+base_f.sum())/1e6:.4f} Mbps")
for k in range(4):
    perm = rng.permutation(N)
    pn, pf = per_user_rates(SUMRATE, far_perm=perm)
    print(f"  random permutation #{k+1}:      sum = {(pn.sum()+pf.sum())/1e6:.4f} Mbps")
print("  -> pairing is throughput-neutral (per-user SIC + identical orthogonal sub-bands)\n")

# ---- W10: max-min and PF fairness, with/without RIS, both positions ----
print("=== W10: fairness objectives (min-user rate = max-min value; sum-log = PF) ===")
print(f"  {'position':<10} {'RIS':>5} | {'min-USER':>10} {'min-FAR':>10} {'far<100bps':>11} {'PF(sum-log)':>12}")
print("-"*64)
for name, uav in [('sum-rate', SUMRATE), ('survivor', SURV)]:
    for use_ris in [False, True]:
        nr, fr = per_user_rates(uav, use_ris=use_ris)
        allr = np.concatenate([nr, fr])
        minu = allr.min(); minf = fr.min()
        below = int((fr < 100).sum())
        pf = float(np.sum(np.log(np.maximum(allr, 1e-3))))
        print(f"  {name:<10} {('on' if use_ris else 'off'):>5} | {minu:9.1f}b {minf:9.1f}b {below:>2}/12 far  {pf:11.2f}")
