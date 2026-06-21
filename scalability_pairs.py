#!/usr/bin/env python3
r"""
scalability_pairs.py  — Q1 / Weakness: scalability beyond 12 NOMA pairs.
Partitions the SAME 20 MHz / 2 W budget into N orthogonal pairs (N = 12..30),
recomputing the per-sub-band noise (20e6/N), and reports how aggregate throughput,
per-pair rate, and far-user SLA reliability scale. Reuses channel_model.py physics
and the exact eval metric of survivor_aware_dql.eval_position.
"""
import numpy as np
import channel_model as cm

BW_TOTAL, P_MAX, NOISE_PSD, NF = 20e6, 2.0, -174, 7
SUMRATE = np.array([450., 577., 108.])      # conventional sum-rate-optimal UAV position
SURV    = np.array([400., 523., 150.])      # survivor-aware position

def set_pairs(N):
    c = cm.CFG
    c.K_NEAR = c.K_FAR = N; c.K_SC1 = 2*N
    c.K_SC2 = c.K_SC3 = 0;  c.K_TOTAL = 2*N
    c.N_PAIRS = N
    c.BW_PAIR = BW_TOTAL / N
    c.P_PAIR  = P_MAX / N
    c.NOISE_W = 10**((NOISE_PSD + NF - 30 + 10*np.log10(BW_TOTAL/N)) / 10)

def eval_at(uav, NEAR, FAR, N, n_mc=60, seed=99):
    np.random.seed(seed)
    near_sum = 0.0; far = []
    for _ in range(n_mc):
        G = cm.G_vector(uav); df = np.zeros(N)
        for i in range(N):
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            hd = cm.chan_scalar(uav, FAR[i],  cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(FAR[i]); phi = cm.opt_phi(G, hr)
            he = cm.composite(hd, G, hr, phi)
            rN, _ = cm.noma_pair(hn, hd); _, rF = cm.noma_pair(hn, he)
            near_sum += rN; df[i] = rF
        far.append(df)
    fs = np.array(far)
    return near_sum/n_mc/1e6, float(fs.mean()), float((fs >= 100.0).mean())

# validation at N=12, sum-rate position
set_pairs(12); np.random.seed(42); POS,_ = cm.generate_devices()
agg,_,rel = eval_at(SUMRATE, POS[:12], POS[12:24], 12)
print(f"[VALIDATE N=12 @ sum-rate pos] reliability={rel*100:.1f}% (paper 64.6%), aggregate={agg:.1f} Mbps\n")

print(f"  {'N':>3} {'BW/pair':>8} {'P/pair':>7} | {'agg(SR)':>8} {'/pair':>6} {'far-rel(SR)':>11} | {'agg(SV)':>8} {'far-rel(SV)':>11}")
print("-"*82)
for N in [12, 18, 24, 30]:
    set_pairs(N); np.random.seed(42); POS,_ = cm.generate_devices()
    NEAR, FAR = POS[:N], POS[N:2*N]
    aSR, _, rSR = eval_at(SUMRATE, NEAR, FAR, N)
    aSV, _, rSV = eval_at(SURV,    NEAR, FAR, N)
    print(f"  {N:>3} {BW_TOTAL/N/1e6:6.2f}MHz {P_MAX/N:6.3f}W | {aSR:7.1f}M {aSR/N:5.1f}M {rSR*100:9.1f}%  | {aSV:7.1f}M {rSV*100:9.1f}%")
