#!/usr/bin/env python3
r"""
optimized_oma.py — Q5 (reviewer): does an OPTIMIZED OMA close the +54.8% NOMA gain?
Replicates channel_model.run_altitude_experiment's per-pair physics EXACTLY
(same oma_pair / noma_pair, same blocked-direct far, centroid @ H=125, seed-42 layout)
to validate the 120.93 Mbps equal-OMA baseline, then maximises each pair's OMA sum
rate over a joint bandwidth(time)+power split.
"""
import numpy as np
import channel_model as cm

np.random.seed(42)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]; FAR = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
K = cm.CFG.N_PAIRS
BW, P, N0 = cm.CFG.BW_PAIR, cm.CFG.P_PAIR, cm.CFG.NOISE_W
UAV = np.array([cm.CFG.UAV_CX, cm.CFG.UAV_CY, 125.0])
N_MC = 100

# grid for the per-pair OMA split optimisation (beta=BW frac to near, t=power frac to near)
betas = np.linspace(0.02, 0.98, 49)
ts    = np.linspace(0.02, 0.98, 49)
BB, TT = np.meshgrid(betas, ts)

def oma_opt_pair(hN2, hF2):
    """max over (beta,t) of beta*BW*log2(1+ tP hN2/(beta N)) + (1-beta)*BW*log2(1+(1-t)P hF2/((1-beta)N))."""
    rN = BB*BW*np.log2(1 + (TT*P)*hN2/(BB*N0))
    rF = (1-BB)*BW*np.log2(1 + ((1-TT)*P)*hF2/((1-BB)*N0))
    tot = rN+rF
    k = np.unravel_index(np.argmax(tot), tot.shape)
    return rN[k], rF[k]            # near, far rates at the optimum

np.random.seed(99)
eq_tot=[]; eq_far=[]; opt_tot=[]; opt_far=[]; noma_tot=[]; noma_far=[]
for _ in range(N_MC):
    G = cm.G_vector(UAV)
    eq=eqf=opt=optf=nm=nmf=0.0
    for i in range(K):
        hN = cm.chan_scalar(UAV, NEAR[i], 0.0)
        hFd= cm.chan_scalar(UAV, FAR[i], cm.CFG.BLOCK_DB)   # blocked direct (no RIS, matches OMA/NOMA baseline)
        rNo, rFo = cm.oma_pair(hN, hFd); eq += rNo+rFo; eqf += rFo          # equal OMA
        rNopt, rFopt = oma_opt_pair(abs(hN)**2, abs(hFd)**2); opt += rNopt+rFopt; optf += rFopt  # optimised OMA
        rNn, rFn = cm.noma_pair(hN, hFd); nm += rNn+rFn; nmf += rFn         # NOMA reference
    eq_tot.append(eq/1e6); eq_far.append(eqf)
    opt_tot.append(opt/1e6); opt_far.append(optf)
    noma_tot.append(nm/1e6); noma_far.append(nmf)

eqM, optM, nmM = np.mean(eq_tot), np.mean(opt_tot), np.mean(noma_tot)
print(f"[VALIDATE] equal-split OMA aggregate @H=125 = {eqM:.2f} Mbps   (paper 120.93)")
print("="*72)
print(f"  {'scheme':<32}{'aggregate':>12}   {'far-user total':>16}")
print("-"*72)
print(f"  {'Equal-split OMA (both served)':<32}{eqM:9.2f} Mb {np.mean(eq_far):13.1f} bps")
print(f"  {'Sum-rate-OPTIMISED OMA':<32}{optM:9.2f} Mb {np.mean(opt_far):13.1f} bps")
print(f"  {'NOMA (alpha=0.85, both served)':<32}{nmM:9.2f} Mb {np.mean(noma_far):13.1f} bps")
print("="*72)
print(f"  NOMA gain over EQUAL OMA   : {100*(nmM-eqM)/eqM:+.1f}%   (= the paper's +54.8%)")
print(f"  NOMA vs sum-rate-OPT OMA   : {100*(nmM-optM)/optM:+.1f}%   (optimised OMA wins on AGGREGATE)")
print(f"  ...but far-user service:  optimised-OMA {np.mean(opt_far):.0f} bps  vs  NOMA {np.mean(noma_far):.0f} bps "
      f"(NOMA serves survivors {np.mean(noma_far)/max(np.mean(opt_far),1e-9):.0f}x better)")
