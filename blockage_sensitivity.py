#!/usr/bin/env python3
r"""
blockage_sensitivity.py  — Q2 (reviewer): far-user telemetry reliability vs blockage depth.
FAITHFUL replica of survivor_aware_dql.py::eval_position (same channel draws, same
noma_pair physics, same SLA_BPS=100, same DEVICE_SEED=42 / EVAL_SEED=99 / N_MC=60).
ONLY addition: sweep cm.CFG.BLOCK_DB over 40..80 dB and also report the no-RIS
(blocked direct path) reliability for contrast.
"""
import numpy as np, json, os
import channel_model as cm

DEVICE_SEED, EVAL_SEED, SLA_BPS, N_MC = 42, 99, 100.0, 60
np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]; FAR = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
K = cm.CFG.N_PAIRS

def eval_pos(uav, n_mc=N_MC, seed=EVAL_SEED):
    """Returns (far_mean_bps_RIS, sla_rel_RIS, far_mean_bps_noRIS, sla_rel_noRIS)."""
    np.random.seed(seed)
    f_ris, f_dir = [], []
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        dr, dd = np.zeros(K), np.zeros(K)
        for i in range(K):
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            hd = cm.chan_scalar(uav, FAR[i], cm.CFG.BLOCK_DB)      # blocked direct
            hr = cm.H_ris_dev(FAR[i]); phi = cm.opt_phi(G, hr)
            he = cm.composite(hd, G, hr, phi)                      # RIS bypass
            _, rF_ris = cm.noma_pair(hn, he)
            _, rF_dir = cm.noma_pair(hn, hd)
            dr[i], dd[i] = rF_ris, rF_dir
        f_ris.append(dr); f_dir.append(dd)
    f_ris, f_dir = np.array(f_ris), np.array(f_dir)
    return (f_ris.mean(), float((f_ris>=SLA_BPS).mean()),
            f_dir.mean(), float((f_dir>=SLA_BPS).mean()))

SUMRATE_OPT  = np.array([450.0, 577.0, 108.0])   # paper: 64.6% reliability @ 60 dB
SURVIVOR_OPT = np.array([400.0, 523.0, 150.0])   # paper: 100% reliability @ 60 dB

# ---- VALIDATION at 60 dB ----
cm.CFG.BLOCK_DB = 60.0
_, rel_sr, _, _ = eval_pos(SUMRATE_OPT)
_, rel_sv, _, _ = eval_pos(SURVIVOR_OPT)
print(f"[VALIDATE @60dB] sum-rate opt reliability = {rel_sr*100:.1f}%  (paper 64.6%)")
print(f"[VALIDATE @60dB] survivor opt reliability = {rel_sv*100:.1f}%  (paper 100%)")
print("-"*70)

# ---- SWEEP 40..80 dB ----
blocks = list(range(40, 81, 5))
res = {"block_dB": blocks, "sumrate_opt": [], "survivor_opt": [], "no_ris_sumrate": []}
for b in blocks:
    cm.CFG.BLOCK_DB = float(b)
    fmr, rel_sr, fmd, rel_dir = eval_pos(SUMRATE_OPT)
    _,   rel_sv, _,   _       = eval_pos(SURVIVOR_OPT)
    res["sumrate_opt"].append(rel_sr*100)
    res["survivor_opt"].append(rel_sv*100)
    res["no_ris_sumrate"].append(rel_dir*100)
    print(f"  block={b:>2}dB | RIS@sumrate-opt={rel_sr*100:5.1f}% | "
          f"RIS@survivor-opt={rel_sv*100:5.1f}% | no-RIS={rel_dir*100:4.1f}% "
          f"| far mean(RIS)={fmr:7.1f} bps")
json.dump(res, open("blockage_sensitivity.json","w"), indent=2)
print("[saved] blockage_sensitivity.json")
