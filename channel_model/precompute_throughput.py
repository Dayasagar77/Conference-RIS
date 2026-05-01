"""
precompute_throughput.py
Run this on the VM ONCE to generate the throughput lookup table.
Copy the output file throughput_table.json to Windows.
Runtime: ~5-10 minutes on VM.
"""

import numpy as np
import json
import time

# ─── System Parameters (must match Module 1) ────────────────────────────────
N_PAIRS     = 12
M_RIS       = 1024
FREQ_HZ     = 3.5e9
BW_TOTAL_HZ = 20e6
BW_PAIR_HZ  = BW_TOTAL_HZ / N_PAIRS
NOISE_PSD   = -174.0
NF_DB       = 7.0
P_TOTAL_W   = 2.0
P_PAIR_W    = P_TOTAL_W / N_PAIRS
ALPHA_NOMA  = 0.85
RIS_POS     = np.array([300.0, 250.0, 20.0])
BLOCKAGE_DB = 60.0
UAV_CENTROID= np.array([400.0, 600.0])
GLOBAL_SEED = 99
N_MC        = 50

NOISE_VAR = 10 ** ((NOISE_PSD + NF_DB + 10 * np.log10(BW_PAIR_HZ) - 30) / 10)

# ─── HARDCODED device positions (from seed=99 on this VM) ───────────────────
NEAR_POS = np.array([
    [261.264467, 539.868917],
    [242.697227, 528.040739],
    [289.461071, 508.486364],
    [313.307249, 670.733423],
    [229.229376, 764.690707],
    [367.794695, 495.101302],
    [302.239459, 503.992924],
    [219.317780, 587.170523],
    [482.978004, 656.538888],
    [306.849451, 740.414840],
    [410.016110, 715.936144],
    [248.677990, 463.319640],
])
FAR_POS = np.array([
    [1019.587445,  490.606108],
    [ -118.407425,  415.157443],
    [  420.050692, 1375.395676],
    [   79.853645, 1040.556627],
    [ 1097.828079,  781.825310],
    [  868.347883,  211.844027],
    [ -131.903456,  217.965542],
    [  869.929035,  817.419009],
    [  306.489390, 1080.582253],
    [ 1105.232695,  444.154215],
    [ -219.987471,  295.469860],
    [  864.578688,  309.679558],
])

# ─── Grid definition ─────────────────────────────────────────────────────────
# Spacing matches DQL step sizes (STEP_XY=30m, STEP_H=25m)
# Slight finer grid for better interpolation accuracy
X_VALS = np.linspace(200, 600, 21)   # 20m spacing
Y_VALS = np.linspace(400, 800, 21)   # 20m spacing
H_VALS = np.array([50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0])

def db2lin(db):
    return 10 ** (db / 10)

def los_probability(d_3d):
    if d_3d < 18.0:
        return 1.0
    return 18.0 / d_3d + np.exp(-d_3d / 63.0) * (1 - 18.0 / d_3d)

def path_loss_uav_device(uav_pos, dev_pos):
    dx   = uav_pos[0] - dev_pos[0]
    dy   = uav_pos[1] - dev_pos[1]
    d_h  = np.sqrt(dx**2 + dy**2)
    d_3d = np.sqrt(d_h**2 + uav_pos[2]**2)
    lam  = 3e8 / FREQ_HZ
    pl_fs = 20 * np.log10(4 * np.pi * max(d_3d, 0.1) / lam)
    p_los = los_probability(d_3d)
    return pl_fs + (1 - p_los) * 20.0 + p_los * 1.0

def path_loss_uav_ris(uav_pos):
    d   = np.linalg.norm(uav_pos - RIS_POS)
    lam = 3e8 / FREQ_HZ
    return 20 * np.log10(4 * np.pi * max(d, 0.1) / lam)

def path_loss_ris_device(dev_pos):
    d_h  = np.linalg.norm(dev_pos[:2] - RIS_POS[:2])
    d_3d = np.sqrt(d_h**2 + RIS_POS[2]**2)
    lam  = 3e8 / FREQ_HZ
    return 20 * np.log10(4 * np.pi * max(d_3d, 0.1) / lam)

def compute_pair_throughput_ris(uav_pos, near_pos, far_pos, rng):
    pl_near = path_loss_uav_device(uav_pos, near_pos)
    pl_far  = path_loss_uav_device(uav_pos, far_pos) + BLOCKAGE_DB

    # Explicit complex construction — platform independent
    _r = rng.standard_normal(2)
    h_dn = (_r[0] + 1j * _r[1]) / np.sqrt(2)
    _r = rng.standard_normal(2)
    h_df = (_r[0] + 1j * _r[1]) / np.sqrt(2)

    pl_ur = path_loss_uav_ris(uav_pos)
    pl_rn = path_loss_ris_device(near_pos)
    pl_rf = path_loss_ris_device(far_pos)

    K     = 10.0
    G_los = np.ones(M_RIS, dtype=complex)   # broadside: sin(0)=0 → exp(0)=1
    _r    = rng.standard_normal((2, M_RIS))
    G_sc  = (_r[0] + 1j * _r[1]) / np.sqrt(2)
    G     = (np.sqrt(K/(K+1)) * G_los + np.sqrt(1/(K+1)) * G_sc) / np.sqrt(db2lin(pl_ur))

    _r    = rng.standard_normal((2, M_RIS))
    h_rn  = (_r[0] + 1j * _r[1]) / (np.sqrt(2) * np.sqrt(db2lin(pl_rn)))
    _r    = rng.standard_normal((2, M_RIS))
    h_rf  = (_r[0] + 1j * _r[1]) / (np.sqrt(2) * np.sqrt(db2lin(pl_rf)))

    theta = np.angle(h_rf) - np.angle(G)
    Phi   = np.exp(1j * theta)

    h_eff_near = abs(h_dn / np.sqrt(db2lin(pl_near)) + (h_rn * Phi) @ G)**2
    h_eff_far  = abs(h_df / np.sqrt(db2lin(pl_far))  + (h_rf * Phi) @ G)**2

    sinr_far  = (ALPHA_NOMA * P_PAIR_W * h_eff_far) / \
                ((1 - ALPHA_NOMA) * P_PAIR_W * h_eff_far + NOISE_VAR)
    sinr_near = (1 - ALPHA_NOMA) * P_PAIR_W * h_eff_near / NOISE_VAR

    return BW_PAIR_HZ * (np.log2(1 + sinr_far) + np.log2(1 + sinr_near)) / 1e6

def evaluate_throughput(uav_pos, seed=GLOBAL_SEED, n_mc=N_MC):
    rng   = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_mc):
        for k in range(N_PAIRS):
            total += compute_pair_throughput_ris(uav_pos, NEAR_POS[k], FAR_POS[k], rng)
    return total / n_mc

# ─── Sanity check vs Module 1 ────────────────────────────────────────────────
print("Sanity check vs Module 1...")
m1 = {50: 194.707, 125: 233.666, 200: 222.262}
for H, exp in m1.items():
    pos = np.array([UAV_CENTROID[0], UAV_CENTROID[1], float(H)])
    got = evaluate_throughput(pos, n_mc=30)
    print(f"  H={H:3d}m: {got:.2f} Mbps (Module1={exp:.3f})")

# ─── Precompute grid ─────────────────────────────────────────────────────────
NX, NY, NH = len(X_VALS), len(Y_VALS), len(H_VALS)
total_pts   = NX * NY * NH
print(f"\nPrecomputing {NX}x{NY}x{NH} = {total_pts} grid points (n_mc={N_MC})...")
print(f"Estimated time: {total_pts * N_MC * N_PAIRS * 0.0004:.0f}s")

table = np.zeros((NX, NY, NH))
t0    = time.time()
done  = 0

for i, x in enumerate(X_VALS):
    for j, y in enumerate(Y_VALS):
        for k, h in enumerate(H_VALS):
            pos           = np.array([x, y, h])
            table[i,j,k]  = evaluate_throughput(pos, seed=GLOBAL_SEED, n_mc=N_MC)
            done += 1
            if done % 50 == 0 or done == total_pts:
                el  = time.time() - t0
                eta = el / done * (total_pts - done)
                print(f"  {done}/{total_pts} | ETA={eta:.0f}s | "
                      f"last={table[i,j,k]:.1f} Mbps")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# ─── Save to JSON ────────────────────────────────────────────────────────────
out = {
    "x_vals"  : X_VALS.tolist(),
    "y_vals"  : Y_VALS.tolist(),
    "h_vals"  : H_VALS.tolist(),
    "table"   : table.tolist(),          # shape [NX, NY, NH]
    "n_mc"    : N_MC,
    "seed"    : GLOBAL_SEED,
    "near_pos": NEAR_POS.tolist(),
    "far_pos" : FAR_POS.tolist(),
    "module1_ref": {
        "50": 194.707, "75": 222.523, "100": 233.290,
        "125": 233.666, "150": 230.646, "175": 226.695, "200": 222.262
    }
}
path = "results/throughput_table.json"
import os; os.makedirs("results", exist_ok=True)
with open(path, 'w') as f:
    json.dump(out, f)
print(f"Saved: {path}")
print("\nNow copy results/throughput_table.json to Windows")
print("D:\\DAYA PHD\\Conference paper RIS\\throughput_table.json")
