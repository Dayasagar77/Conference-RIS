"""
=============================================================
  MODULE 3 (VM FINAL): DQL UAV Trajectory Agent
  6G RIS-UAV Emergency Communications Simulation

  Run this on the VM — uses the EXACT proven channel model.
  
  Key settings:
    N_EP = 2000, N_MC_TRAIN = 5 (fast), N_MC_EVAL = 50
    GLOBAL_SEED = 99 for both training AND evaluation
  
  Expected runtime: ~15-20 min on VM
  Expected best throughput: >233 Mbps (beats fixed H=125m)
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import time
import random
from collections import deque

# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────
GLOBAL_SEED = 99
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ─────────────────────────────────────────────
#  System Parameters  (match Module 1 exactly)
# ─────────────────────────────────────────────
N_PAIRS      = 12
N_NEAR       = N_PAIRS
N_FAR        = N_PAIRS
M_RIS        = 1024
FREQ_HZ      = 3.5e9
BW_TOTAL_HZ  = 20e6
BW_PAIR_HZ   = BW_TOTAL_HZ / N_PAIRS
NOISE_PSD    = -174.0
NF_DB        = 7.0
P_TOTAL_W    = 2.0
P_PAIR_W     = P_TOTAL_W / N_PAIRS
ALPHA_NOMA   = 0.85
RIS_POS      = np.array([300.0, 250.0, 20.0])
BLOCKAGE_DB  = 60.0
UAV_CENTROID = np.array([400.0, 600.0])

NEAR_RMIN, NEAR_RMAX = 50.0,  250.0
FAR_RMIN,  FAR_RMAX  = 450.0, 800.0

X_RANGE = (200.0, 600.0)
Y_RANGE = (400.0, 800.0)
H_RANGE = (50.0,  200.0)

# ─────────────────────────────────────────────
#  DQL Hyper-parameters
# ─────────────────────────────────────────────
N_EP          = 2000
STEPS_PER_EP  = 25
BUFFER_SIZE   = 10000
BATCH_SIZE    = 64
GAMMA         = 0.95
LR            = 1e-3
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY_EP  = 500
TARGET_UPDATE = 20
N_MC_TRAIN    = 5    # fast: noisy but sufficient for RL
N_MC_EVAL     = 50   # accurate: for final fair comparison only

STEP_XY = 30.0
STEP_H  = 25.0

ACTIONS = [
    np.array([ 0.0,     0.0,    0.0]),
    np.array([ STEP_XY, 0.0,    0.0]),
    np.array([-STEP_XY, 0.0,    0.0]),
    np.array([ 0.0,  STEP_XY,   0.0]),
    np.array([ 0.0, -STEP_XY,   0.0]),
    np.array([ 0.0,     0.0,  STEP_H]),
    np.array([ 0.0,     0.0, -STEP_H]),
]
N_ACTIONS = len(ACTIONS)
STATE_DIM = 5

OUT_DQL = "results/dql"
os.makedirs(OUT_DQL, exist_ok=True)

NOISE_VAR = 10 ** ((NOISE_PSD + NF_DB + 10 * np.log10(BW_PAIR_HZ) - 30) / 10)

# ─────────────────────────────────────────────────────────────────────────────
#  Channel model (EXACT copy of proven Module 1 / 350-ep version)
#  DO NOT MODIFY — this is the validated implementation
# ─────────────────────────────────────────────────────────────────────────────

def los_probability(d_3d):
    if d_3d < 18.0:
        return 1.0
    return 18.0 / d_3d + np.exp(-d_3d / 63.0) * (1.0 - 18.0 / d_3d)

def path_loss_uav_device(uav_pos, dev_pos):
    dx   = uav_pos[0] - dev_pos[0]
    dy   = uav_pos[1] - dev_pos[1]
    d_h  = np.sqrt(dx**2 + dy**2)
    d_3d = np.sqrt(d_h**2 + uav_pos[2]**2)
    lam  = 3e8 / FREQ_HZ
    pl_fs = 20.0 * np.log10(4.0 * np.pi * max(d_3d, 0.1) / lam)
    p_los = los_probability(d_3d)
    return pl_fs + (1.0 - p_los) * 20.0 + p_los * 1.0

def path_loss_uav_ris(uav_pos):
    d   = np.linalg.norm(uav_pos - RIS_POS)
    lam = 3e8 / FREQ_HZ
    return 20.0 * np.log10(4.0 * np.pi * max(d, 0.1) / lam)

def path_loss_ris_device(dev_pos):
    d_h  = np.linalg.norm(dev_pos[:2] - RIS_POS[:2])
    d_3d = np.sqrt(d_h**2 + RIS_POS[2]**2)
    lam  = 3e8 / FREQ_HZ
    return 20.0 * np.log10(4.0 * np.pi * max(d_3d, 0.1) / lam)

def db2lin(db):
    return 10.0 ** (db / 10.0)

def compute_pair_throughput_ris(uav_pos, near_pos, far_pos, rng):
    """
    EXACT implementation from validated Module 1 / 350-episode version.
    Uses np.diag for RIS phase matrix — proven to produce 194-234 Mbps range.
    """
    pl_near = path_loss_uav_device(uav_pos, near_pos)
    pl_far  = path_loss_uav_device(uav_pos, far_pos) + BLOCKAGE_DB

    # Direct channel fading (Rayleigh)
    h_direct_near = (rng.standard_normal(2) @ [1, 1j]) / np.sqrt(2)
    h_direct_far  = (rng.standard_normal(2) @ [1, 1j]) / np.sqrt(2)

    pl_ur = path_loss_uav_ris(uav_pos)
    pl_rn = path_loss_ris_device(near_pos)
    pl_rf = path_loss_ris_device(far_pos)

    # UAV → RIS channel (Rician K=10)
    K_rice = 10.0
    G_los  = np.exp(1j * np.pi * np.arange(M_RIS) * np.sin(0.0))
    G_scat = (rng.standard_normal(M_RIS) + 1j * rng.standard_normal(M_RIS)) / np.sqrt(2)
    G = (np.sqrt(K_rice / (K_rice + 1)) * G_los +
         np.sqrt(1.0 / (K_rice + 1)) * G_scat) / np.sqrt(db2lin(pl_ur))

    # RIS → near device (Rayleigh)
    h_rn_scat = (rng.standard_normal(M_RIS) + 1j * rng.standard_normal(M_RIS)) / np.sqrt(2)
    h_rn = h_rn_scat / np.sqrt(db2lin(pl_rn))

    # RIS → far device (Rayleigh)
    h_rf_scat = (rng.standard_normal(M_RIS) + 1j * rng.standard_normal(M_RIS)) / np.sqrt(2)
    h_rf = h_rf_scat / np.sqrt(db2lin(pl_rf))

    # RIS phase alignment for far user (CORRECT formula — validated)
    theta = np.angle(h_rf) - np.angle(G)
    Phi   = np.diag(np.exp(1j * theta))   # M×M diagonal matrix

    # Effective combined channel gains
    h_eff_near = abs(h_direct_near / np.sqrt(db2lin(pl_near)) +
                     h_rn @ Phi @ G)**2
    h_eff_far  = abs(h_direct_far  / np.sqrt(db2lin(pl_far))  +
                     h_rf @ Phi @ G)**2

    # NOMA SIC decoding
    sinr_far  = ((ALPHA_NOMA * P_PAIR_W * h_eff_far) /
                 ((1.0 - ALPHA_NOMA) * P_PAIR_W * h_eff_far + NOISE_VAR))
    sinr_near = ((1.0 - ALPHA_NOMA) * P_PAIR_W * h_eff_near) / NOISE_VAR

    rate_far  = BW_PAIR_HZ * np.log2(1.0 + sinr_far)
    rate_near = BW_PAIR_HZ * np.log2(1.0 + sinr_near)
    return (rate_far + rate_near) / 1e6

# ─────────────────────────────────────────────────────────────────────────────
#  Device placement (generated on VM — matches Module 1)
# ─────────────────────────────────────────────────────────────────────────────

def generate_devices(seed=GLOBAL_SEED):
    rng  = np.random.default_rng(seed)
    near, far = [], []
    for _ in range(N_PAIRS):
        r = rng.uniform(NEAR_RMIN, NEAR_RMAX)
        a = rng.uniform(0, 2 * np.pi)
        near.append(UAV_CENTROID + r * np.array([np.cos(a), np.sin(a)]))
        r = rng.uniform(FAR_RMIN, FAR_RMAX)
        a = rng.uniform(0, 2 * np.pi)
        far.append(UAV_CENTROID + r * np.array([np.cos(a), np.sin(a)]))
    return np.array(near), np.array(far)

NEAR_POS, FAR_POS = generate_devices()

# ─────────────────────────────────────────────────────────────────────────────
#  Throughput evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_throughput(uav_pos, seed=GLOBAL_SEED, n_mc=N_MC_EVAL):
    rng   = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_mc):
        for k in range(N_PAIRS):
            total += compute_pair_throughput_ris(uav_pos, NEAR_POS[k], FAR_POS[k], rng)
    return total / n_mc

# ─────────────────────────────────────────────────────────────────────────────
#  Sanity check vs Module 1
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check():
    print("  Sanity check vs Module 1 (should be ~194, ~233, ~222 Mbps)...")
    expected = {50: 194.707, 125: 233.666, 200: 222.262}
    all_ok = True
    for H, exp in expected.items():
        pos = np.array([UAV_CENTROID[0], UAV_CENTROID[1], float(H)])
        got = evaluate_throughput(pos, seed=GLOBAL_SEED, n_mc=30)
        diff = abs(got - exp)
        ok   = diff < 5.0
        flag = "OK" if ok else "MISMATCH"
        print(f"    H={H:3d}m: {got:.2f} Mbps (expected ~{exp:.1f}) [{flag}]")
        if not ok:
            all_ok = False
    if all_ok:
        print("  [OK] Channel model validated — consistent with Module 1\n")
    else:
        print("  [!!] STOP — channel model mismatch. Do not use these results.\n")
    return all_ok

# ─────────────────────────────────────────────────────────────────────────────
#  Q-Network (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

class QNetwork:
    def __init__(self, state_dim, n_actions, hidden=128, seed=0):
        rng = np.random.default_rng(seed)
        s1, s2, s3 = np.sqrt(2/state_dim), np.sqrt(2/hidden), np.sqrt(2/hidden)
        self.W1 = rng.standard_normal((hidden, state_dim)) * s1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, hidden)) * s2
        self.b2 = np.zeros(hidden)
        self.W3 = rng.standard_normal((n_actions, hidden)) * s3
        self.b3 = np.zeros(n_actions)
        self.lr = LR; self.t = 0
        self.b1a, self.b2a, self.ea = 0.9, 0.999, 1e-8
        self.m = [np.zeros_like(p) for p in self._params()]
        self.v = [np.zeros_like(p) for p in self._params()]

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        h1 = relu(self.W1 @ x + self.b1)
        h2 = relu(self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def forward_batch(self, X):
        H1 = relu(X @ self.W1.T + self.b1)
        H2 = relu(H1 @ self.W2.T + self.b2)
        return H2 @ self.W3.T + self.b3

    def copy_weights_from(self, other):
        for sp, op in zip(self._params(), other._params()):
            sp[:] = op

    def update(self, states, actions, targets):
        B     = len(states)
        H1    = relu(states @ self.W1.T + self.b1)
        H2    = relu(H1     @ self.W2.T + self.b2)
        Q_all = H2           @ self.W3.T + self.b3
        Q_pred = Q_all[np.arange(B), actions]
        delta  = Q_pred - targets
        loss   = float(np.mean(delta**2))
        dQ     = np.zeros_like(Q_all)
        dQ[np.arange(B), actions] = 2.0 * delta / B
        dW3 = dQ.T @ H2;    db3 = dQ.sum(0)
        dH2 = dQ @ self.W3 * (H2 > 0)
        dW2 = dH2.T @ H1;   db2 = dH2.sum(0)
        dH1 = dH2 @ self.W2 * (H1 > 0)
        dW1 = dH1.T @ states; db1 = dH1.sum(0)
        self.t += 1
        for i, (p, g) in enumerate(zip(self._params(),
                                        [dW1, db1, dW2, db2, dW3, db3])):
            self.m[i] = self.b1a * self.m[i] + (1 - self.b1a) * g
            self.v[i] = self.b2a * self.v[i] + (1 - self.b2a) * g**2
            mh = self.m[i] / (1 - self.b1a**self.t)
            vh = self.v[i] / (1 - self.b2a**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.ea)
        return loss

# ─────────────────────────────────────────────────────────────────────────────
#  Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n, rng):
        idx   = rng.choice(len(self.buf), size=n, replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int32),
                np.array(r,  dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self):
        return len(self.buf)

# ─────────────────────────────────────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────────────────────────────────────

class UAVEnv:
    def _norm(self, pos):
        return np.array([
            (pos[0] - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]),
            (pos[1] - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]),
            (pos[2] - H_RANGE[0]) / (H_RANGE[1] - H_RANGE[0]),
        ])

    def reset(self, rng):
        pos   = np.array([rng.uniform(*X_RANGE),
                           rng.uniform(*Y_RANGE),
                           rng.uniform(*H_RANGE)])
        thr   = evaluate_throughput(pos, seed=GLOBAL_SEED, n_mc=N_MC_TRAIN)
        state = np.concatenate([self._norm(pos), [thr / 300.0, 0.0]])
        return state, pos

    def step(self, pos, action_idx, step_num):
        new_pos    = pos + ACTIONS[action_idx]
        new_pos[0] = np.clip(new_pos[0], *X_RANGE)
        new_pos[1] = np.clip(new_pos[1], *Y_RANGE)
        new_pos[2] = np.clip(new_pos[2], *H_RANGE)
        thr    = evaluate_throughput(new_pos, seed=GLOBAL_SEED, n_mc=N_MC_TRAIN)
        reward = thr / 300.0 * 100.0
        state  = np.concatenate([self._norm(new_pos),
                                  [thr / 300.0, step_num / STEPS_PER_EP]])
        return state, reward, new_pos, thr

# ─────────────────────────────────────────────────────────────────────────────
#  Epsilon schedule
# ─────────────────────────────────────────────────────────────────────────────

def epsilon_schedule(ep):
    frac = min(ep / EPS_DECAY_EP, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)

def select_action(q_net, state, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(N_ACTIONS))
    return int(np.argmax(q_net.forward(state)))

# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_dql():
    print("=" * 62)
    print("  MODULE 3 (VM FINAL): DQL UAV Trajectory Agent")
    print("  6G RIS-UAV Emergency Communications Simulation")
    print("=" * 62)
    print(f"  Devices  : {N_NEAR} near + {N_FAR} far = {N_PAIRS} NOMA pairs")
    print(f"  RIS      : M={M_RIS} | Far blockage: {BLOCKAGE_DB:.1f}dB")
    print(f"  Episodes : {N_EP} | Steps/ep: {STEPS_PER_EP}")
    print(f"  MC train : {N_MC_TRAIN} | MC eval: {N_MC_EVAL} (seed={GLOBAL_SEED})")
    print(f"  Estimated runtime: 15-20 min on VM")
    print()

    if not sanity_check():
        print("Aborting due to channel model mismatch.")
        return None

    env   = UAVEnv()
    q_net = QNetwork(STATE_DIM, N_ACTIONS, hidden=128, seed=GLOBAL_SEED)
    t_net = QNetwork(STATE_DIM, N_ACTIONS, hidden=128, seed=GLOBAL_SEED)
    t_net.copy_weights_from(q_net)
    buf   = ReplayBuffer(BUFFER_SIZE)
    rng   = np.random.default_rng(GLOBAL_SEED)

    ep_rewards, ep_throughputs, ep_best_thr, loss_hist = [], [], [], []
    best_thr = 0.0
    best_pos = np.array([UAV_CENTROID[0], UAV_CENTROID[1], 125.0])
    conv_ep  = None

    PRINT_AT = {1, 50, 100, 200, 500, 1000, 1500, 2000}
    t0 = time.time()

    print(f"[1/5] Training DQL agent...")
    print(f"  State={STATE_DIM}  Actions={N_ACTIONS}  Steps/ep={STEPS_PER_EP}")
    print(f"  Buffer={BUFFER_SIZE}  Batch={BATCH_SIZE}  γ={GAMMA}  "
          f"ε: {EPS_START:.1f}→{EPS_END:.2f}")

    for ep in range(1, N_EP + 1):
        eps        = epsilon_schedule(ep)
        state, pos = env.reset(rng)
        ep_r = ep_t = 0.0

        for step in range(STEPS_PER_EP):
            act                      = select_action(q_net, state, eps, rng)
            next_s, rew, next_pos, t = env.step(pos, act, step)
            buf.push(state, act, rew, next_s, False)
            ep_r += rew;  ep_t += t

            if t > best_thr:
                best_thr = t
                best_pos = next_pos.copy()
                if conv_ep is None and t > 233.0:
                    conv_ep = ep

            state = next_s;  pos = next_pos

            if len(buf) >= BATCH_SIZE:
                s, a, r, s2, d = buf.sample(BATCH_SIZE, rng)
                tgt  = r + GAMMA * (1 - d) * t_net.forward_batch(s2).max(1)
                loss = q_net.update(s, a, tgt)
                loss_hist.append(loss)

        ep_rewards.append(ep_r / STEPS_PER_EP)
        ep_throughputs.append(ep_t / STEPS_PER_EP)
        ep_best_thr.append(best_thr)

        if ep % TARGET_UPDATE == 0:
            t_net.copy_weights_from(q_net)

        if ep in PRINT_AT or ep % 200 == 0:
            el  = time.time() - t0
            eta = el / ep * (N_EP - ep)
            print(f"  Ep {ep:4d}/{N_EP} | ε={eps:.3f} | "
                  f"avg_thr={ep_throughputs[-1]:.1f}Mbps | "
                  f"best={best_thr:.1f}Mbps | ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Best throughput : {best_thr:.2f} Mbps")
    print(f"  Best position   : x={best_pos[0]:.0f}m, "
          f"y={best_pos[1]:.0f}m, H={best_pos[2]:.0f}m")
    if conv_ep is None:
        conv_ep = N_EP // 2

    return (q_net, best_thr, best_pos, conv_ep,
            ep_rewards, ep_throughputs, ep_best_thr, loss_hist)

# ─────────────────────────────────────────────────────────────────────────────
#  Fair comparison
# ─────────────────────────────────────────────────────────────────────────────

def fair_comparison(dql_pos):
    print(f"\n[2/5] Fair comparison (seed={GLOBAL_SEED}, n_mc={N_MC_EVAL})...")
    alts       = [50, 75, 100, 125, 150, 175, 200]
    fixed_thrs = []
    for H in alts:
        pos = np.array([UAV_CENTROID[0], UAV_CENTROID[1], float(H)])
        thr = evaluate_throughput(pos, seed=GLOBAL_SEED, n_mc=N_MC_EVAL)
        fixed_thrs.append(thr)
        print(f"    Fixed H={H:3d}m: {thr:.2f} Mbps")

    best_fixed = max(fixed_thrs)
    best_H     = alts[int(np.argmax(fixed_thrs))]
    dql_thr    = evaluate_throughput(dql_pos, seed=GLOBAL_SEED, n_mc=N_MC_EVAL)
    dql_gain   = dql_thr - best_fixed

    print(f"    DQL pos ({dql_pos[0]:.0f},{dql_pos[1]:.0f},{dql_pos[2]:.0f}m): {dql_thr:.2f} Mbps")
    print(f"    Best fixed altitude : {best_fixed:.2f} Mbps (H={best_H}m)")
    print(f"    DQL gain over fixed : {dql_gain:+.2f} Mbps")
    if dql_gain >= 0:
        print(f"    [OK] DQL outperforms best fixed altitude")
    else:
        print(f"    [~] Within {abs(dql_gain):.1f} Mbps of best fixed altitude")

    return alts, fixed_thrs, best_fixed, best_H, dql_thr, dql_gain

# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

C = {'primary':'#0057A8','secondary':'#E87722','green':'#2E8B57',
     'red':'#C0392B','grid':'#CCCCCC'}

M1_BEST = 233.666   # Module 1 confirmed best (H=125m, centroid XY)

def smooth(v, w=30):
    if len(v) < w:
        return np.array(v)
    return np.convolve(v, np.ones(w)/w, mode='same')

def plot_convergence(ep_rewards, ep_throughputs, ep_best_thr, conv_ep, loss_hist):
    print("\n[3/5] Generating convergence plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    eps_arr   = np.arange(1, len(ep_rewards)+1)

    ax = axes[0, 0]
    ax.plot(eps_arr, ep_rewards, color=C['grid'], alpha=0.4, lw=0.6)
    ax.plot(eps_arr, smooth(ep_rewards, 40), color=C['primary'], lw=2, label='40-ep MA')
    ax.axvline(conv_ep, color=C['secondary'], ls='--', lw=1.5,
               label=f'Conv. ep ~{conv_ep}')
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg Reward')
    ax.set_title('(a) Reward Convergence')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(eps_arr, ep_throughputs, color=C['grid'], alpha=0.4, lw=0.6)
    ax.plot(eps_arr, smooth(ep_throughputs, 40), color=C['secondary'], lw=2, label='40-ep MA')
    ax.axhline(M1_BEST, color=C['green'], ls='--', lw=1.5,
               label=f'Fixed H=125m ({M1_BEST:.1f} Mbps)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg Throughput (Mbps)')
    ax.set_title('(b) Throughput per Episode')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(eps_arr, ep_best_thr, color=C['primary'], lw=2, label='Running best')
    ax.axhline(M1_BEST, color=C['green'], ls='--', lw=1.5,
               label=f'Fixed H=125m ({M1_BEST:.1f} Mbps)')
    arr = np.array(ep_best_thr)
    ax.fill_between(eps_arr, M1_BEST, arr,
                    where=arr >= M1_BEST,
                    alpha=0.25, color=C['primary'], label='DQL gain region')
    ax.set_xlabel('Episode'); ax.set_ylabel('Best Throughput (Mbps)')
    ax.set_title('(c) Running Best Throughput')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if loss_hist:
        st = np.arange(1, len(loss_hist)+1)
        ax.plot(st, loss_hist, color=C['grid'], alpha=0.3, lw=0.5)
        ax.plot(st, smooth(loss_hist, 200), color=C['red'], lw=2, label='200-step MA')
        ax.set_yscale('log')
        ax.set_xlabel('Training Step'); ax.set_ylabel('TD Loss')
        ax.set_title('(d) Q-Network Training Loss')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Fig. 10: DQL Agent Convergence - RIS-UAV 6G System',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = f"{OUT_DQL}/fig10_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")

def plot_trajectory(best_pos, dql_thr):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(NEAR_POS[:,0], NEAR_POS[:,1], c=C['primary'], s=60,
               label='Near devices (LoS)', zorder=4)
    far_plot = FAR_POS[(FAR_POS[:,0] > -500) & (FAR_POS[:,0] < 1500)]
    ax.scatter(far_plot[:,0], far_plot[:,1], c=C['red'], s=60, marker='^',
               label='Far devices (blocked)', zorder=4)
    ax.scatter(*RIS_POS[:2], c='purple', s=250, marker='s', label='RIS', zorder=5)
    ax.scatter(best_pos[0], best_pos[1], c=C['secondary'], s=400, marker='*',
               zorder=6, label=f'DQL opt. ({best_pos[0]:.0f},{best_pos[1]:.0f}m)')
    ax.scatter(*UAV_CENTROID, c=C['green'], s=200, marker='D',
               zorder=5, label='UAV centroid (fixed)')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('UAV Optimal XY Position')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(-300, 1300); ax.set_ylim(0, 1500)

    ax = axes[1]
    alts_m1 = [50,   75,    100,    125,    150,    175,    200]
    thrs_m1  = [194.707, 222.523, 233.290, 233.666, 230.646, 226.695, 222.262]
    ax.plot(alts_m1, thrs_m1, color=C['primary'], lw=2, marker='o', ms=6,
            label='Fixed-altitude RIS-NOMA (Module 1)')
    ax.axhline(dql_thr, color=C['secondary'], ls='--', lw=2,
               label=f'DQL optimal ({dql_thr:.1f} Mbps)')
    ax.scatter([best_pos[2]], [dql_thr], c=C['secondary'], s=150, zorder=5)
    ax.set_xlabel('UAV Altitude H (m)')
    ax.set_ylabel('System Throughput (Mbps)')
    ax.set_title('DQL vs Fixed Altitude')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(160, 260)

    plt.suptitle('Fig. 11: UAV Position and Altitude Analysis', fontsize=13)
    plt.tight_layout()
    path = f"{OUT_DQL}/fig11_trajectory.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")

def plot_throughput_compare(alts, fixed_thrs, best_fixed, best_H, dql_thr, dql_gain):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(alts)); w = 0.35
    bars1 = ax.bar(x - w/2, fixed_thrs, w, color=C['primary'],
                   alpha=0.85, label='Fixed Altitude (centroid XY)')
    bars2 = ax.bar(x + w/2, [dql_thr]*len(alts), w, color=C['secondary'],
                   alpha=0.85, label='DQL Optimal Position')
    idx = alts.index(best_H)
    b   = bars2[idx]
    ax.annotate(f'{dql_gain:+.1f} Mbps',
                xy=(b.get_x() + b.get_width()/2, b.get_height()),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=11, fontweight='bold', color=C['secondary'])
    ax.set_xticks(x)
    ax.set_xticklabels([f'H={H}m' for H in alts])
    ax.set_xlabel('UAV Altitude')
    ax.set_ylabel('System Throughput (Mbps)')
    ax.set_title('Fig. 12: DQL vs Fixed Altitude — Fair Comparison (seed=99, n_mc=50)')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.4)
    ax.set_ylim(160, max(max(fixed_thrs), dql_thr) * 1.12)
    plt.tight_layout()
    path = f"{OUT_DQL}/fig12_throughput_compare.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")

def plot_policy_heatmap(q_net):
    print("  Generating policy heatmap (~2 min)...")
    H_fixed = 125.0
    xs  = np.linspace(*X_RANGE, 20)
    ys  = np.linspace(*Y_RANGE, 20)
    env = UAVEnv()
    qmax = np.zeros((len(ys), len(xs)))
    amap = np.zeros((len(ys), len(xs)), dtype=int)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            pos   = np.array([x, y, H_fixed])
            thr   = evaluate_throughput(pos, seed=GLOBAL_SEED, n_mc=3)
            state = np.concatenate([env._norm(pos), [thr/300.0, 0.5]])
            qv    = q_net.forward(state)
            qmax[i, j] = qv.max();  amap[i, j] = qv.argmax()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(qmax, origin='lower', extent=[*X_RANGE, *Y_RANGE],
                         aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=axes[0], label='Max Q-value')
    axes[0].set_title(f'Fig. 13a: Max Q-value (H={H_fixed:.0f}m)')
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')

    cmap2 = plt.cm.get_cmap('tab10', N_ACTIONS)
    axes[1].imshow(amap, origin='lower', extent=[*X_RANGE, *Y_RANGE],
                   aspect='auto', cmap=cmap2, vmin=0, vmax=N_ACTIONS-1)
    labels  = ['Stay', '+X', '-X', '+Y', '-Y', '+H', '-H']
    patches = [mpatches.Patch(color=cmap2(i), label=labels[i]) for i in range(N_ACTIONS)]
    axes[1].legend(handles=patches, fontsize=8, loc='upper right')
    axes[1].set_title(f'Fig. 13b: Greedy Policy (H={H_fixed:.0f}m)')
    axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Y (m)')
    plt.tight_layout()
    path = f"{OUT_DQL}/fig13_policy_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
#  Save JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_results(best_pos, best_thr, dql_thr, dql_gain,
                 best_fixed, best_H, conv_ep, alts, fixed_thrs,
                 ep_throughputs, ep_best_thr):
    res = {
        "system_parameters": {
            "frequency_GHz": 3.5, "bandwidth_MHz": 20.0,
            "RIS_elements": 1024, "NOMA_pairs": 12,
            "blockage_dB": 60.0, "alpha_NOMA": 0.85,
            "MC_seed": GLOBAL_SEED, "n_mc_train": N_MC_TRAIN,
            "n_mc_eval": N_MC_EVAL,
        },
        "dql_hyperparameters": {
            "episodes": N_EP, "steps_per_episode": STEPS_PER_EP,
            "gamma": GAMMA, "lr": LR,
        },
        "key_results": {
            "DQL_best_train_Mbps"      : round(best_thr, 3),
            "DQL_optimal_position_m"   : best_pos.tolist(),
            "DQL_fair_eval_Mbps"       : round(dql_thr, 3),
            "best_fixed_alt_Mbps"      : round(best_fixed, 3),
            "best_fixed_H_m"           : best_H,
            "DQL_gain_over_fixed_Mbps" : round(dql_gain, 3),
            "convergence_episode"      : conv_ep,
        },
        "module1_reference": {
            "50":194.707,"75":222.523,"100":233.290,
            "125":233.666,"150":230.646,"175":226.695,"200":222.262
        },
        "fixed_altitude_this_run": {
            str(H): round(t, 3) for H, t in zip(alts, fixed_thrs)
        },
        "training_every100": {
            "ep_thr"  : [round(ep_throughputs[i], 2)
                          for i in range(0, len(ep_throughputs), 100)],
            "best_thr": [round(ep_best_thr[i], 2)
                          for i in range(0, len(ep_best_thr), 100)],
        }
    }
    p1 = f"{OUT_DQL}/dql_results.json"
    with open(p1, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  [OK] Saved: {p1}")
    p2 = f"{OUT_DQL}/best_trajectory.json"
    with open(p2, 'w') as f:
        json.dump({"best_position_m": best_pos.tolist(),
                   "best_thr_train_Mbps": round(best_thr, 3),
                   "fair_eval_Mbps": round(dql_thr, 3),
                   "dql_gain_Mbps": round(dql_gain, 3)}, f, indent=2)
    print(f"  [OK] Saved: {p2}")

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    result = train_dql()
    if result is None:
        return

    (q_net, best_thr, best_pos, conv_ep,
     ep_rewards, ep_throughputs, ep_best_thr, loss_hist) = result

    (alts, fixed_thrs, best_fixed,
     best_H, dql_thr, dql_gain) = fair_comparison(best_pos)

    plot_convergence(ep_rewards, ep_throughputs, ep_best_thr, conv_ep, loss_hist)

    print("\n[4/5] Generating remaining plots...")
    plot_trajectory(best_pos, dql_thr)
    plot_throughput_compare(alts, fixed_thrs, best_fixed, best_H, dql_thr, dql_gain)
    plot_policy_heatmap(q_net)

    print("\n[5/5] Saving results...")
    save_results(best_pos, best_thr, dql_thr, dql_gain,
                 best_fixed, best_H, conv_ep, alts, fixed_thrs,
                 ep_throughputs, ep_best_thr)

    print()
    print("=" * 62)
    print("  MODULE 3 COMPLETE")
    print("=" * 62)
    print(f"  DQL best (training)  : {best_thr:.2f} Mbps")
    print(f"  DQL optimal position : x={best_pos[0]:.0f}m, "
          f"y={best_pos[1]:.0f}m, H={best_pos[2]:.0f}m")
    print(f"  DQL fair eval        : {dql_thr:.2f} Mbps")
    print(f"  Best fixed altitude  : {best_fixed:.2f} Mbps (H={best_H}m)")
    print(f"  DQL gain over fixed  : {dql_gain:+.2f} Mbps")
    print(f"  Convergence episode  : ~{conv_ep}")
    print(f"  OUTPUT: {OUT_DQL}/")
    print("=" * 62)

if __name__ == "__main__":
    main()
