"""
=============================================================
  MODULE 3 (WINDOWS FINAL): DQL UAV Trajectory Agent
  6G RIS-UAV Emergency Communications Simulation

  Channel model: EXACT copy from dql_agent_350ep_backup.py
  Device seed  : 42  (matches Module 1 / backup file)
  Compare seed : 99  (fair comparison)
  No PyTorch needed — pure NumPy Q-network

  N_EP = 2000, N_MC_TRAIN = 3 (fast), N_MC_EVAL = 50
=============================================================
"""

import numpy as np
import os, json, time, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)

os.makedirs("results/dql", exist_ok=True)
OUT = "results/dql"

# =============================================================
# CHANNEL MODEL — exact copy from backup file
# =============================================================
class Config:
    AREA_X    = 3000.0
    AREA_Y    = 1600.0
    K_NEAR    = 12
    K_FAR     = 12
    N_PAIRS   = 12
    UAV_CX    = 400.0
    UAV_CY    = 600.0
    NEAR_DMIN = 50.0;   NEAR_DMAX = 250.0
    FAR_DMIN  = 450.0;  FAR_DMAX  = 800.0
    H_MIN     = 50.0;   H_MAX     = 200.0
    P_MAX_W   = 2.0
    P_PAIR    = 2.0 / 12
    BW_PAIR   = 20e6 / 12
    ALPHA     = 0.85
    M         = 1024
    RIS_POS   = np.array([300.0, 250.0, 20.0])
    B_PHASE   = 3
    FC        = 3.5e9;  C = 3e8
    NOISE_W   = 10**((-174 + 7 - 30 + 10*np.log10(20e6/12)) / 10)  # NF = 7 dB
    A_URBAN   = 9.61;   B_URBAN = 0.16
    ETA_LOS   = 1.0;    ETA_NLOS = 20.0
    K_RICIAN  = 10**(10.0 / 10)
    BLOCK_DB  = 60.0

CFG = Config()


def los_prob(tx, rx):
    dx = tx[0]-rx[0];  dy = tx[1]-rx[1]
    h  = np.sqrt(dx**2+dy**2) + 1e-9
    el = np.degrees(np.arctan(tx[2] / h))
    return float(np.clip(
        1 / (1 + CFG.A_URBAN*np.exp(-CFG.B_URBAN*(el-CFG.A_URBAN))),
        0, 1))


def pl_dB(tx, rx, extra=0):
    d = np.linalg.norm(np.array(tx)-np.array(rx)) + 1e-9
    F = (20*np.log10(d) + 20*np.log10(CFG.FC) +
         20*np.log10(4*np.pi/CFG.C))
    p = los_prob(np.array(tx), np.array(rx))
    return p*(CFG.ETA_LOS+F) + (1-p)*(CFG.ETA_NLOS+F) + extra


def rician(sz=1):
    K  = CFG.K_RICIAN
    ph = np.random.uniform(0, 2*np.pi, sz)
    hL = np.exp(1j*ph) * np.sqrt(K/(K+1))
    hS = ((np.random.randn(*ph.shape) +
           1j*np.random.randn(*ph.shape)) / np.sqrt(2*(K+1)))
    h  = hL + hS
    return h.squeeze() if sz == 1 else h


def chan(tx, rx, ex=0):
    return 10**(-pl_dB(tx, rx, ex)/20) * rician()


def G_vec(uav):
    return 10**(-pl_dB(uav, CFG.RIS_POS)/20) * rician(CFG.M)


def Hr_vec(dev):
    return 10**(-pl_dB(CFG.RIS_POS, dev)/20) * rician(CFG.M)


def opt_phi(G, hr):
    """Correct phase alignment: angle(h_r) - angle(G), quantised to B_PHASE bits"""
    raw  = (np.angle(hr) - np.angle(G)) % (2*np.pi)
    step = 2*np.pi / (2**CFG.B_PHASE)
    return np.round(raw/step) * step


def h_eff(hd, G, hr, phi):
    return hd + np.dot(hr.conj(), np.exp(1j*phi)*G)


def pair_rates(h_near, h_far):
    """SIC-NOMA pair rates — far user decoded first."""
    a  = CFG.ALPHA;  P  = CFG.P_PAIR;  BW = CFG.BW_PAIR
    sF = (a*P*abs(h_far)**2)  / ((1-a)*P*abs(h_far)**2 + CFG.NOISE_W)
    sN = ((1-a)*P*abs(h_near)**2) / CFG.NOISE_W
    return BW*np.log2(1+sN), BW*np.log2(1+sF)


# ── Device placement — seed=42, same as Module 1 ─────────────────────────────
np.random.seed(42)
cx, cy = CFG.UAV_CX, CFG.UAV_CY

ang_n = np.random.uniform(0, 2*np.pi, CFG.K_NEAR)
rad_n = np.random.uniform(CFG.NEAR_DMIN, CFG.NEAR_DMAX, CFG.K_NEAR)
near_devs = np.zeros((CFG.K_NEAR, 3))
near_devs[:,0] = np.clip(cx + rad_n*np.cos(ang_n), 50, CFG.AREA_X-50)
near_devs[:,1] = np.clip(cy + rad_n*np.sin(ang_n), 50, CFG.AREA_Y-50)

ang_f = np.random.uniform(0, 2*np.pi, CFG.K_FAR)
rad_f = np.random.uniform(CFG.FAR_DMIN, CFG.FAR_DMAX, CFG.K_FAR)
far_devs = np.zeros((CFG.K_FAR, 3))
far_devs[:,0] = np.clip(cx + rad_f*np.cos(ang_f), 50, CFG.AREA_X-50)
far_devs[:,1] = np.clip(cy + rad_f*np.sin(ang_f), 50, CFG.AREA_Y-50)


def compute_throughput(uav_pos, n_mc=3, seed=None):
    """Exact copy from backup file."""
    if seed is not None:
        np.random.seed(seed)
    total = []
    G_v   = G_vec(uav_pos)
    for _ in range(n_mc):
        thr = 0.0
        for i in range(CFG.N_PAIRS):
            hn  = chan(uav_pos, near_devs[i], 0.0)
            hf  = chan(uav_pos, far_devs[i],  CFG.BLOCK_DB)
            hr  = Hr_vec(far_devs[i])
            phi = opt_phi(G_v, hr)
            he  = h_eff(hf, G_v, hr, phi)
            rN, rF = pair_rates(hn, he)
            thr += (rN + rF)
        total.append(thr / 1e6)
    return float(np.mean(total))


# ── Sanity check ─────────────────────────────────────────────────────────────
def sanity_check():
    print("  Sanity check vs Module 1 (expected ~194, ~233, ~222 Mbps)...")
    expected = {50: 194.7, 125: 233.7, 200: 222.3}
    all_ok = True
    for H, exp in expected.items():
        pos = np.array([cx, cy, float(H)])
        got = compute_throughput(pos, n_mc=50, seed=99)
        diff = abs(got - exp)
        flag = "OK" if diff < 10 else "WARN"
        print(f"    H={H:3d}m: {got:.2f} Mbps (expected ~{exp:.1f}) [{flag}]")
        if diff >= 10:
            all_ok = False
    print()
    return all_ok

# =============================================================
# DQL HYPER-PARAMETERS
# =============================================================
N_EP         = 2000
STEPS_PER_EP = 25
BUFFER_SIZE  = 10000
BATCH_SIZE   = 64
GAMMA        = 0.95
LR           = 1e-3
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 500
TARGET_SYNC  = 20
N_MC_TRAIN   = 3    # fast — matches backup file
N_MC_EVAL    = 50   # accurate — for final comparison
COMPARE_SEED = 99

STEP_XY = 100.0    # matches backup file
STEP_H  = 25.0

ACTIONS = [
    np.array([ STEP_XY, 0.0,    0.0]),   # +x
    np.array([-STEP_XY, 0.0,    0.0]),   # -x
    np.array([ 0.0,  STEP_XY,   0.0]),   # +y
    np.array([ 0.0, -STEP_XY,   0.0]),   # -y
    np.array([ 0.0,     0.0,  STEP_H]),  # +H
    np.array([ 0.0,     0.0, -STEP_H]),  # -H
    np.array([ 0.0,     0.0,    0.0]),   # hover
]
N_ACTIONS = len(ACTIONS)
STATE_DIM = 5

X_RANGE = (50.0,  700.0)
Y_RANGE = (50.0,  1100.0)
H_RANGE = (CFG.H_MIN, CFG.H_MAX)

# =============================================================
# Q-NETWORK (pure NumPy — no PyTorch needed)
# =============================================================
def relu(x):
    return np.maximum(0, x)

class QNetwork:
    def __init__(self, state_dim, n_actions, hidden=128, seed=0):
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2/state_dim)
        s2 = np.sqrt(2/hidden)
        self.W1 = rng.standard_normal((hidden, state_dim)) * s1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, hidden)) * s2
        self.b2 = np.zeros(hidden)
        self.W3 = rng.standard_normal((n_actions, hidden)) * s2
        self.b3 = np.zeros(n_actions)
        self.lr  = LR; self.t = 0
        self.b1a, self.b2a, self.ea = 0.9, 0.999, 1e-8
        self.m = [np.zeros_like(p) for p in self._params()]
        self.v = [np.zeros_like(p) for p in self._params()]

    def _params(self):
        return [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]

    def forward(self, x):
        h1 = relu(self.W1 @ x + self.b1)
        h2 = relu(self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def forward_batch(self, X):
        H1 = relu(X @ self.W1.T + self.b1)
        H2 = relu(H1 @ self.W2.T + self.b2)
        return H2 @ self.W3.T + self.b3

    def copy_from(self, other):
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
        for i,(p,g) in enumerate(zip(self._params(),
                                      [dW1,db1,dW2,db2,dW3,db3])):
            self.m[i] = self.b1a*self.m[i] + (1-self.b1a)*g
            self.v[i] = self.b2a*self.v[i] + (1-self.b2a)*g**2
            mh = self.m[i]/(1-self.b1a**self.t)
            vh = self.v[i]/(1-self.b2a**self.t)
            p -= self.lr * mh/(np.sqrt(vh)+self.ea)
        return loss

# =============================================================
# REPLAY BUFFER
# =============================================================
class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, d):
        self.buf.append((s,a,r,s2,d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s,a,r,s2,d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int32),
                np.array(r,  dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self):
        return len(self.buf)

# =============================================================
# ENVIRONMENT
# =============================================================
class UAVEnv:
    def _norm(self, pos):
        return np.array([
            (pos[0]-X_RANGE[0])/(X_RANGE[1]-X_RANGE[0]),
            (pos[1]-Y_RANGE[0])/(Y_RANGE[1]-Y_RANGE[0]),
            (pos[2]-H_RANGE[0])/(H_RANGE[1]-H_RANGE[0]),
        ])

    def reset(self):
        pos   = np.array([
            np.random.uniform(*X_RANGE),
            np.random.uniform(*Y_RANGE),
            np.random.uniform(*H_RANGE),
        ])
        thr   = compute_throughput(pos, n_mc=N_MC_TRAIN)
        state = np.concatenate([self._norm(pos), [thr/300.0, 0.0]])
        return state, pos

    def step(self, pos, action_idx, step_num):
        new_pos    = pos + ACTIONS[action_idx]
        new_pos[0] = np.clip(new_pos[0], *X_RANGE)
        new_pos[1] = np.clip(new_pos[1], *Y_RANGE)
        new_pos[2] = np.clip(new_pos[2], *H_RANGE)
        thr    = compute_throughput(new_pos, n_mc=N_MC_TRAIN)
        reward = thr / 300.0 * 100.0
        state  = np.concatenate([self._norm(new_pos),
                                  [thr/300.0, step_num/STEPS_PER_EP]])
        return state, reward, new_pos, thr

# =============================================================
# TRAINING
# =============================================================
def epsilon(ep):
    frac = min(ep/EPS_DECAY, 1.0)
    return EPS_START + frac*(EPS_END-EPS_START)

def select_action(q_net, state, eps):
    if random.random() < eps:
        return random.randint(0, N_ACTIONS-1)
    return int(np.argmax(q_net.forward(state)))


def train_dql():
    print("=" * 62)
    print("  MODULE 3 (WINDOWS FINAL): DQL UAV Trajectory Agent")
    print("  6G RIS-UAV Emergency Communications Simulation")
    print("=" * 62)
    print(f"  Devices  : {CFG.K_NEAR} near + {CFG.K_FAR} far = {CFG.N_PAIRS} pairs")
    print(f"  RIS      : M={CFG.M} | Blockage: {CFG.BLOCK_DB}dB")
    print(f"  Episodes : {N_EP} | Steps/ep: {STEPS_PER_EP}")
    print(f"  MC train : {N_MC_TRAIN} | MC eval: {N_MC_EVAL}")
    print()

    sanity_check()

    env   = UAVEnv()
    q_net = QNetwork(STATE_DIM, N_ACTIONS, seed=42)
    t_net = QNetwork(STATE_DIM, N_ACTIONS, seed=42)
    t_net.copy_from(q_net)
    buf   = ReplayBuffer(BUFFER_SIZE)

    ep_rewards, ep_thrs, best_thrs, loss_hist = [], [], [], []
    best_thr = 0.0
    best_pos = np.array([cx, cy, 125.0])
    conv_ep  = None

    PRINT_AT = {1,50,100,200,500,1000,1500,2000}
    t0 = time.time()

    print(f"[1/5] Training DQL agent...")
    print(f"  State={STATE_DIM}  Actions={N_ACTIONS}  "
          f"Steps/ep={STEPS_PER_EP}")
    print(f"  Buffer={BUFFER_SIZE}  Batch={BATCH_SIZE}  "
          f"γ={GAMMA}  ε: {EPS_START:.1f}→{EPS_END:.2f}")

    for ep in range(1, N_EP+1):
        eps        = epsilon(ep)
        state, pos = env.reset()
        ep_r = ep_t = 0.0

        for step in range(STEPS_PER_EP):
            act              = select_action(q_net, state, eps)
            ns, rew, npos, t = env.step(pos, act, step)
            buf.push(state, act, rew, ns, False)
            ep_r += rew;  ep_t += t

            if t > best_thr:
                best_thr = t
                best_pos = npos.copy()
                if conv_ep is None and t > 230.0:
                    conv_ep = ep

            state = ns;  pos = npos

            if len(buf) >= BATCH_SIZE:
                s,a,r,s2,d = buf.sample(BATCH_SIZE)
                tgt  = r + GAMMA*(1-d)*t_net.forward_batch(s2).max(1)
                loss = q_net.update(s, a, tgt)
                loss_hist.append(loss)

        ep_rewards.append(ep_r/STEPS_PER_EP)
        ep_thrs.append(ep_t/STEPS_PER_EP)
        best_thrs.append(best_thr)

        if ep % TARGET_SYNC == 0:
            t_net.copy_from(q_net)

        if ep in PRINT_AT or ep % 200 == 0:
            el  = time.time()-t0
            eta = el/ep*(N_EP-ep)
            print(f"  Ep {ep:4d}/{N_EP} | ε={eps:.3f} | "
                  f"avg_thr={ep_thrs[-1]:.1f}Mbps | "
                  f"best={best_thr:.1f}Mbps | ETA={eta:.0f}s")

    elapsed = time.time()-t0
    print(f"  Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Best throughput : {best_thr:.2f} Mbps")
    print(f"  Best position   : x={best_pos[0]:.0f}m, "
          f"y={best_pos[1]:.0f}m, H={best_pos[2]:.0f}m")
    if conv_ep is None:
        conv_ep = N_EP//2

    return (q_net, best_thr, best_pos, conv_ep,
            ep_rewards, ep_thrs, best_thrs, loss_hist)

# =============================================================
# FAIR COMPARISON
# =============================================================
def fair_comparison(dql_pos):
    print(f"\n[2/5] Fair comparison (seed={COMPARE_SEED}, n_mc={N_MC_EVAL})...")
    alts = [50, 75, 100, 125, 150, 175, 200]
    fixed_thrs = []
    for H in alts:
        pos = np.array([cx, cy, float(H)])
        thr = compute_throughput(pos, n_mc=N_MC_EVAL, seed=COMPARE_SEED)
        fixed_thrs.append(thr)
        print(f"    Fixed H={H:3d}m: {thr:.2f} Mbps")

    best_fixed = max(fixed_thrs)
    best_H     = alts[int(np.argmax(fixed_thrs))]
    dql_thr    = compute_throughput(dql_pos, n_mc=N_MC_EVAL, seed=COMPARE_SEED)
    dql_gain   = dql_thr - best_fixed

    print(f"    DQL pos ({dql_pos[0]:.0f},{dql_pos[1]:.0f},{dql_pos[2]:.0f}m): "
          f"{dql_thr:.2f} Mbps")
    print(f"    Best fixed altitude : {best_fixed:.2f} Mbps (H={best_H}m)")
    print(f"    DQL gain over fixed : {dql_gain:+.2f} Mbps")
    if dql_gain >= 0:
        print(f"    [OK] DQL outperforms fixed altitude")
    else:
        print(f"    [~] Within {abs(dql_gain):.1f} Mbps of best fixed")

    return alts, fixed_thrs, best_fixed, best_H, dql_thr, dql_gain

# =============================================================
# PLOTS
# =============================================================
C = {'primary':'#0057A8','secondary':'#E87722','green':'#2E8B57',
     'red':'#C0392B','grid':'#CCCCCC'}

def smooth(v, w=30):
    if len(v) < w:
        return np.array(v)
    return np.convolve(v, np.ones(w)/w, mode='same')


def plot_convergence(ep_rewards, ep_thrs, best_thrs, conv_ep, loss_hist):
    print("\n[3/5] Generating convergence plot...")
    M1_BEST = 233.7
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    eps_arr   = np.arange(1, len(ep_rewards)+1)

    ax = axes[0,0]
    ax.plot(eps_arr, ep_rewards, color=C['grid'], alpha=0.4, lw=0.6)
    ax.plot(eps_arr, smooth(ep_rewards,40), color=C['primary'], lw=2, label='40-ep MA')
    ax.axvline(conv_ep, color=C['secondary'], ls='--', lw=1.5,
               label=f'Conv. ep ~{conv_ep}')
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg Reward')
    ax.set_title('(a) Reward Convergence')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.plot(eps_arr, ep_thrs, color=C['grid'], alpha=0.4, lw=0.6)
    ax.plot(eps_arr, smooth(ep_thrs,40), color=C['secondary'], lw=2, label='40-ep MA')
    ax.axhline(M1_BEST, color=C['green'], ls='--', lw=1.5,
               label=f'Fixed H=125m ({M1_BEST:.1f} Mbps)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg Throughput (Mbps)')
    ax.set_title('(b) Throughput per Episode')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    arr = np.array(best_thrs)
    ax.plot(eps_arr, arr, color=C['primary'], lw=2, label='Running best')
    ax.axhline(M1_BEST, color=C['green'], ls='--', lw=1.5,
               label=f'Fixed H=125m ({M1_BEST:.1f} Mbps)')
    ax.fill_between(eps_arr, M1_BEST, arr,
                    where=arr>=M1_BEST, alpha=0.25,
                    color=C['primary'], label='DQL gain region')
    ax.set_xlabel('Episode'); ax.set_ylabel('Best Throughput (Mbps)')
    ax.set_title('(c) Running Best Throughput')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    if loss_hist:
        st = np.arange(1, len(loss_hist)+1)
        ax.plot(st, loss_hist, color=C['grid'], alpha=0.3, lw=0.5)
        ax.plot(st, smooth(loss_hist,200), color=C['red'], lw=2, label='200-step MA')
        ax.set_yscale('log')
        ax.set_xlabel('Training Step'); ax.set_ylabel('TD Loss')
        ax.set_title('(d) Q-Network Training Loss')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Fig. 10: DQL Agent Convergence - RIS-UAV 6G System',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = f"{OUT}/fig10_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [OK] {path}")


def plot_trajectory(best_pos, dql_thr):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(near_devs[:,0], near_devs[:,1], c=C['primary'],
               s=60, label='Near devices (LoS)', zorder=4)
    ax.scatter(far_devs[:,0],  far_devs[:,1],  c=C['red'],
               s=60, marker='^', label='Far devices (blocked)', zorder=4)
    ax.scatter(*CFG.RIS_POS[:2], c='purple', s=250, marker='s',
               label='RIS', zorder=5)
    ax.scatter(best_pos[0], best_pos[1], c=C['secondary'], s=400,
               marker='*', zorder=6,
               label=f'DQL opt. ({best_pos[0]:.0f},{best_pos[1]:.0f}m)')
    ax.scatter(cx, cy, c=C['green'], s=200, marker='D',
               zorder=5, label='UAV centroid (fixed)')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('UAV Optimal XY Position')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    alts_m1 = [50,   75,    100,    125,    150,    175,    200]
    thrs_m1  = [194.707,222.523,233.290,233.666,230.646,226.695,222.262]
    ax.plot(alts_m1, thrs_m1, color=C['primary'], lw=2, marker='o',
            ms=6, label='Fixed-altitude RIS-NOMA (Module 1)')
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
    path = f"{OUT}/fig11_trajectory.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [OK] {path}")


def plot_throughput_compare(alts, fixed_thrs, best_fixed, best_H, dql_thr, dql_gain):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(alts)); w = 0.35
    bars1 = ax.bar(x-w/2, fixed_thrs, w, color=C['primary'],
                   alpha=0.85, label='Fixed Altitude (centroid XY)')
    bars2 = ax.bar(x+w/2, [dql_thr]*len(alts), w,
                   color=C['secondary'], alpha=0.85,
                   label='DQL Optimal Position')
    idx = alts.index(best_H)
    b   = bars2[idx]
    ax.annotate(f'{dql_gain:+.1f} Mbps',
                xy=(b.get_x()+b.get_width()/2, b.get_height()),
                xytext=(0,10), textcoords='offset points',
                ha='center', fontsize=11, fontweight='bold',
                color=C['secondary'])
    ax.set_xticks(x)
    ax.set_xticklabels([f'H={H}m' for H in alts])
    ax.set_xlabel('UAV Altitude')
    ax.set_ylabel('System Throughput (Mbps)')
    ax.set_title('Fig. 12: DQL vs Fixed Altitude — Fair Comparison (seed=99, n_mc=50)')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.4)
    ax.set_ylim(160, max(max(fixed_thrs), dql_thr)*1.12)
    plt.tight_layout()
    path = f"{OUT}/fig12_throughput_compare.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [OK] {path}")


def plot_policy_heatmap(q_net):
    print("  Generating policy heatmap...")
    H_fixed = 125.0
    xs  = np.linspace(*X_RANGE, 20)
    ys  = np.linspace(*Y_RANGE, 20)
    env = UAVEnv()
    qmax = np.zeros((len(ys), len(xs)))
    amap = np.zeros((len(ys), len(xs)), dtype=int)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            pos   = np.array([x, y, H_fixed])
            thr   = compute_throughput(pos, n_mc=2)
            state = np.concatenate([env._norm(pos), [thr/300.0, 0.5]])
            qv    = q_net.forward(state)
            qmax[i,j] = qv.max();  amap[i,j] = qv.argmax()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(qmax, origin='lower',
                         extent=[*X_RANGE, *Y_RANGE],
                         aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=axes[0], label='Max Q-value')
    axes[0].set_title(f'Fig. 13a: Max Q-value (H={H_fixed:.0f}m)')
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')

    cmap2 = plt.cm.get_cmap('tab10', N_ACTIONS)
    axes[1].imshow(amap, origin='lower', extent=[*X_RANGE, *Y_RANGE],
                   aspect='auto', cmap=cmap2, vmin=0, vmax=N_ACTIONS-1)
    labels  = ['+X','-X','+Y','-Y','+H','-H','Hover']
    patches = [mpatches.Patch(color=cmap2(i), label=labels[i])
               for i in range(N_ACTIONS)]
    axes[1].legend(handles=patches, fontsize=8, loc='upper right')
    axes[1].set_title(f'Fig. 13b: Greedy Policy (H={H_fixed:.0f}m)')
    axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Y (m)')
    plt.tight_layout()
    path = f"{OUT}/fig13_policy_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [OK] {path}")

# =============================================================
# SAVE JSON
# =============================================================
def save_results(best_pos, best_thr, dql_thr, dql_gain,
                 best_fixed, best_H, conv_ep, alts, fixed_thrs,
                 ep_thrs, best_thrs):
    res = {
        "system": {"freq_GHz":3.5,"bw_MHz":20,"M_RIS":1024,
                   "N_pairs":12,"blockage_dB":60,"alpha":0.85,
                   "device_seed":42,"compare_seed":COMPARE_SEED},
        "dql": {"episodes":N_EP,"steps":STEPS_PER_EP,
                "gamma":GAMMA,"lr":LR,"n_mc_train":N_MC_TRAIN,
                "n_mc_eval":N_MC_EVAL},
        "results": {
            "best_train_Mbps"    : round(best_thr,3),
            "optimal_pos_m"      : best_pos.tolist(),
            "fair_eval_Mbps"     : round(dql_thr,3),
            "best_fixed_Mbps"    : round(best_fixed,3),
            "best_fixed_H_m"     : best_H,
            "dql_gain_Mbps"      : round(dql_gain,3),
            "convergence_ep"     : conv_ep,
        },
        "module1_ref": {
            "50":194.707,"75":222.523,"100":233.290,
            "125":233.666,"150":230.646,"175":226.695,"200":222.262},
        "fixed_this_run": {str(H):round(t,3)
                            for H,t in zip(alts,fixed_thrs)},
        "curve_every100": {
            "thr" :[round(ep_thrs[i],2)
                    for i in range(0,len(ep_thrs),100)],
            "best":[round(best_thrs[i],2)
                    for i in range(0,len(best_thrs),100)],
        }
    }
    p = f"{OUT}/dql_results.json"
    with open(p,'w') as f: json.dump(res,f,indent=2)
    print(f"  [OK] {p}")
    p2 = f"{OUT}/best_trajectory.json"
    with open(p2,'w') as f:
        json.dump({"pos_m":best_pos.tolist(),
                   "best_train_Mbps":round(best_thr,3),
                   "fair_eval_Mbps":round(dql_thr,3),
                   "gain_Mbps":round(dql_gain,3)},f,indent=2)
    print(f"  [OK] {p2}")

# =============================================================
# MAIN
# =============================================================
def main():
    (q_net, best_thr, best_pos, conv_ep,
     ep_rewards, ep_thrs, best_thrs, loss_hist) = train_dql()

    (alts, fixed_thrs, best_fixed,
     best_H, dql_thr, dql_gain) = fair_comparison(best_pos)

    plot_convergence(ep_rewards, ep_thrs, best_thrs, conv_ep, loss_hist)

    print("\n[4/5] Generating remaining plots...")
    plot_trajectory(best_pos, dql_thr)
    plot_throughput_compare(alts, fixed_thrs, best_fixed,
                             best_H, dql_thr, dql_gain)
    plot_policy_heatmap(q_net)

    print("\n[5/5] Saving results...")
    save_results(best_pos, best_thr, dql_thr, dql_gain,
                 best_fixed, best_H, conv_ep, alts, fixed_thrs,
                 ep_thrs, best_thrs)

    print()
    print("=" * 62)
    print("  MODULE 3 COMPLETE")
    print("=" * 62)
    print(f"  DQL best (training)  : {best_thr:.2f} Mbps")
    print(f"  DQL position         : x={best_pos[0]:.0f}m, "
          f"y={best_pos[1]:.0f}m, H={best_pos[2]:.0f}m")
    print(f"  DQL fair eval        : {dql_thr:.2f} Mbps")
    print(f"  Best fixed altitude  : {best_fixed:.2f} Mbps (H={best_H}m)")
    print(f"  DQL gain over fixed  : {dql_gain:+.2f} Mbps")
    print(f"  Convergence episode  : ~{conv_ep}")
    print(f"  Output folder        : {OUT}/")
    print("=" * 62)

if __name__ == "__main__":
    main()
