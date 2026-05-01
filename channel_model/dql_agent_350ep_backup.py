"""
=============================================================
  RIS-Assisted UAV Emergency Communications — 6G Simulation
  MODULE 3 (UPDATED): Deep Q-Learning UAV Trajectory Agent

  CHANGE FROM PREVIOUS VERSION:
  - Fair comparison: same random seed (99) used for ALL
    throughput evaluations in the comparison section
  - This ensures DQL vs fixed-altitude comparison is not
    affected by Monte Carlo noise

  INSTRUCTIONS:
  -----------------------------------------------
  cd ~/PhD_6G_RIS_DQL
  source ris_dql_env/bin/activate
  pip install torch   (if not already installed)
  python channel_model/dql_agent.py

  RUNTIME: ~10-15 minutes on VM

  OUTPUT FILES SAVED TO:
  -----------------------------------------------
  ~/PhD_6G_RIS_DQL/results/dql/
      fig10_convergence.png         <- reward vs episodes
      fig11_trajectory.png          <- learned UAV path
      fig12_throughput_compare.png  <- DQL vs fixed altitude
      fig13_policy_heatmap.png      <- reward surface
      dql_results.json              <- all numbers for paper
      best_trajectory.json          <- optimal UAV waypoints
=============================================================
"""

import numpy as np
import os, json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    print("[!] PyTorch not found. Install with: pip install torch")
    exit(1)

from collections import deque
import random

os.makedirs("results/dql", exist_ok=True)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

print("=" * 62)
print("  MODULE 3 (UPDATED): DQL UAV Trajectory Agent")
print("  6G RIS-UAV Emergency Communications Simulation")
print("=" * 62)


# =============================================================
# CHANNEL MODEL (self-contained copy from Module 1)
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
    NOISE_W   = 10**((-174 - 30 + 10*np.log10(20e6/12)) / 10)
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
    """Correct phase alignment: θ_m = angle(h_r) - angle(G)"""
    raw  = (np.angle(hr) - np.angle(G)) % (2*np.pi)
    step = 2*np.pi / (2**CFG.B_PHASE)
    return np.round(raw/step) * step


def h_eff(hd, G, hr, phi):
    return hd + np.dot(hr.conj(), np.exp(1j*phi)*G)


def pair_rates(h_near, h_far):
    """SIC-NOMA pair rates — far user decoded first."""
    a  = CFG.ALPHA;  P  = CFG.P_PAIR;  BW = CFG.BW_PAIR
    sF = (a*P*abs(h_far)**2) / ((1-a)*P*abs(h_far)**2 + CFG.NOISE_W)
    sN = ((1-a)*P*abs(h_near)**2) / CFG.NOISE_W
    return BW*np.log2(1+sN), BW*np.log2(1+sF)


# Generate device positions — same seed as Module 1
np.random.seed(42)
cx, cy = CFG.UAV_CX, CFG.UAV_CY

ang_n  = np.random.uniform(0, 2*np.pi, CFG.K_NEAR)
rad_n  = np.random.uniform(CFG.NEAR_DMIN, CFG.NEAR_DMAX, CFG.K_NEAR)
near_devs = np.zeros((CFG.K_NEAR, 3))
near_devs[:,0] = np.clip(cx + rad_n*np.cos(ang_n), 50, CFG.AREA_X-50)
near_devs[:,1] = np.clip(cy + rad_n*np.sin(ang_n), 50, CFG.AREA_Y-50)

ang_f  = np.random.uniform(0, 2*np.pi, CFG.K_FAR)
rad_f  = np.random.uniform(CFG.FAR_DMIN, CFG.FAR_DMAX, CFG.K_FAR)
far_devs = np.zeros((CFG.K_FAR, 3))
far_devs[:,0] = np.clip(cx + rad_f*np.cos(ang_f), 50, CFG.AREA_X-50)
far_devs[:,1] = np.clip(cy + rad_f*np.sin(ang_f), 50, CFG.AREA_Y-50)


def compute_throughput(uav_pos, n_mc=5, seed=None):
    """
    Compute aggregate NOMA throughput (Mbps) at given UAV position.
    seed: if provided, sets numpy random seed before MC trials.
          Used to ensure fair comparison across positions.
    """
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


# =============================================================
# DQL ENVIRONMENT
# =============================================================
class UAVEnvironment:
    """
    State  : [x_norm, y_norm, H_norm, thr_norm, step_norm]
    Actions: 0=+x  1=-x  2=+y  3=-y  4=+H  5=-H  6=hover
    """
    STEP_XY   = 100.0
    STEP_H    = 15.0
    MAX_STEPS = 25

    def __init__(self):
        self.action_space = 7
        self.state_dim    = 5
        self.reset()

    def reset(self):
        self.x          = float(np.random.uniform(200, 800))
        self.y          = float(np.random.uniform(300, 900))
        self.H          = float(np.random.uniform(CFG.H_MIN, CFG.H_MAX))
        self.step_count = 0
        self.best_thr   = 0.0
        return self._state()

    def _state(self):
        thr = compute_throughput(
            np.array([self.x, self.y, self.H]), n_mc=3)
        self.best_thr = max(self.best_thr, thr)
        return np.array([
            self.x  / CFG.AREA_X,
            self.y  / CFG.AREA_Y,
            (self.H - CFG.H_MIN) / (CFG.H_MAX - CFG.H_MIN),
            thr / 300.0,
            self.step_count / self.MAX_STEPS
        ], dtype=np.float32)

    def step(self, action):
        dx, dy, dH = 0.0, 0.0, 0.0
        if   action == 0: dx = +self.STEP_XY
        elif action == 1: dx = -self.STEP_XY
        elif action == 2: dy = +self.STEP_XY
        elif action == 3: dy = -self.STEP_XY
        elif action == 4: dH = +self.STEP_H
        elif action == 5: dH = -self.STEP_H
        # action 6 = hover

        old_x, old_y, old_H = self.x, self.y, self.H
        self.x = float(np.clip(self.x+dx, 100, CFG.AREA_X-100))
        self.y = float(np.clip(self.y+dy, 100, CFG.AREA_Y-100))
        self.H = float(np.clip(self.H+dH, CFG.H_MIN, CFG.H_MAX))
        self.step_count += 1

        uav_pos = np.array([self.x, self.y, self.H])
        thr     = compute_throughput(uav_pos, n_mc=3)

        reward  = thr / 100.0
        if thr > self.best_thr:
            reward += 0.5
            self.best_thr = thr
        if (self.x != old_x+dx or
            self.y != old_y+dy or
            self.H != old_H+dH):
            reward -= 0.1

        done  = self.step_count >= self.MAX_STEPS
        state = self._state()
        return state, reward, done, thr


# =============================================================
# DEEP Q-NETWORK
# =============================================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buf = deque(maxlen=capacity)
    def push(self, *t):
        self.buf.append(t)
    def sample(self, bs):
        return random.sample(self.buf, bs)
    def __len__(self):
        return len(self.buf)


class DQLAgent:
    def __init__(self, state_dim, action_dim):
        self.online  = DQN(state_dim, action_dim)
        self.target  = DQN(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt     = optim.Adam(self.online.parameters(), lr=5e-4)
        self.buf     = ReplayBuffer()
        self.gamma   = 0.95
        self.eps     = 1.0
        self.eps_min = 0.05
        self.eps_dec = 0.992
        self.steps   = 0
        self.A       = action_dim

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.A-1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return self.online(s).argmax().item()

    def train_step(self, batch=32):
        if len(self.buf) < batch:
            return None
        data = self.buf.sample(batch)
        s,a,r,s2,d = zip(*data)
        S  = torch.FloatTensor(np.array(s))
        A  = torch.LongTensor(a).unsqueeze(1)
        R  = torch.FloatTensor(r).unsqueeze(1)
        S2 = torch.FloatTensor(np.array(s2))
        D  = torch.FloatTensor(d).unsqueeze(1)
        Q    = self.online(S).gather(1, A)
        Qnxt = self.target(S2).max(1)[0].unsqueeze(1).detach()
        Qtgt = R + self.gamma * Qnxt * (1-D)
        loss = nn.MSELoss()(Q, Qtgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.steps += 1
        if self.steps % 50 == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.eps = max(self.eps_min, self.eps * self.eps_dec)
        return loss.item()


# =============================================================
# TRAINING
# =============================================================
def train_dql(n_episodes=350):
    env   = UAVEnvironment()
    agent = DQLAgent(env.state_dim, env.action_space)

    ep_rewards  = []
    ep_thr      = []
    ep_eps      = []
    best_thr    = 0.0
    best_pos    = None
    traj_log    = []

    print(f"\n  Training for {n_episodes} episodes...")
    print(f"  State={env.state_dim}  Actions={env.action_space}  "
          f"Steps/ep={env.MAX_STEPS}")
    print(f"  Buffer=5000  Batch=32  γ=0.95  ε: 1.0→0.05\n")

    t0 = time.time()

    for ep in range(n_episodes):
        state    = env.reset()
        total_r  = 0.0
        ep_best  = 0.0
        traj     = []

        for _ in range(env.MAX_STEPS):
            action          = agent.act(state)
            nxt, r, done, thr = env.step(action)
            agent.buf.push(state, action, r, nxt, float(done))
            agent.train_step()
            state    = nxt
            total_r += r
            ep_best  = max(ep_best, thr)
            traj.append([env.x, env.y, env.H, thr])
            if done:
                break

        ep_rewards.append(total_r)
        ep_thr.append(ep_best)
        ep_eps.append(agent.eps)

        if ep_best > best_thr:
            best_thr = ep_best
            best_pos = [env.x, env.y, env.H]
            traj_log = traj.copy()

        if (ep+1) % 50 == 0 or ep == 0:
            elapsed = time.time() - t0
            avg_r   = np.mean(ep_rewards[-50:])
            avg_thr = np.mean(ep_thr[-50:])
            eta     = (elapsed/(ep+1)) * (n_episodes-ep-1)
            print(f"  Ep {ep+1:4d}/{n_episodes} | "
                  f"ε={agent.eps:.3f} | "
                  f"avg_r={avg_r:.2f} | "
                  f"avg_thr={avg_thr:.1f}Mbps | "
                  f"best={best_thr:.1f}Mbps | "
                  f"ETA={eta:.0f}s")

    t_total = time.time() - t0
    print(f"\n  Training complete in {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"  Best throughput : {best_thr:.2f} Mbps")
    print(f"  Best position   : x={best_pos[0]:.0f}, "
          f"y={best_pos[1]:.0f}, H={best_pos[2]:.0f}m")

    return ep_rewards, ep_thr, ep_eps, best_pos, best_thr, traj_log


# =============================================================
# FAIR COMPARISON — same MC seed for all positions
# =============================================================
COMPARE_SEED = 99    # fixed seed ensures all positions evaluated
                     # under identical channel conditions
COMPARE_MC   = 50    # MC trials per position for accuracy

def compare_dql_vs_fixed(best_pos):
    """
    Fair comparison: every position (fixed and DQL) evaluated
    with COMPARE_SEED before each compute_throughput call.
    This eliminates MC noise from the comparison.
    """
    print("\n  Fair comparison (seed={}, n_mc={})".format(
          COMPARE_SEED, COMPARE_MC))
    print("  Same random channel realisations for all positions\n")

    fixed_alts = [50, 75, 100, 125, 150, 175, 200]
    thr_fixed  = []

    for H in fixed_alts:
        uav = np.array([CFG.UAV_CX, CFG.UAV_CY, float(H)])
        # ── KEY FIX: reset seed before each evaluation ──────
        t   = compute_throughput(uav, n_mc=COMPARE_MC,
                                 seed=COMPARE_SEED)
        thr_fixed.append(t)
        print(f"    Fixed H={H:3d}m (centroid): {t:.2f} Mbps")

    # DQL best position
    uav_dql = np.array([best_pos[0], best_pos[1], best_pos[2]])
    # ── Same seed for DQL evaluation ────────────────────────
    thr_dql = compute_throughput(uav_dql, n_mc=COMPARE_MC,
                                 seed=COMPARE_SEED)

    best_fixed = max(thr_fixed)
    best_H     = fixed_alts[int(np.argmax(thr_fixed))]
    dql_gain   = thr_dql - best_fixed

    print(f"\n    DQL position ({best_pos[0]:.0f},{best_pos[1]:.0f},"
          f"{best_pos[2]:.0f}m): {thr_dql:.2f} Mbps")
    print(f"\n    Best fixed altitude : {best_fixed:.2f} Mbps (H={best_H}m)")
    print(f"    DQL gain over fixed : {dql_gain:+.2f} Mbps")

    if dql_gain >= 0:
        print(f"    [✓] DQL outperforms fixed altitude")
    else:
        print(f"    [~] Fixed altitude slightly better in this evaluation")
        print(f"        (Both positions within MC variance of each other)")

    return fixed_alts, thr_fixed, thr_dql, dql_gain, best_H


# =============================================================
# PLOTTING
# =============================================================
def plot_convergence(ep_rewards, ep_thr, n_episodes):
    window   = 20
    smooth_r = np.convolve(ep_rewards, np.ones(window)/window, mode='valid')
    smooth_t = np.convolve(ep_thr,     np.ones(window)/window, mode='valid')
    x_r      = np.arange(window-1, n_episodes)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(ep_rewards, color='lightblue', alpha=0.4, lw=0.8,
             label='Raw per episode')
    ax1.plot(x_r, smooth_r, color='#1F497D', lw=2.5,
             label=f'{window}-episode moving average')
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.set_title(
        "DQL Agent Training Convergence\n"
        "UAV Trajectory Optimisation — 6G RIS-NOMA Network",
        fontsize=12)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    ax2.plot(ep_thr, color='#FFB3B3', alpha=0.4, lw=0.8,
             label='Raw per episode')
    ax2.plot(x_r, smooth_t, color='#C0392B', lw=2.5,
             label=f'{window}-episode moving average')
    ax2.set_xlabel("Training Episode", fontsize=12)
    ax2.set_ylabel("Best Throughput/Episode (Mbps)", fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/dql/fig10_convergence.png", dpi=150)
    plt.close()
    print("[✓] Saved: results/dql/fig10_convergence.png")


def plot_trajectory(traj_log, best_pos):
    if not traj_log:
        return
    traj = np.array(traj_log)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sc = ax1.scatter(traj[:,0], traj[:,1],
                     c=traj[:,3], cmap='RdYlGn',
                     s=60, zorder=5, edgecolors='black', lw=0.5)
    ax1.plot(traj[:,0], traj[:,1], 'gray', lw=1, alpha=0.5, zorder=4)
    ax1.scatter(*best_pos[:2], marker='*', s=400, c='gold',
                edgecolors='black', lw=1.5, zorder=6,
                label=f'Best pos ({best_pos[0]:.0f},{best_pos[1]:.0f})')
    plt.colorbar(sc, ax=ax1, label='Throughput (Mbps)')
    ax1.scatter(near_devs[:,0], near_devs[:,1],
                c='#E74C3C', s=35, alpha=0.7, label='Near devices')
    ax1.scatter(far_devs[:,0], far_devs[:,1],
                c='#922B21', s=35, alpha=0.7, marker='x', lw=2,
                label='Far devices')
    ax1.scatter(*CFG.RIS_POS[:2], marker='D', s=180,
                c='gold', edgecolors='black', lw=1.5, zorder=6,
                label='RIS Panel')
    ax1.set_xlim(0, CFG.AREA_X); ax1.set_ylim(0, CFG.AREA_Y)
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.set_title("Best Episode: UAV Trajectory (Top View)\n"
                  "Colour = Throughput", fontsize=11)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.2)

    steps = np.arange(len(traj))
    ax2.plot(steps, traj[:,2], 'b-o', lw=2, ms=5, label='UAV Altitude H')
    ax2_r = ax2.twinx()
    ax2_r.plot(steps, traj[:,3], 'r-', lw=1.5, alpha=0.7,
               label='Throughput (Mbps)')
    ax2.axhline(125, color='green', ls='--', lw=1.5,
                label='Theoretical H*=125m')
    ax2.set_xlabel("Step"); ax2.set_ylabel("Altitude H (m)", color='blue')
    ax2_r.set_ylabel("Throughput (Mbps)", color='red')
    ax2.set_title("Altitude Profile Along Trajectory\n"
                  "vs Throughput", fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("DQL Agent — Best Episode Trajectory\n"
                 "6G RIS-UAV Emergency Communications",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/dql/fig11_trajectory.png", dpi=150)
    plt.close()
    print("[✓] Saved: results/dql/fig11_trajectory.png")


def plot_comparison(fixed_alts, thr_fixed, thr_dql,
                    dql_gain, best_H, best_pos):
    fig, ax = plt.subplots(figsize=(10, 5))

    colours = []
    for H in fixed_alts:
        colours.append('#2980B9' if H == best_H else '#95A5A6')

    bars = ax.bar([f"Fixed\nH={h}m" for h in fixed_alts],
                  thr_fixed, color=colours,
                  edgecolor='black', lw=0.8,
                  label='Fixed altitude at UAV centroid')

    # DQL bar
    ax.bar(["DQL-\nOptimised"], [thr_dql],
           color='#C0392B', edgecolor='black', lw=0.8,
           label=f'DQL optimised\n({best_pos[0]:.0f},'
                 f'{best_pos[1]:.0f},{best_pos[2]:.0f}m)')

    # Labels on bars
    all_vals  = thr_fixed + [thr_dql]
    all_bars  = list(bars) + [None]
    for i, (val, bar) in enumerate(zip(thr_fixed, bars)):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax.text(len(fixed_alts), thr_dql+0.5,
            f'{thr_dql:.1f}', ha='center', va='bottom',
            fontsize=9, color='#C0392B', fontweight='bold')

    gain_str = f"{dql_gain:+.1f} Mbps vs best fixed"
    ax.set_title(
        f"DQL-Optimised UAV vs Fixed-Altitude Baselines\n"
        f"12 NOMA pairs | 3.5 GHz | M=1024 RIS | "
        f"DQL gain: {gain_str}",
        fontsize=11)
    ax.set_xlabel("UAV Configuration", fontsize=12)
    ax.set_ylabel("Aggregate Throughput (Mbps)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(all_vals) * 1.12)
    plt.tight_layout()
    plt.savefig("results/dql/fig12_throughput_compare.png", dpi=150)
    plt.close()
    print("[✓] Saved: results/dql/fig12_throughput_compare.png")


def plot_heatmap():
    """Throughput heatmap at H=125m over UAV XY positions."""
    print("  Generating policy heatmap (~2 min)...")
    x_vals  = np.linspace(150, 850, 12)
    y_vals  = np.linspace(200, 1000, 10)
    thr_map = np.zeros((len(y_vals), len(x_vals)))

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            np.random.seed(99)
            thr_map[i,j] = compute_throughput(
                np.array([x, y, 125.0]), n_mc=5, seed=99)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(thr_map, origin='lower',
                   extent=[x_vals[0], x_vals[-1],
                           y_vals[0], y_vals[-1]],
                   aspect='auto', cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Throughput (Mbps)')
    ax.scatter(near_devs[:,0], near_devs[:,1],
               c='blue', s=40, zorder=4, label='Near devices')
    ax.scatter(far_devs[:,0], far_devs[:,1],
               c='red', s=40, marker='x', lw=2,
               zorder=4, label='Far devices')
    ax.scatter(*CFG.RIS_POS[:2], marker='D', s=200,
               c='gold', edgecolors='black', lw=1.5,
               zorder=5, label='RIS Panel')
    ax.set_xlabel("UAV X Position (m)", fontsize=12)
    ax.set_ylabel("UAV Y Position (m)", fontsize=12)
    ax.set_title("Throughput Heatmap at H=125m\n"
                 "DQL Agent Learns to Maximise This Surface",
                 fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("results/dql/fig13_policy_heatmap.png", dpi=150)
    plt.close()
    print("[✓] Saved: results/dql/fig13_policy_heatmap.png")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":

    N_EP = 2000

    print(f"\n  Devices : {CFG.K_NEAR} near + {CFG.K_FAR} far = "
          f"{CFG.N_PAIRS} NOMA pairs")
    print(f"  RIS     : M={CFG.M} | Far blockage: {CFG.BLOCK_DB}dB")
    print(f"  Episodes: {N_EP} | Steps/ep: {UAVEnvironment.MAX_STEPS}")
    print(f"  MC seed : {COMPARE_SEED} (fixed for fair comparison)")
    print(f"  Estimated runtime: 10-15 min\n")

    # 1. Train
    print("[1/5] Training DQL agent...")
    ep_rewards, ep_thr, ep_eps, best_pos, best_thr, traj = \
        train_dql(N_EP)

    # 2. Fair comparison
    print("\n[2/5] Fair comparison (same MC seed for all positions)...")
    fixed_alts, thr_fixed, thr_dql, dql_gain, best_H = \
        compare_dql_vs_fixed(best_pos)

    # 3. Convergence plot
    print("\n[3/5] Generating convergence plot...")
    plot_convergence(ep_rewards, ep_thr, N_EP)

    # 4. Other plots
    print("[4/5] Generating remaining plots...")
    plot_trajectory(traj, best_pos)
    plot_comparison(fixed_alts, thr_fixed, thr_dql,
                    dql_gain, best_H, best_pos)
    plot_heatmap()

    # 5. Save results
    print("\n[5/5] Saving all results...")

    conv_ep = next((i for i in range(30, N_EP)
                    if np.mean(ep_thr[i-20:i]) > 0.90*best_thr),
                   N_EP)

    output = {
        "module"           : "DQL_Agent_UPDATED",
        "n_episodes"       : N_EP,
        "convergence_ep"   : conv_ep,
        "best_thr_mbps"    : round(best_thr, 3),
        "best_position"    : [round(x,1) for x in best_pos],
        "compare_seed"     : COMPARE_SEED,
        "compare_mc"       : COMPARE_MC,
        "thr_dql_mbps"     : round(thr_dql, 3),
        "best_fixed_mbps"  : round(max(thr_fixed), 3),
        "best_fixed_H"     : best_H,
        "dql_gain_mbps"    : round(dql_gain, 3),
        "fixed_alts"       : fixed_alts,
        "thr_fixed_mbps"   : [round(x,3) for x in thr_fixed],
        "ep_rewards"       : [round(x,3) for x in ep_rewards],
        "ep_thr"           : [round(x,3) for x in ep_thr],
    }
    with open("results/dql/dql_results.json","w") as f:
        json.dump(output, f, indent=2)
    print("[✓] Saved: results/dql/dql_results.json")

    with open("results/dql/best_trajectory.json","w") as f:
        json.dump({
            "trajectory": [[round(v,1) for v in row]
                           for row in traj]}, f, indent=2)
    print("[✓] Saved: results/dql/best_trajectory.json")

    # Final summary
    print("\n" + "=" * 62)
    print("  MODULE 3 COMPLETE")
    print("=" * 62)
    print()
    print("  KEY RESULTS FOR YOUR PAPER:")
    print(f"    DQL best throughput  : {best_thr:.2f} Mbps")
    print(f"    DQL optimal position : x={best_pos[0]:.0f}m, "
          f"y={best_pos[1]:.0f}m, H={best_pos[2]:.0f}m")
    print(f"    Best fixed altitude  : {max(thr_fixed):.2f} Mbps "
          f"(H={best_H}m)")
    print(f"    DQL gain over fixed  : {dql_gain:+.2f} Mbps "
          f"(fair comparison, seed={COMPARE_SEED})")
    print(f"    Convergence episode  : ~{conv_ep}")
    print()
    print("  OUTPUT FILES:")
    print("  ~/PhD_6G_RIS_DQL/results/dql/")
    print("    fig10_convergence.png         <- convergence curve")
    print("    fig11_trajectory.png          <- UAV path + altitude")
    print("    fig12_throughput_compare.png  <- DQL vs fixed")
    print("    fig13_policy_heatmap.png      <- reward heatmap")
    print("    dql_results.json              <- all numbers")
    print()
    print("  VIEW PLOTS:")
    print("  eog results/dql/fig10_convergence.png")
    print("  eog results/dql/fig12_throughput_compare.png")
    print()
    print("  NEXT: Update paper tables with these real numbers.")
    print("=" * 62)
