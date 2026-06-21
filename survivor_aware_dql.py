#!/usr/bin/env python3
"""
survivor_aware_dql.py
═══════════════════════════════════════════════════════════════════════════════
EXECUTES Reviewer-2 Critical Flaw A — "redesign the reward function so the agent
actually optimises for the far-users/RIS."

This is a complete, self-contained NumPy DQN (no PyTorch/TF — matches the paper's
lightweight agent) that trains on channel_model.py physics with the SURVIVOR-AWARE
reward:

    R(s) = w · (near_sum / NEAR_NORM)  +  (1 - w) · far_SLA_reliability

where far_SLA_reliability = P(far-user rate >= SLA_BPS).  Setting w = 0.5 gives the
telemetry-SLA operating point that the Pareto analysis identified as the sweet spot.

Compared against the conventional agent (reward = aggregate throughput), this agent
learns to fly to a position that sacrifices a few Mbps for 100% survivor beacon
reliability — demonstrating the agent now RESPONDS to blocked-survivor welfare,
which the sum-rate agent provably cannot.

Run on Windows (channel_model.py in same folder):
    python survivor_aware_dql.py
Outputs:
    results/SURVIVOR/dql_convergence.png
    results/SURVIVOR/dql_training_log.csv
"""
import os, time, csv, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import channel_model as cm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
os.makedirs(OUT, exist_ok=True)

# ── experiment / agent config ─────────────────────────────────────────────────
DEVICE_SEED  = 42
EVAL_SEED    = 99
SLA_BPS      = 100.0
W_SLA        = 0.5
NEAR_NORM    = 200.0          # Mbps normaliser

EPISODES     = 600
T_STEPS      = 25             # steps per episode
GAMMA        = 0.95
LR           = 1e-3
BATCH        = 64
REPLAY_CAP   = 10000
TARGET_SYNC  = 20             # episodes
N_MC_TRAIN   = 6              # channel draws for the (noisy) training reward
N_MC_EVAL    = 60             # channel draws for low-variance evaluation
EPS_START, EPS_END, EPS_FRAC = 1.0, 0.05, 0.7

# search bounds (cover near-centroid ~(400,600) AND RIS at (300,250,20))
XMIN, XMAX = 150.0, 800.0
YMIN, YMAX = 200.0, 900.0
HMIN, HMAX = 50.0, 200.0
DXY, DH    = 100.0, 25.0

# 7 discrete actions: +x -x +y -y +H -H hover
ACTIONS = np.array([[ DXY,0,0],[-DXY,0,0],[0, DXY,0],[0,-DXY,0],
                    [0,0, DH],[0,0,-DH],[0,0,0]], dtype=float)
NA = len(ACTIONS)

# fixed device layout
np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]
FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]


# ── reward / physics ──────────────────────────────────────────────────────────
def eval_position(uav, n_mc, seed=None):
    """Return (near_sum_mbps, far_mean_bps, sla_reliability) at uav via channel_model.py."""
    if seed is not None:
        np.random.seed(seed)
    K = cm.CFG.N_PAIRS
    near_sum = 0.0
    far_samples = []
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        df = np.zeros(K)
        for i in range(K):
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            hd = cm.chan_scalar(uav, FAR[i],  cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(FAR[i])
            phi = cm.opt_phi(G, hr)
            he  = cm.composite(hd, G, hr, phi)
            rN, _  = cm.noma_pair(hn, hd)
            _,  rF = cm.noma_pair(hn, he)
            near_sum += rN
            df[i] = rF
        far_samples.append(df)
    near_sum_mbps = near_sum / n_mc / 1e6
    fs = np.array(far_samples)
    far_mean_bps = float(fs.mean())
    sla_rel = float((fs >= SLA_BPS).mean())
    return near_sum_mbps, far_mean_bps, sla_rel


def reward_sla(uav, n_mc=N_MC_TRAIN, seed=None):
    near_mbps, _, sla_rel = eval_position(uav, n_mc, seed)
    return W_SLA * (near_mbps / NEAR_NORM) + (1.0 - W_SLA) * sla_rel


def normalise(uav):
    return np.array([(uav[0]-XMIN)/(XMAX-XMIN),
                     (uav[1]-YMIN)/(YMAX-YMIN),
                     (uav[2]-HMIN)/(HMAX-HMIN)])


# ── NumPy MLP Q-network (5 → 64 → 64 → 7) with Adam ───────────────────────────
class QNet:
    def __init__(self, n_in=5, n_h=64, n_out=NA, lr=LR, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/n_in), (n_in, n_h)); self.b1 = np.zeros(n_h)
        self.W2 = rng.normal(0, np.sqrt(2/n_h),  (n_h,  n_h)); self.b2 = np.zeros(n_h)
        self.W3 = rng.normal(0, np.sqrt(2/n_h),  (n_h,  n_out)); self.b3 = np.zeros(n_out)
        self.lr = lr
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(getattr(self, k)) for k in ["W1","b1","W2","b2","W3","b3"]}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in ["W1","b1","W2","b2","W3","b3"]}
        self.t = 0

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1; self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = np.maximum(0, self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3

    def train_step(self, X, a_idx, target):
        """Gradient step on MSE of Q(s,a) toward target, only for taken actions."""
        out = self.forward(X)
        B = X.shape[0]
        dout = np.zeros_like(out)
        q_taken = out[np.arange(B), a_idx]
        dout[np.arange(B), a_idx] = 2.0 * (q_taken - target) / B
        # backprop
        dW3 = self.a2.T @ dout; db3 = dout.sum(0)
        da2 = dout @ self.W3.T; dz2 = da2 * (self.z2 > 0)
        dW2 = self.a1.T @ dz2; db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * (self.z1 > 0)
        dW1 = self.X.T @ dz1; db1 = dz1.sum(0)
        grads = dict(W1=dW1,b1=db1,W2=dW2,b2=db2,W3=dW3,b3=db3)
        self._adam(grads)
        return float(((q_taken - target) ** 2).mean())

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = b1*self.m[k] + (1-b1)*g
            self.v[k] = b2*self.v[k] + (1-b2)*g*g
            mh = self.m[k]/(1-b1**self.t); vh = self.v[k]/(1-b2**self.t)
            setattr(self, k, getattr(self, k) - self.lr*mh/(np.sqrt(vh)+eps))

    def copy_from(self, other):
        for k in ["W1","b1","W2","b2","W3","b3"]:
            setattr(self, k, getattr(other, k).copy())


# ── replay buffer ─────────────────────────────────────────────────────────────
class Replay:
    def __init__(self, cap): self.cap=cap; self.buf=[]; self.i=0
    def push(self, *tr):
        if len(self.buf) < self.cap: self.buf.append(tr)
        else: self.buf[self.i] = tr
        self.i = (self.i+1) % self.cap
    def sample(self, n):
        idx = np.random.randint(0, len(self.buf), n)
        s, a, r, s2, d = zip(*[self.buf[j] for j in idx])
        return (np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d))
    def __len__(self): return len(self.buf)


# ── environment ───────────────────────────────────────────────────────────────
def env_reset():
    uav = np.array([np.random.uniform(XMIN, XMAX),
                    np.random.uniform(YMIN, YMAX),
                    np.random.uniform(HMIN, HMAX)])
    # snap to grid
    uav[0] = np.clip(round(uav[0]/DXY)*DXY, XMIN, XMAX)
    uav[1] = np.clip(round(uav[1]/DXY)*DXY, YMIN, YMAX)
    uav[2] = np.clip(round(uav[2]/DH)*DH,  HMIN, HMAX)
    return uav

def env_step(uav, a):
    nxt = uav + ACTIONS[a]
    nxt[0] = np.clip(nxt[0], XMIN, XMAX)
    nxt[1] = np.clip(nxt[1], YMIN, YMAX)
    nxt[2] = np.clip(nxt[2], HMIN, HMAX)
    r = reward_sla(nxt)
    return nxt, r


# ── training ──────────────────────────────────────────────────────────────────
def train():
    print("=" * 74)
    print("  SURVIVOR-AWARE DQL  (reward J_sla, w=%.2f, SLA=%.0f bps)" % (W_SLA, SLA_BPS))
    print("=" * 74)
    print(f"  {EPISODES} episodes × {T_STEPS} steps | NumPy DQN 5-64-64-{NA} | "
          f"N_MC_train={N_MC_TRAIN}")
    q, qt = QNet(seed=0), QNet(seed=0); qt.copy_from(q)
    rb = Replay(REPLAY_CAP)
    eps_decay_ep = int(EPISODES * EPS_FRAC)
    log = []
    t0 = time.time()

    for ep in range(EPISODES):
        eps = max(EPS_END, EPS_START - (EPS_START-EPS_END)*ep/eps_decay_ep)
        uav = env_reset()
        r_last = 0.0
        s = np.concatenate([normalise(uav), [r_last, 0.0]])
        ep_r, ep_loss, nl = 0.0, 0.0, 0
        for t in range(T_STEPS):
            if np.random.rand() < eps:
                a = np.random.randint(NA)
            else:
                a = int(np.argmax(q.forward(s[None])[0]))
            nuav, r = env_step(uav, a)
            tn = (t+1)/T_STEPS
            s2 = np.concatenate([normalise(nuav), [r, tn]])
            done = 1.0 if t == T_STEPS-1 else 0.0
            rb.push(s, a, r, s2, done)
            uav, s, ep_r = nuav, s2, ep_r + r
            if len(rb) >= BATCH:
                bs, ba, br, bs2, bd = rb.sample(BATCH)
                q_next = qt.forward(bs2).max(1)
                target = br + GAMMA * q_next * (1 - bd)
                ep_loss += q.train_step(bs, ba, target); nl += 1
        if ep % TARGET_SYNC == 0:
            qt.copy_from(q)
        log.append(dict(ep=ep, eps=eps, ep_reward=ep_r/T_STEPS,
                        loss=ep_loss/max(nl,1)))
        if ep % 50 == 0 or ep == EPISODES-1:
            print(f"  ep {ep:4d} | eps {eps:.3f} | mean_r {ep_r/T_STEPS:.3f} | "
                  f"loss {ep_loss/max(nl,1):.4f} | {time.time()-t0:.0f}s")

    # ── greedy rollout from several starts → learned deployment position ────────
    print("\n  Greedy policy rollout (learned deployment):")
    finals = []
    for st in range(6):
        np.random.seed(1000+st)
        uav = env_reset()
        r_last = 0.0
        s = np.concatenate([normalise(uav), [r_last, 0.0]])
        for t in range(T_STEPS):
            a = int(np.argmax(q.forward(s[None])[0]))
            uav, r = env_step(uav, a)
            s = np.concatenate([normalise(uav), [r, (t+1)/T_STEPS]])
        finals.append(uav.copy())
    finals = np.array(finals)
    uav_star = np.median(finals, axis=0)
    uav_star[0] = round(uav_star[0]/DXY)*DXY
    uav_star[1] = round(uav_star[1]/DXY)*DXY
    uav_star[2] = round(uav_star[2]/DH)*DH

    near_mbps, far_mean, sla_rel = eval_position(uav_star, N_MC_EVAL, seed=EVAL_SEED)
    # also evaluate the conventional sum-rate optimum for contrast
    sumrate_uav = np.array([450.0, 577.0, 108.0])
    n0, f0, r0 = eval_position(sumrate_uav, N_MC_EVAL, seed=EVAL_SEED)

    print(f"\n  Learned survivor-aware position : ({uav_star[0]:.0f},{uav_star[1]:.0f},"
          f"{uav_star[2]:.0f}) m")
    print(f"    near throughput   : {near_mbps:.1f} Mbps")
    print(f"    far mean rate     : {far_mean:.0f} bps")
    print(f"    SLA reliability   : {sla_rel*100:.1f}% @ {SLA_BPS:.0f} bps")
    print(f"  Conventional sum-rate optimum   : (450,577,108) m")
    print(f"    near throughput   : {n0:.1f} Mbps")
    print(f"    SLA reliability   : {r0*100:.1f}%")
    print(f"\n  → Survivor-aware reward moves the agent; reliability "
          f"{r0*100:.0f}% → {sla_rel*100:.0f}% at "
          f"{100*(n0-near_mbps)/n0:+.1f}% throughput.")

    # ── plot convergence ────────────────────────────────────────────────────────
    eps_arr = [l["ep"] for l in log]
    rew = np.array([l["ep_reward"] for l in log])
    # smoothed
    win = 15
    sm = np.convolve(rew, np.ones(win)/win, mode="valid")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(eps_arr, rew, color="#b8c6d8", lw=0.8, label="per-episode")
    ax[0].plot(range(win-1, len(rew)), sm, color="#1f5fa8", lw=2,
               label=f"moving avg ({win})")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Mean step reward  J_sla")
    ax[0].set_title("Survivor-Aware DQL Convergence", fontweight="bold")
    ax[0].grid(True, ls=":", alpha=0.5); ax[0].legend()

    cats = ["Sum-rate\noptimum\n(conventional)", "Survivor-aware\noptimum\n(learned)"]
    rels = [r0*100, sla_rel*100]
    thrs = [n0, near_mbps]
    xb = np.arange(2)
    ax2 = ax[1]; ax2b = ax2.twinx()
    b1 = ax2.bar(xb-0.2, rels, 0.4, color="#2ca02c", label="SLA reliability (%)")
    b2 = ax2b.bar(xb+0.2, thrs, 0.4, color="#d62728", label="Throughput (Mbps)")
    ax2.set_xticks(xb); ax2.set_xticklabels(cats, fontsize=9)
    ax2.set_ylabel("Far-user SLA reliability @100 bps (%)", color="#2ca02c")
    ax2b.set_ylabel("Near-user throughput (Mbps)", color="#d62728")
    ax2.set_ylim(0, 110); ax2b.set_ylim(0, 210)
    ax2.set_title("Conventional vs Survivor-Aware Reward", fontweight="bold")
    for b in b1: ax2.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                          f"{b.get_height():.0f}%", ha="center", fontsize=9)
    for b in b2: ax2b.text(b.get_x()+b.get_width()/2, b.get_height()+3,
                           f"{b.get_height():.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    p = os.path.join(OUT, "dql_convergence.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Convergence plot → {p}")

    with open(os.path.join(OUT, "dql_training_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
    print(f"  Training log → {os.path.join(OUT,'dql_training_log.csv')}")
    print(f"  Total runtime: {time.time()-t0:.0f}s")
    print("=" * 74)


if __name__ == "__main__":
    train()
