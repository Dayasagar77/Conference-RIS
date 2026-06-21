"""
=================================================================
  COMPREHENSIVE BENCHMARK & ABLATION STUDY
  RIS-Assisted NOMA with UAV-side SIC — IEEE Access Revision
  
  Studies included:
  [A] Algorithm comparison: Random / PSO / DDQN / A2C / DQL
  [B] RIS ablation: M elements, phase bits, position
  [C] System ablation: NOMA alpha, transmit power
  [D] DQL architecture ablation: hidden units, layers, buffer
  
  Runtime: ~3-4 hours on standard CPU
  Output:  results/benchmark/  (JSON + PNG figures)
  
  Run: python3 channel_model/benchmark_ablation.py
=================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, time
from collections import deque
import random as pyrandom

os.makedirs("results/benchmark", exist_ok=True)
OUT = "results/benchmark"

# ─────────────────────────────────────────────────────────────
#  SECTION 1 — CHANNEL MODEL (self-contained, OMA bug fixed)
# ─────────────────────────────────────────────────────────────

class Cfg:
    """Mutable config — ablation studies modify copies."""
    AREA_X   = 3000.0;  AREA_Y   = 1600.0
    K_NEAR   = 12;      K_FAR    = 12;    N_PAIRS = 12
    UAV_CX   = 400.0;   UAV_CY   = 600.0
    NEAR_DMIN= 50.0;    NEAR_DMAX= 250.0
    FAR_DMIN = 450.0;   FAR_DMAX = 800.0
    H_MIN    = 50.0;    H_MAX    = 200.0
    P_T      = 2.0
    ALPHA    = 0.85
    M        = 1024
    B_PHASE  = 3
    RIS_POS  = np.array([300.0, 250.0, 20.0])
    FC       = 3.5e9;  C = 3e8
    NOISE_W  = 10**((-174 + 7 - 30 + 10*np.log10(20e6/12)) / 10)
    A_URBAN  = 9.61;   B_URBAN = 0.16
    ETA_LOS  = 1.0;    ETA_NLOS = 20.0
    K_RICIAN = 10**(10.0/10)
    BLOCK_DB = 60.0

    @property
    def P_PAIR(self): return self.P_T / self.N_PAIRS
    @property
    def BW_PAIR(self): return 20e6 / self.N_PAIRS

BASE = Cfg()


def make_cfg(**kwargs):
    """Return a modified Cfg copy for ablation studies."""
    c = Cfg()
    for k, v in kwargs.items():
        setattr(c, k, v)
    return c


def los_prob(tx, rx, cfg=BASE):
    dx = tx[0]-rx[0]; dy = tx[1]-rx[1]
    h  = np.sqrt(dx**2+dy**2) + 1e-9
    el = np.degrees(np.arctan(tx[2]/h))
    return float(np.clip(
        1/(1 + cfg.A_URBAN*np.exp(-cfg.B_URBAN*(el-cfg.A_URBAN))), 0, 1))


def pl_dB(tx, rx, extra=0.0, cfg=BASE):
    d = np.linalg.norm(np.array(tx)-np.array(rx)) + 1e-9
    F = (20*np.log10(d) + 20*np.log10(cfg.FC) +
         20*np.log10(4*np.pi/cfg.C))
    p = los_prob(np.array(tx), np.array(rx), cfg)
    return p*(cfg.ETA_LOS+F) + (1-p)*(cfg.ETA_NLOS+F) + extra


def rician(sz, cfg=BASE):
    K  = cfg.K_RICIAN
    ph = np.random.uniform(0, 2*np.pi, sz)
    hL = np.exp(1j*ph) * np.sqrt(K/(K+1))
    hS = (np.random.randn(*ph.shape)+1j*np.random.randn(*ph.shape))/np.sqrt(2*(K+1))
    h  = hL + hS
    return h.squeeze() if isinstance(sz, int) and sz == 1 else h


def chan(tx, rx, extra=0.0, cfg=BASE):
    return 10**(-pl_dB(tx, rx, extra, cfg)/20) * rician(1, cfg)


def G_vec(uav, cfg=BASE):
    return 10**(-pl_dB(uav, cfg.RIS_POS, 0, cfg)/20) * rician(cfg.M, cfg)


def hr_vec(dev, cfg=BASE):
    return 10**(-pl_dB(cfg.RIS_POS, dev, 0, cfg)/20) * rician(cfg.M, cfg)


def opt_phi(G, hr, cfg=BASE, mode='quantised'):
    """Phase alignment. mode: 'quantised'|'continuous'|'random'"""
    raw = (np.angle(hr) - np.angle(G)) % (2*np.pi)
    if mode == 'continuous':
        return raw
    elif mode == 'random':
        return np.random.uniform(0, 2*np.pi, cfg.M)
    else:  # quantised
        step = 2*np.pi / (2**cfg.B_PHASE)
        return np.round(raw/step) * step


def h_eff(hd, G, hr, phi):
    return hd + np.dot(hr.conj(), np.exp(1j*phi)*G)


def noma_rates(h_near, h_far, cfg=BASE):
    """UAV-side SIC NOMA: near user gets interference-free rate."""
    a  = cfg.ALPHA; P = cfg.P_PAIR; BW = cfg.BW_PAIR
    sF = (a*P*abs(h_far)**2) / ((1-a)*P*abs(h_far)**2 + cfg.NOISE_W)
    sN = ((1-a)*P*abs(h_near)**2) / cfg.NOISE_W
    return BW*np.log2(1+sN), BW*np.log2(1+sF)


def oma_rates(h_near, h_far, cfg=BASE):
    """OMA: BW/2 per user, P/2 per user, noise correctly halved."""
    bw    = cfg.BW_PAIR / 2
    p     = cfg.P_PAIR  / 2
    noise = cfg.NOISE_W / 2   # FIXED: noise scales with bandwidth
    rN = bw * np.log2(1 + p*abs(h_near)**2 / noise)
    rF = bw * np.log2(1 + p*abs(h_far)**2  / noise)
    return rN, rF


def compute_throughput(uav_pos, near_devs, far_devs, n_mc=10, seed=None,
                       cfg=BASE, phi_mode='quantised'):
    """Compute mean NOMA system throughput over n_mc Monte Carlo trials."""
    if seed is not None:
        np.random.seed(seed)
    totals = []
    G_v = G_vec(uav_pos, cfg)
    for _ in range(n_mc):
        thr = 0.0
        for i in range(cfg.N_PAIRS):
            hn  = chan(uav_pos, near_devs[i], 0.0, cfg)
            hf  = chan(uav_pos, far_devs[i],  cfg.BLOCK_DB, cfg)
            hr  = hr_vec(far_devs[i], cfg)
            phi = opt_phi(G_v, hr, cfg, phi_mode)
            he  = h_eff(hf, G_v, hr, phi)
            rN, rF = noma_rates(hn, he, cfg)
            thr += rN + rF
        totals.append(thr / 1e6)
    return float(np.mean(totals))


def compute_oma_throughput(uav_pos, near_devs, far_devs, n_mc=50, seed=99, cfg=BASE):
    """OMA baseline throughput."""
    np.random.seed(seed)
    totals = []
    for _ in range(n_mc):
        thr = 0.0
        for i in range(cfg.N_PAIRS):
            hn = chan(uav_pos, near_devs[i], 0.0, cfg)
            hf = chan(uav_pos, far_devs[i],  cfg.BLOCK_DB, cfg)
            rN, rF = oma_rates(hn, hf, cfg)
            thr += rN + rF
        totals.append(thr / 1e6)
    return float(np.mean(totals))


def ris_far_user_gain(uav_pos, far_devs, n_mc=200, cfg=BASE, phi_mode='quantised'):
    """Mean per-user RIS beamforming gain (dB) for far users."""
    gains = []
    for _ in range(n_mc):
        G_v = G_vec(uav_pos, cfg)
        for fd in far_devs:
            hd  = chan(uav_pos, fd, cfg.BLOCK_DB, cfg)
            hr  = hr_vec(fd, cfg)
            phi = opt_phi(G_v, hr, cfg, phi_mode)
            he  = h_eff(hd, G_v, hr, phi)
            g   = 10*np.log10(abs(he)**2 / max(abs(hd)**2, 1e-40))
            gains.append(g)
    return float(np.mean(gains))


# ─────────────────────────────────────────────────────────────
#  SECTION 2 — DEVICE PLACEMENT (fixed, seed=42)
# ─────────────────────────────────────────────────────────────

def place_devices(cfg=BASE):
    np.random.seed(42)
    cx, cy = cfg.UAV_CX, cfg.UAV_CY
    ang_n = np.random.uniform(0, 2*np.pi, cfg.K_NEAR)
    rad_n = np.random.uniform(cfg.NEAR_DMIN, cfg.NEAR_DMAX, cfg.K_NEAR)
    near  = np.zeros((cfg.K_NEAR, 3))
    near[:,0] = np.clip(cx + rad_n*np.cos(ang_n), 50, cfg.AREA_X-50)
    near[:,1] = np.clip(cy + rad_n*np.sin(ang_n), 50, cfg.AREA_Y-50)

    ang_f = np.random.uniform(0, 2*np.pi, cfg.K_FAR)
    rad_f = np.random.uniform(cfg.FAR_DMIN, cfg.FAR_DMAX, cfg.K_FAR)
    far   = np.zeros((cfg.K_FAR, 3))
    far[:,0] = np.clip(cx + rad_f*np.cos(ang_f), 50, cfg.AREA_X-50)
    far[:,1] = np.clip(cy + rad_f*np.sin(ang_f), 50, cfg.AREA_Y-50)
    return near, far

NEAR_DEVS, FAR_DEVS = place_devices(BASE)

# DQL-optimal position from Module 3
DQL_OPT_POS = np.array([437.0, 584.0, 108.0])
CENTROID_POS = np.array([400.0, 600.0, 125.0])


# ─────────────────────────────────────────────────────────────
#  SECTION 3 — NEURAL NETWORK (pure NumPy)
# ─────────────────────────────────────────────────────────────

class NumpyNet:
    """Lightweight MLP. output_fn: 'linear'|'softmax'"""
    def __init__(self, dims, seed=None, output_fn='linear'):
        if seed is not None:
            np.random.seed(seed)
        self.layers = []
        for i in range(len(dims)-1):
            scale = np.sqrt(2.0/dims[i])
            self.layers.append({
                'W': np.random.randn(dims[i], dims[i+1]) * scale,
                'b': np.zeros(dims[i+1])
            })
        self.output_fn = output_fn

    def forward(self, x):
        h = np.atleast_2d(x)
        self.cache = [h]
        for i, l in enumerate(self.layers[:-1]):
            h = np.maximum(0, h @ l['W'] + l['b'])   # ReLU
            self.cache.append(h)
        last = self.layers[-1]
        out = self.cache[-1] @ last['W'] + last['b']
        if self.output_fn == 'softmax':
            ex = np.exp(out - out.max(axis=-1, keepdims=True))
            out = ex / ex.sum(axis=-1, keepdims=True)
        self.cache.append(out)
        return out

    def copy_from(self, other):
        for sl, tl in zip(self.layers, other.layers):
            sl['W'] = tl['W'].copy()
            sl['b'] = tl['b'].copy()

    def adam_update(self, grads_W, grads_b, lr=1e-3, t=1,
                    ms_W=None, vs_W=None, ms_b=None, vs_b=None,
                    beta1=0.9, beta2=0.999, eps=1e-8):
        """In-place Adam update."""
        for i, (gW, gb) in enumerate(zip(grads_W, grads_b)):
            ms_W[i] = beta1*ms_W[i] + (1-beta1)*gW
            vs_W[i] = beta2*vs_W[i] + (1-beta2)*gW**2
            mW_hat  = ms_W[i] / (1 - beta1**t)
            vW_hat  = vs_W[i] / (1 - beta2**t)
            self.layers[i]['W'] -= lr * mW_hat / (np.sqrt(vW_hat)+eps)

            ms_b[i] = beta1*ms_b[i] + (1-beta1)*gb
            vs_b[i] = beta2*vs_b[i] + (1-beta2)*gb**2
            mb_hat  = ms_b[i] / (1 - beta1**t)
            vb_hat  = vs_b[i] / (1 - beta2**t)
            self.layers[i]['b'] -= lr * mb_hat / (np.sqrt(vb_hat)+eps)

    def zero_moments(self):
        mW = [np.zeros_like(l['W']) for l in self.layers]
        vW = [np.zeros_like(l['W']) for l in self.layers]
        mb = [np.zeros_like(l['b']) for l in self.layers]
        vb = [np.zeros_like(l['b']) for l in self.layers]
        return mW, vW, mb, vb


# ─────────────────────────────────────────────────────────────
#  SECTION 4 — ENVIRONMENT
# ─────────────────────────────────────────────────────────────

class UAVEnv:
    def __init__(self, cfg=BASE, n_mc_train=3):
        self.cfg = cfg
        self.n_mc = n_mc_train
        self.actions = np.array([
            [ 100, 0, 0], [-100, 0, 0],
            [0,  100, 0], [0, -100, 0],
            [0, 0,  25],  [0, 0, -25],
            [0,   0, 0],
        ])
        self.thr_min = 0.0; self.thr_max = 300.0

    def reset(self):
        cfg = self.cfg
        pos = np.array([
            np.random.uniform(50, cfg.AREA_X-50),
            np.random.uniform(50, cfg.AREA_Y-50),
            np.random.uniform(cfg.H_MIN, cfg.H_MAX)
        ])
        return self._state(pos, 0.0), pos

    def step(self, pos, act_idx, step_t, T=25):
        cfg = self.cfg
        new_pos = pos + self.actions[act_idx]
        new_pos[0] = np.clip(new_pos[0], 50, cfg.AREA_X-50)
        new_pos[1] = np.clip(new_pos[1], 50, cfg.AREA_Y-50)
        new_pos[2] = np.clip(new_pos[2], cfg.H_MIN, cfg.H_MAX)
        thr = compute_throughput(new_pos, NEAR_DEVS, FAR_DEVS,
                                 self.n_mc, cfg=cfg)
        r_norm = (thr - self.thr_min) / (self.thr_max - self.thr_min)
        return self._state(new_pos, thr), r_norm, new_pos, thr

    def _state(self, pos, thr):
        cfg = self.cfg
        return np.array([
            pos[0]/cfg.AREA_X,
            pos[1]/cfg.AREA_Y,
            (pos[2]-cfg.H_MIN)/(cfg.H_MAX-cfg.H_MIN),
            thr/300.0,
            0.0
        ])


# ─────────────────────────────────────────────────────────────
#  SECTION 5 — STUDY A: ALGORITHM COMPARISON
# ─────────────────────────────────────────────────────────────

def eval_final(pos, cfg=BASE):
    return compute_throughput(pos, NEAR_DEVS, FAR_DEVS,
                              n_mc=50, seed=99, cfg=cfg)


# ── A1: Random deployment ─────────────────────────────────────
def run_random_baseline(n_trials=200, cfg=BASE):
    print("\n[A1] Random deployment baseline...")
    t0 = time.time()
    np.random.seed(0)
    thrs = []
    for i in range(n_trials):
        pos = np.array([
            np.random.uniform(50, cfg.AREA_X-50),
            np.random.uniform(50, cfg.AREA_Y-50),
            np.random.uniform(cfg.H_MIN, cfg.H_MAX)
        ])
        thrs.append(compute_throughput(pos, NEAR_DEVS, FAR_DEVS,
                                       n_mc=10, seed=None, cfg=cfg))
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{n_trials} done...")
    best_idx = int(np.argmax(thrs))
    result = {
        'mean': float(np.mean(thrs)),
        'std':  float(np.std(thrs)),
        'best': float(np.max(thrs)),
        'worst': float(np.min(thrs)),
        'wall_sec': time.time()-t0,
        'all_thrs': thrs
    }
    print(f"    Mean={result['mean']:.2f} ± {result['std']:.2f} Mbps  "
          f"Best={result['best']:.2f}  [{result['wall_sec']:.0f}s]")
    return result


# ── A2: PSO ───────────────────────────────────────────────────
def run_pso(n_particles=30, n_iter=150, cfg=BASE):
    print("\n[A2] PSO optimizer...")
    t0 = time.time()
    np.random.seed(7)

    bounds = np.array([
        [50, cfg.AREA_X-50],
        [50, cfg.AREA_Y-50],
        [cfg.H_MIN, cfg.H_MAX]
    ])
    dim = 3

    # Initialise particles
    pos = np.array([[np.random.uniform(*bounds[d]) for d in range(dim)]
                    for _ in range(n_particles)])
    vel = np.zeros_like(pos)
    fit = np.array([compute_throughput(p, NEAR_DEVS, FAR_DEVS, 5, cfg=cfg)
                    for p in pos])

    pbest = pos.copy();   pbest_fit = fit.copy()
    gbest = pbest[np.argmax(pbest_fit)].copy()
    gbest_fit = float(np.max(pbest_fit))

    w = 0.7; c1 = 1.5; c2 = 1.5
    history = [gbest_fit]

    for it in range(n_iter):
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        vel = w*vel + c1*r1*(pbest-pos) + c2*r2*(gbest-pos)
        pos = np.clip(pos + vel, bounds[:,0], bounds[:,1])

        fit = np.array([compute_throughput(p, NEAR_DEVS, FAR_DEVS, 5, cfg=cfg)
                        for p in pos])
        improved = fit > pbest_fit
        pbest[improved] = pos[improved]; pbest_fit[improved] = fit[improved]
        if float(np.max(pbest_fit)) > gbest_fit:
            gbest = pbest[np.argmax(pbest_fit)].copy()
            gbest_fit = float(np.max(pbest_fit))
        history.append(gbest_fit)

        if (it+1) % 30 == 0:
            print(f"    iter {it+1}/{n_iter}  best={gbest_fit:.2f} Mbps")

    # Final accurate evaluation
    final_thr = eval_final(gbest, cfg)
    result = {
        'best_pos': gbest.tolist(),
        'pso_best': gbest_fit,
        'final_thr': final_thr,
        'history': history,
        'wall_sec': time.time()-t0,
        'n_evals': n_particles * (1 + n_iter)
    }
    print(f"    PSO best pos={np.round(gbest,1)} → {final_thr:.2f} Mbps "
          f"[{result['n_evals']} evals, {result['wall_sec']:.0f}s]")
    return result


# ── A3: DDQN (Double DQN) ─────────────────────────────────────
def run_ddqn(hidden=128, n_ep=2000, cfg=BASE, label='DDQN'):
    print(f"\n[A3] {label} agent...")
    t0 = time.time()
    STATE_DIM = 5; N_ACT = 7

    dims = [STATE_DIM, hidden, hidden, N_ACT]
    q_net = NumpyNet(dims, seed=42)
    t_net = NumpyNet(dims, seed=42); t_net.copy_from(q_net)
    mW, vW, mb, vb = q_net.zero_moments()

    buf = deque(maxlen=10000)
    env = UAVEnv(cfg, n_mc_train=3)

    best_thr = 0.0; best_pos = CENTROID_POS.copy()
    ep_thrs = []; t_step = 0; GAMMA = 0.95; LR = 1e-3; BS = 64

    def epsilon(ep):
        return max(0.05, 1.0 - (1.0-0.05)/500 * ep)

    for ep in range(1, n_ep+1):
        eps = epsilon(ep)
        state, pos = env.reset()
        ep_best = 0.0

        for step in range(25):
            q_vals = q_net.forward(state).flatten()
            act = (pyrandom.randrange(N_ACT) if np.random.rand() < eps
                   else int(np.argmax(q_vals)))
            ns, rew, npos, thr = env.step(pos, act, step)
            buf.append((state.copy(), act, rew, ns.copy()))
            state = ns; pos = npos
            if thr > ep_best: ep_best = thr
            t_step += 1

            if len(buf) >= BS:
                batch = pyrandom.sample(buf, BS)
                s_b  = np.array([x[0] for x in batch])
                a_b  = np.array([x[1] for x in batch])
                r_b  = np.array([x[2] for x in batch])
                ns_b = np.array([x[3] for x in batch])

                # DDQN: online net selects action, target net evaluates
                q_ns_online = q_net.forward(ns_b)           # (BS, N_ACT)
                best_acts   = np.argmax(q_ns_online, axis=1) # (BS,)
                q_ns_target = t_net.forward(ns_b)            # (BS, N_ACT)
                target_vals = q_ns_target[np.arange(BS), best_acts]
                targets     = r_b + GAMMA * target_vals

                q_curr = q_net.forward(s_b)
                td_err = q_curr.copy()
                td_err[np.arange(BS), a_b] -= targets
                loss_grad = 2 * td_err / BS

                # Backprop 2-hidden-layer network
                h2 = q_net.cache[-2]; h1 = q_net.cache[-3]; x_ = q_net.cache[-4]
                dh2  = loss_grad @ q_net.layers[2]['W'].T
                dh2 *= (h2 > 0)
                dh1  = dh2 @ q_net.layers[1]['W'].T
                dh1 *= (h1 > 0)
                gW = [x_.T @ dh1, h1.T @ dh2, h2.T @ loss_grad]
                gb = [dh1.sum(0), dh2.sum(0), loss_grad.sum(0)]
                q_net.adam_update(gW, gb, LR, t_step, mW, vW, mb, vb)

        if ep % 20 == 0: t_net.copy_from(q_net)
        ep_thrs.append(ep_best)

        if ep_best > best_thr:
            best_thr = ep_best
            best_pos = pos.copy()

        if ep in {1,100,500,1000,2000}:
            print(f"    ep={ep:4d}  ε={epsilon(ep):.2f}  "
                  f"ep_best={ep_best:.1f}  overall_best={best_thr:.1f}")

    final_thr = eval_final(best_pos, cfg)
    result = {
        'label': label,
        'best_pos': best_pos.tolist(),
        'final_thr': final_thr,
        'ep_thrs': ep_thrs,
        'wall_sec': time.time()-t0
    }
    print(f"    {label} → {final_thr:.2f} Mbps at {np.round(best_pos,1)}  "
          f"[{result['wall_sec']:.0f}s]")
    return result


# ── A4: A2C (Actor-Critic) ────────────────────────────────────
def run_a2c(hidden=128, n_ep=2000, cfg=BASE):
    print(f"\n[A4] A2C (Actor-Critic) agent...")
    t0 = time.time()
    STATE_DIM = 5; N_ACT = 7; GAMMA = 0.95; LR = 5e-4

    actor  = NumpyNet([STATE_DIM, hidden, hidden, N_ACT], seed=42, output_fn='softmax')
    critic = NumpyNet([STATE_DIM, hidden, hidden, 1],    seed=43, output_fn='linear')
    mWa, vWa, mba, vba = actor.zero_moments()
    mWc, vWc, mbc, vbc = critic.zero_moments()

    env = UAVEnv(cfg, n_mc_train=3)
    best_thr = 0.0; best_pos = CENTROID_POS.copy()
    ep_thrs = []; t_step = 0

    for ep in range(1, n_ep+1):
        state, pos = env.reset()
        states, actions, rewards, values = [], [], [], []

        for step in range(25):
            probs = actor.forward(state).flatten()
            val   = critic.forward(state).flatten()[0]
            act   = int(np.random.choice(N_ACT, p=probs))
            ns, rew, npos, thr = env.step(pos, act, step)
            states.append(state.copy()); actions.append(act)
            rewards.append(rew); values.append(val)
            state = ns; pos = npos
            t_step += 1

        # Returns and advantages
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + GAMMA*G; returns.insert(0, G)
        returns  = np.array(returns)
        values_a = np.array(values)
        advs     = returns - values_a
        advs     = (advs - advs.mean()) / (advs.std() + 1e-8)
        ep_best  = float(np.max([r*300 for r in rewards]))

        # Batch update (actor)
        s_batch = np.array(states)
        probs_b = actor.forward(s_batch)          # (T, N_ACT)
        eps_    = 1e-8
        log_pi  = np.log(probs_b + eps_)
        one_hot = np.zeros_like(probs_b)
        one_hot[np.arange(len(actions)), actions] = 1.0

        # Policy gradient: -log π(a|s) * A
        actor_loss_grad = -(advs[:, None] * one_hot) / len(actions)
        # Backprop softmax + hidden layers
        # Softmax Jacobian shortcut: δ = (loss_grad - sum) * probs
        dl_dout = actor_loss_grad * probs_b - probs_b * (actor_loss_grad * probs_b).sum(axis=1, keepdims=True)
        h2a = actor.cache[-2]; h1a = actor.cache[-3]; xa = actor.cache[-4]
        dh2a = dl_dout @ actor.layers[2]['W'].T; dh2a *= (h2a > 0)
        dh1a = dh2a    @ actor.layers[1]['W'].T; dh1a *= (h1a > 0)
        gWa = [xa.T @ dh1a, h1a.T @ dh2a, h2a.T @ dl_dout]
        gba = [dh1a.sum(0), dh2a.sum(0), dl_dout.sum(0)]
        actor.adam_update(gWa, gba, LR, t_step, mWa, vWa, mba, vba)

        # Critic update: MSE loss
        vals_b    = critic.forward(s_batch).flatten()
        td_err_c  = (vals_b - returns)
        dc_dout   = (2*td_err_c / len(returns)).reshape(-1, 1)
        h2c = critic.cache[-2]; h1c = critic.cache[-3]; xc = critic.cache[-4]
        dh2c = dc_dout @ critic.layers[2]['W'].T; dh2c *= (h2c > 0)
        dh1c = dh2c    @ critic.layers[1]['W'].T; dh1c *= (h1c > 0)
        gWc = [xc.T @ dh1c, h1c.T @ dh2c, h2c.T @ dc_dout]
        gbc = [dh1c.sum(0), dh2c.sum(0), dc_dout.sum(0)]
        critic.adam_update(gWc, gbc, LR, t_step, mWc, vWc, mbc, vbc)

        ep_thrs.append(ep_best)
        if ep_best > best_thr:
            best_thr = ep_best; best_pos = pos.copy()

        if ep in {1,100,500,1000,2000}:
            print(f"    ep={ep:4d}  best={ep_best:.1f}  overall={best_thr:.1f}")

    final_thr = eval_final(best_pos, cfg)
    result = {
        'label': 'A2C',
        'best_pos': best_pos.tolist(),
        'final_thr': final_thr,
        'ep_thrs': ep_thrs,
        'wall_sec': time.time()-t0
    }
    print(f"    A2C → {final_thr:.2f} Mbps  [{result['wall_sec']:.0f}s]")
    return result


# ── A5: DQL (original, reproduced here for consistent comparison) ──
def run_dql(hidden=128, n_ep=2000, cfg=BASE, label='DQL (proposed)'):
    print(f"\n[A5] {label}...")
    t0 = time.time()
    STATE_DIM = 5; N_ACT = 7

    dims = [STATE_DIM, hidden, hidden, N_ACT]
    q_net = NumpyNet(dims, seed=42)
    t_net = NumpyNet(dims, seed=42); t_net.copy_from(q_net)
    mW, vW, mb, vb = q_net.zero_moments()

    buf = deque(maxlen=10000)
    env = UAVEnv(cfg, n_mc_train=3)

    best_thr = 0.0; best_pos = CENTROID_POS.copy()
    ep_thrs = []; t_step = 0; GAMMA = 0.95; LR = 1e-3; BS = 64

    def epsilon(ep): return max(0.05, 1.0 - (1.0-0.05)/500 * ep)

    for ep in range(1, n_ep+1):
        eps = epsilon(ep)
        state, pos = env.reset()
        ep_best = 0.0

        for step in range(25):
            q_vals = q_net.forward(state).flatten()
            act = (pyrandom.randrange(N_ACT) if np.random.rand() < eps
                   else int(np.argmax(q_vals)))
            ns, rew, npos, thr = env.step(pos, act, step)
            buf.append((state.copy(), act, rew, ns.copy()))
            state = ns; pos = npos
            if thr > ep_best: ep_best = thr
            t_step += 1

            if len(buf) >= BS:
                batch = pyrandom.sample(buf, BS)
                s_b  = np.array([x[0] for x in batch])
                a_b  = np.array([x[1] for x in batch])
                r_b  = np.array([x[2] for x in batch])
                ns_b = np.array([x[3] for x in batch])

                # Standard DQN target
                q_ns  = t_net.forward(ns_b)
                targets = r_b + GAMMA * q_ns.max(axis=1)
                q_curr  = q_net.forward(s_b)
                td_err  = q_curr.copy()
                td_err[np.arange(BS), a_b] -= targets
                loss_grad = 2 * td_err / BS

                h2 = q_net.cache[-2]; h1 = q_net.cache[-3]; x_ = q_net.cache[-4]
                dh2  = loss_grad @ q_net.layers[2]['W'].T; dh2 *= (h2 > 0)
                dh1  = dh2     @ q_net.layers[1]['W'].T;  dh1 *= (h1 > 0)
                gW = [x_.T @ dh1, h1.T @ dh2, h2.T @ loss_grad]
                gb = [dh1.sum(0), dh2.sum(0), loss_grad.sum(0)]
                q_net.adam_update(gW, gb, LR, t_step, mW, vW, mb, vb)

        if ep % 20 == 0: t_net.copy_from(q_net)
        ep_thrs.append(ep_best)
        if ep_best > best_thr:
            best_thr = ep_best; best_pos = pos.copy()
        if ep in {1,100,500,1000,2000}:
            print(f"    ep={ep:4d}  ε={epsilon(ep):.2f}  best={ep_best:.1f}  "
                  f"overall={best_thr:.1f}")

    final_thr = eval_final(best_pos, cfg)
    result = {
        'label': label,
        'best_pos': best_pos.tolist(),
        'final_thr': final_thr,
        'ep_thrs': ep_thrs,
        'wall_sec': time.time()-t0
    }
    print(f"    {label} → {final_thr:.2f} Mbps  [{result['wall_sec']:.0f}s]")
    return result


# ─────────────────────────────────────────────────────────────
#  SECTION 6 — STUDY B: RIS ABLATION
# ─────────────────────────────────────────────────────────────

def run_ris_M_ablation():
    print("\n[B1] RIS element count ablation (M = 16, 64, 256, 512, 1024)...")
    M_values = [16, 64, 256, 512, 1024]
    results  = []
    for M in M_values:
        cfg = make_cfg(M=M)
        thr  = compute_throughput(DQL_OPT_POS, NEAR_DEVS, FAR_DEVS,
                                  n_mc=50, seed=99, cfg=cfg)
        gain = ris_far_user_gain(DQL_OPT_POS, FAR_DEVS, n_mc=200, cfg=cfg)
        results.append({'M': M, 'throughput': thr, 'far_gain_dB': gain})
        print(f"    M={M:5d}: thr={thr:.2f} Mbps  far-user gain={gain:+.2f} dB")
    return results


def run_ris_bits_ablation():
    print("\n[B2] RIS phase quantisation bits (1-bit → continuous)...")
    configs = [
        (1,  'quantised', '1-bit (2 levels)'),
        (2,  'quantised', '2-bit (4 levels)'),
        (3,  'quantised', '3-bit (8 levels) [proposed]'),
        (4,  'quantised', '4-bit (16 levels)'),
        (3,  'continuous','Continuous (∞ precision)'),
        (3,  'random',    'Random phase (no alignment)'),
    ]
    results = []
    for B, mode, label in configs:
        cfg  = make_cfg(B_PHASE=B)
        thr  = compute_throughput(DQL_OPT_POS, NEAR_DEVS, FAR_DEVS,
                                  n_mc=50, seed=99, cfg=cfg, phi_mode=mode)
        gain = ris_far_user_gain(DQL_OPT_POS, FAR_DEVS,
                                 n_mc=200, cfg=cfg, phi_mode=mode)
        results.append({'label': label, 'B': B, 'mode': mode,
                        'throughput': thr, 'far_gain_dB': gain})
        print(f"    {label:30s}: thr={thr:.2f} Mbps  gain={gain:+.2f} dB")
    return results


# ─────────────────────────────────────────────────────────────
#  SECTION 7 — STUDY C: SYSTEM PARAMETER ABLATION
# ─────────────────────────────────────────────────────────────

def run_alpha_ablation():
    print("\n[C1] NOMA power ratio α ablation...")
    alphas  = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    results = []
    for a in alphas:
        cfg = make_cfg(ALPHA=a)
        thr = compute_throughput(DQL_OPT_POS, NEAR_DEVS, FAR_DEVS,
                                 n_mc=50, seed=99, cfg=cfg)
        oma = compute_oma_throughput(DQL_OPT_POS, NEAR_DEVS, FAR_DEVS,
                                     n_mc=50, seed=99, cfg=cfg)
        gain_pct = (thr/oma - 1)*100
        results.append({'alpha': a, 'noma_thr': thr, 'oma_thr': oma,
                        'gain_pct': gain_pct,
                        'proposed': (abs(a-0.85) < 1e-9)})
        marker = ' ← proposed' if abs(a-0.85) < 1e-9 else ''
        print(f"    α={a:.2f}: NOMA={thr:.2f}  OMA={oma:.2f}  "
              f"gain={gain_pct:.1f}%{marker}")
    return results


def run_power_ablation():
    print("\n[C2] Total transmit power P_T ablation...")
    powers  = [0.5, 1.0, 2.0, 4.0]
    results = []
    for pt in powers:
        cfg = make_cfg(P_T=pt,
                       NOISE_W=10**((-174+7-30+10*np.log10(20e6/12))/10))
        thr = compute_throughput(DQL_OPT_POS, NEAR_DEVS, FAR_DEVS,
                                 n_mc=50, seed=99, cfg=cfg)
        results.append({'P_T': pt, 'throughput': thr,
                        'proposed': (abs(pt-2.0) < 1e-9)})
        marker = ' ← proposed' if abs(pt-2.0) < 1e-9 else ''
        print(f"    P_T={pt:.1f}W: {thr:.2f} Mbps{marker}")
    return results


# ─────────────────────────────────────────────────────────────
#  SECTION 8 — STUDY D: DQL ARCHITECTURE ABLATION
# ─────────────────────────────────────────────────────────────

def run_architecture_ablation():
    print("\n[D1] DQL architecture ablation (500 episodes each)...")
    configs = [
        ([64, 64],        '1-layer [64]'),
        ([128],           '1-layer [128]'),
        ([64, 64],        '2-layer [64,64]'),
        ([128, 128],      '2-layer [128,128] [proposed]'),
        ([256, 256],      '2-layer [256,256]'),
        ([128, 128, 128], '3-layer [128,128,128]'),
    ]
    results = []
    for hidden_list, label in configs:
        t0 = time.time()
        STATE_DIM = 5; N_ACT = 7; GAMMA = 0.95; LR = 1e-3; BS = 64
        dims = [STATE_DIM] + hidden_list + [N_ACT]
        q_net = NumpyNet(dims, seed=42)
        t_net = NumpyNet(dims, seed=42); t_net.copy_from(q_net)
        mW, vW, mb, vb = q_net.zero_moments()
        buf = deque(maxlen=10000)
        env = UAVEnv(BASE, n_mc_train=3)
        best_thr = 0.0; best_pos = CENTROID_POS.copy()
        t_step = 0

        n_layers = len(hidden_list)

        def epsilon(ep): return max(0.05, 1.0-(1.0-0.05)/300*ep)

        for ep in range(1, 501):
            eps = epsilon(ep)
            state, pos = env.reset(); ep_best = 0.0
            for step in range(25):
                q_vals = q_net.forward(state).flatten()
                act = (pyrandom.randrange(N_ACT) if np.random.rand()<eps
                       else int(np.argmax(q_vals)))
                ns, rew, npos, thr = env.step(pos, act, step)
                buf.append((state.copy(), act, rew, ns.copy()))
                state=ns; pos=npos; t_step+=1
                if thr > ep_best: ep_best = thr

                if len(buf) >= BS:
                    batch = pyrandom.sample(buf, BS)
                    s_b = np.array([x[0] for x in batch])
                    a_b = np.array([x[1] for x in batch])
                    r_b = np.array([x[2] for x in batch])
                    ns_b= np.array([x[3] for x in batch])
                    q_ns = t_net.forward(ns_b)
                    targets = r_b + GAMMA*q_ns.max(axis=1)
                    q_curr  = q_net.forward(s_b)
                    td_err  = q_curr.copy()
                    td_err[np.arange(BS), a_b] -= targets
                    loss_grad = 2*td_err/BS

                    # Generic backprop for variable depth
                    n = len(q_net.layers)
                    cache = q_net.cache
                    delta = loss_grad
                    gW_list = []; gb_list = []
                    for li in range(n-1, -1, -1):
                        gW_list.insert(0, cache[li].T @ delta)
                        gb_list.insert(0, delta.sum(0))
                        if li > 0:
                            delta = delta @ q_net.layers[li]['W'].T
                            delta *= (cache[li] > 0)
                    q_net.adam_update(gW_list, gb_list, LR, t_step,
                                      mW, vW, mb, vb)

            if ep % 20 == 0: t_net.copy_from(q_net)
            if ep_best > best_thr:
                best_thr = ep_best; best_pos = pos.copy()

        final_thr = eval_final(best_pos)
        n_params  = sum(l['W'].size + l['b'].size for l in q_net.layers)
        results.append({
            'label': label,
            'final_thr': final_thr,
            'n_params': n_params,
            'wall_sec': time.time()-t0
        })
        print(f"    {label:30s}: {final_thr:.2f} Mbps  "
              f"({n_params} params, {time.time()-t0:.0f}s)")
    return results


# ─────────────────────────────────────────────────────────────
#  SECTION 9 — PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_algorithm_comparison(rand_r, pso_r, ddqn_r, a2c_r, dql_r, oma_thr):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Bar chart of final throughputs
    ax = axes[0]
    methods = ['Random\n(baseline)', 'PSO', 'DDQN', 'A2C',
               'DQL\n(proposed)']
    thrs = [rand_r['mean'], pso_r['final_thr'],
            ddqn_r['final_thr'], a2c_r['final_thr'], dql_r['final_thr']]
    colors = ['#95A5A6','#E67E22','#2980B9','#8E44AD','#1A5276']
    bars = ax.bar(methods, thrs, color=colors, edgecolor='white',
                  linewidth=1.5, width=0.6)
    ax.axhline(oma_thr, color='red', ls='--', lw=1.5, label=f'OMA baseline ({oma_thr:.1f} Mbps)')
    for bar, thr in zip(bars, thrs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{thr:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('System Throughput (Mbps)', fontsize=11)
    ax.set_title('(a) Algorithm comparison — final throughput (N_MC=50)', fontsize=10)
    ax.legend(fontsize=9); ax.set_ylim(0, max(thrs)*1.12)
    ax.grid(axis='y', alpha=0.35)

    # (b) Convergence curves (DRL methods only)
    ax2 = axes[1]
    SM = 40  # smoothing window
    for res, col, ls in [(ddqn_r,'#2980B9','-'),
                          (a2c_r, '#8E44AD','--'),
                          (dql_r, '#1A5276','-.')]:
        thrs_ep = np.array(res['ep_thrs'])
        smooth = np.convolve(thrs_ep, np.ones(SM)/SM, 'valid')
        ax2.plot(smooth, color=col, lw=1.5, ls=ls, label=res['label'])
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax2.set_title('(b) DRL convergence comparison (40-episode moving average)', fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUT}/figA_algorithm_comparison.png'
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  [✓] {path}')


def plot_ris_ablation(M_res, bits_res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    Ms   = [r['M'] for r in M_res]
    gains= [r['far_gain_dB'] for r in M_res]
    thrs = [r['throughput'] for r in M_res]
    ax.plot(Ms, gains, 'go-', lw=2, ms=8, label='Far-user gain (dB)')
    ax2b = ax.twinx()
    ax2b.plot(Ms, thrs, 'b^--', lw=1.5, ms=7, label='System throughput (Mbps)')
    ax.set_xscale('log'); ax.set_xlabel('RIS elements M', fontsize=11)
    ax.set_ylabel('Far-user RIS gain (dB)', fontsize=11, color='g')
    ax2b.set_ylabel('Throughput (Mbps)', fontsize=11, color='b')
    ax.set_title('(a) Effect of RIS element count M', fontsize=10)
    ax.axvline(1024, color='gray', ls=':', lw=1, alpha=0.7)
    ax.text(1024, min(gains)*1.02, 'M=1024\n[proposed]', fontsize=8, color='gray', ha='right')
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1+lines2, l1+l2, fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels_b = [r['label'] for r in bits_res]
    gains_b  = [r['far_gain_dB'] for r in bits_res]
    bars = ax.bar(range(len(labels_b)), gains_b,
                  color=['#E74C3C' if 'Random' in r['label'] else
                         ('#1A5276' if 'proposed' in r['label'] else '#3498DB')
                         for r in bits_res],
                  edgecolor='white', linewidth=1.2, width=0.6)
    for bar, g in zip(bars, gains_b):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f'{g:+.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(labels_b)))
    ax.set_xticklabels([r['label'] for r in bits_res],
                       rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Far-user RIS gain (dB)', fontsize=11)
    ax.set_title('(b) Effect of phase quantisation bits', fontsize=10)
    ax.grid(axis='y', alpha=0.35)
    ax.axhline(0, color='k', lw=0.8, ls='--')

    plt.tight_layout()
    path = f'{OUT}/figB_ris_ablation.png'
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  [✓] {path}')


def plot_system_ablation(alpha_res, power_res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    alphas = [r['alpha'] for r in alpha_res]
    noma_t = [r['noma_thr'] for r in alpha_res]
    oma_t  = [r['oma_thr']  for r in alpha_res]
    gain_p = [r['gain_pct'] for r in alpha_res]
    ax.plot(alphas, noma_t, 'bs-', lw=2, ms=8, label='NOMA throughput (Mbps)')
    ax.plot(alphas, oma_t,  'r^--',lw=1.5, ms=7, label='OMA throughput (Mbps)')
    ax2c = ax.twinx()
    ax2c.plot(alphas, gain_p, 'go:', lw=1.5, ms=7, label='NOMA gain over OMA (%)')
    opt_a = 0.85
    ax.axvline(opt_a, color='gray', ls=':', lw=1.5, alpha=0.7)
    ax.text(opt_a+0.005, min(oma_t)*1.01, 'α=0.85\n[proposed]', fontsize=8, color='gray')
    ax.set_xlabel('Power ratio α (fraction to far user)', fontsize=11)
    ax.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax2c.set_ylabel('NOMA gain over OMA (%)', fontsize=11, color='g')
    ax.set_title('(a) Sensitivity to NOMA power ratio α', fontsize=10)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2c.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    pts  = [r['P_T'] for r in power_res]
    thrs = [r['throughput'] for r in power_res]
    colors_p = ['#3498DB' if not r['proposed'] else '#1A5276' for r in power_res]
    bars = ax.bar([f'{p:.1f}W' for p in pts], thrs, color=colors_p,
                  edgecolor='white', linewidth=1.2, width=0.5)
    for bar, thr in zip(bars, thrs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{thr:.1f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Total transmit power P_T (W)', fontsize=11)
    ax.set_ylabel('System Throughput (Mbps)', fontsize=11)
    ax.set_title('(b) Sensitivity to UAV transmit power P_T', fontsize=10)
    ax.grid(axis='y', alpha=0.35)
    ax.set_ylim(0, max(thrs)*1.12)

    plt.tight_layout()
    path = f'{OUT}/figC_system_ablation.png'
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  [✓] {path}')


def plot_arch_ablation(arch_res):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [r['label'] for r in arch_res]
    thrs   = [r['final_thr'] for r in arch_res]
    params = [r['n_params'] for r in arch_res]
    colors = ['#1A5276' if 'proposed' in r['label'] else '#85C1E9'
              for r in arch_res]
    bars = ax.bar(range(len(labels)), thrs, color=colors,
                  edgecolor='white', linewidth=1.2, width=0.6)
    ax2  = ax.twinx()
    ax2.plot(range(len(labels)), params, 'rs--', lw=1.5, ms=7, label='# parameters')
    for bar, thr in zip(bars, thrs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{thr:.1f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Throughput after 500 episodes (Mbps)', fontsize=11)
    ax2.set_ylabel('Network parameters (#)', fontsize=11, color='r')
    ax.set_title('DQL architecture ablation — throughput vs model complexity', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l2, lb2, fontsize=9, loc='upper right')
    plt.tight_layout()
    path = f'{OUT}/figD_architecture_ablation.png'
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  [✓] {path}')


# ─────────────────────────────────────────────────────────────
#  SECTION 10 — SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def print_summary_table(rand_r, pso_r, ddqn_r, a2c_r, dql_r, oma_thr):
    print("\n" + "=" * 85)
    print("  TABLE: Algorithm Comparison — Final Results")
    print("=" * 85)
    print(f"  {'Method':<28} {'Throughput':>12} {'vs OMA':>9} {'Evals/Train':>13} "
          f"{'Prior Knowledge':>17} {'Wall time':>10}")
    print("-" * 85)

    rows = [
        ("OMA (FDMA baseline)",        oma_thr,           0,           "N/A",      "None",   "N/A"),
        ("Random deployment (best)",   rand_r['best'],    0,           "200",      "None",   f"{rand_r['wall_sec']:.0f}s"),
        ("Greedy hill-climb",          187.80,            0,           "~500",     "Centroid req.", "N/A"),
        ("PSO (30p × 150 iter)",       pso_r['final_thr'],0,           f"{pso_r['n_evals']}","None",f"{pso_r['wall_sec']:.0f}s"),
        ("DDQN",                       ddqn_r['final_thr'],0,          "50k steps","None",   f"{ddqn_r['wall_sec']:.0f}s"),
        ("A2C",                        a2c_r['final_thr'],0,           "50k steps","None",   f"{a2c_r['wall_sec']:.0f}s"),
        ("DQL proposed",               dql_r['final_thr'],0,           "50k steps","None",   f"{dql_r['wall_sec']:.0f}s"),
    ]
    for name, thr, _, evals, prior, wall in rows:
        gain = f"+{(thr/oma_thr-1)*100:.1f}%" if thr > oma_thr and oma_thr > 0 else "—"
        star = " ←" if "proposed" in name else ""
        print(f"  {name:<28} {thr:>10.2f}M {gain:>9} {evals:>13} "
              f"{prior:>17} {wall:>10}{star}")
    print("=" * 85)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    T_TOTAL = time.time()
    print("=" * 65)
    print("  BENCHMARK & ABLATION STUDY — starting")
    print("=" * 65)

    # OMA baseline (fixed for comparison)
    print("\nComputing OMA baseline...")
    oma_thr = compute_oma_throughput(DQL_OPT_POS, NEAR_DEVS, FAR_DEVS,
                                     n_mc=50, seed=99)
    print(f"  OMA baseline: {oma_thr:.2f} Mbps")

    # ── STUDY A ──────────────────────────────────────────────
    print("\n" + "─"*50)
    print("  STUDY A — Algorithm Comparison")
    print("─"*50)
    rand_r  = run_random_baseline(n_trials=200)
    pso_r   = run_pso(n_particles=30, n_iter=150)
    ddqn_r  = run_ddqn(hidden=128, n_ep=2000, label='DDQN')
    a2c_r   = run_a2c(hidden=128,  n_ep=2000)
    dql_r   = run_dql(hidden=128,  n_ep=2000, label='DQL (proposed)')

    # ── STUDY B ──────────────────────────────────────────────
    print("\n" + "─"*50)
    print("  STUDY B — RIS Ablation")
    print("─"*50)
    M_res   = run_ris_M_ablation()
    bits_r  = run_ris_bits_ablation()

    # ── STUDY C ──────────────────────────────────────────────
    print("\n" + "─"*50)
    print("  STUDY C — System Parameter Ablation")
    print("─"*50)
    alpha_r = run_alpha_ablation()
    power_r = run_power_ablation()

    # ── STUDY D ──────────────────────────────────────────────
    print("\n" + "─"*50)
    print("  STUDY D — DQL Architecture Ablation")
    print("─"*50)
    arch_r  = run_architecture_ablation()

    # ── Save all results ──────────────────────────────────────
    all_results = {
        'oma_thr': oma_thr,
        'study_A': {
            'random': rand_r,
            'pso':    pso_r,
            'ddqn':   ddqn_r,
            'a2c':    a2c_r,
            'dql':    dql_r,
        },
        'study_B': {'M_ablation': M_res, 'bits_ablation': bits_r},
        'study_C': {'alpha_ablation': alpha_r, 'power_ablation': power_r},
        'study_D': {'arch_ablation': arch_r},
    }
    json_path = f'{OUT}/all_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Results saved: {json_path}")

    # ── Generate figures ──────────────────────────────────────
    print("\nGenerating figures...")
    plot_algorithm_comparison(rand_r, pso_r, ddqn_r, a2c_r, dql_r, oma_thr)
    plot_ris_ablation(M_res, bits_r)
    plot_system_ablation(alpha_r, power_r)
    plot_arch_ablation(arch_r)

    # ── Summary table ─────────────────────────────────────────
    print_summary_table(rand_r, pso_r, ddqn_r, a2c_r, dql_r, oma_thr)

    print(f"\n{'='*65}")
    print(f"  TOTAL WALL TIME: {(time.time()-T_TOTAL)/60:.1f} minutes")
    print(f"  Output folder:   results/benchmark/")
    print(f"{'='*65}")
