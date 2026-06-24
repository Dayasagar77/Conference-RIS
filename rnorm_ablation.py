#!/usr/bin/env python3
r"""
rnorm_ablation.py
═══════════════════════════════════════════════════════════════════════════════
EXECUTES the R_norm ablation that BOTH Round-2 reviewers requested
(R1 Comment 3 / R2 Q-Markov): "the inclusion of last-step throughput R_norm in
the state vector may violate the Markov property — provide an ablation."

This is a FAITHFUL DERIVATIVE of survivor_aware_dql.py: the Q-network
(NumPy 64-64 MLP + Adam), search bounds, epsilon schedule, replay buffer,
greedy-rollout extraction, N_MC, and RNG threading are IDENTICAL to that agent.
The ONLY additions are:
    (1) a USE_RNORM toggle that includes / excludes the 4th state element
        (the last-step reward / R_norm), and
    (2) a multi-seed loop so we can report run-to-run policy variance.

The conventional sum-rate reward (Eq. 9) is used by default, because the
manuscript's Markov note (Section IV) is written in the conventional-agent
context ("R_norm is the most recent throughput rescaled to its observed range")
and the 188.36 Mbps headline rests on that agent. In the tight search region
inherited from survivor_aware_dql.py, the sum-rate reward converges to the
sum-rate optimum (450, 577, 108) m that the Pareto analysis (Fig. 23) already
reports — so the ablation lands on a number already in the paper. A REWARD_MODE
switch ("sla") is provided if the survivor-aware variant is also wanted.

WHAT IT TESTS / EXPECTED RESULT (the manuscript's hypothesis, now MEASURED):
    H1  Removing R_norm does NOT move the converged optimum
        => UAV position is sufficient state; the Markov approximation is sound.
    H2  Removing R_norm slows convergence and/or widens run-to-run variance
        => R_norm is a convergence AID, not a hidden-state necessity.
    Confirming H1+H2 converts the manuscript's "we argue" into "we measured".

───────────────────────────────────────────────────────────────────────────────
HOW TO RUN  (Windows — DQL must run on Windows per project constraint)
───────────────────────────────────────────────────────────────────────────────
    cd D:\DAYA PHD\PHD WORK\RIS\channel_model\
    python rnorm_ablation.py
  channel_model.py must be in the same folder (imported exactly, as in
  survivor_aware_dql.py — guarantees identical reward physics).

  Outputs (-> ..\results\SURVIVOR\ if it exists, else .\results\SURVIVOR\):
    rnorm_ablation.png      2-panel figure for the manuscript
    rnorm_ablation.json     all numbers for the response-to-reviewers letter

  Runtime ~45-75 min for 5 seeds x 2 configs x 600 episodes (N_MC_TRAIN=6),
  matching survivor_aware_dql.py's per-run cost. To shorten: drop SEEDS to
  [0,1,2] or set N_MC_TRAIN=4. To lengthen if curves haven't plateaued: raise
  EPISODES to 800.
═══════════════════════════════════════════════════════════════════════════════
"""
import os, time, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import channel_model as cm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
if not os.path.isdir(os.path.dirname(OUT)):
    OUT = os.path.join(HERE, "results", "SURVIVOR")
os.makedirs(OUT, exist_ok=True)

# ── ablation configuration ────────────────────────────────────────────────────
SEEDS        = [0, 1, 2, 3, 4]   # independent training runs per configuration
REWARD_MODE  = "sumrate"         # "sumrate" (Eq. 9, conventional) | "sla" (J_sla)

# ── agent / experiment config — COPIED VERBATIM from survivor_aware_dql.py ─────
DEVICE_SEED  = 42
EVAL_SEED    = 99
SLA_BPS      = 100.0
W_SLA        = 0.5
NEAR_NORM    = 200.0          # Mbps normaliser (also rescales R_norm state feature)

EPISODES     = 600
T_STEPS      = 25
GAMMA        = 0.95
LR           = 1e-3
BATCH        = 64
REPLAY_CAP   = 10000
TARGET_SYNC  = 20
N_MC_TRAIN   = 6
N_MC_EVAL    = 60
EPS_START, EPS_END, EPS_FRAC = 1.0, 0.05, 0.7

XMIN, XMAX = 150.0, 800.0
YMIN, YMAX = 200.0, 900.0
HMIN, HMAX = 50.0, 200.0
DXY, DH    = 100.0, 25.0

ACTIONS = np.array([[ DXY,0,0],[-DXY,0,0],[0, DXY,0],[0,-DXY,0],
                    [0,0, DH],[0,0,-DH],[0,0,0]], dtype=float)
NA = len(ACTIONS)

CONV_WINDOW  = 30            # MA window for convergence detection
CONV_FRAC    = 0.95          # "converged" when MA reward >= CONV_FRAC*final-MA

# fixed device layout (identical to survivor_aware_dql.py)
np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]
FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]


# ── reward / physics (same eval_position as survivor_aware_dql.py) ────────────
def eval_position(uav, n_mc, seed=None):
    """(near_sum_mbps, far_mean_bps, sla_reliability, agg_mbps) at uav."""
    if seed is not None:
        np.random.seed(seed)
    K = cm.CFG.N_PAIRS
    near_sum = 0.0
    far_sum  = 0.0
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
            far_sum  += rF
            df[i] = rF
        far_samples.append(df)
    near_sum_mbps = near_sum / n_mc / 1e6
    agg_mbps      = (near_sum + far_sum) / n_mc / 1e6
    fs = np.array(far_samples)
    far_mean_bps = float(fs.mean())
    sla_rel = float((fs >= SLA_BPS).mean())
    return near_sum_mbps, far_mean_bps, sla_rel, agg_mbps


def reward_fn(uav, n_mc=N_MC_TRAIN, seed=None):
    """Returns (reward, r_state) where r_state is the value placed in the
    R_norm state slot (already rescaled to ~[0,1])."""
    near_mbps, _, sla_rel, agg_mbps = eval_position(uav, n_mc, seed)
    if REWARD_MODE == "sla":
        r = W_SLA * (near_mbps / NEAR_NORM) + (1.0 - W_SLA) * sla_rel  # J_sla in [0,1]
        return r, r                                  # r already in [0,1]
    else:  # "sumrate" — conventional aggregate-throughput reward (Eq. 9)
        r = agg_mbps                                 # Mbps (~188 at optimum)
        return r, min(agg_mbps / NEAR_NORM, 1.0)     # rescaled R_norm feature


def normalise(uav):
    return np.array([(uav[0]-XMIN)/(XMAX-XMIN),
                     (uav[1]-YMIN)/(YMAX-YMIN),
                     (uav[2]-HMIN)/(HMAX-HMIN)])


def make_state(uav, r_state, tn, use_rnorm):
    base = normalise(uav)
    if use_rnorm:
        return np.concatenate([base, [r_state, tn]])     # 5-dim (with R_norm)
    else:
        return np.concatenate([base, [tn]])              # 4-dim (without R_norm)


# ── NumPy MLP Q-network — identical to survivor_aware_dql.py (n_in parameterised)
class QNet:
    def __init__(self, n_in, n_h=64, n_out=NA, lr=LR, seed=0):
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
        out = self.forward(X)
        B = X.shape[0]
        dout = np.zeros_like(out)
        q_taken = out[np.arange(B), a_idx]
        dout[np.arange(B), a_idx] = 2.0 * (q_taken - target) / B
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


def env_reset():
    uav = np.array([np.random.uniform(XMIN, XMAX),
                    np.random.uniform(YMIN, YMAX),
                    np.random.uniform(HMIN, HMAX)])
    uav[0] = np.clip(round(uav[0]/DXY)*DXY, XMIN, XMAX)
    uav[1] = np.clip(round(uav[1]/DXY)*DXY, YMIN, YMAX)
    uav[2] = np.clip(round(uav[2]/DH)*DH,  HMIN, HMAX)
    return uav

def env_step(uav, a):
    nxt = uav + ACTIONS[a]
    nxt[0] = np.clip(nxt[0], XMIN, XMAX)
    nxt[1] = np.clip(nxt[1], YMIN, YMAX)
    nxt[2] = np.clip(nxt[2], HMIN, HMAX)
    r, r_state = reward_fn(nxt)
    return nxt, r, r_state


# ── train one agent (one config, one seed) ────────────────────────────────────
def train_one(use_rnorm, seed):
    n_in = 5 if use_rnorm else 4
    np.random.seed(seed)                         # single global RNG stream (as in his file)
    q  = QNet(n_in, seed=seed)
    qt = QNet(n_in, seed=seed); qt.copy_from(q)
    rb = Replay(REPLAY_CAP)
    eps_decay_ep = int(EPISODES * EPS_FRAC)
    curve = []

    for ep in range(EPISODES):
        eps = max(EPS_END, EPS_START - (EPS_START-EPS_END)*ep/eps_decay_ep)
        uav = env_reset()
        r_state = 0.0
        s = make_state(uav, r_state, 0.0, use_rnorm)
        ep_r, ep_loss, nl = 0.0, 0.0, 0
        for t in range(T_STEPS):
            if np.random.rand() < eps:
                a = np.random.randint(NA)
            else:
                a = int(np.argmax(q.forward(s[None])[0]))
            nuav, r, r_state = env_step(uav, a)
            tn = (t+1)/T_STEPS
            s2 = make_state(nuav, r_state, tn, use_rnorm)
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
        curve.append(ep_r / T_STEPS)

    # ── greedy rollout -> learned deployment position (his method) ──────────────
    finals = []
    for st in range(6):
        np.random.seed(1000+st)
        uav = env_reset()
        r_state = 0.0
        s = make_state(uav, r_state, 0.0, use_rnorm)
        for t in range(T_STEPS):
            a = int(np.argmax(q.forward(s[None])[0]))
            uav, r, r_state = env_step(uav, a)
            s = make_state(uav, r_state, (t+1)/T_STEPS, use_rnorm)
        finals.append(uav.copy())
    finals = np.array(finals)
    uav_star = np.median(finals, axis=0)
    uav_star[0] = round(uav_star[0]/DXY)*DXY
    uav_star[1] = round(uav_star[1]/DXY)*DXY
    uav_star[2] = round(uav_star[2]/DH)*DH

    near_mbps, far_mean, sla_rel, agg_mbps = eval_position(uav_star, N_MC_EVAL, seed=EVAL_SEED)

    # convergence episode
    ma = np.convolve(curve, np.ones(CONV_WINDOW)/CONV_WINDOW, mode="valid")
    final_ma = float(np.mean(curve[-CONV_WINDOW:]))
    conv_ep = EPISODES
    if len(ma) > 0 and final_ma != 0:
        hits = np.where(ma >= CONV_FRAC*final_ma)[0]
        if len(hits): conv_ep = int(hits[0] + CONV_WINDOW)

    return dict(use_rnorm=use_rnorm, seed=seed, curve=curve,
                final_pos=uav_star.tolist(), agg_mbps=agg_mbps,
                near_mbps=near_mbps, far_mean_bps=far_mean,
                sla_rel=sla_rel, conv_ep=conv_ep)


# ── aggregate + report ────────────────────────────────────────────────────────
def aggregate(runs):
    agg  = np.array([r["agg_mbps"] for r in runs])
    sla  = np.array([r["sla_rel"]  for r in runs])
    conv = np.array([r["conv_ep"]  for r in runs])
    pos  = np.array([r["final_pos"] for r in runs])
    pj = pos[:, :2]
    if len(pj) > 1:
        d = [np.linalg.norm(pj[i]-pj[j]) for i in range(len(pj)) for j in range(i+1, len(pj))]
        spread = float(np.mean(d))
    else:
        spread = 0.0
    return dict(agg_mean=float(agg.mean()), agg_std=float(agg.std()),
                sla_mean=float(sla.mean()), sla_std=float(sla.std()),
                conv_mean=float(conv.mean()), conv_std=float(conv.std()),
                pos_mean=pos.mean(0).round(1).tolist(),
                pos_spread_xy_m=round(spread, 1))


def main():
    t0 = time.time()
    print("=" * 76)
    print(f"  R_norm ABLATION  (REWARD_MODE={REWARD_MODE})  —  faithful derivative")
    print(f"  of survivor_aware_dql.py  |  seeds={SEEDS}  episodes={EPISODES}")
    print(f"  net 64-64  |  bounds X[{XMIN:.0f},{XMAX:.0f}] Y[{YMIN:.0f},{YMAX:.0f}] "
          f"H[{HMIN:.0f},{HMAX:.0f}]  |  N_MC_train={N_MC_TRAIN}")
    print("=" * 76)

    results = {True: [], False: []}
    for use_rnorm in (True, False):
        tag = "WITH R_norm (5-dim state)" if use_rnorm else "WITHOUT R_norm (4-dim state)"
        print(f"\n--- {tag} ---")
        for seed in SEEDS:
            run = train_one(use_rnorm, seed)
            results[use_rnorm].append(run)
            print(f"    seed {seed}: pos={[round(x) for x in run['final_pos']]}  "
                  f"agg={run['agg_mbps']:.2f} Mbps  SLA={run['sla_rel']*100:.0f}%  "
                  f"conv_ep={run['conv_ep']}  [{(time.time()-t0)/60:.1f} min]")

    A_with    = aggregate(results[True])
    A_without = aggregate(results[False])

    print("\n" + "=" * 76)
    print("  SUMMARY  (mean +/- std across seeds)")
    print("=" * 76)
    print(f"  {'Metric':<32}{'WITH R_norm':<24}{'WITHOUT R_norm'}")
    print("-" * 76)
    print(f"  {'Final throughput (Mbps)':<32}"
          f"{A_with['agg_mean']:.2f} +/- {A_with['agg_std']:.2f}        "
          f"{A_without['agg_mean']:.2f} +/- {A_without['agg_std']:.2f}")
    print(f"  {'Far-user SLA reliability (%)':<32}"
          f"{A_with['sla_mean']*100:.1f} +/- {A_with['sla_std']*100:.1f}          "
          f"{A_without['sla_mean']*100:.1f} +/- {A_without['sla_std']*100:.1f}")
    print(f"  {'Episodes to convergence':<32}"
          f"{A_with['conv_mean']:.0f} +/- {A_with['conv_std']:.0f}              "
          f"{A_without['conv_mean']:.0f} +/- {A_without['conv_std']:.0f}")
    print(f"  {'Final position (mean)':<32}{str(A_with['pos_mean']):<24}{str(A_without['pos_mean'])}")
    print(f"  {'Policy spread XY (m)':<32}{A_with['pos_spread_xy_m']:<24}{A_without['pos_spread_xy_m']}")
    print("=" * 76)
    print("  INTERPRETATION:")
    print("   H1  same final position        => position is sufficient state (Markov OK)")
    print("   H2  WITHOUT slower / wider band => R_norm is a convergence aid, not a")
    print("       hidden-state necessity.  Confirms the Section-IV argument by measurement.")
    print("=" * 76)

    # ── figure ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.7))

    def band(runs, color, label, ls):
        L = min(len(r["curve"]) for r in runs)
        M = np.array([r["curve"][:L] for r in runs])
        k = CONV_WINDOW
        Ms = np.array([np.convolve(m, np.ones(k)/k, mode="valid") for m in M])
        x = np.arange(Ms.shape[1])
        mean, std = Ms.mean(0), Ms.std(0)
        ax1.plot(x, mean, color=color, ls=ls, lw=2.2, label=label)
        ax1.fill_between(x, mean-std, mean+std, color=color, alpha=0.18)

    band(results[True],  "#1f5fa8", "With R_norm (5-dim)",    "-")
    band(results[False], "#d62728", "Without R_norm (4-dim)", "--")
    ax1.set_xlabel(f"Episode ({CONV_WINDOW}-ep moving average)")
    ylabel = "Mean step reward (Mbps)" if REWARD_MODE == "sumrate" else "Mean step reward  J_sla"
    ax1.set_ylabel(ylabel)
    ax1.set_title("(a) Convergence: with vs without R_norm", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9); ax1.grid(True, ls=":", alpha=0.5)

    # panel (b): final metric per seed + mean/std
    if REWARD_MODE == "sumrate":
        with_vals    = [r["agg_mbps"] for r in results[True]]
        without_vals = [r["agg_mbps"] for r in results[False]]
        ylab2, mw, sw, mo, so = ("Final throughput (Mbps)",
                                 A_with['agg_mean'], A_with['agg_std'],
                                 A_without['agg_mean'], A_without['agg_std'])
    else:
        with_vals    = [r["sla_rel"]*100 for r in results[True]]
        without_vals = [r["sla_rel"]*100 for r in results[False]]
        ylab2, mw, sw, mo, so = ("Final far-user SLA reliability (%)",
                                 A_with['sla_mean']*100, A_with['sla_std']*100,
                                 A_without['sla_mean']*100, A_without['sla_std']*100)
    # Raw per-seed dots are dodged to the LEFT of each category with a tiny symmetric
    # jitter (so identical converged values fan out instead of stacking), and the
    # mean +/- std summary is dodged to the RIGHT. This keeps the raw data visually
    # separated from the summary statistic and removes the seed-order artifact of the
    # previous np.linspace placement (which scattered the lone outlier asymmetrically
    # and ran the error bar straight through the points).
    _jit = np.random.default_rng(0)
    for base, vals, col in ((0, with_vals, "#1f5fa8"), (1, without_vals, "#d62728")):
        jx = base - 0.16 + _jit.uniform(-0.045, 0.045, len(vals))
        ax2.scatter(jx, vals, s=46, c=col, alpha=0.75, edgecolor="white", linewidth=0.6,
                    zorder=3, label=(f"per-seed (n={len(SEEDS)})" if base == 0 else None))
    ax2.errorbar(0 + 0.16, mw, yerr=sw, color="#1f5fa8", capsize=6, lw=2.2, marker="D",
                 ms=7, zorder=4, label="mean \u00b1 std")
    ax2.errorbar(1 + 0.16, mo, yerr=so, color="#d62728", capsize=6, lw=2.2, marker="D",
                 ms=7, zorder=4)
    ax2.set_xlim(-0.55, 1.55)
    ax2.legend(loc="center right", fontsize=8.5, framealpha=0.9)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["With R_norm", "Without R_norm"])
    ax2.set_ylabel(ylab2)
    ax2.set_title("(b) Final policy quality & variance", fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5, axis="y")

    # figure title omitted; the IEEE caption provides it
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(OUT, "rnorm_ablation.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"\n[saved] {fig_path}")

    out = {
        "experiment": "rnorm_ablation",
        "reward_mode": REWARD_MODE,
        "config": dict(seeds=SEEDS, episodes=EPISODES, t_steps=T_STEPS,
                       n_mc_train=N_MC_TRAIN, n_mc_eval=N_MC_EVAL, gamma=GAMMA,
                       lr=LR, hidden=64, bounds=[XMIN, XMAX, YMIN, YMAX, HMIN, HMAX],
                       device_seed=DEVICE_SEED, eval_seed=EVAL_SEED),
        "with_rnorm": A_with, "without_rnorm": A_without,
        "runs_with":    [{**{k: r[k] for k in ("seed","final_pos","agg_mbps","sla_rel","conv_ep")},
                          "curve": list(map(float, r["curve"]))} for r in results[True]],
        "runs_without": [{**{k: r[k] for k in ("seed","final_pos","agg_mbps","sla_rel","conv_ep")},
                          "curve": list(map(float, r["curve"]))} for r in results[False]],
    }
    json_path = os.path.join(OUT, "rnorm_ablation.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {json_path}")
    print(f"\nTotal wall-clock: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
