#!/usr/bin/env python3
r"""
continuous_rl_baseline.py  (v2 — VecNormalize)
═══════════════════════════════════════════════════════════════════════════════
Reviewer-1: "DQN is a relatively old algorithm; adopt newer versions such as
DDPG, PPO, A3C." + "absence of strong benchmarks."

Trains CONTINUOUS-ACTION deep-RL agents — PPO, DDPG, A2C (the synchronous A3C
variant) — on the SAME channel_model.py physics and sum-rate objective as the
proposed NumPy DQN, then compares converged throughput. Standard
Stable-Baselines3 implementations.

v2 FIX: the reward is raw throughput (~60-190 Mbps). Unnormalised large rewards
destabilise continuous-control value learning (agents collapsed to the altitude
ceiling in v1). We therefore wrap the env in VecNormalize (observation + reward
normalisation), the standard remedy, and give a longer budget. Evaluation rolls
out the greedy policy on a RAW env, normalising observations through the frozen
VecNormalize statistics, and reports RAW throughput (Mbps) from channel_model.py.

CONTEXT FOR THE PAPER: v24 Table II already reports DQN 188.36 / DDQN 186.08 /
A2C 186.45 / PSO 188.55 (all within ~1%). These continuous agents are a
reinforcing benchmark, not the sole answer to R1.

ENVIRONMENT (Gymnasium):
    obs    : normalised [x, y, H, t/T]            (VecNormalize-standardised)
    action : continuous [dx, dy, dH] in [-1,1]^3 scaled to (+-100 m xy, +-25 m H)
    reward : aggregate throughput (Mbps) from channel_model.py  (== DQN objective)
    horizon: 25 steps
eval_position() is COPIED VERBATIM from survivor_aware_dql.py (no import side effects).

───────────────────────────────────────────────────────────────────────────────
SETUP + RUN  (Windows — RL training runs on Windows per project constraint)
───────────────────────────────────────────────────────────────────────────────
    pip install stable-baselines3 gymnasium torch
    cd "D:\DAYA PHD\PHD WORK\RIS\channel_model"
    python continuous_rl_baseline.py
  channel_model.py must be in the same folder.
  Outputs (-> ..\results\SURVIVOR\): continuous_rl_baseline.png / .json
  Runtime ~2-3 h for 3 agents x 3 seeds x 60k timesteps. Reduce SEEDS to [0,1] or
  TIMESTEPS to 40000 to shorten. Smoke-test:  SMOKE=1 python continuous_rl_baseline.py
═══════════════════════════════════════════════════════════════════════════════
"""
import os, time, json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import channel_model as cm
warnings.filterwarnings("ignore")

SMOKE = os.environ.get("SMOKE") == "1"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
if not os.path.isdir(os.path.dirname(OUT)):
    OUT = os.path.join(HERE, "results", "SURVIVOR")
os.makedirs(OUT, exist_ok=True)

DEVICE_SEED, EVAL_SEED, SLA_BPS = 42, 99, 100.0
GAMMA      = 0.95
T_STEPS    = 25
N_MC_TRAIN = 4 if not SMOKE else 2
N_MC_EVAL  = 60 if not SMOKE else 10
XMIN, XMAX = 150.0, 800.0
YMIN, YMAX = 200.0, 900.0
HMIN, HMAX = 50.0, 200.0
STEP_XY, STEP_H = 100.0, 25.0

AGENTS    = (["PPO"] if SMOKE else ["PPO", "DDPG", "A2C"])
SEEDS     = ([0] if SMOKE else [0, 1, 2])
TIMESTEPS = (1500 if SMOKE else 60000)
EVAL_EVERY = (500 if SMOKE else 6000)

DQN_FINE   = 188.35
DQN_COARSE = 184.98

np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]
FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
K    = cm.CFG.N_PAIRS


def eval_position(uav, n_mc, seed=None):
    """COPIED VERBATIM from survivor_aware_dql.py — bit-identical reward physics."""
    if seed is not None:
        np.random.seed(seed)
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
    return near_sum_mbps, float(fs.mean()), float((fs >= SLA_BPS).mean())


def aggregate_mbps(uav, n_mc, seed=None):
    near_mbps, far_mean_bps, _ = eval_position(uav, n_mc, seed)
    return near_mbps + far_mean_bps * K / 1e6


import gymnasium as gym
from gymnasium import spaces


class UAVPositionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_mc=N_MC_TRAIN):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, (4,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.n_mc = n_mc
        self.lo = np.array([XMIN, YMIN, HMIN]); self.hi = np.array([XMAX, YMAX, HMAX])

    def _obs(self):
        return np.array([(self.pos[0]-XMIN)/(XMAX-XMIN),
                         (self.pos[1]-YMIN)/(YMAX-YMIN),
                         (self.pos[2]-HMIN)/(HMAX-HMIN),
                         self.t / T_STEPS], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([(XMIN+XMAX)/2, (YMIN+YMAX)/2, (HMIN+HMAX)/2])
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        mv = np.array([action[0]*STEP_XY, action[1]*STEP_XY, action[2]*STEP_H])
        self.pos = np.clip(self.pos + mv, self.lo, self.hi)
        reward = aggregate_mbps(self.pos, self.n_mc)
        self.t += 1
        return self._obs(), float(reward), self.t >= T_STEPS, False, {}


def run():
    from stable_baselines3 import PPO, DDPG, A2C
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    print("=" * 86)
    print("  CONTINUOUS-RL BASELINES (PPO / DDPG / A2C) vs proposed DQN  [VecNormalize]")
    print(f"  agents={AGENTS} seeds={SEEDS} timesteps={TIMESTEPS}  "
          f"N_MC_train={N_MC_TRAIN}{'  [SMOKE]' if SMOKE else ''}")
    print("=" * 86)
    t0 = time.time()

    def greedy_pos(model, venv):
        raw = UAVPositionEnv(); obs, _ = raw.reset(); done = False
        while not done:
            nobs = venv.normalize_obs(obs)
            a, _ = model.predict(nobs, deterministic=True)
            obs, _, done, _, _ = raw.step(a)
        return raw.pos.copy()

    class TputCB(BaseCallback):
        def __init__(self, venv, every):
            super().__init__(); self.venv = venv; self.every = every; self.rec = []
        def _on_step(self):
            if self.num_timesteps % self.every == 0:
                pos = greedy_pos(self.model, self.venv)
                self.rec.append((int(self.num_timesteps), aggregate_mbps(pos, N_MC_EVAL, EVAL_SEED)))
            return True

    def make(agent, venv, seed):
        if agent == "PPO":
            return PPO("MlpPolicy", venv, seed=seed, n_steps=(256 if SMOKE else 1024), verbose=0)
        if agent == "A2C":
            return A2C("MlpPolicy", venv, seed=seed, verbose=0)
        if agent == "DDPG":
            n = NormalActionNoise(np.zeros(3), 0.1 * np.ones(3))
            return DDPG("MlpPolicy", venv, seed=seed, action_noise=n,
                        learning_starts=1000, verbose=0)
        raise ValueError(agent)

    results = {a: dict(final=[], sla=[], pos=[], curves=[]) for a in AGENTS}
    for agent in AGENTS:
        print(f"\n--- {agent} ---")
        for seed in SEEDS:
            venv = VecNormalize(DummyVecEnv([lambda: UAVPositionEnv()]),
                                norm_obs=True, norm_reward=True, gamma=GAMMA, clip_obs=10.0)
            model = make(agent, venv, seed)
            cb = TputCB(venv, EVAL_EVERY)
            model.learn(total_timesteps=TIMESTEPS, callback=cb, progress_bar=False)
            venv.training = False; venv.norm_reward = False
            pos = greedy_pos(model, venv)
            near, fmean, sla = eval_position(pos, N_MC_EVAL, EVAL_SEED)
            agg = near + fmean * K / 1e6
            results[agent]["final"].append(agg); results[agent]["sla"].append(sla * 100)
            results[agent]["pos"].append([float(x) for x in pos])
            results[agent]["curves"].append(cb.rec)
            print(f"    seed {seed}: pos=[{pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}]  "
                  f"agg={agg:.2f} Mbps  SLA={sla*100:.0f}%  [{(time.time()-t0)/60:.1f} min]")

    print("\n" + "=" * 86)
    print(f"  {'method':<14}{'throughput (Mbps)':<24}{'vs DQN-fine':<14}{'far-SLA (%)'}")
    print("-" * 86)
    summary = {}
    for agent in AGENTS:
        f = np.array(results[agent]["final"]); s = np.array(results[agent]["sla"])
        gap = (f.mean() - DQN_FINE) / DQN_FINE * 100
        summary[agent] = dict(mean=float(f.mean()), std=float(f.std()),
                              sla_mean=float(s.mean()), gap_vs_dqn_fine_pct=float(gap))
        print(f"  {agent:<14}{f.mean():>7.2f} +/- {f.std():<11.2f}{gap:>+8.2f} %    {s.mean():>6.1f}")
    print(f"  {'DQN (proposed)':<14}{DQN_FINE:>7.2f}  (finer-grid)        {0.0:>+8.2f} %      --")
    print("=" * 86)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    cols = {"PPO": "#1f5fa8", "DDPG": "#2ca02c", "A2C": "#9467bd"}
    for agent in AGENTS:
        curves = results[agent]["curves"]
        if curves and curves[0]:
            ln = min(len(c) for c in curves)
            ts = np.array([t for t, _ in curves[0][:ln]])
            M = np.array([[v for _, v in c[:ln]] for c in curves])
            if M.size:
                mu = M.mean(0); sd = M.std(0)
                ax[0].plot(ts, mu, color=cols[agent], lw=2.2, label=agent)
                ax[0].fill_between(ts, mu - sd, mu + sd, color=cols[agent], alpha=0.18)
    ax[0].axhline(DQN_FINE, color="#d62728", ls="--", lw=1.8, label=f"DQN {DQN_FINE:.1f} (proposed)")
    ax[0].set_xlabel("Training timesteps"); ax[0].set_ylabel("Greedy throughput (Mbps)")
    ax[0].set_title("(a) Continuous-RL convergence vs DQN", fontweight="bold")
    ax[0].grid(True, ls=":", alpha=0.5); ax[0].legend(fontsize=9, loc="lower right")

    labels = AGENTS + ["DQN"]
    means = [summary[a]["mean"] for a in AGENTS] + [DQN_FINE]
    stds  = [summary[a]["std"] for a in AGENTS] + [0.0]
    bcols = [cols[a] for a in AGENTS] + ["#d62728"]
    xb = np.arange(len(labels))
    ax[1].bar(xb, means, yerr=stds, capsize=5, color=bcols, alpha=0.9)
    ax[1].axhline(DQN_FINE, color="#d62728", ls=":", lw=1.2)
    ax[1].set_xticks(xb); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("Converged throughput (Mbps)")
    ax[1].set_title("(b) Converged throughput (mean +/- std)", fontweight="bold")
    for i, m in enumerate(means):
        ax[1].text(i, m + 1, f"{m:.1f}", ha="center", fontsize=9)
    fig.suptitle("Continuous-action RL baselines vs proposed DQN "
                 "(same channel_model environment & sum-rate objective)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "continuous_rl_baseline.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {p}")

    with open(os.path.join(OUT, "continuous_rl_baseline.json"), "w") as f:
        json.dump(dict(summary=summary, dqn_ref=dict(fine=DQN_FINE, coarse_lattice=DQN_COARSE),
                       per_seed={a: dict(final=results[a]["final"], sla=results[a]["sla"],
                                         pos=results[a]["pos"]) for a in AGENTS},
                       config=dict(agents=AGENTS, seeds=SEEDS, timesteps=TIMESTEPS,
                                   n_mc_train=N_MC_TRAIN, vecnormalize=True,
                                   step_xy=STEP_XY, step_h=STEP_H, t_steps=T_STEPS)),
                  f, indent=2)
    print(f"[saved] {os.path.join(OUT,'continuous_rl_baseline.json')}")
    print(f"  Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    run()
