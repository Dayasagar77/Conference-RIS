# Survivor-Aware UAV Positioning for RIS-Assisted 6G Emergency Communications

Simulation source code for the IEEE Access manuscript
**"Survivor-Aware UAV Positioning for RIS-Assisted 6G Emergency Communications: A Deep Reinforcement Learning Approach"**
by Dayasagar G and Deepa Nivethika S, VIT Chennai Campus.

The code reproduces all figures and tables in the paper: a single UAV flying base
station serves 24 survivor IoT devices (12 NOMA pairs) with a 1024-element passive
RIS providing a reflected bypass for far survivors blocked by ~60 dB of rubble, and
a Deep Q-Learning agent placing the UAV in 3D.

## Key results reproduced

- DQL-optimal UAV position **(437, 584, 108) m → 188.36 Mbps** (**+55.8%** over an equal-split OMA baseline).
- RIS far-user beamforming gain **+7.55 dB** (3-bit phase quantisation, M = 1024).
- HSDC auto-determines **K\* = 10** clusters (100-device scalability population; K\* = 15 for the 99-device two-hop relay population).
- UAV–RIS geometric coupling law: far-user rate governed by UAV→RIS separation (**Spearman ρ = −0.84**) vs UAV→survivor separation (+0.54).
- Survivor-aware reward: **100% far-user telemetry-beacon reliability at ~7% aggregate-throughput cost** (4.7% at the Pareto-optimal grid point), 7.1× far-user mean-rate gain.

## Repository layout

```
channel_model/
  channel_model.py            Core physics: RIS phase alignment, path loss, NOMA rates (import only; do not modify)
  hsdc.py                     HSDC clustering (HAC Ward linkage + k-means refinement, auto K*)
  dql_agent.py                Deep Q-Learning UAV 3D positioning (sum-rate reward)
  survivor_aware_dql.py       DQL under the survivor-aware reward
  survivor_aware_pareto.py    Throughput–reliability Pareto frontier
  uav_ris_coupling.py         UAV–RIS coupling-law analysis (Spearman/Pearson)
  coupling_robustness.py      Coupling robustness across topologies
  coupling_topology_robustness.py / coupling_boundary_map.py / coupling_conditional_robustness.py
  continuous_rl_baseline.py   PPO / DDPG / A2C baselines (Stable-Baselines3)
  hsdc_d2d_relay.py           Two-hop UAV→head→D2D relay architecture
  ris_scheduling.py           On-demand RIS M/G/1 scheduling
  benchmark_ablation.py       Algorithm/benchmark comparison (PSO, grid, greedy, DDQN, A2C)
  optimized_oma.py            Proportional-fair OMA baseline
  fig13_per_user_cdf_FAITHFUL.py, fig14_multi_geometry_FAITHFUL.py,
  fig15_hsdc_sc1_FAITHFUL.py, fig16_telemetry_outage_FAITHFUL.py, run_all_figs.py   Figure generation
  results/                    Generated datasets, JSON, and figure outputs
```

## Requirements

```
pip install numpy scipy scikit-learn matplotlib
# additionally for continuous_rl_baseline.py:
pip install stable-baselines3 gymnasium torch
```

Python 3.9+ recommended.

## How to run

```
cd channel_model
python hsdc.py                 # clustering -> K*
python dql_agent.py            # sum-rate DQL positioning
python survivor_aware_dql.py   # survivor-aware DQL
python uav_ris_coupling.py     # coupling-law analysis
python run_all_figs.py         # regenerate paper figures
```

## Reproducibility

Device placement uses `seed = 42`; evaluation uses `seed = 99`. Results in the paper
are reported over 50 Monte Carlo trials with standard deviation < 0.5 Mbps. A permanent
archived release is deposited at Zenodo: https://doi.org/10.5281/zenodo.20047405.

## Authors

- Dayasagar G — dayasagar.g2025@vitstudent.ac.in
- Deepa Nivethika S — deepanivethika.s@vit.ac.in
- VIT University, Chennai Campus

## Citation

If you use this code, please cite the IEEE Access paper (citation details to be added on acceptance).
