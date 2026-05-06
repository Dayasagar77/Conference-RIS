# RIS-Assisted Hybrid DSF-NOMA for UAV-Enabled Emergency Communications in 6G HetNets

Simulation source code and dataset for the journal paper:

**Dayasagar G and Deepa Nivethika S**, VIT Chennai Campus.
Submitted to IEEE Access, 2025.

## Repository Structure

- `channel_model/channel_model.py` — RIS phase alignment, path-loss, NOMA throughput
- `channel_model/hsdc.py` — HSDC Clustering (HAC + k-means, auto K*)
- `channel_model/dql_agent.py` — Deep Q-Learning UAV 3D positioning
- `channel_model/precompute_throughput.py` — Throughput precomputation
- `results/` — Full simulation dataset (Figs. 1-13, Tables III-VII, JSON logs)

## Requirements

pip install numpy scipy scikit-learn matplotlib

## How to Run

cd channel_model
python3 channel_model.py
python3 hsdc.py
python3 dql_agent.py

## Key Results

- K* = 10 clusters, 100% device coverage, 25.8% lower energy than k-means
- 188.36 +/- 0.4 Mbps system throughput (50 Monte Carlo trials)
- 69.8% gain over OMA, 0.63% over NOMA without RIS
- +7.55 dB RIS beamforming gain to blocked far users
- DQL convergence in ~450 episodes (7.4 min, Intel i9 CPU)

## Authors

Dayasagar G (dayasagar.g2025@vitstudent.ac.in)
Deepa Nivethika S (deepanivethika.s@vit.ac.in)
School of Computer Science and Engineering, VIT University, Chennai Campus
