# RIS-Assisted Hybrid DSF-NOMA for UAV-Enabled Emergency Communications in 6G HetNets

Simulation source code for the journal paper by Dayasagar G and Deepa Nivethika S,
VIT Chennai Campus.

## Modules
- `channel_model/hsdc.py` — HSDC Clustering (HAC + k-means, auto K*)
- `channel_model/dql_agent.py` — Deep Q-Learning UAV 3D positioning
- `channel_model/` — RIS phase alignment, path-loss, NOMA throughput

## Requirements
pip install numpy scipy scikit-learn matplotlib

## How to Run
cd channel_model
python3 hsdc.py
python3 dql_agent.py

## Key Results
- K* = 15 auto-determined clusters, 100% coverage
- 234.77 Mbps system throughput
- 74.9% gain over OMA

## Authors
Dayasagar G (dayasagar.g2025@vitstudent.ac.in)
Deepa Nivethika S (deepanivethika.s@vit.ac.in)
VIT University, Chennai Campus
