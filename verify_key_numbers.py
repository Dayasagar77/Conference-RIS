#!/usr/bin/env python3
r"""
verify_key_numbers.py  —  reproducibility check using the paper's OWN eval functions
═══════════════════════════════════════════════════════════════════════════════
This version is apples-to-apples: the headline throughput is computed with
dql_agent.compute_throughput() (the authoritative Module-3 evaluator, n_mc=50,
seed=99) — NOT a re-implementation — so it should match the paper's 188.36 Mbps
to the decimal. OMA/NOMA/RIS-gain are read from channel_model.py's own output,
and the RL / blockage results from their committed JSONs.

    python verify_key_numbers.py

Prereqs: run `python channel_model.py` once (creates results/altitude_throughput.json),
and the continuous_rl_baseline.json / blockage_sensitivity.json already exist.
Importing dql_agent below does NOT train — it only loads the module and its
device setup so we can call its evaluator.
"""
import os
import json
import numpy as np

results = []


def check(name, got, expect, tol, unit=""):
    ok = (got is not None) and (abs(got - expect) <= tol)
    results.append(ok)
    g = "   n/a" if got is None else f"{got:9.3f}"
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<44} got={g}{unit}  expected={expect}{unit}  (tol +/-{tol})")


# ── 1. Headline throughput via the paper's OWN evaluator ────────────────────────
print("=" * 80)
print("  1. Throughput at (437,584,108) m  —  dql_agent.compute_throughput (authoritative)")
print("=" * 80)
try:
    import dql_agent  # module-level setup only (no training); __main__ is guarded
    thr_opt = dql_agent.compute_throughput(np.array([437.0, 584.0, 108.0]), n_mc=50, seed=99)
    check("DQL-optimal aggregate throughput", thr_opt, 188.36, 0.30, " Mbps")
except Exception as e:  # noqa: BLE001
    print(f"  [FAIL] could not call dql_agent.compute_throughput: {e}")
    results.append(False)
    thr_opt = None

# ── 2. OMA / NOMA / RIS gain  —  from channel_model.py's own output ──────────────
print("\n" + "=" * 80)
print("  2. OMA / NOMA / RIS gain  —  results/altitude_throughput.json (channel_model output)")
print("=" * 80)
p = os.path.join("results", "altitude_throughput.json")
if os.path.isfile(p):
    d = json.load(open(p))
    i = d["altitudes_m"].index(125)
    oma125 = d["thr_oma_mbps"][i]
    check("OMA throughput @ H=125 m", oma125, 120.93, 0.10, " Mbps")
    check("NOMA throughput @ H=125 m", d["thr_noma_mbps"][i], 187.18, 0.10, " Mbps")
    check("RIS far-user gain", d["ris_gain_far_dB"], 7.55, 0.05, " dB")
    if thr_opt is not None:
        check("DQL gain over equal-split OMA", 100 * (thr_opt / oma125 - 1), 55.8, 0.40, " %")
else:
    print("  [skip] run  python channel_model.py  first to create this file")

# ── 3. Continuous-action RL baselines  (continuous_rl_baseline.json) ─────────────
print("\n" + "=" * 80)
print("  3. Continuous-action RL baselines  (continuous_rl_baseline.json)")
print("=" * 80)
p = "continuous_rl_baseline.json"
if os.path.isfile(p):
    s = json.load(open(p))["summary"]
    ref = json.load(open(p))["dqn_ref"]["fine"]
    check("PPO converged throughput", s["PPO"]["mean"], 188.85, 0.30, " Mbps")
    check("A2C converged throughput", s["A2C"]["mean"], 181.47, 0.30, " Mbps")
    check("DDPG converged throughput", s["DDPG"]["mean"], 142.75, 1.00, " Mbps")
    check("DQN reference (fine grid)", ref, 188.35, 0.30, " Mbps")
else:
    print("  [skip] continuous_rl_baseline.json not found")

# ── 4. Far-user reliability vs blockage  (blockage_sensitivity.json) ─────────────
print("\n" + "=" * 80)
print("  4. Far-user reliability vs blockage  (blockage_sensitivity.json)")
print("=" * 80)
p = "blockage_sensitivity.json"
if os.path.isfile(p):
    d = json.load(open(p)); b = d["block_dB"]; i60 = b.index(60); i80 = b.index(80)
    check("Survivor-aware reliability @ 60 dB", d["survivor_opt"][i60], 100.0, 2.0, " %")
    check("Sum-rate reliability @ 60 dB", d["sumrate_opt"][i60], 64.58, 2.0, " %")
    check("No-RIS reliability @ 60 dB", d["no_ris_sumrate"][i60], 1.39, 2.0, " %")
    check("Sum-rate reliability @ 80 dB", d["sumrate_opt"][i80], 55.28, 2.0, " %")
else:
    print("  [skip] blockage_sensitivity.json not found")

# ── Summary ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"  RESULT: {sum(results)}/{len(results)} checks PASSED")
print("=" * 80)
print("""
Numbers that still need a full run (compare the printed value to the paper):
  python hsdc.py                  # K* = 10
  python uav_ris_coupling.py      # Spearman rho = -0.84 ; near/far optima ~533 m apart
  python survivor_aware_pareto.py # grid optima: sum-rate 187.7/64.6% , telemetry-SLA 178.9/100% (4.7%)
  python survivor_aware_dql.py    # survivor-aware agent: 100% far-SLA at ~7% throughput cost
  python benchmark_ablation.py    # PSO 188.55 / DDQN 186.08 / A2C 186.45     [~50 min]
""")
