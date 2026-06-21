r"""
verify_oma_gain.py
------------------
Reproduces the self-consistent NOMA-vs-OMA throughput gain for the
IEEE Access paper, with BOTH NOMA and OMA evaluated at the SAME UAV
position so the percentage is internally consistent.

It imports the verified channel model from benchmark_ablation_fixed.py
(the same code that produces the 188.36 Mbps headline) -- it does NOT
re-implement any physics.

USAGE
  Put this file in the same folder as benchmark_ablation_fixed.py:
      D:\DAYA PHD\PHD WORK\RIS\channel_model\
  then run:
      python verify_oma_gain.py

EXPECTED (container, numpy 2.4.4):
  HEADLINE DQL position (437, 584, 108):  NOMA 188.35 | OMA 121.47 | +55.05%
  Centroid/best-fixed   (400, 600, 125):  NOMA 187.34 | OMA 121.00 | +54.83%
"""
import numpy as np
from benchmark_ablation_fixed import (
    compute_throughput, compute_oma_throughput,
    NEAR_DEVS, FAR_DEVS, DQL_OPT_POS, CENTROID_POS,
)

# Headline Monte-Carlo settings (identical to the throughput headline)
N_MC = 50
SEED = 99


def report(label, pos):
    noma = compute_throughput(pos, NEAR_DEVS, FAR_DEVS, n_mc=N_MC, seed=SEED)
    oma  = compute_oma_throughput(pos, NEAR_DEVS, FAR_DEVS, n_mc=N_MC, seed=SEED)
    gain = (noma / oma - 1.0) * 100.0
    p = tuple(int(round(x)) for x in pos)
    print(f"{label}  @ {p}")
    print(f"    NOMA throughput : {noma:8.4f} Mbps")
    print(f"    OMA  throughput : {oma:8.4f} Mbps")
    print(f"    NOMA gain / OMA : +{gain:.2f}%")
    print()
    return noma, oma, gain


if __name__ == "__main__":
    print("=" * 62)
    print("  NOMA-vs-OMA gain  (devices seed=42, n_mc=50, seed=99,")
    print("                     quantised 3-bit RIS phase)")
    print("=" * 62)
    print()
    n_h, o_h, g_h = report("HEADLINE DQL position", DQL_OPT_POS)
    n_c, o_c, g_c = report("Centroid / best-fixed ", CENTROID_POS)

    print("-" * 62)
    print("  PAPER NUMBER TO USE (headline position, consistent):")
    print(f"    NOMA = {n_h:.2f} Mbps,  OMA = {o_h:.2f} Mbps,  gain = +{g_h:.1f}%")
    print("-" * 62)
