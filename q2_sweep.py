#!/usr/bin/env python3
r"""
q2_sweep.py — Q2 sensitivity of survivor capacity to beacon payload (L_b),
period (T_b), and RIS reconfiguration time (t_recfg), using the author's own
on-demand RIS scheduling model (ris_scheduling.py) on channel_model.py physics.
"""
import sys; sys.path.insert(0, '/home/claude/work')
import numpy as np
import ris_scheduling as rs

fr = rs.per_beacon_far_rate()          # per-beacon far rates (RIS per-survivor)
print(f"per-beacon far rate: mean={fr.mean():.0f} bps, median={np.median(fr):.0f} bps\n")

# validate baseline (12-byte / 96-bit, 60 s, 1 ms reconfig)
rs.RIS_RECONFIG_MS = 1.0
cap0 = rs.capacity_at_sla(fr, 96, 60.0)
print(f"[VALIDATE] capacity @ 96-bit, 60 s, 1 ms reconfig = {cap0} survivors @95% SLA\n")

print("=== (a) capacity vs RIS reconfiguration time t_recfg (12-byte, 60 s) ===")
for trec in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    rs.RIS_RECONFIG_MS = trec
    cap = rs.capacity_at_sla(fr, 96, 60.0)
    print(f"    t_recfg = {trec:5.1f} ms : {cap:5d} survivors")

print("\n=== (b) capacity vs beacon payload L_b (60 s, 1 ms reconfig) ===")
rs.RIS_RECONFIG_MS = 1.0
for Lb_bytes in [12, 25, 50, 100]:
    cap = rs.capacity_at_sla(fr, Lb_bytes*8, 60.0)
    print(f"    L_b = {Lb_bytes:3d} B ({Lb_bytes*8:4d} bits) : {cap:5d} survivors")

print("\n=== (c) capacity vs beacon period T_b (12-byte, 1 ms reconfig) ===")
for Tb in [30.0, 60.0, 120.0, 240.0]:
    cap = rs.capacity_at_sla(fr, 96, Tb)
    print(f"    T_b = {Tb:5.0f} s : {cap:5d} survivors")
