#!/usr/bin/env python3
"""
ris_scheduling_v2.py
═══════════════════════════════════════════════════════════════════════════════
Realizable multiuser RIS via ON-DEMAND scheduling — answers Reviewer 2's
"single global phase cannot serve 12 users simultaneously" objection with a
physically-grounded control model.

PARAMETERS ARE GROUNDED IN THE LPWAN EMERGENCY-TELEMETRY LITERATURE (not assumed):
  • Beacon payload: LPWAN emergency telemetry uses very small messages — Sigfox-class
    devices carry ~12-byte uplinks; NB-IoT emergency trackers (e.g., Spain's V16 road
    beacon, GPS+ID+status) use compact payloads. We sweep 12-50 bytes (96-400 bits).
  • Reporting cadence: commercial NB-IoT trackers default to ~240 s reporting; SAR
    survivor beacons report every tens of seconds to minutes. We sweep 30-240 s.
  • Reconfiguration: passive-RIS reconfiguration is sub-ms to a few ms; we use 1 ms.

MODEL: the RIS is reconfigured on demand to serve whichever survivor has an active
beacon, each served under ITS OWN optimal phase (the realizable, full +7.55 dB case
— one beacon at a time, not 12 simultaneous streams). A round-robin/FIFO scheduler
serves beacons as they arrive. We measure the beacon delivery ratio (fraction of
beacons delivered before that survivor's next beacon) vs. number of survivors and
beacon cadence, and extract the survivor capacity at a 95% delivery SLA.

Run on Windows (channel_model.py in same folder):
    python ris_scheduling_v2.py
Outputs:
    results/SURVIVOR/ris_scheduling_grounded.png
    results/SURVIVOR/ris_scheduling_grounded.csv
"""
import os, time, csv, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import channel_model as cm
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
os.makedirs(OUT, exist_ok=True)

DEVICE_SEED = 42
EVAL_SEED   = 99
N_MC        = 300
UAV         = np.array([400.0, 523.0, 150.0])

# ── literature-grounded parameters ─────────────────────────────────────────────
PAYLOAD_BITS_DEFAULT = 96       # 12-byte beacon (GPS+ID+status), Sigfox-class
PERIOD_S_DEFAULT     = 60.0     # survivor beacon every 60 s (between SAR 30 s and NB-IoT 240 s)
RIS_RECONFIG_MS      = 1.0      # passive-RIS reconfiguration overhead
SLA_DELIVERY         = 0.95

np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]
FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]


def per_beacon_far_rate(n_mc=N_MC, seed=EVAL_SEED):
    """Instantaneous far rate when the RIS is configured for THAT survivor (its own slot)."""
    np.random.seed(seed)
    K = cm.CFG.N_PAIRS
    rates = []
    for _ in range(n_mc):
        G = cm.G_vector(UAV)
        for i in range(K):
            hn = cm.chan_scalar(UAV, NEAR[i], 0.0)
            hd = cm.chan_scalar(UAV, FAR[i], cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(FAR[i])
            phi = cm.opt_phi(G, hr)
            he  = cm.composite(hd, G, hr, phi)
            _, rF = cm.noma_pair(hn, he)
            rates.append(rF)
    return np.array(rates)


def airtime_ms(rate_bps, payload_bits):
    return payload_bits / np.maximum(rate_bps, 1e-9) * 1e3 + RIS_RECONFIG_MS


def sim_delivery(n_users, far_rates, payload_bits, period_s, sim_s=600.0, seed=1):
    """FIFO on-demand RIS scheduler. A beacon is delivered on time if it completes
    before that survivor's next beacon (period_s). Returns delivery ratio."""
    rng = np.random.default_rng(seed)
    events = []
    for u in range(n_users):
        t = rng.uniform(0, period_s)
        while t < sim_s:
            events.append((t, u)); t += rng.exponential(period_s)
    events.sort()
    ris_free = 0.0; on_time = 0; total = 0
    for arr, u in events:
        total += 1
        rate = far_rates[rng.integers(len(far_rates))]
        air = airtime_ms(rate, payload_bits) / 1e3
        start = max(arr, ris_free); finish = start + air
        ris_free = finish
        if (finish - arr) <= period_s:
            on_time += 1
    return on_time / max(total, 1)


def capacity_at_sla(far_rates, payload_bits, period_s, seeds=6):
    """Largest survivor count with delivery >= SLA (search a grid, then refine)."""
    grid = [12, 24, 48, 96, 192, 384, 768, 1536, 3072]
    best = 0
    for n in grid:
        d = np.mean([sim_delivery(n, far_rates, payload_bits, period_s, seed=s) for s in range(seeds)])
        if d >= SLA_DELIVERY:
            best = n
        else:
            break
    return best


def main():
    print("=" * 74)
    print("  ON-DEMAND RIS SCHEDULING (literature-grounded LPWAN beacon parameters)")
    print("=" * 74)
    t0 = time.time()
    fr = per_beacon_far_rate()
    print(f"  Per-beacon far rate (RIS configured for that survivor): "
          f"mean={fr.mean():.0f} bps, median={np.median(fr):.0f} bps, "
          f"10th pct={np.percentile(fr,10):.0f} bps")
    print(f"  Grounded defaults: payload={PAYLOAD_BITS_DEFAULT} bits "
          f"({PAYLOAD_BITS_DEFAULT//8} B), period={PERIOD_S_DEFAULT:.0f} s, "
          f"reconfig={RIS_RECONFIG_MS:.0f} ms")
    air = airtime_ms(fr, PAYLOAD_BITS_DEFAULT)
    print(f"  Mean beacon airtime (incl. reconfig) = {air.mean():.1f} ms "
          f"(median {np.median(air):.1f} ms)")
    print("-" * 74)

    # (1) delivery ratio vs survivor count at grounded defaults
    counts = [12, 24, 48, 96, 192, 384, 768, 1536]
    deliv = []
    for n in counts:
        d = np.mean([sim_delivery(n, fr, PAYLOAD_BITS_DEFAULT, PERIOD_S_DEFAULT, seed=s) for s in range(8)])
        deliv.append(d)
        print(f"  {n:5d} survivors: delivery {d*100:5.1f}%  {'OK' if d>=SLA_DELIVERY else '<'}")
    cap_default = max([n for n,d in zip(counts,deliv) if d>=SLA_DELIVERY], default=0)
    print(f"\n  Capacity at default (96-bit, 60 s): {cap_default} survivors @ {SLA_DELIVERY*100:.0f}% SLA")

    # (2) capacity vs beacon period (sparsity) for two payloads
    periods = [30, 60, 120, 240]
    cap_96  = [capacity_at_sla(fr, 96,  T) for T in periods]
    cap_400 = [capacity_at_sla(fr, 400, T) for T in periods]
    print("\n  Capacity vs beacon period:")
    for T, c9, c4 in zip(periods, cap_96, cap_400):
        print(f"    period {T:3d}s : {c9:5d} survivors (12 B) | {c4:5d} survivors (50 B)")

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax[0].plot(counts, [d*100 for d in deliv], "-o", color="#1f5fa8", lw=2, ms=6)
    ax[0].axhline(SLA_DELIVERY*100, color="#d62728", ls="--", lw=1.5,
                  label=f"{SLA_DELIVERY*100:.0f}% SLA")
    ax[0].axvline(cap_default, color="#2ca02c", ls=":", lw=1.5,
                  label=f"capacity ≈ {cap_default}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Survivors sharing one RIS-scheduled UAV")
    ax[0].set_ylabel("Beacon delivery ratio (%)")
    ax[0].set_title(f"On-demand beacon delivery\n(12-byte beacon, 60 s period — grounded)",
                    fontweight="bold")
    ax[0].grid(True, ls=":", alpha=0.5); ax[0].legend()

    x = np.arange(len(periods)); w = 0.36
    ax[1].bar(x-w/2, cap_96,  w, color="#2ca02c", label="12-byte beacon")
    ax[1].bar(x+w/2, cap_400, w, color="#9467bd", label="50-byte beacon")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{T}s" for T in periods])
    ax[1].set_xlabel("Beacon period per survivor")
    ax[1].set_ylabel(f"Survivors at {SLA_DELIVERY*100:.0f}% SLA")
    ax[1].set_title("Survivor capacity scales with beacon sparsity\n(RIS time-shared on demand)",
                    fontweight="bold")
    ax[1].set_yscale("log"); ax[1].grid(axis="y", ls=":", alpha=0.5); ax[1].legend()
    for i,(c9,c4) in enumerate(zip(cap_96,cap_400)):
        ax[1].text(i-w/2, c9, str(c9), ha="center", va="bottom", fontsize=8)
        ax[1].text(i+w/2, c4, str(c4), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Realizable multiuser RIS: full per-beacon gain via on-demand reconfiguration; "
                 "sparse LPWAN telemetry makes large survivor populations feasible",
                 fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "ris_scheduling_grounded.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Plot → {p}")

    with open(os.path.join(OUT, "ris_scheduling_grounded.csv"), "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["per_beacon_mean_bps", f"{fr.mean():.0f}"])
        w_.writerow(["n_survivors", "delivery_ratio_at_default"])
        for n,d in zip(counts,deliv): w_.writerow([n, f"{d:.3f}"])
        w_.writerow([]); w_.writerow(["period_s","cap_12B","cap_50B"])
        for T,c9,c4 in zip(periods,cap_96,cap_400): w_.writerow([T,c9,c4])
    print(f"  CSV → {os.path.join(OUT,'ris_scheduling_grounded.csv')}")
    print(f"  Runtime: {time.time()-t0:.0f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
