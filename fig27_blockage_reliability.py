#!/usr/bin/env python3
r"""
fig27_blockage_reliability.py  —  Paper Fig. 27
═══════════════════════════════════════════════════════════════════════════════
Far-user telemetry reliability versus direct-path blockage depth.

Plots the three reliability curves over a 40–80 dB blockage sweep:
  * survivor-aware UAV position (served via the RIS)   — holds ~100%
  * sum-rate UAV position       (served via the RIS)   — degrades to ~55%
  * unaided direct path         (no RIS)               — collapses to 0 beyond ~55 dB

Data source: blockage_sensitivity.json (produced by blockage_sensitivity.py).
If the JSON is absent, the sweep is recomputed here with the IDENTICAL physics
(channel_model.py, DEVICE_SEED=42, EVAL_SEED=99, N_MC=60, SLA=100 bps), so the
script is fully self-contained.

No figure title is drawn — the IEEE caption provides it.

Output (300 dpi):  ../results/FIGS_13_16/fig27_blockage_reliability.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "results", "FIGS_13_16"))
os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(HERE, "blockage_sensitivity.json")


def compute_sweep():
    """Recompute the sweep if the JSON is missing — same physics as blockage_sensitivity.py."""
    import channel_model as cm
    DEVICE_SEED, EVAL_SEED, SLA_BPS, N_MC = 42, 99, 100.0, 60
    np.random.seed(DEVICE_SEED)
    POS, _ = cm.generate_devices()
    NEAR = POS[:cm.CFG.K_NEAR]
    FAR = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
    K = cm.CFG.N_PAIRS

    def eval_pos(uav, seed=EVAL_SEED):
        np.random.seed(seed)
        f_ris, f_dir = [], []
        for _ in range(N_MC):
            G = cm.G_vector(uav)
            dr, dd = np.zeros(K), np.zeros(K)
            for i in range(K):
                hn = cm.chan_scalar(uav, NEAR[i], 0.0)
                hd = cm.chan_scalar(uav, FAR[i], cm.CFG.BLOCK_DB)   # blocked direct
                hr = cm.H_ris_dev(FAR[i])
                phi = cm.opt_phi(G, hr)
                he = cm.composite(hd, G, hr, phi)                   # RIS bypass
                _, rF_ris = cm.noma_pair(hn, he)
                _, rF_dir = cm.noma_pair(hn, hd)
                dr[i], dd[i] = rF_ris, rF_dir
            f_ris.append(dr)
            f_dir.append(dd)
        f_ris, f_dir = np.array(f_ris), np.array(f_dir)
        return (float((f_ris >= SLA_BPS).mean() * 100),
                float((f_dir >= SLA_BPS).mean() * 100))

    SUMRATE_OPT = np.array([450.0, 577.0, 108.0])
    SURVIVOR_OPT = np.array([400.0, 523.0, 150.0])
    blocks = list(range(40, 81, 5))
    res = {"block_dB": blocks, "sumrate_opt": [], "survivor_opt": [], "no_ris_sumrate": []}
    for b in blocks:
        cm.CFG.BLOCK_DB = float(b)
        rel_sr, rel_dir = eval_pos(SUMRATE_OPT)
        rel_sv, _ = eval_pos(SURVIVOR_OPT)
        res["sumrate_opt"].append(rel_sr)
        res["survivor_opt"].append(rel_sv)
        res["no_ris_sumrate"].append(rel_dir)
    return res


# ── load (or recompute) the sweep data ──────────────────────────────────────────
if os.path.isfile(JSON_PATH):
    res = json.load(open(JSON_PATH, encoding="utf-8"))
    print(f"[loaded] {JSON_PATH}")
else:
    print("[info] blockage_sensitivity.json not found — recomputing the sweep ...")
    res = compute_sweep()

x = np.array(res["block_dB"], dtype=float)
sv = np.array(res["survivor_opt"], dtype=float)
sr = np.array(res["sumrate_opt"], dtype=float)
nr = np.array(res["no_ris_sumrate"], dtype=float)

# ── figure ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10})
fig, ax = plt.subplots(figsize=(8.6, 5.0))

# Regime shading: where the direct path alone suffices vs. where the RIS is essential.
ax.axvspan(38, 52, color="#e8f4ea", zorder=0)
ax.axvspan(55, 82, color="#fdecea", zorder=0)

ax.plot(x, sv, "-o", color="#2ca02c", lw=2.2, ms=6, zorder=4,
        label="Survivor-aware position (RIS)")
ax.plot(x, sr, "--s", color="#1f77b4", lw=2.0, ms=5, zorder=4,
        label="Sum-rate position (RIS)")
ax.plot(x, nr, "-.^", color="#d62728", lw=2.0, ms=5, zorder=4,
        label="Unaided direct path (no RIS)")

ax.axvline(60, color="#666", lw=1.0, ls=":", zorder=2)
ax.text(60.4, 4, "nominal\n60 dB", fontsize=8.5, color="#444", va="bottom")

ax.text(45, 106, "direct path\nsuffices", ha="center", va="top",
        fontsize=9, color="#2e7d32")
ax.text(68.5, 106, "RIS essential (deep blockage)", ha="center", va="top",
        fontsize=9, color="#b03a2e")

ax.set_xlabel("Direct-path blockage depth (dB)")
ax.set_ylabel("Far-user telemetry reliability at 100 bps SLA (%)")
ax.set_xlim(38, 82)
ax.set_ylim(-3, 110)
ax.set_xticks(x)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="center left")
# Figure title intentionally omitted; the IEEE caption provides it.

fig.text(0.5, -0.02,
         "Monte Carlo over 60 channel realisations (seed 99); SLA = 100 bps; "
         "channel_model.py physics.",
         ha="center", fontsize=8, style="italic", color="#555")
plt.tight_layout()

out = os.path.join(OUT, "fig27_blockage_reliability.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out}")
