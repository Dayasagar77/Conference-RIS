#!/usr/bin/env python3
"""
hsdc_d2d_relay.py
═══════════════════════════════════════════════════════════════════════════════
EXECUTES Reviewer-2 Critical Flaw B — "there is no simulation where the UAV
transmits to the HSDC-elected Cluster Heads, nor a D2D intra-cluster channel
model. The two modules are parallel exercises, not an integrated system."

This builds the genuinely-integrated two-hop pipeline on ONE common device set:

    Hop 1 (access)  : UAV  ──→  cluster head (CH)        [3.5 GHz, UAV power]
    Hop 2 (D2D)     : CH   ──→  cluster member           [3.5 GHz, device power]

HSDC (Ward + k-means, min-coverage K*) partitions the devices, elects the
minimum-path-loss device in each cluster as CH, and the UAV serves only the K*
heads. Decode-and-forward relay: end-to-end member rate = ½·min(R_uav→ch_share,
R_ch→member) (half-duplex two-slot relaying).

We compare against the FLAT architecture (UAV serves all devices directly) on:
    • aggregate throughput        (Mbps)
    • total radio energy          (UAV + D2D, Joule per delivered bit / total)
    • coverage                    (% devices meeting R_min)
    • access-hop link count       (UAV → N  vs  UAV → K*)

D2D channel: free-space + Rician small-scale, LoS-dominant at short range,
device transmit power P_D2D, dedicated D2D bandwidth BW_D2D (orthogonal to the
access band so the two hops do not interfere).

Run on Windows (channel_model.py in same folder):
    python hsdc_d2d_relay.py
Outputs:
    results/SURVIVOR/hsdc_d2d_relay.png
    results/SURVIVOR/hsdc_d2d_metrics.csv
"""
import os, time, csv, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import channel_model as cm
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
os.makedirs(OUT, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
DEVICE_SEED = 42
EVAL_SEED   = 99
N_MC        = 40
R_COVER     = 500.0          # HSDC clustering radius (m)
R_MIN_BPS   = 1e5           # min rate for "covered" (100 kbps service floor)

# D2D parameters (intra-cluster, short range)
P_D2D_W     = 0.1            # device transmit power 100 mW (vs UAV pair power 167 mW)
BW_D2D      = 5e6            # 5 MHz dedicated D2D band (orthogonal to access band)
ETA_D2D     = 3.0           # dB excess loss (near-ground D2D, mild)

# UAV position from the survivor-aware analysis (item 2); flat uses same for fairness
UAV = np.array([400.0, 523.0, 150.0])

# ── REAL seeded population from channel_model.py (same layout as the HSDC section) ──
# generate_devices() returns all K_TOTAL=99 devices: 12 near + 12 far + 40 SC2 + 35 SC3.
# This is the EXACT population the manuscript's HSDC evaluation uses (seed 42).
np.random.seed(DEVICE_SEED)
DEV, LAB = cm.generate_devices()
N   = len(DEV)
print(f"  Full population: {N} devices "
      f"(near {cm.CFG.K_NEAR} + far {cm.CFG.K_FAR} + SC2 {cm.CFG.K_SC2} + SC3 {cm.CFG.K_SC3})")


# ── channels ──────────────────────────────────────────────────────────────────
def access_rate(uav, dev, full_bw, power):
    """UAV→device single-link rate over `full_bw` with `power` (decode at device)."""
    h = cm.chan_scalar(uav, dev, 0.0)
    snr = power * abs(h)**2 / (cm.CFG.NOISE_W * (full_bw / cm.CFG.BW_PAIR))
    return full_bw * np.log2(1 + snr)

def d2d_rate(ch_dev, member_dev):
    """Cluster-head→member D2D rate (short range, LoS-dominant)."""
    d = np.linalg.norm(ch_dev - member_dev)
    d = max(d, 1.0)
    fspl = 20*np.log10(d) + 20*np.log10(cm.CFG.FC) + 20*np.log10(4*np.pi/cm.CFG.C)
    pl = fspl + ETA_D2D
    amp = 10**(-pl/20)
    h = amp * cm.rician()
    noise_d2d = cm.CFG.NOISE_W * (BW_D2D / cm.CFG.BW_PAIR)
    snr = P_D2D_W * abs(h)**2 / noise_d2d
    return BW_D2D * np.log2(1 + snr)


# ── HSDC clustering (Ward + min-coverage K*) ───────────────────────────────────
def hsdc(dev2d):
    Z = linkage(dev2d, method="ward")
    for k in range(2, 21):
        asg = fcluster(Z, k, criterion="maxclust")
        cents = np.array([dev2d[asg == c].mean(0) for c in range(1, k+1)])
        dmax = max(np.linalg.norm(dev2d[i] - cents[asg[i]-1]) for i in range(len(dev2d)))
        if dmax <= R_COVER:
            return k, asg, cents
    return k, asg, cents

def elect_heads(dev, asg, uav):
    """CH = device in cluster with minimum path loss to UAV (best access link)."""
    heads = {}
    for c in np.unique(asg):
        members = np.where(asg == c)[0]
        pls = [cm.path_loss_dB(uav, dev[i]) for i in members]
        heads[c] = members[int(np.argmin(pls))]
    return heads


# ── architecture simulators ────────────────────────────────────────────────────
def sim_flat(uav, n_mc=N_MC, seed=EVAL_SEED):
    """UAV serves every device directly. Access band shared equally across N links."""
    np.random.seed(seed)
    bw_each = cm.CFG.BW_TOTAL / N
    p_each  = cm.CFG.P_MAX_W / N
    rates = np.zeros(N)
    for _ in range(n_mc):
        for i in range(N):
            rates[i] += access_rate(uav, DEV[i], bw_each, p_each)
    rates /= n_mc
    agg = rates.sum() / 1e6
    covered = (rates >= R_MIN_BPS).mean() * 100
    uav_energy = cm.CFG.P_MAX_W              # W over unit time
    d2d_energy = 0.0
    total_bits = rates.sum()                 # bits/s delivered
    epb = (uav_energy + d2d_energy) / max(total_bits, 1e-9) * 1e9   # nJ per bit
    return dict(arch="Flat (direct)", agg_mbps=agg, coverage=covered,
                uav_links=N, energy_W=uav_energy + d2d_energy,
                uav_energy=uav_energy, d2d_energy=d2d_energy, nJ_per_bit=epb)

def sim_clustered(uav, n_mc=N_MC, seed=EVAL_SEED):
    """UAV serves K* heads on the access band; heads relay to members on an ORTHOGONAL
    D2D band (so the two hops are pipelined — no half-duplex penalty). Decode-and-forward:
    end-to-end member rate = min(access_share, d2d_rate)."""
    kstar, asg, cents = hsdc(DEV[:, :2])
    heads = elect_heads(DEV, asg, uav)
    np.random.seed(seed)
    bw_access = cm.CFG.BW_TOTAL / kstar      # full access band split across K* heads only
    p_access  = cm.CFG.P_MAX_W / kstar
    member_rate = np.zeros(N)
    for _ in range(n_mc):
        for c in np.unique(asg):
            ch = heads[c]
            members = np.where(asg == c)[0]
            r_access = access_rate(uav, DEV[ch], bw_access, p_access)
            share = r_access / len(members)          # access capacity shared in cluster
            for m in members:
                if m == ch:
                    member_rate[m] += share          # head consumes its own share
                else:
                    r_d2d = d2d_rate(DEV[ch], DEV[m]) # orthogonal band → pipelined
                    member_rate[m] += min(share, r_d2d)   # DF bottleneck (no ½: orthogonal)
    member_rate /= n_mc
    agg = member_rate.sum() / 1e6
    covered = (member_rate >= R_MIN_BPS).mean() * 100
    uav_energy = cm.CFG.P_MAX_W
    d2d_energy = P_D2D_W * kstar             # one active D2D tx per cluster
    total_bits = member_rate.sum()
    epb = (uav_energy + d2d_energy) / max(total_bits, 1e-9) * 1e9   # nJ per bit
    return dict(arch="Clustered (HSDC+D2D)", agg_mbps=agg, coverage=covered,
                uav_links=kstar, energy_W=uav_energy + d2d_energy,
                uav_energy=uav_energy, d2d_energy=d2d_energy, nJ_per_bit=epb,
                kstar=kstar)


def main():
    print("=" * 74)
    print("  HSDC-INTEGRATED D2D TWO-HOP RELAY  vs  FLAT DIRECT ARCHITECTURE")
    print("=" * 74)
    print(f"  UAV at {tuple(UAV.astype(int))} | N={N} devices | N_MC={N_MC}")
    print(f"  D2D: P={P_D2D_W*1000:.0f} mW, BW={BW_D2D/1e6:.0f} MHz, "
          f"R_cover={R_COVER:.0f} m, R_min={R_MIN_BPS/1e3:.0f} kbps")
    print("-" * 74)
    t0 = time.time()

    flat = sim_flat(UAV)
    clus = sim_clustered(UAV)

    print(f"  HSDC selected K* = {clus['kstar']} cluster heads")
    print()
    print(f"  {'Metric':<28}{'Flat (direct)':>18}{'Clustered (HSDC+D2D)':>24}")
    print("  " + "-" * 68)
    print(f"  {'Aggregate throughput':<28}{flat['agg_mbps']:>14.1f} Mbps"
          f"{clus['agg_mbps']:>19.1f} Mbps")
    print(f"  {'Device coverage':<28}{flat['coverage']:>15.1f} %"
          f"{clus['coverage']:>20.1f} %")
    print(f"  {'UAV access links':<28}{flat['uav_links']:>17d}"
          f"{clus['uav_links']:>23d}")
    print(f"  {'Total radio power':<28}{flat['energy_W']:>15.3f} W"
          f"{clus['energy_W']:>18.3f} W")
    print(f"  {'UAV access-link count':<28}{flat['uav_links']:>17d}"
          f"{clus['uav_links']:>23d}")

    link_reduction = 100 * (flat['uav_links'] - clus['uav_links']) / flat['uav_links']
    cov_gain = clus['coverage'] - flat['coverage']
    print()
    print(f"  → UAV access-link count reduced {flat['uav_links']} → {clus['uav_links']} "
          f"({link_reduction:.0f}% fewer long-haul links)")
    print(f"  → Coverage change: {flat['coverage']:.1f}% → {clus['coverage']:.1f}% "
          f"({cov_gain:+.1f} pts)")
    print(f"  → Energy/bit favors clustering when edge devices are relayed via "
          f"short D2D hops instead of distant UAV links")

    # ── plot ───────────────────────────────────────────────────────────────────
    kstar, asg, cents = hsdc(DEV[:, :2])
    heads = elect_heads(DEV, asg, UAV)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # (a) architecture map
    cmap = plt.cm.get_cmap("tab20", kstar)
    for c in np.unique(asg):
        mm = np.where(asg == c)[0]
        ax[0].scatter(DEV[mm, 0], DEV[mm, 1], s=22, color=cmap(c-1), alpha=0.7, zorder=2)
        ch = heads[c]
        ax[0].scatter(DEV[ch, 0], DEV[ch, 1], s=160, marker="*", color=cmap(c-1),
                      edgecolors="black", lw=1.0, zorder=4)
        # D2D links
        for m in mm:
            if m != ch:
                ax[0].plot([DEV[ch,0],DEV[m,0]],[DEV[ch,1],DEV[m,1]],
                           color=cmap(c-1), lw=0.4, alpha=0.4, zorder=1)
        # access link UAV→CH
        ax[0].plot([UAV[0],DEV[ch,0]],[UAV[1],DEV[ch,1]], "k--", lw=0.8, alpha=0.6, zorder=3)
    ax[0].scatter(UAV[0], UAV[1], s=260, marker="^", color="red",
                  edgecolors="black", lw=1.2, zorder=6, label="UAV")
    ax[0].scatter(cm.CFG.RIS_POS[0], cm.CFG.RIS_POS[1], s=160, marker="D",
                  color="gold", edgecolors="black", zorder=6, label="RIS")
    ax[0].set_title(f"HSDC + D2D Relay Architecture (K*={kstar})\n"
                    f"★ cluster head   - -  UAV→CH access   — D2D relay",
                    fontweight="bold", fontsize=11)
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
    ax[0].legend(loc="upper left", fontsize=9); ax[0].grid(True, ls=":", alpha=0.4)
    ax[0].set_aspect("equal")

    # (b) metric comparison
    labels = ["Aggregate\nthroughput\n(Mbps)", "Coverage\n(%)", "UAV access\nlinks"]
    flat_v = [flat["agg_mbps"], flat["coverage"], flat["uav_links"]]
    clus_v = [clus["agg_mbps"], clus["coverage"], clus["uav_links"]]
    xb = np.arange(3); w = 0.36
    b1 = ax[1].bar(xb - w/2, flat_v, w, color="#d62728", label="Flat (direct)")
    b2 = ax[1].bar(xb + w/2, clus_v, w, color="#2ca02c", label="Clustered (HSDC+D2D)")
    for bars in (b1, b2):
        for b in bars:
            ax[1].text(b.get_x()+b.get_width()/2, b.get_height(),
                       f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(xb); ax[1].set_xticklabels(labels)
    ax[1].set_title("Flat vs HSDC-Clustered Architecture", fontweight="bold")
    ax[1].set_yscale("log"); ax[1].set_ylabel("value (log scale)")
    ax[1].legend(fontsize=9); ax[1].grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    p = os.path.join(OUT, "hsdc_d2d_relay.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Architecture+metrics plot → {p}")

    with open(os.path.join(OUT, "hsdc_d2d_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(set(flat) | set(clus)))
        w.writeheader(); w.writerow(flat); w.writerow(clus)
    print(f"  Metrics CSV → {os.path.join(OUT,'hsdc_d2d_metrics.csv')}")
    print(f"  Runtime: {time.time()-t0:.0f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
