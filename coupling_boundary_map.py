#!/usr/bin/env python3
"""
coupling_boundary_map.py
═══════════════════════════════════════════════════════════════════════════════
Reviewer Q1 — explicit conditions under which the "fly toward the reflector"
coupling law HOLDS or FAILS, mapped over BLOCKAGE DEPTH and RICIAN K-FACTOR.

The headline coupling result (far-user rate governed by the UAV->RIS separation,
not the UAV->far separation; rho(d_RIS) = -0.84) was established at ONE operating
point: 60 dB blockage, K = 10 dB. The reviewer asks for the boundary of that
regime in terms of UAV-RIS distance ranges, blockage levels and K-factors. The
distance dependence IS what the correlation measures (rho is computed over a grid
of UAV->RIS distances); this script maps, for each (blockage, K) cell, whether
that distance law dominates.

FAITHFUL EXTENSION of uav_ris_coupling.py (TEST B). IDENTICAL:
    • per_user physics — every channel quantity comes from channel_model.py;
      nothing is reimplemented here
    • 26x26 grid GX in [150,800], GY in [150,900], H = 120 m
    • EVAL_SEED = 99, device seed = 42, RIS at baseline (300,250)
    • classification rule
        holds     : |rho(d_RIS)| > |rho(d_far)| + 0.2   (rule dominates)
        inverts   : |rho(d_RIS)| < |rho(d_far)|         (rule fails)
        conflated : otherwise                            (predictors ambiguous)
The ONLY additions are two outer sweeps:
    • blockage depth   cm.CFG.BLOCK_DB  in {40,50,60,70,80} dB
    • Rician K-factor  cm.CFG.K_RICIAN  in {0,5,10,15,20} dB   (system-wide)
Both are config knobs in channel_model.py, set then restored per cell — the
device layout (generated once at seed 42, before any change) is held fixed, so
only the channel realisations vary with K and only the direct far-path varies
with blockage (the RIS path is unblocked, exactly as in the manuscript model).

VALIDATION ANCHOR: the (BLOCK=60 dB, K=10 dB) cell is the published operating
point and MUST reproduce rho(d_RIS) ~ -0.84 at N_MC=30 (~ -0.82 at N_MC=10; the
<0.02 N_MC deviation is the one verified in coupling_conditional_robustness.py).
Run this cell first; if it does not match, the harness is wrong — do not trust
the rest of the map.

Resumable: each (blockage,K) cell is appended to
results/SURVIVOR/boundary_cells.json immediately, so a timeout never loses a
completed cell. Re-running skips cells already present.

USAGE
  python coupling_boundary_map.py <cell_start> <cell_count> [n_mc]
      -> computes that block of cells (row-major over blockage), appends to JSON
  python coupling_boundary_map.py finalize
      -> reads the JSON, prints the rho/regime tables, writes the heatmap figure

WINDOWS (authoritative): channel_model.py in the same folder.
  python coupling_boundary_map.py 0 25 30      # full 5x5 map, N_MC=30 (~42 min)
  python coupling_boundary_map.py finalize
Physics-only -> reproduces exactly in-container; N_MC=10 is a valid <0.02 preview.
"""
import os, sys, json, time, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import channel_model as cm
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "SURVIVOR"))
if not os.path.isdir(os.path.dirname(OUT)):
    OUT = os.path.join(HERE, "results", "SURVIVOR")
os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(OUT, "boundary_cells.json")

# ── EXACT uav_ris_coupling.py settings ────────────────────────────────────────
EVAL_SEED   = 99
N_MC_DEF    = 30
H_FIXED     = 120.0
GX = np.linspace(150, 800, 26)
GY = np.linspace(150, 900, 26)
RIS_XY      = cm.CFG.RIS_POS[:2].copy()        # baseline (300,250)
DEVICE_SEED = 42

# ── boundary sweep axes ───────────────────────────────────────────────────────
BLOCK_DB_VALUES = [40.0, 50.0, 60.0, 70.0, 80.0]
KDB_VALUES      = [0.0, 5.0, 10.0, 15.0, 20.0]
CELLS  = [(b, k) for b in BLOCK_DB_VALUES for k in KDB_VALUES]   # row-major over blockage
ANCHOR = (60.0, 10.0)                          # published operating point

# fixed device layout (seed 42) — identical to uav_ris_coupling.py
np.random.seed(DEVICE_SEED)
POS, _ = cm.generate_devices()
NEAR = POS[:cm.CFG.K_NEAR]
FAR  = POS[cm.CFG.K_NEAR:cm.CFG.K_SC1]
FAR_CENTROID = FAR[:, :2].mean(0)


def per_user_far(uav, n_mc, seed=EVAL_SEED):
    """Far-user mean rate at uav. Identical orchestration to uav_ris_coupling.py's
    per_user (RIS-assisted far branch); all physics from channel_model.py."""
    np.random.seed(seed)
    K = cm.CFG.N_PAIRS
    far_samp = []
    for _ in range(n_mc):
        G = cm.G_vector(uav)
        df = np.zeros(K)
        for i in range(K):
            hn = cm.chan_scalar(uav, NEAR[i], 0.0)
            hd = cm.chan_scalar(uav, FAR[i],  cm.CFG.BLOCK_DB)
            hr = cm.H_ris_dev(FAR[i])
            phi = cm.opt_phi(G, hr)
            he  = cm.composite(hd, G, hr, phi)
            _,  rF = cm.noma_pair(hn, he)
            df[i] = rF
        far_samp.append(df)
    return float(np.array(far_samp).mean())


def classify(sr_ris, sr_far):
    a, b = abs(sr_ris), abs(sr_far)
    if a > b + 0.2:   return "holds"
    elif a < b:       return "inverts"
    else:             return "conflated"


def sweep_one_cell(block_db, kdb, n_mc):
    """Full 26x26 grid -> coupling rho for one (blockage, K) operating point."""
    orig_block = cm.CFG.BLOCK_DB
    orig_k     = cm.CFG.K_RICIAN
    cm.CFG.BLOCK_DB = float(block_db)
    cm.CFG.K_RICIAN = float(10 ** (kdb / 10.0))
    far_rate, d_ris, d_far = [], [], []
    for y in GY:
        for x in GX:
            uav = np.array([x, y, H_FIXED])
            far_rate.append(per_user_far(uav, n_mc))
            d_ris.append(np.linalg.norm(uav[:2] - RIS_XY))
            d_far.append(np.linalg.norm(uav[:2] - FAR_CENTROID))
    cm.CFG.BLOCK_DB = orig_block
    cm.CFG.K_RICIAN = orig_k
    far_rate = np.array(far_rate)
    log_far  = np.log10(np.maximum(far_rate, 1e-3))
    sr_ris, _ = spearmanr(np.array(d_ris), far_rate)
    sr_far, _ = spearmanr(np.array(d_far), far_rate)
    pr_ris, _ = pearsonr(np.array(d_ris), log_far)
    return dict(block_db=float(block_db), kdb=float(kdb), n_mc=int(n_mc),
                spearman_dris=float(sr_ris), spearman_dfar=float(sr_far),
                pearson_dris=float(pr_ris),
                far_rate_mean=float(far_rate.mean()),
                far_rate_max=float(far_rate.max()),
                regime=classify(sr_ris, sr_far))


def _cid(block_db, kdb):
    return f"{block_db:.0f}_{kdb:.0f}"


def load_results():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            return json.load(f)
    return []


def save_result(rec):
    cid = _cid(rec["block_db"], rec["kdb"])
    data = [d for d in load_results() if _cid(d["block_db"], d["kdb"]) != cid]
    data.append(rec)
    data.sort(key=lambda d: (d["block_db"], d["kdb"]))
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)


def run_block(cell_start, cell_count, n_mc):
    done = {_cid(d["block_db"], d["kdb"]) for d in load_results()}
    sel  = CELLS[cell_start:cell_start + cell_count]
    print(f"[Boundary] cells {cell_start}..{cell_start + len(sel) - 1} of "
          f"{len(CELLS)} (N_MC={n_mc}); {len(done)} already done")
    t0 = time.time()
    for (b, k) in sel:
        if _cid(b, k) in done:
            print(f"  cell block={b:>2.0f}dB K={k:>2.0f}dB: present, skipping")
            continue
        ts  = time.time()
        rec = sweep_one_cell(b, k, n_mc)
        save_result(rec)
        flag = "  <-- ANCHOR (expect rho_RIS ~ -0.84)" if (b, k) == ANCHOR else ""
        print(f"  cell block={b:>2.0f}dB K={k:>2.0f}dB | rho(d_RIS)={rec['spearman_dris']:+.3f} "
              f"rho(d_far)={rec['spearman_dfar']:+.3f} -> {rec['regime']:<9} "
              f"far_mean={rec['far_rate_mean']:.0f}bps [{time.time()-ts:.0f}s, "
              f"total {(time.time()-t0)/60:.1f}min]{flag}")


def finalize():
    data = load_results()
    if not data:
        print("No cells computed yet."); return
    look = {_cid(d["block_db"], d["kdb"]): d for d in data}
    nb, nk = len(BLOCK_DB_VALUES), len(KDB_VALUES)
    DRIS = np.full((nb, nk), np.nan)
    DFAR = np.full((nb, nk), np.nan)
    MARG = np.full((nb, nk), np.nan)        # |rho_RIS| - |rho_far| (dominance margin)
    REG  = np.full((nb, nk), "", dtype=object)
    for ib, b in enumerate(BLOCK_DB_VALUES):
        for ik, k in enumerate(KDB_VALUES):
            d = look.get(_cid(b, k))
            if d is None:
                continue
            DRIS[ib, ik] = d["spearman_dris"]
            DFAR[ib, ik] = d["spearman_dfar"]
            MARG[ib, ik] = abs(d["spearman_dris"]) - abs(d["spearman_dfar"])
            REG[ib, ik]  = d["regime"]

    a = look.get(_cid(*ANCHOR))
    print("=" * 88)
    print("  Q1 COUPLING-LAW BOUNDARY MAP  (Spearman rho over 26x26 grid, per cell)")
    print("=" * 88)
    if a:
        print(f"  ANCHOR (block=60 dB, K=10 dB): rho(d_RIS)={a['spearman_dris']:+.3f}  "
              f"(published -0.84; N_MC={a['n_mc']})  ->  {a['regime']}")
    print(f"  Completed {len(data)}/{len(CELLS)} cells")
    print("-" * 88)
    print("  rho(d_RIS)   rows = blockage (dB), cols = Rician K (dB):")
    print("           " + "".join(f"K={k:>4.0f} " for k in KDB_VALUES))
    for ib, b in enumerate(BLOCK_DB_VALUES):
        row = "".join(f"{DRIS[ib,ik]:+7.3f}" if not np.isnan(DRIS[ib,ik]) else "   --  "
                      for ik in range(nk))
        print(f"  {b:>4.0f}dB {row}")
    print()
    print("  regime   rows = blockage (dB), cols = Rician K (dB):")
    print("           " + "".join(f"K={k:>4.0f} " for k in KDB_VALUES))
    for ib, b in enumerate(BLOCK_DB_VALUES):
        row = "".join(f"{(REG[ib,ik][:5] if REG[ib,ik] else '-'):>7}" for ik in range(nk))
        print(f"  {b:>4.0f}dB {row}")
    holds = int((REG == "holds").sum())
    inv   = int((REG == "inverts").sum())
    conf  = int((REG == "conflated").sum())
    filled = holds + inv + conf
    print("-" * 88)
    print(f"  holds {holds}/{filled} | inverts {inv}/{filled} | conflated {conf}/{filled}")
    print("=" * 88)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    im = ax.imshow(MARG, origin="lower", cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(nk)); ax.set_xticklabels([f"{k:.0f}" for k in KDB_VALUES])
    ax.set_yticks(range(nb)); ax.set_yticklabels([f"{b:.0f}" for b in BLOCK_DB_VALUES])
    ax.set_xlabel("Rician K-factor (dB)")
    ax.set_ylabel("Blockage depth (dB)")
    # figure title omitted; the IEEE caption provides it
    for ib in range(nb):
        for ik in range(nk):
            if not np.isnan(MARG[ib, ik]):
                ax.text(ik, ib, f"{MARG[ib,ik]:+.2f}\n{REG[ib,ik][:4]}",
                        ha="center", va="center", fontsize=7.5, color="black")
    plt.colorbar(im, ax=ax, label="dominance margin")
    plt.tight_layout()
    p = os.path.join(OUT, "coupling_boundary_map.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[saved] {p}")

    with open(os.path.join(OUT, "boundary_summary.json"), "w") as f:
        json.dump(dict(block_db=BLOCK_DB_VALUES, kdb=KDB_VALUES,
                       rho_dris=DRIS.tolist(), rho_dfar=DFAR.tolist(),
                       margin=MARG.tolist(), regime=REG.tolist(),
                       counts=dict(holds=holds, inverts=inv, conflated=conf),
                       anchor=a), f, indent=2)
    print(f"[saved] {os.path.join(OUT,'boundary_summary.json')}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "finalize":
        finalize()
    elif len(sys.argv) >= 3:
        start = int(sys.argv[1]); count = int(sys.argv[2])
        nmc = int(sys.argv[3]) if len(sys.argv) >= 4 else N_MC_DEF
        run_block(start, count, nmc)
    else:
        print("No arguments -> running full default 5x5 boundary map at "
              "N_MC=%d, then finalize.\n"
              "(For blocks or resuming a partial run, see USAGE in the header.)\n"
              % N_MC_DEF)
        run_block(0, len(CELLS), N_MC_DEF)
        finalize()
