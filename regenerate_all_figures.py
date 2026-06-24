#!/usr/bin/env python3
r"""
regenerate_all_figures.py
═══════════════════════════════════════════════════════════════════════════════
Regenerate every paper figure and collect them ALL into ONE folder:

        results/ALL_FIGURES/

Why this script exists
----------------------
The figure-producing scripts each save into their own sub-folder, and across
TWO different "results" roots:

    <repo>/results/channel_plots/   (channel_model.py  -> Fig 1,2,3,4,9)
    <repo>/results/hsdc/            (hsdc.py           -> Fig 5,6,7,8)
    <repo>/results/dql/             (dql_agent.py      -> Fig 10-13)
    <repo>/results/benchmark/       (benchmark_ablation.py)
    <repo>/../results/FIGS_13_16/   (fig13-16_FAITHFUL -> paper Fig 14-17)
    <repo>/../results/SURVIVOR/     (coupling / pareto / fairness / system model)
    <repo>/../results/hq_figures/   (per_user_cdf.py)

This driver runs each generator, then copies every PNG produced during this run
into one tidy folder so you can drop them straight into the manuscript.

Usage
-----
    python regenerate_all_figures.py                # FAST group (most figures, minutes)
    python regenerate_all_figures.py --all          # FAST + SLOW (retraining + seed sweeps)
    python regenerate_all_figures.py --include-rl    # also run the 2-3 h SB3 RL baseline
    python regenerate_all_figures.py --only hsdc.py fig15_hsdc_sc1_FAITHFUL.py
    python regenerate_all_figures.py --fresh         # wipe ALL_FIGURES/ before collecting
    python regenerate_all_figures.py --collect-existing   # also copy already-present PNGs
    python regenerate_all_figures.py --timeout 1800  # kill any single script after 30 min

Notes
-----
* Runs on Windows (per the project constraint that the DQL modules run on Windows).
* Each script runs in its own subprocess with MPLBACKEND=Agg, so no windows pop up
  and one script's state can't corrupt another's.
* A failing script is reported but does NOT stop the run.
* channel_model.py must be importable from this folder (it is, by default).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "results", "ALL_FIGURES")

# The two "results" roots the various scripts write into.
RESULT_ROOTS = [
    os.path.join(HERE, "results"),
    os.path.normpath(os.path.join(HERE, "..", "results")),
]

# ── FAST: deterministic figure generators (seconds to a couple of minutes each) ──
FAST = [
    "channel_model.py",                 # Fig 1,2,3,4,9
    "hsdc.py",                          # Fig 5,6,7,8
    "system_model_figure.py",          # system-model schematic
    "fig13_per_user_cdf_FAITHFUL.py",  # paper Fig 14
    "fig14_multi_geometry_FAITHFUL.py",# paper Fig 15
    "fig15_hsdc_sc1_FAITHFUL.py",      # paper Fig 16
    "fig16_telemetry_outage_FAITHFUL.py",  # paper Fig 17
    "fig27_blockage_reliability.py",   # paper Fig 27 (reads blockage_sensitivity.json)
    "survivor_aware_pareto.py",        # Fig 22 (Pareto frontier)
    "baseline_comparison_figure.py",
    "hsdc_d2d_relay.py",               # Fig 24
    "per_user_cdf.py",
    "fairness_oma_pareto.py",
    "fairness_reward_comparison.py",
    "phase_error_sweep.py",
    "hsdc_sensitivity.py",
    "ris_m_blockage_surface.py",
    "uav_ris_coupling.py",             # Fig 18-21 (coupling surfaces) — a few minutes
]

# ── SLOW: retraining and multi-seed sweeps (many minutes each) ──
SLOW = [
    "dql_agent.py",                    # Fig 10-13 — retrains the DQL agent (~minutes)
    "survivor_aware_dql.py",           # Fig 23 — retrains under the survivor-aware reward
    "rnorm_ablation.py",               # Fig 25 — R_norm ablation (multi-seed)
    "blockage_sensitivity.py",         # Fig 27 — blockage-depth sweep
    "coupling_robustness.py",
    "coupling_topology_robustness.py", # Fig 26 — 20-topology robustness
    "coupling_conditional_robustness.py",
    "coupling_boundary_map.py",
    "finer_grid_search.py",
    "benchmark_ablation.py",           # PSO + grid + greedy benchmarks (long)
]

# ── RL: very slow, needs stable-baselines3 + torch, ~2-3 hours ──
RL = ["continuous_rl_baseline.py"]


def snapshot_pngs():
    """Map every existing PNG path -> its mtime, across both results roots."""
    seen = {}
    for root in RESULT_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if os.path.normpath(dirpath).startswith(os.path.normpath(DEST)):
                continue  # never scan the destination itself
            for f in files:
                if f.lower().endswith(".png"):
                    p = os.path.join(dirpath, f)
                    try:
                        seen[p] = os.path.getmtime(p)
                    except OSError:
                        pass
    return seen


def run_script(name, timeout):
    """Run one generator in its own subprocess. Returns (ok, seconds, tail)."""
    path = os.path.join(HERE, name)
    if not os.path.isfile(path):
        return False, 0.0, f"(file not found: {name})"
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"          # headless: no figure windows, no blocking
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"            # force UTF-8 I/O so Unicode prints (a, checkmarks)
    env["PYTHONIOENCODING"] = "utf-8"  #   don't crash on the Windows cp1252 console
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, path],
            cwd=HERE,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=(timeout if timeout and timeout > 0 else None),
        )
        dt = time.time() - t0
        if proc.returncode == 0:
            return True, dt, ""
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        return False, dt, "\n      ".join(tail)
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, f"(timed out after {timeout}s)"
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t0, f"({e})"


def collect(before, run_start, collect_existing):
    """Copy newly produced (or, if requested, all) PNGs into DEST. Returns manifest rows."""
    os.makedirs(DEST, exist_ok=True)
    rows = []
    for root in RESULT_ROOTS:
        if not os.path.isdir(root):
            continue
        label = os.path.relpath(root, os.path.dirname(HERE))
        for dirpath, _dirs, files in os.walk(root):
            if os.path.normpath(dirpath).startswith(os.path.normpath(DEST)):
                continue
            for f in sorted(files):
                if not f.lower().endswith(".png"):
                    continue
                src = os.path.join(dirpath, f)
                try:
                    mtime = os.path.getmtime(src)
                except OSError:
                    continue
                fresh = mtime >= (run_start - 2.0)          # produced this run
                is_new = src not in before
                if not (fresh or (collect_existing)):
                    if not is_new:
                        continue
                # Collision-safe destination name.
                dst = os.path.join(DEST, f)
                if os.path.exists(dst) and os.path.abspath(dst) != os.path.abspath(src):
                    sub = os.path.basename(dirpath)
                    dst = os.path.join(DEST, f"{sub}__{f}")
                shutil.copy2(src, dst)
                rows.append((os.path.basename(dst),
                             os.path.relpath(src, os.path.dirname(HERE))))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Regenerate all paper figures into one folder.")
    ap.add_argument("--all", action="store_true", help="also run the SLOW group (retraining + sweeps)")
    ap.add_argument("--include-rl", action="store_true", help="also run continuous_rl_baseline.py (~2-3 h)")
    ap.add_argument("--only", nargs="+", metavar="SCRIPT", help="run only these script(s)")
    ap.add_argument("--fresh", action="store_true", help="empty results/ALL_FIGURES/ before collecting")
    ap.add_argument("--collect-existing", action="store_true", help="also copy PNGs not regenerated this run")
    ap.add_argument("--timeout", type=int, default=0, help="per-script timeout in seconds (0 = no limit)")
    args = ap.parse_args()

    if args.only:
        scripts = list(args.only)
    else:
        scripts = list(FAST)
        if args.all:
            scripts += SLOW
        if args.include_rl:
            scripts += RL

    if args.fresh and os.path.isdir(DEST):
        shutil.rmtree(DEST)
    os.makedirs(DEST, exist_ok=True)

    print("=" * 74)
    print(f"  Regenerating {len(scripts)} figure script(s)")
    print(f"  Collecting into: {DEST}")
    print("=" * 74)

    before = snapshot_pngs()
    run_start = time.time()
    ok, failed = [], []

    for i, name in enumerate(scripts, 1):
        print(f"\n[{i:2d}/{len(scripts)}] {name} ...", flush=True)
        success, dt, tail = run_script(name, args.timeout)
        if success:
            print(f"        done in {dt:5.1f}s")
            ok.append(name)
        else:
            print(f"        FAILED after {dt:5.1f}s")
            if tail:
                print(f"      {tail}")
            failed.append(name)

    rows = collect(before, run_start, args.collect_existing)

    # Write a manifest mapping each collected figure to its source path.
    manifest = os.path.join(DEST, "MANIFEST.txt")
    with open(manifest, "w", encoding="utf-8") as fh:
        fh.write("Collected figures (filename  <-  source path)\n")
        fh.write("=" * 60 + "\n")
        for dst, src in sorted(rows):
            fh.write(f"{dst:48s} <-  {src}\n")

    print("\n" + "=" * 74)
    print(f"  SCRIPTS:  {len(ok)} ok, {len(failed)} failed")
    if failed:
        print("  Failed:  " + ", ".join(failed))
    print(f"  FIGURES: {len(rows)} PNG(s) collected into {DEST}")
    print(f"  Manifest: {manifest}")
    print("=" * 74)


if __name__ == "__main__":
    main()
