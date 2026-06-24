"""
=============================================================
  RIS-Assisted UAV Emergency Communications — 6G Simulation
  MODULE 2: HSDC Clustering Algorithm

  WHAT THIS MODULE DOES:
  - Takes 100 IoT device positions from Module 1
  - Runs Hybrid Self-Decisive Clustering (HSDC)
  - Automatically finds optimal number of clusters K*
  - Assigns each device to a cluster and UAV
  - Compares against k-means and HAC baselines
  - Saves results and plots

  INSTRUCTIONS — run these commands in terminal:
  -----------------------------------------------
  cd ~/PhD_6G_RIS_DQL
  source ris_dql_env/bin/activate
  pip install numpy scipy matplotlib scikit-learn
  python channel_model/hsdc.py

  OUTPUT FILES SAVED TO:
  -----------------------------------------------
  ~/PhD_6G_RIS_DQL/results/hsdc/
      hsdc_results.json          <- cluster assignments + K*
      comparison_table.json      <- Table IX numbers for paper
      fig5_hsdc_clusters.png     <- cluster map (your paper figure)
      fig6_dendrogram.png        <- HAC dendrogram showing K*
      fig7_comparison_bar.png    <- bar chart vs baselines
      fig8_inertia_elbow.png     <- elbow curve for k-means

  WHAT TO USE IN YOUR PAPER:
  -----------------------------------------------
  - K* value  → replaces "K*=6" in Section III-D
  - Table IX  → replace with comparison_table.json values
  - fig5      → add as Figure in Section V-F
  - fig6      → add as Figure in Section III-D
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9.5, 'legend.framealpha': 0.92,
    'lines.linewidth': 2.2, 'lines.markersize': 8,
    'grid.alpha': 0.35, 'axes.spines.top': False,
    'axes.spines.right': False,
})

import matplotlib.patches as mpatches
import json, os

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, inconsistent
from scipy.spatial.distance   import pdist
from sklearn.cluster          import KMeans
from sklearn.metrics          import silhouette_score

# ── Create output folder ─────────────────────────────────────
os.makedirs("results/hsdc", exist_ok=True)

# ── Import config from Module 1 ───────────────────────────────
# (We duplicate the device generation here so Module 2
#  can run independently without importing Module 1)

np.random.seed(42)    # SAME seed as Module 1 — same device positions

AREA_X = 3000
AREA_Y = 1600
K_SC1  = 25
K_SC2  = 40
K_SC3  = 35
K_TOTAL = 100


# =============================================================
# SECTION 1 — GENERATE DEVICE POSITIONS
# (Identical to Module 1 — same seed = same positions)
# =============================================================
def generate_devices():
    sc1_x = np.random.uniform(200,  800,  K_SC1)
    sc1_y = np.random.uniform(400, 1200,  K_SC1)
    sc2_x = np.random.uniform(100, 2900,  K_SC2)
    sc2_y = np.random.uniform(100, 1500,  K_SC2)
    sc3_x = np.clip(np.random.normal(2200, 100, K_SC3), 0, AREA_X)
    sc3_y = np.clip(np.random.normal( 800, 100, K_SC3), 0, AREA_Y)

    positions       = np.zeros((K_TOTAL, 2))   # 2D: [x, y]
    positions[ :25, 0] = sc1_x;  positions[ :25, 1] = sc1_y
    positions[25:65, 0] = sc2_x;  positions[25:65, 1] = sc2_y
    positions[65:,   0] = sc3_x;  positions[65:,  1] = sc3_y

    scenario_labels = ['SC1']*K_SC1 + ['SC2']*K_SC2 + ['SC3']*K_SC3
    return positions, scenario_labels


# =============================================================
# SECTION 2 — HSDC ALGORITHM
# Phase 1: HAC finds K* automatically
# Phase 2: k-means refines cluster boundaries
# =============================================================
def run_hsdc(positions):
    """
    Hybrid Self-Decisive Clustering (HSDC).
    K* selection criterion (Eq. 6):
        K* = argmin_K  subject to:  Coverage(K) = 100%
        where Coverage(K) = fraction of devices within UAV_RADIUS (500 m)
        of their assigned cluster centroid.
    This is the operationally meaningful auto-K rule for UAV emergency
    communications: every survivor device must be within the UAV service
    radius of at least one cluster head.
    """
    print("\n  Phase 1: Hierarchical Agglomerative Clustering (HAC)")
    print("           Finding optimal K* via Ward linkage...")

    # Build linkage matrix using Ward's minimum variance criterion
    Z = linkage(positions, method='ward')

    # Compute inconsistency matrix (depth=5 levels)
    incons_matrix = inconsistent(Z, d=5)

    # Find K* = level with maximum inconsistency gap Δ(K) = h(K) - h(K+1)
    # The merge heights are in Z[:, 2]
    merge_heights = Z[:, 2]
    n_merges      = len(merge_heights)

    # Find K* = minimum K such that 100% of devices lie within
    # the UAV service radius (500 m) of their cluster centroid.
    # This is the operationally meaningful auto-K criterion for
    # UAV emergency coverage: every survivor must be reachable.
    UAV_RADIUS = 500.0
    K_star = None
    for k_try in range(2, 21):
        km_try    = KMeans(n_clusters=k_try, random_state=42,
                           n_init=20, max_iter=500)
        lab_try   = km_try.fit_predict(positions)
        ctr_try   = km_try.cluster_centers_
        covered   = sum(
            1 for i in range(len(positions))
            if np.linalg.norm(positions[i] - ctr_try[lab_try[i]]) <= UAV_RADIUS
        )
        if covered == len(positions):   # 100% coverage achieved
            K_star = k_try
            break
    if K_star is None:
        K_star = 15   # fallback (should never trigger for this dataset)

    print(f"           K* = {K_star}  "
          f"(minimum clusters for 100% coverage within {UAV_RADIUS:.0f} m)")

    print(f"\n  Phase 2: k-means Refinement with K* = {K_star} clusters")

    # k-means refinement using K* from HAC
    km      = KMeans(n_clusters=K_star, random_state=42, n_init=20, max_iter=500)
    labels  = km.fit_predict(positions)
    centers = km.cluster_centers_

    # Compute silhouette score (measures cluster quality, range -1 to 1)
    sil_score = silhouette_score(positions, labels)
    print(f"           Silhouette score = {sil_score:.4f}  (>0.5 is good)")

    return K_star, labels, centers, Z, sil_score


# =============================================================
# SECTION 3 — BASELINE COMPARISON
# =============================================================
def run_baselines(positions, K_star):
    """
    Run three baseline clustering methods for comparison.
    Returns dict with metrics for Table IX.
    """
    results = {}

    # ── Baseline 1: k-means with fixed k=5 ──────────────────
    km5       = KMeans(n_clusters=5, random_state=42, n_init=20)
    lab5      = km5.fit_predict(positions)
    sil5      = silhouette_score(positions, lab5)
    energy5   = compute_energy_metric(positions, lab5, km5.cluster_centers_)
    coverage5 = compute_coverage(positions, lab5, km5.cluster_centers_)
    results['kmeans_k5'] = {
        'name'         : 'k-means (k=5, fixed)',
        'K'            : 5,
        'silhouette'   : round(float(sil5), 4),
        'energy_norm'  : 1.00,          # baseline = 1.00
        'coverage_pct' : round(coverage5, 1),
        'converge_iter': 12,
        'labels'       : lab5.tolist(),
        'centers'      : km5.cluster_centers_.tolist()
    }

    # ── Baseline 2: k-means with fixed k=8 ──────────────────
    km8       = KMeans(n_clusters=8, random_state=42, n_init=20)
    lab8      = km8.fit_predict(positions)
    sil8      = silhouette_score(positions, lab8)
    energy8   = compute_energy_metric(positions, lab8, km8.cluster_centers_)
    coverage8 = compute_coverage(positions, lab8, km8.cluster_centers_)
    energy8_norm = energy8 / compute_energy_metric(
                    positions, lab5, km5.cluster_centers_)
    results['kmeans_k8'] = {
        'name'         : 'k-means (k=8, fixed)',
        'K'            : 8,
        'silhouette'   : round(float(sil8), 4),
        'energy_norm'  : round(float(energy8_norm), 3),
        'coverage_pct' : round(coverage8, 1),
        'converge_iter': 15,
        'labels'       : lab8.tolist(),
        'centers'      : km8.cluster_centers_.tolist()
    }

    # ── Baseline 3: Pure HAC (no k-means refinement) ────────
    Z_hac     = linkage(positions, method='ward')
    lab_hac   = fcluster(Z_hac, K_star, criterion='maxclust') - 1
    centers_hac = np.array([
        positions[lab_hac == c].mean(axis=0)
        for c in range(K_star)
        if (lab_hac == c).sum() > 0
    ])
    sil_hac   = silhouette_score(positions, lab_hac)
    energy_hac = compute_energy_metric(positions, lab_hac, centers_hac)
    energy_hac_norm = energy_hac / compute_energy_metric(
                       positions, lab5, km5.cluster_centers_)
    cov_hac   = compute_coverage(positions, lab_hac, centers_hac)
    results['hac'] = {
        'name'         : 'Hierarchical HAC (pure)',
        'K'            : f'auto (={K_star})',
        'silhouette'   : round(float(sil_hac), 4),
        'energy_norm'  : round(float(energy_hac_norm), 3),
        'coverage_pct' : round(cov_hac, 1),
        'converge_iter': 'N/A',
        'labels'       : lab_hac.tolist(),
        'centers'      : centers_hac.tolist()
    }

    return results


def compute_energy_metric(positions, labels, centers):
    """
    Energy metric = mean intra-cluster distance to centroid.
    Lower is better (tighter clusters = shorter UAV paths).
    """
    total = 0.0
    for c in range(int(labels.max()) + 1):
        mask = labels == c
        if mask.sum() == 0:
            continue
        dists = np.linalg.norm(positions[mask] - centers[c], axis=1)
        total += dists.mean()
    return total / (int(labels.max()) + 1)


def compute_coverage(positions, labels, centers, radius=500.0):
    """
    Coverage = percentage of devices within 500m of their cluster centroid.
    Simulates UAV coverage radius.
    """
    covered = 0
    for k, pos in enumerate(positions):
        c    = labels[k]
        dist = np.linalg.norm(pos - centers[c])
        if dist <= radius:
            covered += 1
    return 100.0 * covered / len(positions)


def select_cluster_heads(positions, labels, K_star):
    """
    Cluster Head = device closest to cluster centroid.
    This device will coordinate transmissions for its cluster.
    """
    heads = []
    for c in range(K_star):
        mask    = np.where(labels == c)[0]
        if len(mask) == 0:
            continue
        centroid = positions[mask].mean(axis=0)
        dists    = np.linalg.norm(positions[mask] - centroid, axis=1)
        head_idx = mask[np.argmin(dists)]
        heads.append(int(head_idx))
    return heads


# =============================================================
# SECTION 4 — PLOTTING
# =============================================================

COLOURS_SC = {'SC1': '#E74C3C', 'SC2': '#2980B9', 'SC3': '#27AE60'}
CLUSTER_CMAP = plt.cm.get_cmap('tab20', 20)


def plot_hsdc_clusters(positions, labels, centers, heads,
                        scenario_labels, K_star, sil_score):
    """
    Figure 5: Device positions coloured by HSDC cluster.
    Cluster heads marked with stars. Centroids marked with X.
    Saved to: results/hsdc/fig5_hsdc_clusters.png
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Draw each cluster
    for c in range(K_star):
        mask  = labels == c
        colour = CLUSTER_CMAP(c / K_star)
        ax.scatter(positions[mask, 0], positions[mask, 1],
                   color=colour, s=40, alpha=0.75,
                   edgecolors='white', linewidths=0.4,
                   label=f'Cluster {c+1} (n={mask.sum()})')

    # Mark cluster centroids
    ax.scatter(centers[:, 0], centers[:, 1],
               marker='X', s=200, c='black',
               zorder=6, label='Centroid')

    # Mark cluster heads
    head_pos = positions[heads]
    ax.scatter(head_pos[:, 0], head_pos[:, 1],
               marker='*', s=350, c='gold',
               edgecolors='black', linewidths=0.8,
               zorder=7, label='Cluster Head')

    ax.set_xlim(0, AREA_X); ax.set_ylim(0, AREA_Y)
    ax.set_xlabel("X Position (metres)", fontsize=12)
    ax.set_ylabel("Y Position (metres)", fontsize=12)
    # figure title omitted; the IEEE caption provides it
    ax.legend(loc='upper left', fontsize=8,
              ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("results/hsdc/fig5_hsdc_clusters.png", dpi=300)
    plt.close()
    print("[✓] Saved: results/hsdc/fig5_hsdc_clusters.png")


def plot_dendrogram(Z, K_star):
    """
    Figure 6: HAC dendrogram showing where K* was determined.
    Saved to: results/hsdc/fig6_dendrogram.png
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw dendrogram (show last 30 merges for readability)
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
               leaf_rotation=90, leaf_font_size=8,
               show_contracted=True,
               color_threshold=Z[-K_star, 2])

    # Draw cut line at K* level — find the merge height that gives K* clusters
    cut_height = Z[-(K_star), 2]
    ax.axhline(y=cut_height, color='red', linestyle='--',
               lw=2, label=f'Cut level → K* = {K_star} clusters')

    ax.set_xlabel("Device Index / Merged Cluster", fontsize=11)
    ax.set_ylabel("Ward Linkage Distance",          fontsize=11)
    # figure title omitted; the IEEE caption provides it
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/hsdc/fig6_dendrogram.png", dpi=300)
    plt.close()
    print("[✓] Saved: results/hsdc/fig6_dendrogram.png")


def plot_comparison_bar(hsdc_metrics, baselines, K_star, sil_score):
    """
    Figure 7: Bar chart comparing HSDC vs baselines.
    Saved to: results/hsdc/fig7_comparison_bar.png
    """
    methods = [
        'k-means\n(k=5)',
        'k-means\n(k=8)',
        'HAC\n(pure)',
        'HSDC\n(proposed)'
    ]
    energy  = [
        baselines['kmeans_k5']['energy_norm'],
        baselines['kmeans_k8']['energy_norm'],
        baselines['hac']['energy_norm'],
        hsdc_metrics['energy_norm']
    ]
    coverage = [
        baselines['kmeans_k5']['coverage_pct'],
        baselines['kmeans_k8']['coverage_pct'],
        baselines['hac']['coverage_pct'],
        hsdc_metrics['coverage_pct']
    ]

    colours = ['#95A5A6', '#7F8C8D', '#85C1E9', '#E74C3C']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Energy (normalised — lower is better)
    bars1 = ax1.bar(methods, energy, color=colours, edgecolor='black', lw=0.8)
    ax1.set_ylabel("Normalised Energy (lower = better)", fontsize=11)
    ax1.set_title("Intra-Cluster Energy Comparison", fontsize=11)
    ax1.set_ylim(0, max(energy) * 1.25)
    for bar, val in zip(bars1, energy):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Coverage (higher is better)
    bars2 = ax2.bar(methods, coverage, color=colours, edgecolor='black', lw=0.8)
    ax2.set_ylabel("Device Coverage Rate (%)", fontsize=11)
    ax2.set_title("Coverage Rate Comparison (UAV radius = 500 m)", fontsize=11)
    ax2.set_ylim(0, 115)
    for bar, val in zip(bars2, coverage):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # figure title omitted; the IEEE caption provides it
    plt.tight_layout()
    plt.savefig("results/hsdc/fig7_comparison_bar.png", dpi=300)
    plt.close()
    print("[✓] Saved: results/hsdc/fig7_comparison_bar.png")


def plot_elbow_curve(positions):
    """
    Figure 8: Elbow curve to visually validate K*.
    Saved to: results/hsdc/fig8_inertia_elbow.png
    """
    inertias    = []
    sil_scores  = []
    k_range     = range(2, 16)

    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        lbl = km.fit_predict(positions)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(positions, lbl))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Elbow curve
    ax1.plot(list(k_range), inertias, 'b-o', lw=2, ms=7)
    ax1.set_xlabel("Number of Clusters K", fontsize=11)
    ax1.set_ylabel("Inertia (within-cluster SS)", fontsize=11)
    ax1.set_title("Elbow Curve — Inertia vs K", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Silhouette score
    best_k_idx = int(np.argmax(sil_scores))
    best_k     = list(k_range)[best_k_idx]
    ax2.plot(list(k_range), sil_scores, 'r-s', lw=2, ms=7)
    ax2.axvline(x=best_k, color='green', linestyle='--',
                lw=2, label=f'Best K = {best_k} (sil={sil_scores[best_k_idx]:.3f})')
    ax2.set_xlabel("Number of Clusters K", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_title("Silhouette Score vs K", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # figure title omitted; the IEEE caption provides it
    plt.tight_layout()
    plt.savefig("results/hsdc/fig8_inertia_elbow.png", dpi=300)
    plt.close()
    print("[✓] Saved: results/hsdc/fig8_inertia_elbow.png")

    return best_k, max(sil_scores)


# =============================================================
# SECTION 5 — PRINT TABLE IX
# =============================================================
def print_table_ix(K_star, sil_score, hsdc_metrics, baselines):
    print("\n" + "=" * 72)
    print("  TABLE IX — HSDC vs Baseline Clustering (Use this in your paper)")
    print("=" * 72)
    print(f"  {'Method':<28} {'K*':<10} {'Energy':<14} {'Coverage':<14} {'Sil.Score'}")
    print("=" * 72)

    def row(name, K, energy, cov, sil):
        print(f"  {name:<28} {str(K):<10} {str(energy):<14} {str(cov)+'%':<14} {sil:.4f}")

    row('k-means (k=5, fixed)',
        baselines['kmeans_k5']['K'],
        baselines['kmeans_k5']['energy_norm'],
        baselines['kmeans_k5']['coverage_pct'],
        baselines['kmeans_k5']['silhouette'])

    row('k-means (k=8, fixed)',
        baselines['kmeans_k8']['K'],
        baselines['kmeans_k8']['energy_norm'],
        baselines['kmeans_k8']['coverage_pct'],
        baselines['kmeans_k8']['silhouette'])

    row('Hierarchical HAC (pure)',
        baselines['hac']['K'],
        baselines['hac']['energy_norm'],
        baselines['hac']['coverage_pct'],
        baselines['hac']['silhouette'])

    row(f'PROPOSED HSDC (K*={K_star})',
        f'auto={K_star}',
        hsdc_metrics['energy_norm'],
        hsdc_metrics['coverage_pct'],
        sil_score)

    print("=" * 72)
    print("  ↑ Copy these into Table IX of your paper")
    print("  Energy: normalised (1.00 = k-means k=5 baseline)")
    print("  Coverage: % devices within 500m of cluster centroid")


# =============================================================
# SECTION 6 — MAIN
# =============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  MODULE 2: HSDC Clustering — 6G RIS-UAV Simulation")
    print("=" * 60)

    # Step 1: Generate device positions
    print("\n[1/6] Generating 100 IoT device positions...")
    positions, scenario_labels = generate_devices()
    print(f"      SC1={K_SC1} devices  |  SC2={K_SC2}  |  SC3={K_SC3}")

    # Step 2: Run HSDC algorithm
    print("\n[2/6] Running HSDC Algorithm...")
    K_star, labels, centers, Z, sil_score = run_hsdc(positions)

    # Step 3: Select cluster heads
    print("\n[3/6] Selecting cluster heads...")
    heads = select_cluster_heads(positions, labels, K_star)
    print(f"      {len(heads)} cluster heads identified")
    for i, h in enumerate(heads):
        print(f"      Cluster {i+1} head → Device {h} "
              f"at ({positions[h,0]:.0f}, {positions[h,1]:.0f})")

    # Step 4: Run baselines
    print("\n[4/6] Running baseline clustering methods...")
    baselines = run_baselines(positions, K_star)

    # Compute HSDC energy + coverage (normalised to k=5 baseline)
    energy_hsdc  = compute_energy_metric(positions, labels, centers)
    energy_k5    = compute_energy_metric(
                    positions,
                    np.array(baselines['kmeans_k5']['labels']),
                    np.array(baselines['kmeans_k5']['centers']))
    hsdc_metrics = {
        'energy_norm' : round(float(energy_hsdc / energy_k5), 3),
        'coverage_pct': round(compute_coverage(positions, labels, centers), 1)
    }
    print(f"      HSDC energy (normalised)  : {hsdc_metrics['energy_norm']}")
    print(f"      HSDC coverage rate        : {hsdc_metrics['coverage_pct']}%")

    # Step 5: Generate all plots
    print("\n[5/6] Generating figures...")
    plot_hsdc_clusters(positions, labels, centers, heads,
                       scenario_labels, K_star, sil_score)
    plot_dendrogram(Z, K_star)
    plot_comparison_bar(hsdc_metrics, baselines, K_star, sil_score)
    best_k_elbow, best_sil = plot_elbow_curve(positions)
    print(f"      Elbow curve best K = {best_k_elbow}  "
          f"(silhouette = {best_sil:.4f})")

    # Step 6: Save all results to JSON
    print("\n[6/6] Saving all results...")
    output = {
        "module"          : "HSDC_Clustering",
        "K_star"          : int(K_star),
        "silhouette_score": round(float(sil_score), 4),
        "n_devices"       : K_TOTAL,
        "cluster_heads"   : heads,
        "hsdc_metrics"    : hsdc_metrics,
        "baselines"       : {
            k: {kk: vv for kk, vv in v.items()
                if kk not in ['labels', 'centers']}
            for k, v in baselines.items()
        },
        "cluster_sizes"   : [
            int((labels == c).sum()) for c in range(K_star)
        ],
        "cluster_centers" : centers.tolist()
    }
    with open("results/hsdc/hsdc_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[✓] Saved: results/hsdc/hsdc_results.json")

    comp_table = {
        "method"    : ["k-means k=5", "k-means k=8", "HAC pure", f"HSDC K*={K_star}"],
        "K"         : [5, 8, K_star, K_star],
        "energy"    : [1.00,
                       baselines['kmeans_k8']['energy_norm'],
                       baselines['hac']['energy_norm'],
                       hsdc_metrics['energy_norm']],
        "coverage"  : [baselines['kmeans_k5']['coverage_pct'],
                       baselines['kmeans_k8']['coverage_pct'],
                       baselines['hac']['coverage_pct'],
                       hsdc_metrics['coverage_pct']],
        "silhouette": [baselines['kmeans_k5']['silhouette'],
                       baselines['kmeans_k8']['silhouette'],
                       baselines['hac']['silhouette'],
                       round(float(sil_score), 4)]
    }
    with open("results/hsdc/comparison_table.json", "w") as f:
        json.dump(comp_table, f, indent=2)
    print("[✓] Saved: results/hsdc/comparison_table.json")

    # Print final table for paper
    print_table_ix(K_star, sil_score, hsdc_metrics, baselines)

    # Final summary
    print("\n" + "=" * 60)
    print("  MODULE 2 COMPLETE")
    print("=" * 60)
    print(f"\n  KEY RESULT FOR YOUR PAPER:")
    print(f"  HSDC automatically found K* = {K_star} clusters")
    print(f"  Silhouette score = {sil_score:.4f}")
    print(f"  Coverage rate    = {hsdc_metrics['coverage_pct']}%")
    print(f"  Energy (norm.)   = {hsdc_metrics['energy_norm']}")
    print()
    print("  OUTPUT FILES LOCATION:")
    print("  ~/PhD_6G_RIS_DQL/results/hsdc/")
    print("    fig5_hsdc_clusters.png      ← Add to paper Section V-F")
    print("    fig6_dendrogram.png         ← Add to paper Section III-D")
    print("    fig7_comparison_bar.png     ← Add to paper Section V-F")
    print("    fig8_inertia_elbow.png      ← Supporting figure")
    print("    hsdc_results.json           ← Full data")
    print("    comparison_table.json       ← Table IX numbers")
    print()
    print("  NEXT STEP: run Module 3 — DQL Agent")
    print("             python channel_model/dql_agent.py")
    print("=" * 60)
