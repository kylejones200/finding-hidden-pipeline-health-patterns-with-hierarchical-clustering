import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def plot_dendrogram(X_scaled, Z, df, features, logger) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-5, 2], above_threshold_color="gray")
    ax.set_xlabel("Segment Index (sorted by similarity)", fontsize=11)
    ax.set_ylabel("Linkage Distance", fontsize=11)
    ax.set_title("Pipeline Segment Hierarchical Clustering Dendrogram", fontsize=12, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    plt.tight_layout()
    plt.savefig("23_pipeline_dendrogram.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("✓ Dendrogram saved")
    logger.info("Generating cluster spatial distribution map...")
    n_clusters = 5
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    df["cluster_id"] = clustering.fit_predict(X_scaled) + 1
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#3498db", "#e74c3c"]
    cluster_names = ["Healthy", "Moderate", "High Risk", "Stable", "Critical"]
    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(1, n_clusters + 1):
        cluster_data = df[df["cluster_id"] == i]
        ax.scatter(
            cluster_data["start_chainage_km"],
            cluster_data["max_wall_loss_pct"],
            c=colors[i - 1],
            label=f"C{i}: {cluster_names[i - 1]}",
            s=30,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.3,
        )

    ax.set_xlabel("Chainage (km)", fontsize=11)
    ax.set_ylabel("Max Wall Loss (%)", fontsize=11)
    ax.set_title("Pipeline Segment Clusters by Location and Wall Loss", fontsize=12, pad=15)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    plt.tight_layout()
    plt.savefig("23_pipeline_clusters_spatial.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("✓ Spatial distribution map saved")
    logger.info("Generating cluster profile comparison...")
    cluster_profiles = df.groupby("cluster_id")[features].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(features))
    width = 0.15
    for i in range(1, n_clusters + 1):
        profile = cluster_profiles.loc[i]
        profile_norm = (profile - cluster_profiles.min()) / (
            cluster_profiles.max() - cluster_profiles.min()
        )
        offset = (i - 3) * width
        ax.bar(
            x + offset,
            profile_norm,
            width,
            label=f"C{i}",
            color=colors[i - 1],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Normalized Value (0-1)", fontsize=11)
    ax.set_title("Cluster Feature Profiles (Normalized)", fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Avg Wall Loss", "Max Wall Loss", "CP Potential", "Soil Resistivity", "Anomaly Count"],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    plt.tight_layout()
    plt.savefig("23_pipeline_cluster_profiles.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("✓ Cluster profiles saved")
    logger.info("=== All visualizations generated successfully! ===")
    logger.info("\nFiles created:")
    logger.info("  - 23_pipeline_dendrogram.png")
    logger.info("  - 23_pipeline_clusters_spatial.png")
    logger.info("  - 23_pipeline_cluster_profiles.png")
    logger.info("\nCluster Statistics:")
    for i in range(1, n_clusters + 1):
        cluster_data = df[df["cluster_id"] == i]
        logger.info(f"\n  Cluster {i} ({cluster_names[i - 1]}):")
        logger.info(f"    Segments: {len(cluster_data)}")
        logger.info(f"    Avg Wall Loss: {cluster_data['avg_wall_loss_pct'].mean():.1f}%")
        logger.info(f"    Avg CP Potential: {cluster_data['avg_cp_potential_mv'].mean():.0f} mV")
        logger.info(
            f"    Avg Soil Resistivity: {cluster_data['avg_soil_resistivity_ohm_cm'].mean():.0f} Ω·cm"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    "\n    Blog 23: Pipeline Health Clustering - Visualization Generator\n    Generates dendrogram and spatial clustering visualizations\n    "
    np.random.seed(42)
    logger.info("Blog 23: Pipeline Health Clustering - Visualizations")
    logger.info("\nGenerating synthetic pipeline segment data...")
    n_segments = 500
    cluster_centers = [
        {"wall_loss": 5, "cp_potential": -1050, "soil_resist": 2100},
        {"wall_loss": 18, "cp_potential": -980, "soil_resist": 3800},
        {"wall_loss": 23, "cp_potential": -810, "soil_resist": 7200},
        {"wall_loss": 15, "cp_potential": -1100, "soil_resist": 1600},
        {"wall_loss": 28, "cp_potential": -750, "soil_resist": 9500},
    ]
    segments = []
    for i in range(n_segments):
        cluster = np.random.choice(5, p=[0.35, 0.25, 0.15, 0.2, 0.05])
        center = cluster_centers[cluster]
        wall_loss = center["wall_loss"] + np.random.normal(0, 3)
        cp_potential = center["cp_potential"] + np.random.normal(0, 40)
        soil_resist = center["soil_resist"] + np.random.normal(0, 800)
        chainage = i * 1.0
        segments.append(
            {
                "segment_id": f"SEG-{i:04d}",
                "start_chainage_km": chainage,
                "avg_wall_loss_pct": np.clip(wall_loss, 0, 40),
                "max_wall_loss_pct": np.clip(wall_loss + np.random.uniform(2, 8), 0, 50),
                "avg_cp_potential_mv": cp_potential,
                "avg_soil_resistivity_ohm_cm": np.clip(soil_resist, 500, 12000),
                "anomaly_count": np.random.poisson(center["wall_loss"] / 5),
                "true_cluster": cluster,
            }
        )

    df = pd.DataFrame(segments)
    logger.info(f"✓ Generated {len(df)} segments")
    logger.info("\nGenerating hierarchical clustering dendrogram...")
    features = [
        "avg_wall_loss_pct",
        "max_wall_loss_pct",
        "avg_cp_potential_mv",
        "avg_soil_resistivity_ohm_cm",
        "anomaly_count",
    ]
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Z = linkage(X_scaled, method="ward")
    plot_dendrogram(X_scaled, Z, df, features, logger)


if __name__ == "__main__":
    main()
