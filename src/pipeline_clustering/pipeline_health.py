"""Blog 23: pipeline segment hierarchical clustering visualizations."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from pipeline_clustering.paths import figure_path

logger = logging.getLogger(__name__)

FEATURES = [
    "avg_wall_loss_pct",
    "max_wall_loss_pct",
    "avg_cp_potential_mv",
    "avg_soil_resistivity_ohm_cm",
    "anomaly_count",
]

CLUSTER_NAMES = ["Healthy", "Moderate", "High Risk", "Stable", "Critical"]
CLUSTER_COLORS = ["#2ecc71", "#f39c12", "#e67e22", "#3498db", "#e74c3c"]


def _generate_segments(n_segments: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cluster_centers = [
        {"wall_loss": 5, "cp_potential": -1050, "soil_resist": 2100},
        {"wall_loss": 18, "cp_potential": -980, "soil_resist": 3800},
        {"wall_loss": 23, "cp_potential": -810, "soil_resist": 7200},
        {"wall_loss": 15, "cp_potential": -1100, "soil_resist": 1600},
        {"wall_loss": 28, "cp_potential": -750, "soil_resist": 9500},
    ]

    segments = []
    for i in range(n_segments):
        cluster = rng.choice(5, p=[0.35, 0.25, 0.15, 0.20, 0.05])
        center = cluster_centers[cluster]
        wall_loss = center["wall_loss"] + rng.normal(0, 3)
        cp_potential = center["cp_potential"] + rng.normal(0, 40)
        soil_resist = center["soil_resist"] + rng.normal(0, 800)
        chainage = i * 1.0

        segments.append(
            {
                "segment_id": f"SEG-{i:04d}",
                "start_chainage_km": chainage,
                "avg_wall_loss_pct": np.clip(wall_loss, 0, 40),
                "max_wall_loss_pct": np.clip(
                    wall_loss + rng.uniform(2, 8), 0, 50
                ),
                "avg_cp_potential_mv": cp_potential,
                "avg_soil_resistivity_ohm_cm": np.clip(soil_resist, 500, 12000),
                "anomaly_count": rng.poisson(center["wall_loss"] / 5),
                "true_cluster": cluster,
            }
        )

    return pd.DataFrame(segments)


def _apply_tufte_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))


def plot_dendrogram(X_scaled: np.ndarray, out: Path) -> None:
    z = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        z, ax=ax, no_labels=True, color_threshold=z[-5, 2], above_threshold_color="gray"
    )
    ax.set_xlabel("Segment Index (sorted by similarity)", fontsize=11)
    ax.set_ylabel("Linkage Distance", fontsize=11)
    ax.set_title("Pipeline Segment Hierarchical Clustering Dendrogram", fontsize=12, pad=15)
    _apply_tufte_spines(ax)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_clusters(df: pd.DataFrame, out: Path, n_clusters: int = 5) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(1, n_clusters + 1):
        cluster_data = df[df["cluster_id"] == i]
        ax.scatter(
            cluster_data["start_chainage_km"],
            cluster_data["max_wall_loss_pct"],
            c=CLUSTER_COLORS[i - 1],
            label=f"C{i}: {CLUSTER_NAMES[i - 1]}",
            s=30,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.3,
        )
    ax.set_xlabel("Chainage (km)", fontsize=11)
    ax.set_ylabel("Max Wall Loss (%)", fontsize=11)
    ax.set_title("Pipeline Segment Clusters by Location and Wall Loss", fontsize=12, pad=15)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=5)
    _apply_tufte_spines(ax)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return df


def plot_cluster_profiles(df: pd.DataFrame, out: Path, n_clusters: int = 5) -> None:
    cluster_profiles = df.groupby("cluster_id")[FEATURES].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(FEATURES))
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
            color=CLUSTER_COLORS[i - 1],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Normalized Value (0-1)", fontsize=11)
    ax.set_title("Cluster Feature Profiles (Normalized)", fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Avg Wall Loss",
            "Max Wall Loss",
            "CP Potential",
            "Soil Resistivity",
            "Anomaly Count",
        ],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=5)
    _apply_tufte_spines(ax)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(seed: int = 42, n_segments: int = 500, n_clusters: int = 5) -> list[Path]:
    """Generate pipeline health clustering figures; return written paths."""
    logger.info("Pipeline health clustering visualizations")
    df = _generate_segments(n_segments=n_segments, seed=seed)
    logger.info("Generated %d synthetic segments", len(df))

    x_scaled = StandardScaler().fit_transform(df[FEATURES].values)

    paths = [
        figure_path("23_pipeline_dendrogram.png"),
        figure_path("23_pipeline_clusters_spatial.png"),
        figure_path("23_pipeline_cluster_profiles.png"),
    ]

    plot_dendrogram(x_scaled, paths[0])
    logger.info("Wrote %s", paths[0])

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    df["cluster_id"] = clustering.fit_predict(x_scaled) + 1

    plot_spatial_clusters(df, paths[1], n_clusters=n_clusters)
    logger.info("Wrote %s", paths[1])

    plot_cluster_profiles(df, paths[2], n_clusters=n_clusters)
    logger.info("Wrote %s", paths[2])

    for i in range(1, n_clusters + 1):
        cluster_data = df[df["cluster_id"] == i]
        logger.info(
            "Cluster %d (%s): %d segments, avg wall loss %.1f%%",
            i,
            CLUSTER_NAMES[i - 1],
            len(cluster_data),
            cluster_data["avg_wall_loss_pct"].mean(),
        )

    return paths
