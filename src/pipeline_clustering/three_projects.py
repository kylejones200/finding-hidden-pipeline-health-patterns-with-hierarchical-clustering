"""Blog 25: dendrograms for pipeline health, compressor regimes, and ROW vegetation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from pipeline_clustering.paths import figure_path

logger = logging.getLogger(__name__)


def _apply_tufte_spines(ax: plt.Axes, hide_left: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))


def _plot_pipeline_health_dendrogram(out: Path, n: int = 200) -> None:
    df_health = pd.DataFrame(
        {
            "avg_wall_loss_pct": np.random.normal(15, 8, n).clip(0, 70),
            "cp_potential_mv": np.random.normal(-900, 50, n),
            "soil_resistivity_ohm_cm": np.random.normal(3000, 600, n).clip(200, 8000),
            "coating": np.random.choice(
                ["FBE", "PE", "CoalTar", "Tape"], n, p=[0.4, 0.3, 0.2, 0.1]
            ),
            "near_water": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        }
    )
    coating_score = {"FBE": 1.0, "PE": 0.75, "CoalTar": 0.5, "Tape": 0.25}
    df_health["coating_score"] = df_health["coating"].map(coating_score)
    features = [
        "avg_wall_loss_pct",
        "cp_potential_mv",
        "soil_resistivity_ohm_cm",
        "coating_score",
        "near_water",
    ]
    x_health = StandardScaler().fit_transform(df_health[features])
    z_health = linkage(x_health, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        z_health,
        ax=ax,
        truncate_mode="level",
        p=5,
        color_threshold=8,
        above_threshold_color="gray",
        no_labels=True,
    )
    ax.set_xlabel("Segment Group (sorted by similarity)", fontsize=11)
    ax.set_ylabel("Linkage Distance", fontsize=11)
    ax.set_title("Pipeline Health Signature Dendrogram", fontsize=12, pad=15)
    _apply_tufte_spines(ax)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_compressor_regimes(out_dendrogram: Path, out_timeline: Path) -> int:
    t = 24 * 30
    time = pd.date_range("2024-07-01", periods=t, freq="h")
    flow = 200 + 10 * np.sin(2 * np.pi * np.arange(t) / 24) + np.random.normal(0, 4, t)
    p_suction = 600 + 5 * np.sin(2 * np.pi * np.arange(t) / 12) + np.random.normal(0, 2, t)
    p_discharge = 640 + 6 * np.sin(2 * np.pi * np.arange(t) / 12) + np.random.normal(0, 3, t)
    for i in range(5, 8):
        idx = slice(i * 24, (i + 1) * 24)
        flow[idx] += np.random.normal(0, 15, 24)
        p_discharge[idx] += np.random.normal(0, 10, 24)

    for i in range(12, 15):
        idx = slice(i * 24, (i + 1) * 24)
        flow[idx] += np.random.choice([-30, 0, 30], 24, p=[0.15, 0.70, 0.15])

    df_scada = pd.DataFrame(
        {"timestamp": time, "flow": flow, "p_suction": p_suction, "p_discharge": p_discharge}
    )
    window_size = 24
    daily_features = []
    for i in range(0, len(df_scada) - window_size, window_size):
        window = df_scada.iloc[i : i + window_size]
        daily_features.append(
            {
                "day": i // window_size,
                "flow_mean": window["flow"].mean(),
                "flow_std": window["flow"].std(),
                "flow_kurtosis": window["flow"].kurtosis(),
                "dp_mean": (window["p_discharge"] - window["p_suction"]).mean(),
                "dp_std": (window["p_discharge"] - window["p_suction"]).std(),
            }
        )

    df_daily = pd.DataFrame(daily_features)
    features_ops = ["flow_mean", "flow_std", "flow_kurtosis", "dp_mean", "dp_std"]
    x_ops = StandardScaler().fit_transform(df_daily[features_ops])
    z_ops = linkage(x_ops, method="ward")
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(
        z_ops,
        ax=ax,
        truncate_mode="level",
        p=4,
        color_threshold=5,
        above_threshold_color="gray",
        no_labels=True,
    )
    ax.set_xlabel("Day Window (sorted by similarity)", fontsize=11)
    ax.set_ylabel("Linkage Distance", fontsize=11)
    ax.set_title("Compressor Operating Regimes Dendrogram", fontsize=12, pad=15)
    _apply_tufte_spines(ax)
    plt.tight_layout()
    fig.savefig(out_dendrogram, dpi=300, bbox_inches="tight")
    plt.close(fig)
    df_daily["cluster_id"] = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(
        x_ops
    )
    colors_regimes = ["#2ecc71", "#e67e22", "#95a5a6", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(12, 4))
    for day, cluster in enumerate(df_daily["cluster_id"]):
        ax.bar(day, 1, color=colors_regimes[cluster], edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Operational Regime", fontsize=11)
    ax.set_title("Daily Operational Cluster Index Over Time", fontsize=12, pad=15)
    ax.set_yticks([])
    legend_elements = [
        Patch(facecolor=colors_regimes[0], edgecolor="black", label="Steady-State"),
        Patch(facecolor=colors_regimes[1], edgecolor="black", label="Transient"),
        Patch(facecolor=colors_regimes[2], edgecolor="black", label="Low-Flow/Idle"),
        Patch(facecolor=colors_regimes[3], edgecolor="black", label="Surge-Prone"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=9, ncol=4)
    _apply_tufte_spines(ax, hide_left=True)
    plt.tight_layout()
    fig.savefig(out_timeline, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return len(df_daily)


def _plot_row_vegetation(out: Path, n_tiles: int = 150) -> None:
    df_row = pd.DataFrame(
        {
            "tile_id": range(1, n_tiles + 1),
            "chainage_km": np.linspace(0, 150, n_tiles),
            "ndvi_mean": np.random.uniform(0.1, 0.8, n_tiles),
            "ndvi_std": np.random.uniform(0.01, 0.2, n_tiles),
            "texture_glcm": np.random.uniform(0.05, 0.5, n_tiles),
            "thermal_anomaly_score": np.random.normal(0, 0.3, n_tiles),
            "bare_soil_fraction": np.random.uniform(0, 0.7, n_tiles),
        }
    )
    forest_indices = np.random.choice(n_tiles, size=40, replace=False)
    df_row.loc[forest_indices, "ndvi_mean"] = np.random.uniform(0.65, 0.80, 40)
    df_row.loc[forest_indices, "ndvi_std"] = np.random.uniform(0.10, 0.20, 40)
    df_row.loc[forest_indices, "bare_soil_fraction"] = np.random.uniform(0.0, 0.10, 40)
    bare_indices = np.random.choice(
        [i for i in range(n_tiles) if i not in forest_indices], size=25, replace=False
    )
    df_row.loc[bare_indices, "ndvi_mean"] = np.random.uniform(0.1, 0.25, 25)
    df_row.loc[bare_indices, "bare_soil_fraction"] = np.random.uniform(0.60, 0.85, 25)
    df_row.loc[bare_indices, "thermal_anomaly_score"] = np.random.uniform(0.3, 0.8, 25)
    features_row = [
        "ndvi_mean",
        "ndvi_std",
        "texture_glcm",
        "thermal_anomaly_score",
        "bare_soil_fraction",
    ]
    x_row = StandardScaler().fit_transform(df_row[features_row])
    z_row = linkage(x_row, method="ward")
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(
        z_row,
        ax=ax,
        truncate_mode="level",
        p=6,
        color_threshold=4,
        above_threshold_color="gray",
        no_labels=True,
    )
    ax.set_xlabel("Tile Index (sorted by similarity)", fontsize=11)
    ax.set_ylabel("Linkage Distance", fontsize=11)
    ax.set_title("Right-of-Way Environmental Clusters Dendrogram", fontsize=12, pad=15)
    _apply_tufte_spines(ax)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(seed: int = 42) -> list[Path]:
    """Generate three-project clustering figures; return written paths."""
    np.random.seed(seed)
    logger.info("Three clustering projects visualizations")
    paths = [
        figure_path("25_pipeline_health_dendrogram.png"),
        figure_path("25_compressor_regimes_dendrogram.png"),
        figure_path("25_compressor_cluster_timeline.png"),
        figure_path("25_row_vegetation_dendrogram.png"),
    ]
    _plot_pipeline_health_dendrogram(paths[0])
    logger.info("Wrote %s", paths[0])
    n_days = _plot_compressor_regimes(paths[1], paths[2])
    logger.info("Wrote %s", paths[1])
    logger.info("Wrote %s", paths[2])
    _plot_row_vegetation(paths[3])
    logger.info("Wrote %s", paths[3])
    logger.info(
        "Summary: 200 pipeline segments, %d compressor days, 150 ROW tiles",
        n_days,
    )
    return paths
