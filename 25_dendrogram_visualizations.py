import sys
import os

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

"""
Blog 25: Three Clustering Projects - Visualization Generator
Generates dendrograms for pipeline health, compressor regimes, and ROW vegetation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger.info("Blog 25: Three Clustering Projects - Visualizations")

plt.rcParams['font.family'] = 'serif'

# ============================================================================
# Visualization 1: Pipeline Health Dendrogram
# ============================================================================
logger.info("\nGenerating pipeline health clustering dendrogram...")

np.random.seed(42)
N = 200

df_health = pd.DataFrame({
    'avg_wall_loss_pct': np.random.normal(15, 8, N).clip(0, 70),
    'cp_potential_mv': np.random.normal(-900, 50, N),
    'soil_resistivity_ohm_cm': np.random.normal(3000, 600, N).clip(200, 8000),
    'coating': np.random.choice(['FBE', 'PE', 'CoalTar', 'Tape'], N, p=[0.4, 0.3, 0.2, 0.1]),
    'near_water': np.random.choice([0, 1], N, p=[0.8, 0.2])
})

coating_score = {'FBE': 1.0, 'PE': 0.75, 'CoalTar': 0.5, 'Tape': 0.25}
df_health['coating_score'] = df_health['coating'].map(coating_score)

features_health = ['avg_wall_loss_pct', 'cp_potential_mv', 'soil_resistivity_ohm_cm', 
                   'coating_score', 'near_water']
X_health = StandardScaler().fit_transform(df_health[features_health])
Z_health = linkage(X_health, method='ward')

fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(Z_health, ax=ax, truncate_mode='level', p=5, color_threshold=8,
           above_threshold_color='gray', no_labels=True)

ax.set_xlabel('Segment Group (sorted by similarity)', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Pipeline Health Signature Dendrogram', fontsize=12, pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('25_pipeline_health_dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("✓ Pipeline health dendrogram saved")

# ============================================================================
# Visualization 2: Compressor Operational Regimes Dendrogram
# ============================================================================
logger.info("Generating compressor operational regimes dendrogram...")

np.random.seed(77)
T = 24 * 30
time = pd.date_range('2024-07-01', periods=T, freq='h')

flow = 200 + 10 * np.sin(2 * np.pi * np.arange(T) / 24) + np.random.normal(0, 4, T)
p_suction = 600 + 5 * np.sin(2 * np.pi * np.arange(T) / 12) + np.random.normal(0, 2, T)
p_discharge = 640 + 6 * np.sin(2 * np.pi * np.arange(T) / 12) + np.random.normal(0, 3, T)

# Add some variation to create distinct regimes
for i in range(5, 8):  # Days 5-7: high variance (transient)
    idx = slice(i * 24, (i + 1) * 24)
    flow[idx] += np.random.normal(0, 15, 24)
    p_discharge[idx] += np.random.normal(0, 10, 24)

for i in range(12, 15):  # Days 12-14: surge-prone
    idx = slice(i * 24, (i + 1) * 24)
    flow[idx] += np.random.choice([-30, 0, 30], 24, p=[0.15, 0.70, 0.15])

df_scada = pd.DataFrame({'timestamp': time, 'flow': flow, 'p_suction': p_suction, 'p_discharge': p_discharge})

window_size = 24
daily_features = []
for i in range(0, len(df_scada) - window_size, window_size):
    window = df_scada.iloc[i:i + window_size]
    daily_features.append({
        'day': i // window_size,
        'flow_mean': window['flow'].mean(),
        'flow_std': window['flow'].std(),
        'flow_kurtosis': window['flow'].kurtosis(),
        'dp_mean': (window['p_discharge'] - window['p_suction']).mean(),
        'dp_std': (window['p_discharge'] - window['p_suction']).std()
    })

df_daily = pd.DataFrame(daily_features)
features_ops = ['flow_mean', 'flow_std', 'flow_kurtosis', 'dp_mean', 'dp_std']
X_ops = StandardScaler().fit_transform(df_daily[features_ops])
Z_ops = linkage(X_ops, method='ward')

fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z_ops, ax=ax, truncate_mode='level', p=4, color_threshold=5,
           above_threshold_color='gray', no_labels=True)

ax.set_xlabel('Day Window (sorted by similarity)', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Compressor Operating Regimes Dendrogram', fontsize=12, pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('25_compressor_regimes_dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("✓ Compressor regimes dendrogram saved")

# ============================================================================
# Visualization 3: Compressor Cluster Timeline
# ============================================================================
logger.info("Generating compressor cluster timeline...")

df_daily['cluster_id'] = AgglomerativeClustering(n_clusters=4, linkage='ward').fit_predict(X_ops)

fig, ax = plt.subplots(figsize=(12, 4))

colors_regimes = ['#2ecc71', '#e67e22', '#95a5a6', '#e74c3c']
for day, cluster in enumerate(df_daily['cluster_id']):
    ax.bar(day, 1, color=colors_regimes[cluster], edgecolor='black', linewidth=0.3)

ax.set_xlabel('Day', fontsize=11)
ax.set_ylabel('Operational Regime', fontsize=11)
ax.set_title('Daily Operational Cluster Index Over Time', fontsize=12, pad=15)
ax.set_yticks([])

# Legend
from matplotlib.patches import Patch

# Import Tufte plotting utilities
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors

legend_elements = [
    Patch(facecolor=colors_regimes[0], edgecolor='black', label='Steady-State'),
    Patch(facecolor=colors_regimes[1], edgecolor='black', label='Transient'),
    Patch(facecolor=colors_regimes[2], edgecolor='black', label='Low-Flow/Idle'),
    Patch(facecolor=colors_regimes[3], edgecolor='black', label='Surge-Prone')
]
ax.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=9, ncol=4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('25_compressor_cluster_timeline.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("✓ Compressor cluster timeline saved")

# ============================================================================
# Visualization 4: ROW Vegetation Dendrogram
# ============================================================================
logger.info("Generating ROW vegetation clustering dendrogram...")

np.random.seed(9)
N_tiles = 150

df_row = pd.DataFrame({
    'tile_id': range(1, N_tiles + 1),
    'chainage_km': np.linspace(0, 150, N_tiles),
    'ndvi_mean': np.random.uniform(0.1, 0.8, N_tiles),
    'ndvi_std': np.random.uniform(0.01, 0.2, N_tiles),
    'texture_glcm': np.random.uniform(0.05, 0.5, N_tiles),
    'thermal_anomaly_score': np.random.normal(0, 0.3, N_tiles),
    'bare_soil_fraction': np.random.uniform(0, 0.7, N_tiles)
})

# Create realistic patterns
forest_indices = np.random.choice(N_tiles, size=40, replace=False)
df_row.loc[forest_indices, 'ndvi_mean'] = np.random.uniform(0.65, 0.80, 40)
df_row.loc[forest_indices, 'ndvi_std'] = np.random.uniform(0.10, 0.20, 40)
df_row.loc[forest_indices, 'bare_soil_fraction'] = np.random.uniform(0.0, 0.10, 40)

bare_indices = np.random.choice([i for i in range(N_tiles) if i not in forest_indices], size=25, replace=False)
df_row.loc[bare_indices, 'ndvi_mean'] = np.random.uniform(0.1, 0.25, 25)
df_row.loc[bare_indices, 'bare_soil_fraction'] = np.random.uniform(0.60, 0.85, 25)
df_row.loc[bare_indices, 'thermal_anomaly_score'] = np.random.uniform(0.3, 0.8, 25)

features_row = ['ndvi_mean', 'ndvi_std', 'texture_glcm', 'thermal_anomaly_score', 'bare_soil_fraction']
X_row = StandardScaler().fit_transform(df_row[features_row])
Z_row = linkage(X_row, method='ward')

fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z_row, ax=ax, truncate_mode='level', p=6, color_threshold=4,
           above_threshold_color='gray', no_labels=True)

ax.set_xlabel('Tile Index (sorted by similarity)', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Right-of-Way Environmental Clusters Dendrogram', fontsize=12, pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('25_row_vegetation_dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("✓ ROW vegetation dendrogram saved")

# ============================================================================
# Summary
# ============================================================================
logger.info("=== All visualizations generated successfully! ===")
logger.info("\nFiles created:")
logger.info("  - 25_pipeline_health_dendrogram.png")
logger.info("  - 25_compressor_regimes_dendrogram.png")
logger.info("  - 25_compressor_cluster_timeline.png")
logger.info("  - 25_row_vegetation_dendrogram.png")
logger.info("\nProject Statistics:")
logger.info(f"  Pipeline segments clustered: {N}")
logger.info(f"  Compressor operational days: {len(df_daily)}")
logger.info(f"  ROW tiles clustered: {N_tiles}")

