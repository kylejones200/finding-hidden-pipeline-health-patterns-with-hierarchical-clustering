import sys
import os

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""
Blog 23: Pipeline Health Clustering - Visualization Generator
Generates dendrogram and spatial clustering visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering



from pathlib import Path
np.random.seed(42)
logger.info("Blog 23: Pipeline Health Clustering - Visualizations")

# Set style

# ============================================================================
# Generate Synthetic Pipeline Segment Data
# ============================================================================
logger.info("\nGenerating synthetic pipeline segment data...")

n_segments = 500

# Generate features with realistic patterns
# Create 5 natural clusters with different characteristics
cluster_centers = [
    {'wall_loss': 5, 'cp_potential': -1050, 'soil_resist': 2100},   # Healthy
    {'wall_loss': 18, 'cp_potential': -980, 'soil_resist': 3800},   # Moderate
    {'wall_loss': 23, 'cp_potential': -810, 'soil_resist': 7200},   # High risk
    {'wall_loss': 15, 'cp_potential': -1100, 'soil_resist': 1600},  # Stable
    {'wall_loss': 28, 'cp_potential': -750, 'soil_resist': 9500}    # Critical
]

segments = []
for i in range(n_segments):
    # Assign to cluster
    cluster = np.random.choice(5, p=[0.35, 0.25, 0.15, 0.20, 0.05])
    center = cluster_centers[cluster]
    
    # Add noise
    wall_loss = center['wall_loss'] + np.random.normal(0, 3)
    cp_potential = center['cp_potential'] + np.random.normal(0, 40)
    soil_resist = center['soil_resist'] + np.random.normal(0, 800)
    
    # Add spatial location (chainage)
    chainage = i * 1.0  # 500 km pipeline, 1km segments
    
    segments.append({
        'segment_id': f'SEG-{i:04d}',
        'start_chainage_km': chainage,
        'avg_wall_loss_pct': np.clip(wall_loss, 0, 40),
        'max_wall_loss_pct': np.clip(wall_loss + np.random.uniform(2, 8), 0, 50),
        'avg_cp_potential_mv': cp_potential,
        'avg_soil_resistivity_ohm_cm': np.clip(soil_resist, 500, 12000),
        'anomaly_count': np.random.poisson(center['wall_loss'] / 5),
        'true_cluster': cluster
    })

df = pd.DataFrame(segments)
logger.info(f"✓ Generated {len(df)} segments")

# ============================================================================
# Visualization 1: Dendrogram
# ============================================================================
logger.info("\nGenerating hierarchical clustering dendrogram...")

# Select features for clustering
features = ['avg_wall_loss_pct', 'max_wall_loss_pct', 'avg_cp_potential_mv', 
            'avg_soil_resistivity_ohm_cm', 'anomaly_count']
X = df[features].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute linkage
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
fig, ax = plt.subplots(figsize=(12, 6))

dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-5, 2],
           above_threshold_color='gray')

ax.set_xlabel('Segment Index (sorted by similarity)', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Pipeline Segment Hierarchical Clustering Dendrogram', fontsize=12, pad=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('23_pipeline_dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("✓ Dendrogram saved")

# ============================================================================
# Visualization 2: Cluster Spatial Distribution
# ============================================================================
logger.info("Generating cluster spatial distribution map...")

# Perform clustering
n_clusters = 5
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df['cluster_id'] = clustering.fit_predict(X_scaled) + 1  # 1-indexed

# Colors matching blog post
colors = ['#2ecc71', '#f39c12', '#e67e22', '#3498db', '#e74c3c']
cluster_names = ['Healthy', 'Moderate', 'High Risk', 'Stable', 'Critical']

fig, ax = plt.subplots(figsize=(12, 4))

for i in range(1, n_clusters + 1):
    cluster_data = df[df['cluster_id'] == i]
    ax.scatter(cluster_data['start_chainage_km'], 
               cluster_data['max_wall_loss_pct'],
               c=colors[i-1], label=f'C{i}: {cluster_names[i-1]}',
               s=30, alpha=0.7, edgecolors='black', linewidth=0.3)

ax.set_xlabel('Chainage (km)', fontsize=11)
ax.set_ylabel('Max Wall Loss (%)', fontsize=11)
ax.set_title('Pipeline Segment Clusters by Location and Wall Loss', fontsize=12, pad=15)
ax.legend(loc='upper left', frameon=False, fontsize=9, ncol=5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('23_pipeline_clusters_spatial.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("✓ Spatial distribution map saved")

# ============================================================================
# Visualization 3: Cluster Profiles (Radar Chart)
# ============================================================================
logger.info("Generating cluster profile comparison...")

# Compute cluster statistics
cluster_profiles = df.groupby('cluster_id')[features].mean()

fig, ax = plt.subplots(figsize=(8, 6))

# Create grouped bar chart instead of radar for better readability
x = np.arange(len(features))
width = 0.15

for i in range(1, n_clusters + 1):
    # Normalize to 0-1 scale for comparison
    profile = cluster_profiles.loc[i]
    profile_norm = (profile - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
    
    offset = (i - 3) * width  # Center around 0
    ax.bar(x + offset, profile_norm, width, label=f'C{i}', 
           color=colors[i-1], alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Feature', fontsize=11)
ax.set_ylabel('Normalized Value (0-1)', fontsize=11)
ax.set_title('Cluster Feature Profiles (Normalized)', fontsize=12, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(['Avg Wall Loss', 'Max Wall Loss', 'CP Potential', 
                     'Soil Resistivity', 'Anomaly Count'], rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper left', frameon=False, fontsize=9, ncol=5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('23_pipeline_cluster_profiles.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("✓ Cluster profiles saved")

# ============================================================================
# Summary Statistics
# ============================================================================
logger.info("=== All visualizations generated successfully! ===")
logger.info("\nFiles created:")
logger.info("  - 23_pipeline_dendrogram.png")
logger.info("  - 23_pipeline_clusters_spatial.png")
logger.info("  - 23_pipeline_cluster_profiles.png")
logger.info("\nCluster Statistics:")
for i in range(1, n_clusters + 1):
    cluster_data = df[df['cluster_id'] == i]
    logger.info(f"\n  Cluster {i} ({cluster_names[i-1]}):")
    logger.info(f"    Segments: {len(cluster_data)}")
    logger.info(f"    Avg Wall Loss: {cluster_data['avg_wall_loss_pct'].mean():.1f}%")
    logger.info(f"    Avg CP Potential: {cluster_data['avg_cp_potential_mv'].mean():.0f} mV")
    logger.info(f"    Avg Soil Resistivity: {cluster_data['avg_soil_resistivity_ohm_cm'].mean():.0f} Ω·cm")

