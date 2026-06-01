---
author: "Kyle Jones"
date_published: "October 28, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/finding-hidden-pipeline-health-patterns-with-hierarchical-clustering-e9d3d08e2931"
---

# Finding Hidden Pipeline Health Patterns with Hierarchical Clustering A pipeline integrity engineer reviews inline inspection (ILI) data for
500 km of pipeline divided into 2,000 segments. The average wall...

### Finding Hidden Pipeline Health Patterns with Hierarchical Clustering 

A pipeline integrity engineer reviews inline inspection (ILI) data for 500 km of pipeline divided into 2,000 segments. The average wall loss is 12%. Management asks: "Is this acceptable?"

The answer? It depends. What are the average hides? Are 90% of segments pristine with 10% severely corroded? Or is every segment uniformly degraded? Are coastal segments behaving differently from desert segments? Traditional dashboards show summary statistics --- mean wall loss, maximum pit depth, total anomalies --- but these metrics obscure natural groups that share similar degradation signatures.

An operator might flag segments exceeding a single threshold (e.g., wall loss \> 20%), but this binary classification misses nuance. A segment with 18% wall loss, poor coating, and high soil resistivity is riskier than a 22% wall loss segment with excellent CP and recent remediation. Thresholds can't capture these multivariate patterns.


Hierarchical clustering solves this. By grouping segments based on multiple features --- wall loss, cathodic protection (CP) potential, soil resistivity, coating condition, and historical inspection trends --- you uncover natural health regimes that inform targeted integrity management. This article demonstrates a working implementation using Apache Spark, SciPy, and Databricks.

### Binary Thresholds vs. Complex Degradation Signatures
Let explore this through two scenarios.

Scenario 1: Two segments both have 15% average wall loss:

- Segment A: Stable CP (-1,050 mV), dense coating, low soil resistivity (1,500 Ω·cm), no active corrosion.
- Segment B: Marginal CP (-820 mV), degraded coating, high soil resistivity (8,000 Ω·cm), active pitting.

A threshold-based system would treat these segments identically. But really, Segment B requires immediate attention while Segment A can wait for the next scheduled inspection.

Scenario 2: Three segments exceed 20% wall loss:

- Segment C: Localized external corrosion at a road crossing, otherwise stable.
- Segment D: Widespread internal corrosion from wet gas, accelerating.
- Segment E: Manufacturing anomaly (lamination), not progressing.

All three exceed the threshold, but root causes and risk profiles differ. Treating them uniformly wastes resources.

Hierarchical clustering identifies groups of segments with similar multivariate signatures, enabling:

- Differentiated inspection intervals: Low-risk clusters get 5-year cycles; high-risk clusters get annual digs.
- Root cause analysis: Clusters often map to physical causes (coating failure, soil chemistry, operational stress).
- Resource optimization: Focus CP upgrades on clusters where coating degradation dominates.
- Regulatory compliance: Demonstrate risk-based decision-making with transparent groupings.

### Aggregating ILI Anomalies Per Segment
``` 
CREATE OR REPLACE TABLE silver.segment_ili_features AS
SELECT
  segment_id,
  AVG(metal_loss_pct) AS avg_wall_loss_pct,
  MAX(metal_loss_pct) AS max_wall_loss_pct,
  COUNT(*) AS anomaly_count,
  SUM(CASE WHEN metal_loss_pct > 30 THEN 1 ELSE 0 END) AS critical_anomaly_count
FROM bronze.ili_anomalies
WHERE inspection_date = (SELECT MAX(inspection_date) FROM bronze.ili_anomalies)
GROUP BY segment_id;
```

### Joining CP Survey Data
``` 
CREATE OR REPLACE TABLE silver.segment_cp_features AS
SELECT
  segment_id,
  AVG(cp_potential_mv) AS avg_cp_potential_mv,
  STDDEV(cp_potential_mv) AS cp_std_mv,
  MIN(cp_potential_mv) AS min_cp_potential_mv
FROM bronze.cp_surveys
WHERE survey_date >= DATE_SUB(CURRENT_DATE(), 365)  -- Last year
GROUP BY segment_id;
```

### Unified Feature Table
``` 
CREATE OR REPLACE TABLE silver.segment_features AS
SELECT
  s.segment_id,
  s.segment_length_m,
  s.operating_pressure_mpa,
  s.coating_condition,
  s.years_since_last_ili,
  ili.avg_wall_loss_pct,
  ili.max_wall_loss_pct,
  ili.anomaly_count,
  ili.critical_anomaly_count,
  cp.avg_cp_potential_mv,
  cp.cp_std_mv,
  cp.min_cp_potential_mv,
  soil.avg_soil_resistivity_ohm_cm
FROM bronze.segments s
LEFT JOIN silver.segment_ili_features ili ON s.segment_id = ili.segment_id
LEFT JOIN silver.segment_cp_features cp ON s.segment_id = cp.segment_id
LEFT JOIN silver.segment_soil_features soil ON s.segment_id = soil.segment_id
WHERE ili.avg_wall_loss_pct IS NOT NULL;  -- Only segments with ILI data
```

### Hierarchical Clustering with SciPy
Hierarchical Clustering is an unsupervised learning method that looks to see how similar groups are. Unlike K-means, you don't need to know the number of clusters upfront. This approaches reveals nested groupings (in our case "High Risk" splits into "Coating Failure" vs. "CP Deficiency").

The Dendrogram is a visual repersentaitons of how segments merge into different clusters.

```python
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load segment features from Delta table
df = spark.table('silver.segment_features').toPandas()
# Select numeric features for clustering
features = [
    'avg_wall_loss_pct',
    'max_wall_loss_pct',
    'anomaly_count',
    'critical_anomaly_count',
    'avg_cp_potential_mv',
    'cp_std_mv',
    'avg_soil_resistivity_ohm_cm',
    'years_since_last_ili',
    'operating_pressure_mpa'
]
X = df[features].fillna(df[features].median())
# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Compute linkage (Ward minimizes within-cluster variance)
Z = linkage(X_scaled, method='ward')
# Cut dendrogram at height that yields ~5 clusters
cluster_labels = fcluster(Z, t=5, criterion='maxclust')
df['cluster_id'] = cluster_labels
# Save to Delta
result_df = spark.createDataFrame(df[['segment_id', 'cluster_id']])
result_df.write.mode('overwrite').saveAsTable('gold.segment_clusters')
```

Key choices:

- Ward linkage: Minimizes within-cluster variance (similar to K-means objective).
- StandardScaler: Prevents features with large scales (e.g., soil resistivity in Ω·cm) from dominating.
- Median imputation: Handles missing values conservatively.

### Visualizing the Dendrogram
``` 
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(12, 6))

dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-5, 2])
ax.set_xlabel('Segment Index (sorted by similarity)', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Pipeline Segment Hierarchical Clustering Dendrogram', fontsize=12, pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('pipeline_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()
```


What does this mean? Here is how to read the dendogram.

- Horizontal lines: Represent cluster merges. Height indicates dissimilarity.
- Vertical lines: Show which segments/clusters merge.
- Color threshold: Cuts at a specific height to define final clusters (shown in different colors).

### Cluster Profiling: What Do the Groups Mean?
Let's interpret what the different groups mean.

Cluster 1: "Healthy --- Low Risk"

- Low wall loss (5.2%), excellent CP (-1,050 mV), low soil resistivity.
- Action: Standard 5-year inspection cycle.

Cluster 2: "Moderate --- Coating Degradation"

- Moderate wall loss (18.3%), adequate CP (-980 mV), moderate soil resistivity.
- Action: Coating repair program, 3-year inspection cycle.

Cluster 3: "High Risk --- CP Deficiency"

- High wall loss (22.7%), poor CP (-810 mV), high soil resistivity.
- Action: Immediate CP rectifier upgrades, annual inspections.

Cluster 4: "Stable --- Well Protected"

- Moderate wall loss (14.5%), excellent CP (-1,100 mV), low soil resistivity.
- Action: Continue current CP program, 4-year inspection cycle.

Cluster 5: "Critical --- Multi-Factor"

- Very high wall loss (28.4%), very poor CP (-750 mV), very high soil resistivity.
- Action: Emergency digs, CP overhaul, consider replacement.


### Cool. But can I see this on a map? Yes!
```python
import matplotlib.pyplot as plt
import numpy as np

# Assuming df has 'start_chainage_km' for spatial location
fig, ax = plt.subplots(figsize=(12, 4))
# Color map for clusters
colors = ['#2ecc71', '#f39c12', '#e67e22', '#3498db', '#e74c3c']
cluster_names = ['Healthy', 'Moderate', 'High Risk', 'Stable', 'Critical']
for i, cluster_id in enumerate(range(1, 6)):
    cluster_data = df[df['cluster_id'] == cluster_id]
    ax.scatter(cluster_data['start_chainage_km'], 
               cluster_data['max_wall_loss_pct'],
               c=colors[i], label=f'C{cluster_id}: {cluster_names[i]}',
               s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
ax.set_xlabel('Chainage (km)', fontsize=11)
ax.set_ylabel('Max Wall Loss (%)', fontsize=11)
ax.set_title('Pipeline Segment Clusters by Location and Wall Loss', fontsize=12, pad=15)
ax.legend(loc='upper left', frameon=False, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('pipeline_clusters_spatial.png', dpi=300, bbox_inches='tight')
plt.show()
```

Insights:

- Cluster 5 (Critical) segments concentrate at km 150--180 (likely a river crossing with poor CP coverage).
- Cluster 1 (Healthy) dominates km 0--100 (recent coating rehabilitation project).
- Cluster 3 (High Risk) appears sporadically (isolated CP rectifier failures).

### Tracking Cluster Migration Over Time
Segments can move between clusters as conditions change. Track this with a versioned Delta table:

``` 
CREATE OR REPLACE TABLE gold.segment_cluster_history (
  segment_id STRING,
  cluster_id INT,
  clustering_date DATE,
  avg_wall_loss_pct DOUBLE,
  avg_cp_potential_mv DOUBLE
) USING DELTA
PARTITIONED BY (clustering_date);
```

### Flagging High-Risk Transitions
```python
-- Find segments that moved from Cluster 1 (Healthy) to Cluster 3/5 (High Risk)
WITH transitions AS (
  SELECT
    curr.segment_id,
    prev.cluster_id AS prev_cluster,
    curr.cluster_id AS curr_cluster,
    curr.avg_wall_loss_pct - prev.avg_wall_loss_pct AS wall_loss_increase
  FROM gold.segment_cluster_history curr
  JOIN gold.segment_cluster_history prev
    ON curr.segment_id = prev.segment_id
    AND prev.clustering_date = DATE_SUB(curr.clustering_date, 365)
  WHERE curr.clustering_date = CURRENT_DATE()
)
SELECT * FROM transitions
WHERE prev_cluster = 1 AND curr_cluster IN (3, 5)
ORDER BY wall_loss_increase DESC;
```

This identifies segments with accelerating degradation that require immediate investigation.

### Features for Compressor Station Clustering
``` 
# Rolling 24-hour window features
features_operational = [
    'mean_suction_pressure_mpa',
    'std_suction_pressure_mpa',
    'mean_discharge_pressure_mpa',
    'std_discharge_pressure_mpa',
    'mean_flow_rate_m3h',
    'flow_rate_kurtosis',  # Detects spikes
    'pressure_ratio_mean',
    'vibration_rms_avg'
]
```

### Identified Operational Clusters
- Cluster A: Steady-state operation (low variance).
- Cluster B: Transient operation (high variance, frequent starts/stops).
- Cluster C: Surge-prone (high kurtosis in flow rate).
- Cluster D: Low-flow / idle (near-zero flow for \>12 hours).

Use case: Flag Cluster C days for surge analysis. Correlate with compressor failures.

### Prerequisites
This project assumes you are using a Databricks workspace with Unity Catalog and that you have ILI data ingested into Delta tables.

- CP survey data with (segment_id, cp_potential_mv, survey_date).
- Soil resistivity data with (segment_id, resistivity_ohm_cm).
### So what? 

This approach demosntate that we should look beyond simle averages. Hierarchical clustering reveals natural groupings in pipeline health that simple thresholds miss. We can built features that help reveal degredation and we can visuzlie this with a dendrogram.

From a business point of view, this helps us move to risk-based inspection beause we can assign differentiated inspection intervals based on cluster profiles (5-year for low-risk, annual for high-risk). This data-driven clustering supports risk-based integrity management plans.
### Complete Implementation 

```python
# Databricks Notebook: Pipeline Health Clustering
# Prereqs: ILI, CP, and soil data in bronze tables

# COMMAND ----------
# Install dependencies
%pip install -q scipy scikit-learn matplotlib pandas
dbutils.library.restartPython()
# COMMAND ----------
# Configuration
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
spark = SparkSession.builder.getOrCreate()
CATALOG = 'pipeline'
SCHEMA = 'integrity'
TABLE_FEATURES = f'{CATALOG}.{SCHEMA}.segment_features'
TABLE_CLUSTERS = f'{CATALOG}.{SCHEMA}.segment_clusters'
# COMMAND ----------
# Feature engineering (run this in SQL notebook cell or via spark.sql)
spark.sql(f"""
CREATE OR REPLACE TABLE {TABLE_FEATURES} AS
SELECT
  s.segment_id,
  s.segment_length_m,
  s.operating_pressure_mpa,
  s.coating_condition,
  s.years_since_last_ili,
  ili.avg_wall_loss_pct,
  ili.max_wall_loss_pct,
  ili.anomaly_count,
  ili.critical_anomaly_count,
  cp.avg_cp_potential_mv,
  cp.cp_std_mv,
  cp.min_cp_potential_mv,
  soil.avg_soil_resistivity_ohm_cm,
  s.start_chainage_km
FROM {CATALOG}.bronze.segments s
LEFT JOIN {CATALOG}.silver.segment_ili_features ili ON s.segment_id = ili.segment_id
LEFT JOIN {CATALOG}.silver.segment_cp_features cp ON s.segment_id = cp.segment_id
LEFT JOIN {CATALOG}.silver.segment_soil_features soil ON s.segment_id = soil.segment_id
WHERE ili.avg_wall_loss_pct IS NOT NULL
""")
print(f'✓ Feature table created: {TABLE_FEATURES}')
# COMMAND ----------
# Load features
df = spark.table(TABLE_FEATURES).toPandas()
print(f'Loaded {len(df):,} segments')
# Select numeric features
features = [
    'avg_wall_loss_pct',
    'max_wall_loss_pct',
    'anomaly_count',
    'critical_anomaly_count',
    'avg_cp_potential_mv',
    'cp_std_mv',
    'avg_soil_resistivity_ohm_cm',
    'years_since_last_ili',
    'operating_pressure_mpa'
]
X = df[features].fillna(df[features].median())
# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('✓ Features normalized')
# COMMAND ----------
# Hierarchical clustering
Z = linkage(X_scaled, method='ward')
print('✓ Linkage computed')
# Cut dendrogram to form 5 clusters
n_clusters = 5
cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
df['cluster_id'] = cluster_labels
# Save to Delta
result_df = spark.createDataFrame(df[['segment_id', 'cluster_id']])
result_df.write.mode('overwrite').saveAsTable(TABLE_CLUSTERS)
print(f'✓ Clusters saved to {TABLE_CLUSTERS}')
# COMMAND ----------
# Dendrogram visualization
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-n_clusters, 2])
ax.set_xlabel('Segment Index', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Pipeline Segment Hierarchical Clustering', fontsize=12, pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('/dbfs/FileStore/pipeline_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Dendrogram saved')
# COMMAND ----------
# Cluster profiling
cluster_profiles = df.groupby('cluster_id')[features].mean()
cluster_profiles['segment_count'] = df.groupby('cluster_id').size()
print('\nCluster Profiles:')
print(cluster_profiles.round(2))
# COMMAND ----------
# Spatial visualization
fig, ax = plt.subplots(figsize=(12, 4))
colors = ['#2ecc71', '#f39c12', '#e67e22', '#3498db', '#e74c3c']
cluster_names = ['Healthy', 'Moderate', 'High Risk', 'Stable', 'Critical']
for i, cluster_id in enumerate(range(1, n_clusters + 1)):
    cluster_data = df[df['cluster_id'] == cluster_id]
    ax.scatter(cluster_data['start_chainage_km'], 
               cluster_data['max_wall_loss_pct'],
               c=colors[i], label=f'C{cluster_id}: {cluster_names[i]}',
               s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
ax.set_xlabel('Chainage (km)', fontsize=11)
ax.set_ylabel('Max Wall Loss (%)', fontsize=11)
ax.set_title('Cluster Distribution Along Pipeline', fontsize=12, pad=15)
ax.legend(loc='upper left', frameon=False, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('/dbfs/FileStore/pipeline_clusters_spatial.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Spatial map saved')
```
