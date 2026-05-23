import numpy as np

CLUSTER_WALL = [5, 18, 23, 15, 28]
CLUSTER_CP = [-1050, -980, -810, -1100, -750]
CLUSTER_SOIL = [2100, 3800, 7200, 1600, 9500]

def generate_segment_features(n_segments=500, seed=42):
    rng = np.random.default_rng(seed)
    wall, cp, soil, anom, cl = [], [], [], [], []
    for _ in range(n_segments):
        c = rng.choice(5, p=[0.35, 0.25, 0.15, 0.20, 0.05])
        w = np.clip(CLUSTER_WALL[c] + rng.normal(0, 3), 0, 40)
        wall.append(w)
        cp.append(CLUSTER_CP[c] + rng.normal(0, 40))
        soil.append(np.clip(CLUSTER_SOIL[c] + rng.normal(0, 800), 500, 12000))
        anom.append(max(0, round(w / 5)))
        cl.append(c)
    return np.array(wall), np.array(cp), np.array(soil), np.array(anom), np.array(cl, dtype=np.int32)
