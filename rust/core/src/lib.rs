//! Synthetic pipeline segment integrity features.

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as u32
    }

    fn uniform(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn choice5(&mut self) -> usize {
        let u = self.uniform();
        if u < 0.35 {
            0
        } else if u < 0.60 {
            1
        } else if u < 0.75 {
            2
        } else if u < 0.95 {
            3
        } else {
            4
        }
    }
}

const CLUSTER_WALL: [f64; 5] = [5.0, 18.0, 23.0, 15.0, 28.0];
const CLUSTER_CP: [f64; 5] = [-1050.0, -980.0, -810.0, -1100.0, -750.0];
const CLUSTER_SOIL: [f64; 5] = [2100.0, 3800.0, 7200.0, 1600.0, 9500.0];

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentFeatures {
    pub wall_loss_pct: Vec<f64>,
    pub cp_potential_mv: Vec<f64>,
    pub soil_resistivity: Vec<f64>,
    pub anomaly_count: Vec<f64>,
    pub true_cluster: Vec<i32>,
}

pub fn generate_segment_features(n_segments: usize, seed: u64) -> SegmentFeatures {
    let mut rng = Lcg::new(seed);
    let mut wall_loss_pct = Vec::with_capacity(n_segments);
    let mut cp_potential_mv = Vec::with_capacity(n_segments);
    let mut soil_resistivity = Vec::with_capacity(n_segments);
    let mut anomaly_count = Vec::with_capacity(n_segments);
    let mut true_cluster = Vec::with_capacity(n_segments);

    for _ in 0..n_segments {
        let c = rng.choice5();
        let wall = (CLUSTER_WALL[c] + rng.normal() * 3.0).clamp(0.0, 40.0);
        let cp = CLUSTER_CP[c] + rng.normal() * 40.0;
        let soil = (CLUSTER_SOIL[c] + rng.normal() * 800.0).clamp(500.0, 12000.0);
        let anomalies = (wall / 5.0).max(0.0).round();
        wall_loss_pct.push(wall);
        cp_potential_mv.push(cp);
        soil_resistivity.push(soil);
        anomaly_count.push(anomalies);
        true_cluster.push(c as i32);
    }

    SegmentFeatures {
        wall_loss_pct,
        cp_potential_mv,
        soil_resistivity,
        anomaly_count,
        true_cluster,
    }
}
