use finding_hidden_pipeline_health_patterns_with_hierarchical_clustering_core::generate_segment_features;
use numpy::{PyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (n_segments, seed=42))]
fn generate_segment_features_py<'py>(py: Python<'py>, n_segments: usize, seed: u64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>)> {
    let s = generate_segment_features(n_segments, seed);
    Ok((s.wall_loss_pct.into_pyarray(py), s.cp_potential_mv.into_pyarray(py), s.soil_resistivity.into_pyarray(py), s.anomaly_count.into_pyarray(py), s.true_cluster.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (n_segments=500, seed=42, iterations=200))]
fn bench_kernel_py(n_segments: usize, seed: u64, iterations: usize) -> PyResult<f64> {
    let start = std::time::Instant::now();
    for _ in 0..iterations { let _ = generate_segment_features(n_segments, seed); }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn finding_hidden_pipeline_health_patterns_with_hierarchical_clustering_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_segment_features_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
