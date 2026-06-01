# Finding Hidden Pipeline Health Patterns with Hierarchical Clustering

Published: 2025-10-28  
Medium: [Finding Hidden Pipeline Health Patterns with Hierarchical Clustering](https://medium.com/@kyle-t-jones/finding-hidden-pipeline-health-patterns-with-hierarchical-clustering-e9d3d08e2931)

Companion code for the article (`article.md`). Generates dendrograms and cluster maps from synthetic pipeline, compressor, and right-of-way data.

## Business context

A pipeline integrity engineer reviews inline inspection (ILI) data for 500 km of pipeline divided into 2,000 segments. The average wall loss is 12%. Management asks: "Is this acceptable?"

The answer? It depends. What are the average hides? Are 90% of segments pristine with 10% severely corroded? Or is every segment uniformly degraded? Are coastal segments behaving differently from desert segments? Traditional dashboards show summary statistics --- mean wall loss, maximum pit depth, total anomalies --- but these metrics obscure natural groups that share similar degradation signatures.

An operator might flag segments exceeding a single threshold (e.g., wall loss > 20%), but this binary classification misses nuance. A segment with 18% wall loss, poor coating, and high soil resistivity is riskier than a 22% wall loss segment with excellent CP and recent remediation. Thresholds can't capture these multivariate patterns.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pipeline-clustering-run all
```

Generate one figure set:

```bash
uv run pipeline-clustering-run pipeline-health
uv run pipeline-clustering-run three-projects
```

Figures are written to `outputs/figures/`.

## Outputs

| Command | Figures |
|---------|---------|
| `pipeline-health` | `23_pipeline_dendrogram.png`, `23_pipeline_clusters_spatial.png`, `23_pipeline_cluster_profiles.png` |
| `three-projects` | `25_pipeline_health_dendrogram.png`, `25_compressor_regimes_dendrogram.png`, `25_compressor_cluster_timeline.png`, `25_row_vegetation_dendrogram.png` |
| `all` | All of the above |

## Project layout

```
pyproject.toml / uv.lock
src/pipeline_clustering/   # clustering logic and plot generators
outputs/figures/           # generated PNGs (gitignored except .gitkeep)
content/blog/              # companion blog drafts
content/linkedin/          # LinkedIn post drafts
legacy/                    # original root-level scripts (reference)
tests/
article.md
├── rust/                   # Rust port (core + PyO3 + CLI bench)
├── benchmark_rust.py       # Python vs Rust benchmark
├── src/compute_kernel.py   # Python/numpy reference kernel
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — synthetic pipeline segment feature generation. Reference PyO3 benchmark: **~430×** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p finding_hidden_pipeline_health_patterns_with_hierarchical_clustering_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).