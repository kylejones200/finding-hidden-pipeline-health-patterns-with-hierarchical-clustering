from pipeline_clustering import pipeline_health, three_projects
from pipeline_clustering.paths import FIGURES_DIR


def test_pipeline_health_writes_figures() -> None:
    paths = pipeline_health.run(seed=0, n_segments=50)
    assert len(paths) == 3
    for path in paths:
        assert path.is_file()
        assert path.parent == FIGURES_DIR


def test_three_projects_writes_figures() -> None:
    paths = three_projects.run(seed=0)
    assert len(paths) == 4
    for path in paths:
        assert path.is_file()
