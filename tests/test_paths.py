from pipeline_clustering.paths import FIGURES_DIR, PROJECT_ROOT


def test_project_root_exists() -> None:
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_figures_dir_under_outputs() -> None:
    assert FIGURES_DIR == PROJECT_ROOT / "outputs" / "figures"
