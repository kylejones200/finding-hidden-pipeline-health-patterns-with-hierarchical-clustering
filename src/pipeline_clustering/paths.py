"""Repository root and default output paths."""

from pathlib import Path

# src/pipeline_clustering/paths.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def figure_path(name: str) -> Path:
    """Return path under outputs/figures/, creating the directory if needed."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name
