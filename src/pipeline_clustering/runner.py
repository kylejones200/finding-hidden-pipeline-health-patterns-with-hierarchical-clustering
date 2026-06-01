from __future__ import annotations

import argparse
import logging

from pipeline_clustering import __version__, pipeline_health, three_projects

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def run_pipeline_health(seed: int = 42) -> None:
    pipeline_health.run(seed=seed)


def run_three_projects(seed: int = 42) -> None:
    three_projects.run(seed=seed)


def run_all(seed: int = 42) -> None:
    run_pipeline_health(seed=seed)
    run_three_projects(seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical clustering visualizations for pipeline assets"
    )
    parser.add_argument(
        "command",
        choices=["pipeline-health", "three-projects", "all"],
        help="Which figure set to generate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    if args.command == "pipeline-health":
        run_pipeline_health(seed=args.seed)
    elif args.command == "three-projects":
        run_three_projects(seed=args.seed)
    else:
        run_all(seed=args.seed)


if __name__ == "__main__":
    main()
