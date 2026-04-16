from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Neural CDE lake-ice training CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one experiment config or one batch config end to end.")
    run_parser.add_argument("--config", type=str, required=True, help="Experiment config path.")
    run_parser.add_argument("--override", type=str, action="append", default=[], help="Extra YAML override path.")
    run_parser.add_argument("--set", dest="set_values", type=str, action="append", default=[], help="Dotted key=value override.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command != "run":
        raise ValueError(f"Unsupported command: {args.command}")

    from lakeice_ncde.batch import is_batch_config, run_batch_experiments
    from lakeice_ncde.pipeline import resolve_runtime
    from lakeice_ncde.workflows.xiaoxingkai_transfer import run as run_xiaoxingkai_transfer

    project_root = Path(__file__).resolve().parents[2]
    config, paths, logger = resolve_runtime(project_root, args.config, args.override, args.set_values)
    if is_batch_config(config):
        run_batch_experiments(config, paths, project_root, logger)
    else:
        run_xiaoxingkai_transfer(config, paths, logger)


if __name__ == "__main__":
    main()
