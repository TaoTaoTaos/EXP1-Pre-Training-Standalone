from __future__ import annotations

from typing import Any


LEGACY_WORKFLOW = "xiaoxingkai_transfer"


def resolve_workflow_name(config: dict[str, Any]) -> str:
    experiment_cfg = config.get("experiment") or {}
    return str(experiment_cfg.get("workflow", LEGACY_WORKFLOW))


def run_configured_workflow(config: dict[str, Any], paths, logger):
    workflow_name = resolve_workflow_name(config)
    if workflow_name == LEGACY_WORKFLOW:
        from lakeice_ncde.workflows.xiaoxingkai_transfer import run as run_xiaoxingkai_transfer

        return run_xiaoxingkai_transfer(config, paths, logger)

    raise ValueError(f"Unsupported experiment workflow: {workflow_name}")
