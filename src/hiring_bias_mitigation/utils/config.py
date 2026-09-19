"""Config loading and output-path resolution.

Every stage is driven by a YAML file under `configs/`. Configs hold short relative paths
(`outputs/sft/qwen3.5-4b_military`); `resolve_output_path` relocates that whole tree under
$HBM_OUTPUT_ROOT so no config file contains a machine-specific absolute path, and the same
config works on the Spark, in Docker and in CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Loads a config and splices in any config it references.

    A mitigation config names the audit config it is evaluated under (`audit_config`) and,
    for the training stages, the dataset config it trains on (`data_config`). Resolving them
    here means a runner receives one fully-formed dict and never has to know the layout.
    """
    path = Path(path)
    cfg = load_yaml(path)
    cfg["_config_path"] = str(path)
    for key, target in (("audit_config", "audit"), ("data_config", "data")):
        ref = cfg.get(key)
        if ref:
            cfg[target] = load_yaml(_resolve_ref(ref, path))
            cfg[target]["_config_path"] = str(ref)
    return cfg


def _resolve_ref(ref: str, parent: Path) -> Path:
    """Config cross-references are repo-relative; fall back to parent-relative."""
    candidate = REPO_ROOT / ref
    if candidate.exists():
        return candidate
    sibling = parent.parent / ref
    if sibling.exists():
        return sibling
    raise FileNotFoundError(f"config reference {ref!r} (from {parent}) not found")


def output_root() -> Path:
    root = os.environ.get("HBM_OUTPUT_ROOT", "").strip()
    return Path(root) if root else REPO_ROOT


def resolve_output_path(relative: str | Path) -> Path:
    """Maps a config's relative output path onto the model drive."""
    relative = Path(relative)
    if relative.is_absolute():
        return relative
    return output_root() / relative


def require_output_root() -> Path:
    """Refuses to start a long job whose checkpoints would land somewhere unintended.

    The failure this guards against is silent and expensive: a drive that is configured but
    not actually mounted leaves $HBM_OUTPUT_ROOT pointing at a path Docker will happily
    create as an empty directory on the system SSD, and a week of checkpoints goes there
    while every path still reads like the external drive. The marker file only exists on the
    real drive, so its absence catches exactly that case.
    """
    if os.environ.get("HBM_ALLOW_LOCAL_OUTPUT", "").strip() in {"1", "true", "yes"}:
        return output_root()
    root = output_root()
    marker = root / ".hbm-models-root"
    if not marker.exists():
        raise SystemExit(
            f"Output root {root} has no .hbm-models-root marker.\n"
            "Mount the model drive and `touch` the marker there, or set "
            "HBM_ALLOW_LOCAL_OUTPUT=1 to write into the repo instead."
        )
    return root


@dataclass
class RunPaths:
    """Where a single run's artifacts go."""

    run_name: str
    output_dir: Path
    raw_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.raw_dir = self.output_dir / "raw"

    def ensure(self) -> RunPaths:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        return self


def sanitize(name: str) -> str:
    """Turns a model id or attribute into something safe for a filename."""
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-")
