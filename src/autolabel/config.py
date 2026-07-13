from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    default_path = files("autolabel").joinpath("default.yaml")
    base = yaml.safe_load(default_path.read_text())
    if path is None:
        return base
    override = yaml.safe_load(Path(path).read_text()) or {}
    merged = _merge(base, override)
    _validate(merged)
    return merged


def _validate(config: dict[str, Any]) -> None:
    for name in ("tonality", "rhythmicity"):
        total = sum(float(value) for value in config["axes"][name]["weights"].values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"axes.{name}.weights must sum to 1.0 (got {total})")
    for name in ("key", "tempo"):
        if any(float(value) <= 0 for value in config["fusion"][name]["reliability"].values()):
            raise ValueError(f"fusion.{name}.reliability values must be positive")
    if int(config["report"]["top_k"]) <= 0:
        raise ValueError("report.top_k must be positive")
