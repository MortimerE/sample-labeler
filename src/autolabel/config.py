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
    for section, expected in (("key", 1.0), ("tempo", 1.0)):
        total = sum(float(v) for v in config[section]["weights"].values())
        if abs(total - expected) > 1e-6:
            raise ValueError(f"{section}.weights must sum to 1.0 (got {total})")
    for section, low_name in (("key", "atonal"), ("tempo", "arhythmic")):
        thresholds = config[section]["thresholds"]
        if not 0 <= thresholds[low_name] < thresholds["accept"] <= 1:
            raise ValueError(f"invalid {section} thresholds")

