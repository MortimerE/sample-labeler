from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def export_params(
    model: object, output: str | Path, dataset_hash: str, metrics: dict[str, float],
    config: dict[str, Any], heads: int = 2,
) -> tuple[Path, Path]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()  # type: ignore[attr-defined]
    arrays = {name: value.detach().cpu().numpy() for name, value in state.items()}
    np.savez_compressed(output, **arrays)
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    manifest = {
        "schema": 1,
        "git_sha": git_sha,
        "dataset_hash": dataset_hash,
        "metrics": metrics,
        "config": config,
        "heads": heads,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, manifest_path


def main() -> None:
    import torch
    from train.model import FusionModel

    parser = argparse.ArgumentParser(description="Export a FusionModel checkpoint to NumPy runtime arrays.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-hash", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FusionModel()
    model.load_state_dict(checkpoint)
    export_params(model, args.out, args.dataset_hash, {}, {})


if __name__ == "__main__":
    main()
