from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Beat This inference and emit compact JSON evidence.")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("BEAT_THIS_CHECKPOINT", "final0"),
        help="Checkpoint name/path/URL for Beat This",
    )
    parser.add_argument("--device", default=os.environ.get("BEAT_THIS_DEVICE", "cpu"), help="torch device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from beat_this.inference import Audio2Frames, Postprocessor

        torch.set_num_threads(min(8, os.cpu_count() or 1))
        signal, sample_rate = sf.read(args.audio)
        audio2frames = Audio2Frames(checkpoint_path=args.checkpoint, device=args.device)
        beat_logits, downbeat_logits = audio2frames(signal, sample_rate)
        beats, downbeats = Postprocessor(type="minimal", fps=50)(beat_logits, downbeat_logits)

        beat_prob = torch.sigmoid(beat_logits).cpu().numpy()
        downbeat_prob = torch.sigmoid(downbeat_logits).cpu().numpy()
        activation = beat_prob + downbeat_prob
        dispersion = float(np.std(activation) / (np.mean(activation) + 1e-9))
        flatness = float(1.0 / (1.0 + dispersion))

        payload = {
            "beats": [float(value) for value in np.asarray(beats, dtype=float)],
            "downbeats": [float(value) for value in np.asarray(downbeats, dtype=float)],
            "activations_stats": {"flatness": flatness},
        }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
