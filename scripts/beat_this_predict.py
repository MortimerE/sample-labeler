from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
from beat_this.inference import Audio2Frames, File2Beats


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
    file2beats = File2Beats(checkpoint_path=args.checkpoint, device=args.device, dbn=False)
    beats, downbeats = file2beats(args.audio)

    signal, sample_rate = sf.read(args.audio)
    audio2frames = Audio2Frames(checkpoint_path=args.checkpoint, device=args.device)
    beat_logits, downbeat_logits = audio2frames(signal, sample_rate)

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
