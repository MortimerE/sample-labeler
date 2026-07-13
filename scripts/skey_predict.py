"""Emit S-KEY's full 24-class probability vector as JSON."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np


def tile_short_signal(signal: np.ndarray, sample_rate: int, min_seconds: float) -> tuple[np.ndarray, bool]:
    target = int(math.ceil(float(min_seconds) * int(sample_rate)))
    if signal.shape[-1] >= target:
        return signal, False
    if signal.shape[-1] == 0:
        return np.zeros((*signal.shape[:-1], target), dtype=signal.dtype), True
    repetitions = math.ceil(target / signal.shape[-1])
    tiled = np.tile(signal, (*([1] * (signal.ndim - 1)), repetitions))[..., :target]
    return np.ascontiguousarray(tiled), True


def analyze(audio_path: str, min_seconds: float) -> dict[str, object]:
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import torchaudio
        from skey.key_detection import DEFAULT_CHECKPOINT_PATH, load_checkpoint, load_model_components

        device = torch.device("cpu")
        checkpoint = load_checkpoint(DEFAULT_CHECKPOINT_PATH)
        sample_rate = int(checkpoint["audio"]["sr"])
        waveform, source_rate = torchaudio.load(audio_path, backend="soundfile")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if source_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
        signal, tiled = tile_short_signal(waveform.cpu().numpy(), sample_rate, min_seconds)
        waveform = torch.from_numpy(signal)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        hcqt, chromanet, crop = load_model_components(checkpoint, device)
        batch = waveform.to(device).unsqueeze(0)
        with torch.no_grad():
            features = crop(hcqt(batch), torch.zeros(1, device=device))
            probabilities = chromanet(features).mean(dim=0)
        assert abs(float(probabilities.sum()) - 1.0) < 1e-3, "expected a probability vector from ChromaNet"
        return {
            "probabilities": [float(value) for value in probabilities.cpu()],
            "tiled": tiled,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=float(os.environ.get("SKEY_MIN_SECONDS", "3.75")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(analyze(args.audio_path, args.min_seconds)))


if __name__ == "__main__":
    main()
