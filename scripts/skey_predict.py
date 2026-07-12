"""Emit S-KEY's full 24-class probability vector as JSON."""

from __future__ import annotations

import json
import sys

import torch
import torchaudio
from skey.key_detection import DEFAULT_CHECKPOINT_PATH, load_checkpoint, load_model_components


def main(audio_path: str) -> None:
    device = torch.device("cpu")
    checkpoint = load_checkpoint(DEFAULT_CHECKPOINT_PATH)
    sample_rate = int(checkpoint["audio"]["sr"])
    waveform, source_rate = torchaudio.load(audio_path, backend="soundfile")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    hcqt, chromanet, crop = load_model_components(checkpoint, device)
    batch = waveform.to(device).unsqueeze(0)
    with torch.no_grad():
        features = crop(hcqt(batch), torch.zeros(1, device=device))
        probabilities = chromanet(features).mean(dim=0)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-3, "expected a probability vector from ChromaNet"
    print(json.dumps([float(value) for value in probabilities.cpu()]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: skey_predict.py AUDIO_FILE")
    main(sys.argv[1])

