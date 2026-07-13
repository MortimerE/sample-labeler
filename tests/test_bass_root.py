import numpy as np

from autolabel.backends import _bass_root_histogram
from autolabel.config import load_config


def test_sustained_sub_line_yields_root_histogram():
    sample_rate = 44100
    pieces = []
    for _ in range(4):
        time = np.arange(int(0.36 * sample_rate)) / sample_rate
        pieces.extend((0.8 * np.sin(2 * np.pi * 55.0 * time), np.zeros(int(0.32 * sample_rate))))
    samples = np.concatenate(pieces).astype(np.float32)
    evidence = _bass_root_histogram(
        samples, sample_rate, samples.size / sample_rate,
        load_config()["fusion"]["key"]["bass_root"],
    )
    assert evidence.histogram is not None
    assert int(np.argmax(evidence.histogram)) == 9  # A
    assert evidence.segments >= 2


def test_short_decaying_kick_sweeps_abstain():
    sample_rate = 44100
    samples = np.zeros(int(2.0 * sample_rate), dtype=np.float32)
    for onset in np.arange(0.0, 2.0, 60.0 / 174.0):
        length = int(0.12 * sample_rate)
        time = np.arange(length) / sample_rate
        phase = 2 * np.pi * (50.0 * time - 0.5 * 10.0 / 0.12 * time * time)
        kick = np.sin(phase) * np.exp(-35.0 * time)
        start = int(onset * sample_rate)
        end = min(start + length, samples.size)
        samples[start:end] += kick[: end - start]
    evidence = _bass_root_histogram(
        samples, sample_rate, 2.0, load_config()["fusion"]["key"]["bass_root"],
    )
    assert evidence.histogram is None
