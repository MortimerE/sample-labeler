"""Generate a deterministic A-minor, 100-BPM WAV for the offline CI smoke test."""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    sample_rate = 44_100
    duration_s = 12
    beat_period_s = 60 / 100
    frequencies = (220.0, 261.625565, 329.627557)
    frames = bytearray()
    for index in range(sample_rate * duration_s):
        time_s = index / sample_rate
        chord = sum(0.12 * math.sin(2 * math.pi * frequency * time_s) for frequency in frequencies)
        beat_phase = time_s % beat_period_s
        click = 0.0
        if beat_phase < 0.02:
            click = 0.35 * math.exp(-120 * beat_phase) * math.sin(2 * math.pi * 1_000 * beat_phase)
        sample = max(-1.0, min(1.0, chord + click))
        frames.extend(struct.pack("<h", round(sample * 32_767)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(frames)


if __name__ == "__main__":
    main()
