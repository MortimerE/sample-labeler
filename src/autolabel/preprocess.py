from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .domain import AudioBuffer, FileContext


class DecodeError(RuntimeError):
    pass


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> tuple[np.ndarray, int]:
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return samples, sample_rate
    except (sf.SoundFileError, RuntimeError) as original:
        with tempfile.NamedTemporaryFile(suffix=".wav") as converted:
            process = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", converted.name],
                capture_output=True,
                text=True,
                check=False,
            )
            if process.returncode != 0:
                raise DecodeError(process.stderr.strip() or str(original)) from original
            try:
                samples, sample_rate = sf.read(converted.name, dtype="float32", always_2d=True)
                return samples, sample_rate
            except sf.SoundFileError as error:
                raise DecodeError(str(error)) from error


def active_duration(samples: np.ndarray, sample_rate: int, trim_db: float, hysteresis_ms: float) -> float:
    if samples.size == 0:
        return 0.0
    window = max(1, round(sample_rate * hysteresis_ms / 1000.0))
    threshold = 10.0 ** (trim_db / 20.0)
    squared = np.square(samples.astype(np.float64))
    kernel = np.ones(window, dtype=float) / window
    rms = np.sqrt(np.convolve(squared, kernel, mode="same"))
    active = np.flatnonzero(rms >= threshold)
    return 0.0 if active.size == 0 else min(samples.size, int(active[-1]) + window // 2 + 1) / sample_rate


def decode(path: str | Path, config: dict) -> tuple[AudioBuffer, FileContext, list[str]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DecodeError(f"audio file does not exist: {source}")
    raw, source_rate = _read(source)
    if raw.shape[0] == 0:
        raise DecodeError("decoded audio contains no frames")
    channels = raw.shape[1]
    duration = raw.shape[0] / source_rate
    mono = raw.mean(axis=1, dtype=np.float32)
    active = active_duration(mono, source_rate, config["trim_db"], config["trim_hysteresis_ms"])
    peak = float(np.max(np.abs(mono)))
    flags: list[str] = []
    if peak < 10.0 ** (config["trim_db"] / 20.0):
        flags.append("SILENT_FILE")
    if duration < config["min_duration_s"]:
        flags.append("SHORT_FILE")
    target_rate = int(config["sample_rate"])
    if source_rate != target_rate:
        divisor = np.gcd(source_rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, source_rate // divisor).astype(np.float32)
    # Resampling can overshoot at transients, so normalization must happen after
    # resampling rather than using the source-domain peak.
    working_peak = float(np.max(np.abs(mono)))
    if working_peak > 0:
        mono = mono / working_peak
    audio = AudioBuffer(mono, target_rate, source_rate, channels, duration, active)
    return audio, FileContext(str(source), sha1_file(source)), flags


def chroma_and_tonalness(audio: AudioBuffer) -> tuple[np.ndarray, float]:
    samples = audio.samples
    if samples.size < 2048 or not np.any(samples):
        return np.zeros(12), 0.0
    window = np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(samples * window))
    frequencies = np.fft.rfftfreq(samples.size, 1 / audio.sample_rate)
    valid = (frequencies >= 40) & (frequencies <= 5000) & (spectrum > 0)
    midi = np.rint(69 + 12 * np.log2(frequencies[valid] / 440.0)).astype(int)
    chroma = np.bincount(midi % 12, weights=spectrum[valid], minlength=12).astype(float)
    if chroma.sum() == 0:
        return chroma, 0.0
    chroma /= chroma.sum()
    geometric = float(np.exp(np.mean(np.log(np.maximum(chroma, 1e-12)))))
    flatness = geometric / float(np.mean(chroma))
    return chroma, max(0.0, min(1.0, 1.0 - flatness))
