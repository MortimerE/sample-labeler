from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from threading import Event
from typing import Any, Protocol

import numpy as np
import soundfile as sf
from scipy.signal import butter, find_peaks, istft, medfilt2d, sosfiltfilt, stft

from .domain import AudioBuffer, Key
from .scoring import KeyVote, TempoEvidence


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class KeyEvidence:
    votes: tuple[KeyVote, ...]
    chroma: np.ndarray
    tonalness: float
    flags: tuple[str, ...] = ()
    harmonic_ratio: float = 0.0
    bass_root_histogram: tuple[float, ...] | None = None
    sub_prominence: float = 0.0
    sub_coverage: float = 0.0
    bass_segments: int = 0


@dataclass(frozen=True, slots=True)
class BassRootEvidence:
    histogram: tuple[float, ...] | None
    sub_prominence: float
    sub_coverage: float
    segments: int


class DetectorSuite(Protocol):
    def key_votes(self, audio: AudioBuffer) -> KeyEvidence: ...
    def tempo_evidence(self, audio: AudioBuffer) -> TempoEvidence: ...
    def versions(self) -> dict[str, str]: ...


_NOTE_TO_PC = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
    "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}

_SKEY_KEYS = tuple(
    [Key(pc, "major") for pc in (9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8)]
    + [Key(pc, "minor") for pc in (11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
)

_PROFILE = {
    "major": np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "minor": np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
}


def parse_key(tonic: str, mode: str | None = None) -> Key:
    text = tonic.strip().replace("♭", "b").replace("♯", "#")
    if mode is None:
        lowered = text.lower()
        mode = "minor" if lowered.endswith(("minor", "min", "m")) else "major"
        for suffix in (" minor", " major", "min", "maj", "m"):
            if text.lower().endswith(suffix):
                text = text[: -len(suffix)].strip()
                break
    normalized = text[0].upper() + text[1:]
    try:
        pitch_class = _NOTE_TO_PC[normalized.upper()]
    except (IndexError, KeyError) as error:
        raise ValueError(f"unsupported key name: {tonic!r}") from error
    normalized_mode = "minor" if mode.lower().startswith("min") else "major"
    return Key(pitch_class, normalized_mode)  # type: ignore[arg-type]


def _version(module: object) -> str:
    root_name = str(getattr(module, "__name__", "")).split(".")[0]
    try:
        root = import_module(root_name)
    except ImportError:
        root = module
    return str(getattr(root, "__version__", "unknown"))


def _profile_candidates(hpcp: np.ndarray) -> list[tuple[float, Key]]:
    candidates = [
        (float(np.dot(hpcp, np.roll(_PROFILE[mode], pitch_class))), Key(pitch_class, mode))
        for mode in ("major", "minor")
        for pitch_class in range(12)
    ]
    return sorted(candidates, reverse=True, key=lambda item: item[0])


def _tonalness_from_hpcp(hpcp: np.ndarray, uniform_floor: float) -> tuple[np.ndarray, float]:
    total = float(np.sum(hpcp))
    if total <= 0:
        return np.zeros(12, dtype=float), 0.0
    chroma = (hpcp / total).astype(float)
    top3 = float(np.sort(chroma)[::-1][:3].sum())
    denominator = max(1e-9, 1.0 - float(uniform_floor))
    tonalness = max(0.0, min(1.0, (top3 - float(uniform_floor)) / denominator))
    return chroma, tonalness


def _odd_kernel(target: int, limit: int) -> int:
    value = min(max(1, int(target)), max(1, int(limit)))
    if value % 2 == 0:
        value = max(1, value - 1)
    return value


def _harmonic_component(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    if samples.size < 64:
        return samples.astype(np.float32, copy=True), 0.0
    frame_size = min(4096, samples.size)
    hop_size = min(512, max(1, frame_size // 4))
    _, _, spectrum = stft(
        samples,
        fs=sample_rate,
        nperseg=frame_size,
        noverlap=frame_size - hop_size,
        boundary="zeros",
    )
    magnitude = np.abs(spectrum)
    time_kernel = _odd_kernel(round(0.35 * sample_rate / hop_size), magnitude.shape[1])
    bin_hz = sample_rate / frame_size
    frequency_kernel = _odd_kernel(round(500.0 / bin_hz), magnitude.shape[0])
    harmonic_smooth = medfilt2d(magnitude, kernel_size=(1, time_kernel))
    percussive_smooth = medfilt2d(magnitude, kernel_size=(frequency_kernel, 1))
    harmonic_power = np.square(harmonic_smooth)
    percussive_power = np.square(percussive_smooth)
    harmonic_mask = harmonic_power / np.maximum(harmonic_power + percussive_power, 1e-12)
    harmonic_spectrum = spectrum * harmonic_mask
    harmonic_energy = float(harmonic_power.sum())
    percussive_energy = float(percussive_power.sum())
    _, harmonic = istft(
        harmonic_spectrum,
        fs=sample_rate,
        nperseg=frame_size,
        noverlap=frame_size - hop_size,
        input_onesided=True,
        boundary=True,
    )
    if harmonic.size < samples.size:
        harmonic = np.pad(harmonic, (0, samples.size - harmonic.size))
    return harmonic[: samples.size].astype(np.float32), clamp01(
        harmonic_energy / max(harmonic_energy + percussive_energy, 1e-12)
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bass_root_histogram(
    samples: np.ndarray,
    sample_rate: int,
    active_duration_s: float,
    config: dict,
    downbeats: tuple[float, ...] = (),
) -> BassRootEvidence:
    if not config.get("enabled", True) or samples.size < 4096:
        return BassRootEvidence(None, 0.0, 0.0, 0)
    cutoff = min(float(config["lowpass_hz"]), sample_rate * 0.45)
    low = sosfiltfilt(butter(4, cutoff, btype="lowpass", fs=sample_rate, output="sos"), samples)
    frame_size = 4096
    hop_size = 1024
    low_hz, high_hz = map(float, config["f0_range"])
    min_lag = max(1, int(sample_rate / high_hz))
    max_lag = min(frame_size - 1, int(sample_rate / low_hz))
    full_reference = float(np.sqrt(np.mean(np.square(samples.astype(float)))))
    records: list[tuple[int, float, float]] = []
    window = np.hanning(frame_size)
    for offset in range(0, samples.size - frame_size + 1, hop_size):
        low_frame = low[offset : offset + frame_size]
        full_frame = samples[offset : offset + frame_size]
        low_rms = float(np.sqrt(np.mean(np.square(low_frame))))
        full_rms = float(np.sqrt(np.mean(np.square(full_frame))))
        if low_rms < float(config.get("energy_floor", 0.05)) * max(full_reference, 1e-9):
            continue
        if low_rms / max(full_rms, 1e-9) < float(config.get("energy_floor", 0.05)):
            continue
        centered = (low_frame - np.mean(low_frame)) * window
        fft_size = 1 << (2 * frame_size - 1).bit_length()
        transformed = np.fft.rfft(centered, n=fft_size)
        autocorrelation = np.fft.irfft(np.square(np.abs(transformed)), n=fft_size)[:frame_size]
        if autocorrelation[0] <= 0:
            continue
        search = autocorrelation[min_lag : max_lag + 1]
        lag = min_lag + int(np.argmax(search))
        left_energy = float(np.dot(centered[:-lag], centered[:-lag]))
        right_energy = float(np.dot(centered[lag:], centered[lag:]))
        peak = float(autocorrelation[lag] / max(math.sqrt(left_energy * right_energy), 1e-12))
        if peak >= float(config["voicing_threshold"]):
            records.append((offset, sample_rate / lag, low_rms))

    raw_segments: list[list[tuple[int, float, float]]] = []
    for record in records:
        if not raw_segments:
            raw_segments.append([record])
            continue
        previous = raw_segments[-1][-1]
        median_f0 = float(np.median([item[1] for item in raw_segments[-1]]))
        cents = abs(1200.0 * math.log2(record[1] / median_f0))
        # Bridge brief ACF dropouts inside a sustained note; the stability test
        # still prevents kick sweeps or successive notes from merging.
        if record[0] - previous[0] <= 12 * hop_size and cents <= float(config["stability_cents"]):
            raw_segments[-1].append(record)
        else:
            raw_segments.append([record])

    histogram = np.zeros(12, dtype=float)
    kept_ranges: list[tuple[int, int]] = []
    downbeat_window = float(config["downbeat"]["window_ms"]) / 1000.0
    kept = 0
    for segment in raw_segments:
        start = segment[0][0]
        end = segment[-1][0] + frame_size
        duration = (end - start) / sample_rate
        f0s = np.asarray([item[1] for item in segment])
        median_f0 = float(np.median(f0s))
        stability = np.max(np.abs(1200.0 * np.log2(f0s / median_f0))) if f0s.size else float("inf")
        if duration < float(config["min_note_s"]) or stability > float(config["stability_cents"]):
            continue
        pitch_class = int(round(69 + 12 * math.log2(median_f0 / 440.0))) % 12
        weight = duration * float(np.mean([item[2] for item in segment]))
        onset_s = start / sample_rate
        if any(abs(onset_s - downbeat) <= downbeat_window for downbeat in downbeats):
            weight *= float(config["downbeat"]["boost"])
        histogram[pitch_class] += weight
        kept_ranges.append((start, min(end, low.size)))
        kept += 1

    total_low_energy = float(np.square(low).sum())
    kept_energy = sum(float(np.square(low[start:end]).sum()) for start, end in kept_ranges)
    sub_prominence = clamp01(kept_energy / max(total_low_energy, 1e-12))
    kept_time = sum((end - start) / sample_rate for start, end in kept_ranges)
    sub_coverage = clamp01(kept_time / max(active_duration_s, 1e-9))
    gate = config["gate"]
    passes = (
        kept >= int(gate["min_segments"])
        and sub_prominence >= float(gate["sub_prominence"])
        and sub_coverage >= float(gate["sub_coverage"])
        and histogram.sum() > 0
    )
    normalized = tuple(float(value) for value in histogram / histogram.sum()) if passes else None
    return BassRootEvidence(normalized, sub_prominence, sub_coverage, kept)


def _normalize_essentia_margin(relative_strength: float, scale: float) -> float:
    return max(0.0, min(1.0, (float(relative_strength) - 1.0) / max(float(scale), 1e-9)))


def _beat_grid_evidence(beats: np.ndarray) -> tuple[float | None, float]:
    """Estimate tempo without bias from frame-quantized median intervals."""
    intervals = np.diff(np.asarray(beats, dtype=float))
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        return None, 0.0
    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        return None, 0.0
    inliers = intervals[np.abs(intervals - median_interval) <= 0.25 * median_interval]
    if inliers.size == 0:
        return None, 0.0
    mean_interval = float(np.mean(inliers))
    iqr = float(np.percentile(inliers, 75) - np.percentile(inliers, 25))
    stability = 1.0 - clamp01(iqr / max(mean_interval, 1e-9))
    return 60.0 / mean_interval, stability


def _run(command: list[str], backend: str) -> str:
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise BackendUnavailable(f"{backend} executable is unavailable: {error}") from error
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or f"exit {process.returncode}"
        raise BackendUnavailable(f"{backend} failed: {detail}")
    return process.stdout.strip()


def _last_json_line(stdout: str, backend: str) -> object:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith(("{", "[")):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise BackendUnavailable(f"{backend} produced no parseable JSON line")


@lru_cache(maxsize=None)
def _tempocnn_labels(metadata_path: str) -> tuple[float, ...]:
    try:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        class_count = int(metadata["schema"]["outputs"][0]["shape"][0])
        description = str(metadata["description"])
        first_match = re.search(r"first index represents.*?([0-9]+(?:\.[0-9]+)?)\s*BPM", description)
        if first_match is None:
            raise ValueError("missing first-index BPM")
        first_bpm = float(first_match.group(1))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise BackendUnavailable(f"invalid TempoCNN metadata at {metadata_path}: {error}") from error
    labels = tuple(first_bpm + index for index in range(class_count))
    if any(label != 30.0 + index for index, label in enumerate(labels)):
        raise BackendUnavailable("TempoCNN metadata does not match the expected 30 BPM class offset")
    return labels


def _artifact_digest(graph_path: str, sums_path: str) -> str:
    graph_name = Path(graph_path).name
    try:
        if not Path(graph_path).is_file():
            raise FileNotFoundError(graph_path)
        lines = Path(sums_path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BackendUnavailable(f"TempoCNN artifact integrity metadata is unavailable: {error}") from error
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and Path(parts[-1].lstrip("*")).name == graph_name:
            digest = parts[0].lower()
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                actual = hashlib.sha256(Path(graph_path).read_bytes()).hexdigest()
                if actual != digest:
                    raise BackendUnavailable(
                        f"TempoCNN checksum mismatch for {graph_path}: expected {digest}, got {actual}"
                    )
                return digest
    raise BackendUnavailable(f"TempoCNN digest for {graph_name} not found in {sums_path}")


@dataclass(slots=True)
class ProductionDetectors:
    """Container-backed adapters for Essentia, Beat This, libKeyFinder, and S-KEY."""

    essentia_margin_scale: float = 2.0
    tonalness_uniform_floor: float = 0.25
    skey_min_seconds: float = 3.75
    bass_root_config: dict[str, Any] | None = None
    _downbeats_ready: Event = field(default_factory=Event, init=False, repr=False)
    _downbeats: tuple[float, ...] = field(default=(), init=False, repr=False)

    def _imports(self) -> object:
        try:
            return import_module("essentia.standard")
        except ImportError as error:
            raise BackendUnavailable(
                f"analysis backend {error.name!r} is not installed; use the provided container image"
            ) from error

    def key_votes(self, audio: AudioBuffer) -> KeyEvidence:
        essentia = self._imports()
        harmonic_samples, harmonic_ratio = _harmonic_component(audio.samples, audio.sample_rate)
        window = essentia.Windowing(type="blackmanharris62")
        spectrum = essentia.Spectrum()
        peaks = essentia.SpectralPeaks()
        hpcp_algorithm = essentia.HPCP(size=12)
        frames = []
        for frame in essentia.FrameGenerator(harmonic_samples, frameSize=4096, hopSize=2048, startFromZero=True):
            frequencies, magnitudes = peaks(spectrum(window(frame)))
            frames.append(hpcp_algorithm(frequencies, magnitudes))
        if not frames:
            raise BackendUnavailable("audio is too short for key feature extraction")
        hpcp_a = np.mean(frames, axis=0).astype("float32")
        hpcp = np.roll(hpcp_a, 9)
        chroma, tonalness = _tonalness_from_hpcp(hpcp, self.tonalness_uniform_floor)
        candidates = _profile_candidates(hpcp)

        tonic, mode, strength, relative_strength = essentia.Key(profileType="edma")(hpcp_a)
        essentia_key = parse_key(tonic, mode)
        essentia_runner = next(candidate for _, candidate in candidates if candidate != essentia_key)
        essentia_margin = _normalize_essentia_margin(float(relative_strength), float(self.essentia_margin_scale))

        with tempfile.NamedTemporaryFile(suffix=".wav") as working_file:
            sf.write(working_file.name, audio.samples, audio.sample_rate, subtype="PCM_16")
            keyfinder_command = ["keyfinder-cli", "-n", "standard", working_file.name]
            skey_python = os.environ.get("SKEY_PYTHON", "/opt/ml-venv/bin/python")
            skey_runner = os.environ.get("SKEY_RUNNER", "/app/scripts/skey_predict.py")
            skey_command = [
                skey_python,
                skey_runner,
                "--min-seconds",
                str(self.skey_min_seconds),
                working_file.name,
            ]
            with ThreadPoolExecutor(max_workers=2) as executor:
                keyfinder_future = executor.submit(_run, keyfinder_command, "libKeyFinder")
                skey_future = executor.submit(_run, skey_command, "S-KEY")
                keyfinder_name = keyfinder_future.result()
                try:
                    skey_stdout = skey_future.result()
                except BackendUnavailable:
                    skey_stdout = None
            if not keyfinder_name:
                raise BackendUnavailable("libKeyFinder returned no key")
            keyfinder_key = parse_key(keyfinder_name)
        flags: list[str] = []
        skey_vote: KeyVote | None = None
        if skey_stdout is None:
            flags.append("SKEY_UNAVAILABLE")
        else:
            try:
                payload = _last_json_line(skey_stdout, "S-KEY")
                if isinstance(payload, dict):
                    raw_probabilities = payload.get("probabilities")
                    if payload.get("tiled") is True:
                        flags.append("SKEY_INPUT_TILED")
                else:
                    raw_probabilities = payload
                if not isinstance(raw_probabilities, list) or len(raw_probabilities) != 24:
                    raise BackendUnavailable("S-KEY must return 24 class probabilities")
                probabilities = np.zeros(24, dtype=float)
                for probability, key in zip(raw_probabilities, _SKEY_KEYS):
                    offset = 0 if key.mode == "major" else 12
                    probabilities[offset + key.pitch_class] = float(probability)
                skey_order = np.argsort(probabilities)[::-1]
                skey_index = int(skey_order[0])
                skey_runner_index = int(skey_order[1])
                skey_key = Key(skey_index % 12, "major" if skey_index < 12 else "minor")
                skey_runner = Key(skey_runner_index % 12, "major" if skey_runner_index < 12 else "minor")
                skey_vote = KeyVote(
                    "skey",
                    skey_key,
                    margin=max(0.0, float(probabilities[skey_index] - probabilities[skey_runner_index])),
                    runner_up=skey_runner,
                    probabilities=tuple(float(value) for value in probabilities),
                )
            except (BackendUnavailable, TypeError, ValueError):
                flags.append("SKEY_UNAVAILABLE")

        _, keyfinder_runner = next(
            (score, candidate) for score, candidate in candidates if candidate != keyfinder_key
        )

        votes = [
            KeyVote("libkeyfinder", keyfinder_key, runner_up=keyfinder_runner),
            KeyVote(
                    "essentia",
                    essentia_key,
                    strength=float(strength),
                    margin=float(essentia_margin),
                    runner_up=essentia_runner,
                    margin_ratio_raw=float(relative_strength),
                ),
        ]
        if skey_vote is not None:
            votes.append(skey_vote)
        bass = BassRootEvidence(None, 0.0, 0.0, 0)
        if self.bass_root_config is not None:
            wait_seconds = float(self.bass_root_config["downbeat"]["wait_ms"]) / 1000.0
            self._downbeats_ready.wait(wait_seconds)
            bass = _bass_root_histogram(
                audio.samples,
                audio.sample_rate,
                audio.active_duration_s,
                self.bass_root_config,
                self._downbeats,
            )
        return KeyEvidence(
            votes=tuple(votes),
            chroma=chroma,
            tonalness=tonalness,
            flags=tuple(dict.fromkeys(flags)),
            harmonic_ratio=harmonic_ratio,
            bass_root_histogram=bass.histogram,
            sub_prominence=bass.sub_prominence,
            sub_coverage=bass.sub_coverage,
            bass_segments=bass.segments,
        )

    def tempo_evidence(self, audio: AudioBuffer) -> TempoEvidence:
        essentia = self._imports()
        tempocnn_graph = os.environ.get("TEMPOCNN_GRAPH", "/app/artifacts/deeptemp-k16-3.pb")
        tempocnn_metadata = os.environ.get("TEMPOCNN_METADATA", "/app/artifacts/deeptemp-k16-3.json")
        if not os.path.isfile(tempocnn_graph):
            raise BackendUnavailable(f"TempoCNN graph not found at {tempocnn_graph}; rebuild the image")

        beat_this_python = os.environ.get("BEAT_THIS_PYTHON", "/opt/ml-venv/bin/python")
        beat_this_runner = os.environ.get("BEAT_THIS_RUNNER", "/app/scripts/beat_this_predict.py")
        with tempfile.NamedTemporaryFile(suffix=".wav") as beat_this_file:
            sf.write(beat_this_file.name, audio.samples, audio.sample_rate, subtype="PCM_16")
            beat_this_command = [beat_this_python, beat_this_runner, "--audio", beat_this_file.name]
            with ThreadPoolExecutor(max_workers=2) as executor:
                rhythm_future = executor.submit(
                    essentia.RhythmExtractor2013(method="multifeature"), audio.samples
                )
                beat_this_future = executor.submit(_run, beat_this_command, "Beat This")
                bpm, _, confidence, _, _ = rhythm_future.result()
                beat_this_stdout = beat_this_future.result()

        beat_this_output = _last_json_line(beat_this_stdout, "Beat This")
        if not isinstance(beat_this_output, dict):
            raise BackendUnavailable("Beat This must return a JSON object")
        downbeats = tuple(float(value) for value in beat_this_output.get("downbeats", []))
        self._downbeats = downbeats
        self._downbeats_ready.set()

        hypotheses: tuple[tuple[float, float], ...]
        tempocnn_peakedness: float
        flags: list[str] = []
        labels = _tempocnn_labels(tempocnn_metadata)
        resampled = essentia.Resample(inputSampleRate=audio.sample_rate, outputSampleRate=11025)(audio.samples)
        predictions = essentia.TensorflowPredictTempoCNN(graphFilename=tempocnn_graph)(resampled)
        distribution = np.asarray(predictions, dtype=float)
        if distribution.ndim == 0:
            distribution = distribution.reshape(1)
        if distribution.ndim > 1:
            distribution = distribution.reshape(-1, distribution.shape[-1]).mean(axis=0)
        distribution = np.maximum(distribution, 0.0)
        if distribution.size == 0 or distribution.sum() <= 0:
            hypotheses = ()
            tempocnn_peakedness = 0.0
            flags.append("TEMPOCNN_DEGENERATE")
        elif distribution.size != len(labels):
            raise BackendUnavailable(
                f"TempoCNN returned {distribution.size} classes; metadata declares {len(labels)}"
            )
        else:
            distribution /= distribution.sum()
            order = np.argsort(distribution)[::-1]
            hypotheses = tuple(
                (float(labels[index]), float(distribution[index]))
                for index in order[:5]
            )
            entropy = -np.sum(distribution * np.log(np.maximum(distribution, 1e-12)))
            tempocnn_peakedness = 1.0 - float(entropy / np.log(len(distribution)))
        beats = np.asarray(beat_this_output.get("beats", []), dtype=float)
        beat_this_n_beats = int(beats.size)
        beat_this_bpm, beat_this_stability = (
            _beat_grid_evidence(beats) if beat_this_n_beats >= 4 else (None, 0.0)
        )

        activation_flatness = float(
            beat_this_output.get("activations_stats", {}).get("flatness", 1.0 - beat_this_stability)
        )

        pulse_clarity, onset_events = _pulse_evidence(audio.samples, audio.sample_rate)
        return TempoEvidence(
            hypotheses,
            tempocnn_peakedness,
            beat_this_bpm,
            beat_this_n_beats,
            beat_this_stability,
            float(bpm),
            float(confidence),
            pulse_clarity,
            activation_flatness,
            tuple(flags),
            downbeats,
            onset_events,
        )

    def versions(self) -> dict[str, str]:
        essentia = self._imports()
        graph_path = os.environ.get("TEMPOCNN_GRAPH", "/app/artifacts/deeptemp-k16-3.pb")
        sums_path = os.environ.get("MODEL_SHA256SUMS", "/app/artifacts/SHA256SUMS")
        tempocnn_digest = _artifact_digest(graph_path, sums_path)
        return {
            "essentia": _version(essentia),
            "libkeyfinder": os.environ.get("LIBKEYFINDER_VERSION", "2.2.8"),
            "skey": os.environ.get("SKEY_VERSION", "0.1.0"),
            "beat_this": os.environ.get("BEAT_THIS_VERSION", "1.1.0"),
            "tempocnn": f"{Path(graph_path).stem}@{tempocnn_digest[:12]}",
        }


def _pulse_evidence(samples: np.ndarray, sample_rate: int) -> tuple[float, tuple[float, ...]]:
    hop = 512
    frame = 1024
    if len(samples) < frame * 2:
        return 0.0, ()

    window = np.hanning(frame).astype(float)
    frames = []
    for offset in range(0, len(samples) - frame + 1, hop):
        segment = samples[offset:offset + frame] * window
        magnitude = np.abs(np.fft.rfft(segment))
        frames.append(np.log1p(magnitude))
    if len(frames) < 4:
        return 0.0, ()
    spectrogram = np.asarray(frames)
    flux = np.diff(spectrogram, axis=0)
    onset = np.maximum(flux, 0.0).sum(axis=1)
    onset = np.concatenate(([onset[0]], onset))
    threshold = float(np.median(onset) + 0.5 * np.std(onset))
    peak_indices, _ = find_peaks(onset, height=threshold, distance=2)
    onset_events = tuple(float(index * hop / sample_rate) for index in peak_indices)
    onset -= onset.mean()

    correlation = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    frame_rate = sample_rate / hop
    min_lag = max(1, round(60 * frame_rate / 200))
    max_lag = min(len(correlation), round(60 * frame_rate / 50))
    if correlation[0] <= 0 or max_lag <= min_lag:
        return 0.0, onset_events
    clarity = max(0.0, min(1.0, float(np.max(correlation[min_lag:max_lag]) / correlation[0])))
    return clarity, onset_events


def _pulse_clarity(samples: np.ndarray, sample_rate: int) -> float:
    return _pulse_evidence(samples, sample_rate)[0]
