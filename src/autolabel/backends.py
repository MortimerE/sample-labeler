from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

import numpy as np
import soundfile as sf

from .domain import AudioBuffer, Key
from .scoring import KeyVote, TempoEvidence


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class KeyEvidence:
    votes: tuple[KeyVote, ...]
    chroma: np.ndarray
    tonalness: float


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


def _normalize_essentia_margin(relative_strength: float, scale: float) -> float:
    return max(0.0, min(1.0, (float(relative_strength) - 1.0) / max(float(scale), 1e-9)))


def _run(command: list[str], backend: str) -> str:
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise BackendUnavailable(f"{backend} executable is unavailable: {error}") from error
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or f"exit {process.returncode}"
        raise BackendUnavailable(f"{backend} failed: {detail}")
    return process.stdout.strip()


@dataclass(slots=True)
class ProductionDetectors:
    """Container-backed adapters for Essentia, Beat This, libKeyFinder, and S-KEY."""

    essentia_margin_scale: float = 2.0
    tonalness_uniform_floor: float = 0.25

    def _imports(self) -> object:
        try:
            return import_module("essentia.standard")
        except ImportError as error:
            raise BackendUnavailable(
                f"analysis backend {error.name!r} is not installed; use the provided container image"
            ) from error

    def key_votes(self, audio: AudioBuffer) -> KeyEvidence:
        essentia = self._imports()
        window = essentia.Windowing(type="blackmanharris62")
        spectrum = essentia.Spectrum()
        peaks = essentia.SpectralPeaks()
        hpcp_algorithm = essentia.HPCP(size=12)
        frames = []
        for frame in essentia.FrameGenerator(audio.samples, frameSize=4096, hopSize=2048, startFromZero=True):
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
            keyfinder_name = _run(["keyfinder-cli", "-n", "standard", working_file.name], "libKeyFinder")
            if not keyfinder_name:
                raise BackendUnavailable("libKeyFinder returned no key")
            keyfinder_key = parse_key(keyfinder_name)
            skey_python = os.environ.get("SKEY_PYTHON", "/opt/skey-venv/bin/python")
            skey_runner = os.environ.get("SKEY_RUNNER", "/app/scripts/skey_predict.py")
            raw_probabilities = json.loads(_run([skey_python, skey_runner, working_file.name], "S-KEY").splitlines()[-1])

        if len(raw_probabilities) != 24:
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
        skey_margin = max(0.0, float(probabilities[skey_index] - probabilities[skey_runner_index]))

        _, keyfinder_runner = next(
            (score, candidate) for score, candidate in candidates if candidate != keyfinder_key
        )

        return KeyEvidence(
            votes=(
                KeyVote("libkeyfinder", keyfinder_key, runner_up=keyfinder_runner),
                KeyVote(
                    "essentia",
                    essentia_key,
                    strength=float(strength),
                    margin=float(essentia_margin),
                    runner_up=essentia_runner,
                    margin_ratio_raw=float(relative_strength),
                ),
                KeyVote(
                    "skey",
                    skey_key,
                    margin=skey_margin,
                    runner_up=skey_runner,
                    probabilities=tuple(float(value) for value in probabilities),
                ),
            ),
            chroma=chroma,
            tonalness=tonalness,
        )

    def tempo_evidence(self, audio: AudioBuffer) -> TempoEvidence:
        essentia = self._imports()
        tempocnn_graph = os.environ.get("TEMPOCNN_GRAPH", "/app/artifacts/deeptemp-k16.pb")
        bpm, _, confidence, _, _ = essentia.RhythmExtractor2013(method="multifeature")(audio.samples)

        hypotheses: tuple[tuple[float, float], ...]
        tempocnn_peakedness: float
        if os.path.exists(tempocnn_graph):
            resampled = essentia.Resample(inputSampleRate=audio.sample_rate, outputSampleRate=11025)(audio.samples)
            predictions = essentia.TensorflowPredictTempoCNN(graphFilename=tempocnn_graph)(resampled)
            distribution = np.asarray(predictions, dtype=float)
            if distribution.ndim == 0:
                distribution = distribution.reshape(1)
            if distribution.ndim > 1:
                distribution = distribution.reshape(-1, distribution.shape[-1]).mean(axis=0)
            distribution = np.maximum(distribution, 0.0)
            if distribution.sum() > 0:
                distribution /= distribution.sum()
                order = np.argsort(distribution)[::-1]
                hypotheses = tuple(
                    (float(_tempocnn_index_to_bpm(index)), float(distribution[index]))
                    for index in order[:5]
                )
                entropy = -np.sum(distribution * np.log(np.maximum(distribution, 1e-12)))
                tempocnn_peakedness = 1.0 - float(entropy / np.log(len(distribution)))
            else:
                hypotheses = ((float(bpm), 1.0),)
                tempocnn_peakedness = 0.0
        else:
            hypotheses = ((float(bpm), 1.0),)
            tempocnn_peakedness = 0.0

        beat_this_python = os.environ.get("BEAT_THIS_PYTHON", "/opt/ml-venv/bin/python")
        beat_this_runner = os.environ.get("BEAT_THIS_RUNNER", "/app/scripts/beat_this_predict.py")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as beat_this_file:
                sf.write(beat_this_file.name, audio.samples, audio.sample_rate, subtype="PCM_16")
                beat_this_output = json.loads(
                    _run([beat_this_python, beat_this_runner, "--audio", beat_this_file.name], "Beat This")
                )
        except BackendUnavailable:
            beat_this_output = {"beats": [], "activations_stats": {"flatness": 1.0}}
        beats = np.asarray(beat_this_output.get("beats", []), dtype=float)
        beat_this_n_beats = int(beats.size)
        beat_this_bpm: float | None = None
        beat_this_stability = 0.0
        if beat_this_n_beats >= 4:
            intervals = np.diff(beats)
            intervals = intervals[intervals > 0]
            if intervals.size > 0:
                median_interval = float(np.median(intervals))
                if median_interval > 0:
                    beat_this_bpm = 60.0 / median_interval
                    iqr = float(np.percentile(intervals, 75) - np.percentile(intervals, 25))
                    beat_this_stability = 1.0 - max(0.0, min(1.0, iqr / median_interval))

        activation_flatness = float(
            beat_this_output.get("activations_stats", {}).get("flatness", 1.0 - beat_this_stability)
        )

        return TempoEvidence(
            hypotheses,
            tempocnn_peakedness,
            beat_this_bpm,
            beat_this_n_beats,
            beat_this_stability,
            float(bpm),
            float(confidence),
            _pulse_clarity(audio.samples, audio.sample_rate),
            activation_flatness,
        )

    def versions(self) -> dict[str, str]:
        essentia = self._imports()
        return {
            "essentia": _version(essentia),
            "libkeyfinder": os.environ.get("LIBKEYFINDER_VERSION", "2.2.8"),
            "skey": os.environ.get("SKEY_VERSION", "0.1.0"),
            "beat_this": os.environ.get("BEAT_THIS_VERSION", "unknown"),
            "tempocnn": os.path.basename(os.environ.get("TEMPOCNN_GRAPH", "/app/artifacts/deeptemp-k16.pb")),
        }


def _tempocnn_index_to_bpm(index: int) -> float:
    # Essentia TempoCNN classes are 30..285 BPM (256 classes).
    return float(30 + int(index))


def _pulse_clarity(samples: np.ndarray, sample_rate: int) -> float:
    hop = 512
    frame = 1024
    if len(samples) < frame * 2:
        return 0.0

    window = np.hanning(frame).astype(float)
    frames = []
    for offset in range(0, len(samples) - frame + 1, hop):
        segment = samples[offset:offset + frame] * window
        magnitude = np.abs(np.fft.rfft(segment))
        frames.append(np.log1p(magnitude))
    if len(frames) < 4:
        return 0.0
    spectrogram = np.asarray(frames)
    flux = np.diff(spectrogram, axis=0)
    onset = np.maximum(flux, 0.0).sum(axis=1)
    onset = np.concatenate(([onset[0]], onset))
    onset -= onset.mean()

    correlation = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    frame_rate = sample_rate / hop
    min_lag = max(1, round(60 * frame_rate / 200))
    max_lag = min(len(correlation), round(60 * frame_rate / 50))
    if correlation[0] <= 0 or max_lag <= min_lag:
        return 0.0
    return max(0.0, min(1.0, float(np.max(correlation[min_lag:max_lag]) / correlation[0])))

