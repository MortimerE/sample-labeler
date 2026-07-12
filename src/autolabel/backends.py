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


class DetectorSuite(Protocol):
    def key_votes(self, audio: AudioBuffer) -> list[KeyVote]: ...
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
    """Container-backed adapters for Essentia, madmom, libKeyFinder, and S-KEY."""

    def _imports(self) -> tuple[object, object]:
        try:
            return import_module("essentia.standard"), import_module("madmom.features.beats")
        except ImportError as error:
            raise BackendUnavailable(
                f"analysis backend {error.name!r} is not installed; use the provided container image"
            ) from error

    def key_votes(self, audio: AudioBuffer) -> list[KeyVote]:
        essentia, _ = self._imports()
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
        hpcp = np.mean(frames, axis=0).astype("float32")
        candidates = _profile_candidates(hpcp)

        tonic, mode, strength, relative_strength = essentia.Key(profileType="edma")(hpcp)
        essentia_key = parse_key(tonic, mode)
        essentia_runner = next(candidate for _, candidate in candidates if candidate != essentia_key)

        with tempfile.NamedTemporaryFile(suffix=".wav") as working_file:
            sf.write(working_file.name, audio.samples, audio.sample_rate, subtype="FLOAT")
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
        skey_index = int(np.argmax(probabilities))
        skey_key = Key(skey_index % 12, "major" if skey_index < 12 else "minor")

        keyfinder_runner_score, keyfinder_runner = next(
            (score, candidate) for score, candidate in candidates if candidate != keyfinder_key
        )
        keyfinder_score = next(
            (score for score, candidate in candidates if candidate == keyfinder_key), candidates[0][0]
        )
        keyfinder_margin = max(0.0, (keyfinder_score - keyfinder_runner_score) / (abs(keyfinder_score) + 1e-9))
        return [
            KeyVote("libkeyfinder", keyfinder_key, margin=keyfinder_margin, runner_up=keyfinder_runner),
            KeyVote(
                "essentia", essentia_key, strength=float(strength), margin=float(relative_strength),
                runner_up=essentia_runner,
            ),
            KeyVote("skey", skey_key, probabilities=tuple(float(value) for value in probabilities)),
        ]

    def tempo_evidence(self, audio: AudioBuffer) -> TempoEvidence:
        essentia, beats = self._imports()
        tempo_module = import_module("madmom.features.tempo")
        activations = beats.RNNBeatProcessor()(audio.samples)
        tempo = tempo_module.TempoEstimationProcessor(fps=100)(activations)
        hypotheses = tuple((float(row[0]), float(row[1])) for row in np.atleast_2d(tempo))
        bpm, _, confidence, _, _ = essentia.RhythmExtractor2013(method="multifeature")(audio.samples)
        dispersion = float(np.std(activations) / (np.mean(activations) + 1e-9))
        activation_flatness = 1.0 / (1.0 + dispersion)
        return TempoEvidence(
            hypotheses, float(bpm), float(confidence),
            _pulse_clarity(audio.samples, audio.sample_rate), activation_flatness,
        )

    def versions(self) -> dict[str, str]:
        essentia, beats = self._imports()
        return {
            "essentia": _version(essentia),
            "madmom": _version(beats),
            "libkeyfinder": os.environ.get("LIBKEYFINDER_VERSION", "2.2.8"),
            "skey": os.environ.get("SKEY_VERSION", "0.1.0"),
        }


def _pulse_clarity(samples: np.ndarray, sample_rate: int) -> float:
    hop = 512
    frames = max(0, (len(samples) - 1024) // hop)
    if frames < 4:
        return 0.0
    energy = np.asarray([np.sum(samples[index * hop:index * hop + 1024] ** 2) for index in range(frames)])
    onset = np.maximum(0, np.diff(energy, prepend=energy[0]))
    onset -= onset.mean()
    correlation = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    min_lag = max(1, round(60 * sample_rate / (200 * hop)))
    max_lag = min(len(correlation), round(60 * sample_rate / (50 * hop)))
    if correlation[0] <= 0 or max_lag <= min_lag:
        return 0.0
    return max(0.0, min(1.0, float(np.max(correlation[min_lag:max_lag]) / correlation[0])))

