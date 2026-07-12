from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

import numpy as np

from .domain import AudioBuffer, Key
from .scoring import KeyVote, TempoEvidence


class BackendUnavailable(RuntimeError):
    pass


class DetectorSuite(Protocol):
    def key_votes(self, audio: AudioBuffer) -> list[KeyVote]: ...
    def tempo_evidence(self, audio: AudioBuffer) -> TempoEvidence: ...
    def versions(self) -> dict[str, str]: ...


_NOTE_TO_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
               "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}


def parse_key(tonic: str, mode: str | None = None) -> Key:
    text = tonic.strip().replace("♭", "b").replace("♯", "#")
    if mode is None:
        lowered = text.lower()
        is_minor = lowered.endswith("minor") or lowered.endswith("min") or text.endswith("m")
        mode = "minor" if is_minor else "major"
        for suffix in (" minor", " major", "min", "maj", "m"):
            if text.lower().endswith(suffix):
                text = text[: -len(suffix)].strip()
                break
    normalized = text[0].upper() + text[1:]
    try:
        pc = _NOTE_TO_PC[normalized.upper()]
    except (IndexError, KeyError) as error:
        raise ValueError(f"unsupported key name: {tonic!r}") from error
    normalized_mode = "minor" if mode.lower().startswith("min") else "major"
    return Key(pc, normalized_mode)  # type: ignore[arg-type]


def _version(module: object) -> str:
    return str(getattr(module, "__version__", "unknown"))


@dataclass(slots=True)
class ProductionDetectors:
    """Lazy adapters around the four optional native/model dependencies.

    S-KEY and libkeyfinder have changed public APIs across releases. This adapter
    intentionally validates the small API surface required by the pinned image
    and fails with an actionable message instead of silently substituting a model.
    """

    def _imports(self) -> tuple[object, object, object, object]:
        try:
            return (import_module("essentia.standard"), import_module("madmom.features.beats"),
                    import_module("keyfinder"), import_module("skey"))
        except ImportError as error:
            raise BackendUnavailable(
                f"analysis backend {error.name!r} is not installed; use the provided Docker image or install the analysis dependencies"
            ) from error

    def key_votes(self, audio: AudioBuffer) -> list[KeyVote]:
        es, _, keyfinder, skey = self._imports()
        window = es.Windowing(type="blackmanharris62")
        spectrum = es.Spectrum()
        peaks = es.SpectralPeaks()
        hpcp_algorithm = es.HPCP()
        pool = []
        for frame in es.FrameGenerator(audio.samples, frameSize=4096, hopSize=2048, startFromZero=True):
            frequencies, magnitudes = peaks(spectrum(window(frame)))
            pool.append(hpcp_algorithm(frequencies, magnitudes))
        hpcp = np.mean(pool, axis=0).astype("float32")
        key_algorithm = es.Key(profileType="edma")
        tonic, mode, strength, relative_strength = key_algorithm(hpcp)
        # Essentia exposes first-to-second strength but not the runner-up label.
        # Re-run the same HPCP against all candidate rotations to recover it.
        candidates = []
        for profile in ("major", "minor"):
            for pc in range(12):
                candidates.append((float(np.dot(hpcp, np.roll(_PROFILE[profile], pc))), Key(pc, profile)))
        candidates.sort(reverse=True, key=lambda item: item[0])
        ess_key = parse_key(tonic, mode)
        runner = next(candidate for _, candidate in candidates if candidate != ess_key)

        if not hasattr(keyfinder, "analyze"):
            raise BackendUnavailable("pinned libkeyfinder binding must expose analyze(samples, sample_rate, top_k=2)")
        kf_result = keyfinder.analyze(audio.samples, audio.sample_rate, top_k=2)
        if not hasattr(skey, "predict_proba"):
            raise BackendUnavailable("pinned skey package must expose predict_proba(samples, sample_rate)")
        probabilities = np.asarray(skey.predict_proba(audio.samples, audio.sample_rate), dtype=float).reshape(-1)
        if probabilities.size != 24:
            raise BackendUnavailable("skey predict_proba must return 24 class probabilities")
        skey_index = int(np.argmax(probabilities))
        skey_key = Key(skey_index % 12, "major" if skey_index < 12 else "minor")
        return [
            KeyVote("libkeyfinder", parse_key(kf_result[0][0]), margin=float(kf_result[0][1] - kf_result[1][1]), runner_up=parse_key(kf_result[1][0])),
            KeyVote("essentia", ess_key, strength=float(strength), margin=float(relative_strength), runner_up=runner),
            KeyVote("skey", skey_key, probabilities=tuple(float(x) for x in probabilities)),
        ]

    def tempo_evidence(self, audio: AudioBuffer) -> TempoEvidence:
        es, beats, _, _ = self._imports()
        tempo_module = import_module("madmom.features.tempo")
        activations = beats.RNNBeatProcessor()(audio.samples)
        tempo = tempo_module.TempoEstimationProcessor(fps=100)(activations)
        hypotheses = tuple((float(row[0]), float(row[1])) for row in np.atleast_2d(tempo))
        bpm, _, confidence, _, _ = es.RhythmExtractor2013(method="multifeature")(audio.samples)
        flatness = float(np.std(activations) / (np.mean(activations) + 1e-9))
        activation_flatness = 1.0 / (1.0 + flatness)
        pulse_clarity = _pulse_clarity(audio.samples, audio.sample_rate)
        return TempoEvidence(hypotheses, float(bpm), float(confidence), pulse_clarity, activation_flatness)

    def versions(self) -> dict[str, str]:
        es, beats, keyfinder, skey = self._imports()
        return {"essentia": _version(es), "madmom": _version(beats), "libkeyfinder": _version(keyfinder), "skey": _version(skey)}


_PROFILE = {
    "major": np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "minor": np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
}


def _pulse_clarity(samples: np.ndarray, sample_rate: int) -> float:
    hop = 512
    frames = max(0, (len(samples) - 1024) // hop)
    if frames < 4:
        return 0.0
    energy = np.asarray([np.sum(samples[i * hop:i * hop + 1024] ** 2) for i in range(frames)])
    onset = np.maximum(0, np.diff(energy, prepend=energy[0]))
    onset -= onset.mean()
    correlation = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    min_lag = max(1, round(60 * sample_rate / (200 * hop)))
    max_lag = min(len(correlation), round(60 * sample_rate / (50 * hop)))
    if correlation[0] <= 0 or max_lag <= min_lag:
        return 0.0
    return max(0.0, min(1.0, float(np.max(correlation[min_lag:max_lag]) / correlation[0])))
