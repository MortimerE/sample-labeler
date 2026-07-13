from __future__ import annotations

from .backends import DetectorSuite
from .domain import AudioBuffer, FieldResult, FileContext
from .scoring import score_key, score_tempo


class KeyEnsembleAnalyzer:
    field = "key"

    def __init__(self, detectors: DetectorSuite, config: dict, emit_legacy_confidence: bool = False) -> None:
        self.detectors = detectors
        self.config = config
        self.emit_legacy_confidence = emit_legacy_confidence

    def analyze(self, audio: AudioBuffer, ctx: FileContext) -> FieldResult:
        del ctx
        evidence = self.detectors.key_votes(audio)
        return score_key(
            evidence.votes,
            evidence.tonalness,
            self.config,
            evidence.chroma,
            evidence_flags=evidence.flags,
            active_duration_s=audio.active_duration_s,
            harmonic_ratio=evidence.harmonic_ratio,
            bass_root_histogram=evidence.bass_root_histogram,
            sub_prominence=evidence.sub_prominence,
            sub_coverage=evidence.sub_coverage,
            bass_segments=evidence.bass_segments,
            emit_legacy_confidence=self.emit_legacy_confidence,
        )


class TempoEnsembleAnalyzer:
    field = "bpm"

    def __init__(self, detectors: DetectorSuite, config: dict, emit_legacy_confidence: bool = False) -> None:
        self.detectors = detectors
        self.config = config
        self.emit_legacy_confidence = emit_legacy_confidence

    def analyze(self, audio: AudioBuffer, ctx: FileContext) -> FieldResult:
        del ctx
        return score_tempo(
            self.detectors.tempo_evidence(audio),
            audio.active_duration_s,
            self.config,
            emit_legacy_confidence=self.emit_legacy_confidence,
        )
