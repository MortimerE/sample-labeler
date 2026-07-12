from __future__ import annotations

from .backends import DetectorSuite
from .domain import AudioBuffer, FieldResult, FileContext
from .preprocess import chroma_and_tonalness
from .scoring import score_key, score_tempo


class KeyEnsembleAnalyzer:
    field = "key"

    def __init__(self, detectors: DetectorSuite, config: dict) -> None:
        self.detectors = detectors
        self.config = config

    def analyze(self, audio: AudioBuffer, ctx: FileContext) -> FieldResult:
        del ctx
        chroma, tonalness = chroma_and_tonalness(audio)
        return score_key(self.detectors.key_votes(audio), tonalness, self.config, chroma)


class TempoEnsembleAnalyzer:
    field = "bpm"

    def __init__(self, detectors: DetectorSuite, config: dict) -> None:
        self.detectors = detectors
        self.config = config

    def analyze(self, audio: AudioBuffer, ctx: FileContext) -> FieldResult:
        del ctx
        return score_tempo(self.detectors.tempo_evidence(audio), audio.active_duration_s, self.config)

