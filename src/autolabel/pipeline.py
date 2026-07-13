from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .analyzers import KeyEnsembleAnalyzer, TempoEnsembleAnalyzer
from .backends import BackendUnavailable, DetectorSuite, ProductionDetectors
from .config import load_config
from .domain import FieldResult
from .preprocess import decode
from .schema import AnalysisRecord
from .scoring import REVIEW_FLAGS


def _degenerate(flags: list[str]) -> tuple[FieldResult, FieldResult]:
    signals = {"short_circuit": flags.copy()}
    key = FieldResult("atonal", None, 0.0, [], signals, flags.copy())
    tempo = FieldResult("tempoless", None, 0.0, [], signals, flags.copy())
    return key, tempo


def analyze_file(
    path: str | Path,
    config_path: str | Path | None = None,
    detectors: DetectorSuite | None = None,
    emit_legacy_confidence: bool = False,
) -> AnalysisRecord:
    config = load_config(config_path)
    audio, context, file_flags = decode(path, config["preprocess"])
    suite = detectors or ProductionDetectors(
        essentia_margin_scale=float(config["key"].get("essentia_margin_scale", 2.0)),
        tonalness_uniform_floor=float(config["key"].get("tonalness_uniform_floor", 0.25)),
        skey_min_seconds=float(config["skey"]["min_seconds"]),
        bass_root_config=config["fusion"]["key"]["bass_root"],
    )
    if "SILENT_FILE" in file_flags:
        key_result, tempo_result = _degenerate(file_flags)
    else:
        key_analyzer = KeyEnsembleAnalyzer(suite, config, emit_legacy_confidence)
        tempo_analyzer = TempoEnsembleAnalyzer(suite, config, emit_legacy_confidence)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="autolabel") as executor:
            key_future = executor.submit(key_analyzer.analyze, audio, context)
            tempo_future = executor.submit(tempo_analyzer.analyze, audio, context)
            key_result = key_future.result()
            tempo_result = tempo_future.result()
    try:
        versions = suite.versions()
    except BackendUnavailable:
        versions = {name: "unavailable" for name in ("libkeyfinder", "essentia", "skey", "beat_this", "tempocnn")}
    versions = {"pipeline": __version__, **versions}
    all_field_flags = [*key_result.flags, *tempo_result.flags]
    return AnalysisRecord.model_validate({
        "analyzed_at": datetime.now(timezone.utc),
        "analyzer_versions": versions,
        "file": {
            "path": context.path,
            "sha1": context.sha1,
            "duration_s": audio.duration_s,
            "active_duration_s": audio.active_duration_s,
            "sample_rate": audio.source_sample_rate,
            "channels": audio.channels,
        },
        "key": {
            "status": key_result.status,
            "value": key_result.value,
            "confidence": key_result.confidence,
            "top_k": key_result.top_k,
            "signals": key_result.signals,
            "flags": key_result.flags,
        },
        "tempo": {
            "status": tempo_result.status,
            "bpm": tempo_result.value,
            "confidence": tempo_result.confidence,
            "top_k": tempo_result.top_k,
            "signals": tempo_result.signals,
            "flags": tempo_result.flags,
        },
        "review_required": (
            key_result.status == "review"
            or tempo_result.status == "review"
            or any(flag in REVIEW_FLAGS for flag in all_field_flags)
        ),
        "flags": file_flags,
    })
