from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileRecord(StrictModel):
    path: str
    sha1: str
    duration_s: float = Field(ge=0)
    active_duration_s: float = Field(ge=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)


class KeyValue(StrictModel):
    tonic: str
    mode: Literal["major", "minor"]
    pitch_class: int = Field(ge=0, le=11)
    camelot: str
    rendering: Literal["single"] = "single"


class DualKeyValue(StrictModel):
    rendering: Literal["dual"] = "dual"
    primary: KeyValue
    relative: KeyValue
    display: str


class KeyRecord(StrictModel):
    status: Literal["detected", "atonal", "review"]
    value: KeyValue | DualKeyValue | None
    confidence: float = Field(ge=0, le=1)
    signals: dict[str, Any]
    flags: list[str]


class TempoRecord(StrictModel):
    status: Literal["detected", "tempoless", "review"]
    bpm: float | None = Field(default=None, gt=0)
    confidence: float = Field(ge=0, le=1)
    signals: dict[str, Any]
    flags: list[str]


class AnalysisRecord(StrictModel):
    schema_version: Literal["1.1"] = "1.1"
    analyzed_at: datetime
    analyzer_versions: dict[str, str]
    file: FileRecord
    key: KeyRecord
    tempo: TempoRecord
    review_required: bool
    flags: list[str]

