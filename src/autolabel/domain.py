from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

Mode = Literal["major", "minor"]
Status = Literal["detected", "atonal", "tempoless", "review"]


@dataclass(frozen=True, slots=True)
class Key:
    pitch_class: int
    mode: Mode

    def __post_init__(self) -> None:
        if not 0 <= self.pitch_class <= 11:
            raise ValueError("pitch_class must be between 0 and 11")


@dataclass(slots=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int
    source_sample_rate: int
    channels: int
    duration_s: float
    active_duration_s: float


@dataclass(frozen=True, slots=True)
class FileContext:
    path: str
    sha1: str


@dataclass(slots=True)
class FieldResult:
    status: Status
    value: Any | None
    confidence: float
    signals: dict[str, Any]
    flags: list[str] = field(default_factory=list)


class Analyzer(Protocol):
    field: str

    def analyze(self, audio: AudioBuffer, ctx: FileContext) -> FieldResult: ...

