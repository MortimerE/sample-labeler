from __future__ import annotations

import math

import numpy as np

from .domain import Key
from .music import relative

_MAJOR = {0, 2, 4, 5, 7, 9, 11}
_MINOR = {0, 2, 3, 5, 7, 8, 10}


def _scale(key: Key) -> set[int]:
    intervals = _MAJOR if key.mode == "major" else _MINOR
    return {(key.pitch_class + interval) % 12 for interval in intervals}


def _fifths_position(pitch_class: int) -> int:
    return (pitch_class * 7) % 12


def harmonic_similarity(
    truth: Key, candidate: Key, ring_weight: float = 0.55,
    overlap_weight: float = 0.30, root_weight: float = 0.15,
) -> float:
    if truth == candidate:
        return 1.0
    if relative(truth) == candidate:
        return 0.93
    distance = abs(_fifths_position(truth.pitch_class) - _fifths_position(candidate.pitch_class))
    distance = min(distance, 12 - distance)
    mode_factor = 1.0 if truth.mode == candidate.mode else 0.8
    ring = math.exp(-distance / 1.5) * mode_factor
    overlap = len(_scale(truth) & _scale(candidate)) / 7.0
    root_interval = (candidate.pitch_class - truth.pitch_class) % 12
    root = 1.0 if root_interval == 0 else 0.5 if root_interval in (5, 7) else 0.0
    return ring_weight * ring + overlap_weight * overlap + root_weight * root


def tempo_circular_target(
    index: int, bins: int = 72, sigma_bins: float = 1.0, three_two_weight: float = 0.15,
) -> np.ndarray:
    positions = np.arange(bins, dtype=float)

    def bump(offset: float) -> np.ndarray:
        target = (index + offset) % bins
        distance = np.abs(positions - target)
        distance = np.minimum(distance, bins - distance)
        return np.exp(-0.5 * np.square(distance / sigma_bins))

    three_two = bins * math.log2(1.5)
    values = bump(0.0) + three_two_weight * (bump(three_two) + bump(-three_two))
    return values / values.sum()
