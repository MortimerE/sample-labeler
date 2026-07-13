import numpy as np
import pytest

from autolabel.backends import _beat_grid_evidence


def test_beat_grid_average_removes_50_fps_median_bias():
    intervals = np.asarray([0.34, 0.34, 0.34, 0.36] * 12)
    beats = np.concatenate(([0.0], np.cumsum(intervals)))
    bpm, stability = _beat_grid_evidence(beats)
    assert bpm == pytest.approx(60.0 / np.mean(intervals))
    assert bpm == pytest.approx(173.913043478, rel=1e-8)
    assert stability > 0.9


def test_beat_grid_ignores_missing_beat_outlier():
    intervals = np.asarray([0.34, 0.34, 0.70, 0.36, 0.34])
    beats = np.concatenate(([0.0], np.cumsum(intervals)))
    bpm, _ = _beat_grid_evidence(beats)
    assert bpm == pytest.approx(60.0 / 0.345)
