import pytest
import numpy as np

from autolabel.backends import (
    _normalize_essentia_margin,
    _profile_candidates,
    _tempocnn_index_to_bpm,
    _tonalness_from_hpcp,
)
from autolabel.domain import Key
from autolabel.music import key_dict, relation, relative


def test_relative_relation_is_symmetric():
    g_minor = Key(7, "minor")
    b_flat_major = Key(10, "major")
    assert relative(g_minor) == b_flat_major
    assert relative(b_flat_major) == g_minor
    assert relation(g_minor, b_flat_major) == "relative"
    assert relation(b_flat_major, g_minor) == "relative"


@pytest.mark.parametrize(
    ("key", "camelot"),
    [(Key(7, "minor"), "6A"), (Key(10, "major"), "6B"), (Key(0, "major"), "8B")],
)
def test_camelot_mapping(key, camelot):
    assert key_dict(key)["camelot"] == camelot


def test_fifth_relation_requires_same_mode():
    assert relation(Key(0, "major"), Key(7, "major")) == "fifth"
    assert relation(Key(0, "major"), Key(7, "minor")) is None


def test_profile_candidates_expect_c_referenced_hpcp():
    c_referenced = np.zeros(12, dtype=float)
    c_referenced[9] = 1.4   # A
    c_referenced[0] = 1.0   # C
    c_referenced[4] = 0.6   # E
    top_key = _profile_candidates(c_referenced)[0][1]
    assert top_key == Key(9, "minor")

    a_referenced = np.roll(c_referenced, -9)
    wrong_top = _profile_candidates(a_referenced)[0][1]
    assert wrong_top != Key(9, "minor")


def test_peaks_hpcp_tonalness_separates_tonal_from_noise():
    tonal_hpcp = np.zeros(12, dtype=float)
    tonal_hpcp[9] = 1.4
    tonal_hpcp[0] = 1.0
    tonal_hpcp[4] = 0.6
    _, tonalness_tonal = _tonalness_from_hpcp(tonal_hpcp, 0.25)

    noise_hpcp = np.ones(12, dtype=float)
    _, tonalness_noise = _tonalness_from_hpcp(noise_hpcp, 0.25)

    assert tonalness_tonal > tonalness_noise
    assert tonalness_noise < 0.15


def test_essentia_margin_ratio_normalization():
    assert _normalize_essentia_margin(1.0, 2.0) == pytest.approx(0.0)
    assert _normalize_essentia_margin(3.0, 2.0) == pytest.approx(1.0)


def test_tempocnn_index_to_bpm_linear_mapping():
    assert _tempocnn_index_to_bpm(0) == pytest.approx(30.0)
    assert _tempocnn_index_to_bpm(127) == pytest.approx(157.0)
    assert _tempocnn_index_to_bpm(255) == pytest.approx(285.0)

