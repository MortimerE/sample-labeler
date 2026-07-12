import numpy as np
import pytest

from autolabel.config import load_config
from autolabel.domain import Key
from autolabel.scoring import KeyVote, TempoEvidence, score_key, score_tempo, tempo_relation


def probabilities(index=19, peak=0.82):
    values = np.full(24, (1 - peak) / 23)
    values[index] = peak
    return tuple(values)


def key_config():
    return load_config()["key"]


def tempo_config():
    return load_config()["tempo"]


def test_relative_runner_up_uses_neighbor_floor_and_sets_info_flag():
    votes = [
        KeyVote("libkeyfinder", Key(7, "minor"), margin=0.1, runner_up=Key(10, "major")),
        KeyVote("essentia", Key(7, "minor"), strength=0.9, margin=0.08, runner_up=Key(10, "major")),
        KeyVote("skey", Key(7, "minor"), probabilities=probabilities()),
    ]
    result = score_key(votes, 0.9, key_config())
    assert result.status == "detected"
    assert "KEY_MODE_AMBIGUOUS" in result.flags
    assert result.signals["essentia"]["margin_eff"] == 0.25
    assert result.signals["agreement"] == 1.0
    assert result.value["camelot"] == "6A"


def test_dual_mode_output_preserves_both_relative_keys():
    config = key_config()
    config["dual_mode_output"] = True
    votes = [
        KeyVote("libkeyfinder", Key(7, "minor"), margin=0.1, runner_up=Key(10, "major")),
        KeyVote("essentia", Key(7, "minor"), strength=0.9, margin=0.1, runner_up=Key(10, "major")),
        KeyVote("skey", Key(10, "major"), probabilities=probabilities(index=10)),
    ]
    result = score_key(votes, 0.9, config)
    assert result.value["rendering"] == "dual"
    assert {result.value["primary"]["pitch_class"], result.value["relative"]["pitch_class"]} == {7, 10}


def test_key_disagreement_forces_review_when_confident():
    votes = [
        KeyVote("libkeyfinder", Key(0, "major"), margin=0.9, runner_up=Key(1, "major")),
        KeyVote("essentia", Key(0, "major"), strength=1, margin=0.9, runner_up=Key(1, "major")),
        KeyVote("skey", Key(6, "minor"), probabilities=probabilities(index=18, peak=0.99)),
    ]
    result = score_key(votes, 1, key_config())
    assert result.status == "review"
    assert "KEY_MODEL_DISAGREEMENT" in result.flags


def test_low_key_evidence_abstains_as_atonal():
    uniform = tuple([1 / 24] * 24)
    votes = [
        KeyVote("libkeyfinder", Key(0, "major"), margin=0),
        KeyVote("essentia", Key(4, "minor"), strength=0, margin=0),
        KeyVote("skey", Key(8, "major"), probabilities=uniform),
    ]
    result = score_key(votes, 0, key_config())
    assert result.status == "atonal"
    assert result.value is None


@pytest.mark.parametrize(
    ("a", "b", "name", "credit"),
    [(120, 120, "1:1", 1), (120, 60, "2:1", 0.7), (90, 120, "3:4", 0.7), (121, 120, "1:1", 1)],
)
def test_tempo_metrical_relations(a, b, name, credit):
    assert tempo_relation(a, b) == (name, credit)


def test_bar_snap_breaks_close_half_double_tie():
    evidence = TempoEvidence(((120, 0.51), (60, 0.49)), 60, 2.6, 0.8, 0.1)
    result = score_tempo(evidence, active_duration_s=8.0, config=tempo_config())
    assert result.status == "detected"
    assert result.value == 120.0  # 4 bars at 4/4; 60 BPM only gives 2, also exact, raw winner stands
    assert result.signals["octave_relation"] == "2:1"


def test_tempoless_abstention_clears_hallucinated_bpm():
    config = tempo_config()
    evidence = TempoEvidence(((120, 0.01), (87, 0.99)), 173, 0, 0, 1)
    result = score_tempo(evidence, 7.3, config)
    assert result.status == "tempoless"
    assert result.value is None
    assert "LOW_PULSE_CLARITY" in result.flags


def test_tempo_disagreement_forces_review():
    evidence = TempoEvidence(((128, 1.0),), 103, 5.32, 1, 0)
    result = score_tempo(evidence, 7.5, tempo_config())
    assert result.status == "review"
    assert "TEMPO_MODEL_DISAGREEMENT" in result.flags
