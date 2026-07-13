import numpy as np
import pytest

from autolabel.config import load_config
from autolabel.domain import Key
from autolabel.scoring import (
    KEY_LABELS,
    KeyVote,
    TempoEvidence,
    TempoLobe,
    _posterior_status,
    bass_root_likelihood,
    centroid_lobes,
    extract_lobes,
    fold_tempo_likelihood,
    harmonic_kernel,
    log_gaussian,
    log_linear_pool,
    point_voter_likelihood,
    octave_decision,
    reported_bpm,
    score_key,
    score_tempo,
)


def config():
    return load_config()


def probabilities(key: Key, peak: float = 0.82, runner: Key | None = None, runner_p: float = 0.0):
    values = np.full(24, max(0.0, 1.0 - peak - runner_p) / (23 if runner is None else 22))
    values[key.pitch_class + (12 if key.mode == "minor" else 0)] = peak
    if runner is not None:
        values[runner.pitch_class + (12 if runner.mode == "minor" else 0)] = runner_p
    return tuple(values)


def test_harmonic_kernel_table_includes_parallel():
    cfg = config()["fusion"]["key"]
    center = Key(0, "major")
    likelihood = harmonic_kernel(center, cfg["kernel"])
    exact = likelihood[KEY_LABELS.index(center)]
    assert likelihood[KEY_LABELS.index(Key(9, "minor"))] / exact == pytest.approx(0.55)
    assert likelihood[KEY_LABELS.index(Key(0, "minor"))] / exact == pytest.approx(0.45)
    assert likelihood[KEY_LABELS.index(Key(7, "major"))] / exact == pytest.approx(0.35)
    assert likelihood[KEY_LABELS.index(Key(6, "minor"))] / exact == pytest.approx(0.02)


def test_runner_up_mix_shrinks_monotonically_with_margin():
    cfg = config()["fusion"]["key"]
    runner = Key(6, "minor")
    low = point_voter_likelihood(KeyVote("essentia", Key(0, "major"), margin=0.1, runner_up=runner), cfg)
    high = point_voter_likelihood(KeyVote("essentia", Key(0, "major"), margin=0.9, runner_up=runner), cfg)
    assert low[KEY_LABELS.index(runner)] > high[KEY_LABELS.index(runner)]


def test_log_linear_pool_is_order_independent_and_handles_dropout():
    a = np.asarray([0.8, 0.2])
    b = np.asarray([0.3, 0.7])
    assert log_linear_pool([a, b], [1.0, 0.7]) == pytest.approx(log_linear_pool([b, a], [0.7, 1.0]))
    assert log_linear_pool([a], [1.0]) == pytest.approx(a)


def test_tempo_folding_moves_89_bpm_support_to_178():
    cfg = config()["fusion"]["tempo"]
    grid = np.geomspace(40, 220, 1200)
    source = log_gaussian(grid, 89.0, 0.008)
    folded = fold_tempo_likelihood(grid, source, cfg["folding"])
    at_178 = int(np.argmin(np.abs(grid - 178.0)))
    assert folded[at_178] > source[at_178]


def test_lobe_nms_separates_distant_modes():
    grid = np.geomspace(40, 220, 1200)
    posterior = 0.7 * log_gaussian(grid, 174, 0.008) + 0.3 * log_gaussian(grid, 87, 0.008)
    lobes = extract_lobes(grid, posterior / posterior.sum(), 0.06)
    assert lobes[0].bpm == pytest.approx(174, rel=0.01)
    assert lobes[1].bpm == pytest.approx(87, rel=0.01)


def test_lobe_centroid_uses_reference_mass_not_grid_peak():
    grid = np.asarray([173.0, 174.0, 175.0, 176.0])
    posterior = np.asarray([0.05, 0.70, 0.20, 0.05])
    lobe = TempoLobe(175.0, 0.9, 173.0, 176.0)
    refined = centroid_lobes([lobe], grid, posterior)[0]
    assert refined.bpm == pytest.approx(174.25)
    assert refined.mass == 0.9


def test_lobe_centroid_preserves_negligible_secondary_peak():
    grid = np.asarray([173.0, 174.0, 185.0, 186.0])
    posterior = np.asarray([0.49, 0.49, 0.01, 0.01])
    lobe = TempoLobe(186.0, 1e-8, 175.0, 197.0)
    assert centroid_lobes([lobe], grid, posterior)[0].bpm == 186.0


@pytest.mark.parametrize(
    ("p1", "p2", "related", "expected"),
    [(0.5, 0.1, False, "detected"), (0.36, 0.25, True, "detected"), (0.36, 0.25, False, "review")],
)
def test_status_matrix(p1, p2, related, expected):
    assert _posterior_status(p1, p2, related, config()["fusion"]["key"]["detect"]) == expected


def test_bass_key_runner_up_evidence_fuses_to_ab_minor():
    ab_minor = Key(8, "minor")
    cs_minor = Key(1, "minor")
    votes = [
        KeyVote("libkeyfinder", ab_minor, runner_up=Key(11, "major")),
        KeyVote("essentia", ab_minor, strength=0.8, margin=0.5, runner_up=cs_minor),
        KeyVote("skey", cs_minor, margin=0.3, runner_up=ab_minor, probabilities=probabilities(cs_minor, 0.55, ab_minor, 0.30)),
    ]
    result = score_key(votes, 0.6, config())
    assert result.top_k[0]["tonic"] == "Ab"
    assert result.top_k[0]["mode"] == "minor"
    assert result.confidence > 1 / 24
    assert result.status in {"detected", "review"}


def test_bass_tempo_fuses_adjacent_classes_and_uses_precise_essentia_bpm():
    evidence = TempoEvidence(
        ((174, .28), (173, .22), (175, .14), (89, .09)),
        0.4,
        166.0,
        20,
        0.8,
        174.0,
        3.5,
        0.12,
        0.5,
    )
    result = score_tempo(evidence, 8.0, config())
    assert result.top_k[0]["bpm"] == 174.0
    assert result.confidence >= 0.6
    assert result.status != "tempoless"


def test_true_key_ambiguity_reviews_with_ranked_candidates():
    am = Key(9, "minor")
    fs_major = Key(6, "major")
    eb_major = Key(3, "major")
    split = np.full(24, 1e-4)
    split[KEY_LABELS.index(am)] = 0.15
    split[KEY_LABELS.index(fs_major)] = 0.05
    split[KEY_LABELS.index(eb_major)] = 0.80
    split /= split.sum()
    votes = [
        KeyVote("libkeyfinder", am),
        KeyVote("essentia", fs_major, strength=0.8, margin=1.0),
        KeyVote("skey", eb_major, probabilities=tuple(split)),
    ]
    result = score_key(votes, 0.8, config())
    assert result.status == "review"
    assert result.top_k
    assert "KEY_MODEL_DISAGREEMENT" in result.flags


def test_noise_keeps_ranked_candidates_but_uses_materiality_axes():
    flat = tuple([1 / 24] * 24)
    key_result = score_key(
        [
            KeyVote("libkeyfinder", Key(0, "major")),
            KeyVote("essentia", Key(4, "minor"), strength=0.0, margin=0.0),
            KeyVote("skey", Key(8, "major"), probabilities=flat),
        ],
        0.0,
        config(),
    )
    tempo_result = score_tempo(TempoEvidence(((120, 0.01),), 0.0, None, 1, 0.0, 173, 0, 0, 1), 7.3, config())
    assert key_result.status == "atonal" and key_result.top_k
    assert tempo_result.status == "tempoless" and tempo_result.top_k


def test_bass_kernel_prefers_tonic_but_allows_fifth_swap():
    cfg = config()["fusion"]["key"]["bass_root"]["kernel"]
    histogram = np.zeros(12)
    histogram[1] = 0.7  # C# root
    histogram[8] = 0.3  # G# fifth
    likelihood = bass_root_likelihood(histogram, cfg)
    assert likelihood[KEY_LABELS.index(Key(1, "minor"))] > likelihood[KEY_LABELS.index(Key(8, "minor"))]


def test_background_tempering_deflates_single_tempo_voter():
    evidence = TempoEvidence((), 0.0, None, 0, 0.0, 120.0, 3.5, 0.5, 0.8)
    result = score_tempo(evidence, 0.9, config())
    assert result.confidence <= 0.55
    assert result.status != "detected"
    assert result.signals["n_effective_voters"] == 1


def test_octave_decision_prefers_dense_174_grid():
    events = tuple(np.arange(0.0, 12.0, 60.0 / 174.0))
    evidence = TempoEvidence((), 0.0, 174.0, len(events), 0.9, 87.0, 1.0, 0.8, 0.1, onset_events=events)
    lobes = [TempoLobe(87.0, 0.55, 82.0, 92.0), TempoLobe(174.0, 0.45, 164.0, 184.0)]
    decided, margin, ambiguous, _ = octave_decision(
        lobes, evidence, 12.0, config()["fusion"]["tempo"]["octave_decision"],
        config()["tempo"]["bar_snap"], config()["tempo"]["relation_tolerance"],
    )
    assert decided[0].bpm == 174.0
    assert margin is not None and margin > 0.15
    assert not ambiguous


@pytest.mark.parametrize(("centroid", "essentia", "expected"), [
    (173.995, 172.266, 174.0),
    (174.243, 172.266, 174.0),
    (92.5, 91.0, 92.5),
    (99.56, 98.0, 99.56),
])
def test_soft_bpm_reporting_rules(centroid, essentia, expected):
    lobe = TempoLobe(centroid, 0.7, centroid * 0.94, centroid * 1.06)
    assert reported_bpm(lobe, essentia, config()["reporting"]) == pytest.approx(expected)
