import json

import numpy as np
import pytest

from autolabel.domain import Key
from autolabel.config import load_config
from autolabel.learned_fusion import (
    KEY_VOTERS,
    LearnedFusion,
    invariant_token_features,
    pool_likelihoods,
    tempo_circle_likelihood,
)
from autolabel.objectives import harmonic_similarity, tempo_circular_target
from autolabel.scoring import log_linear_pool
from autolabel.scoring import KeyVote, score_key


def parameters(prefix: str, identities: int, width: int = 8, seed: int = 3):
    rng = np.random.default_rng(seed)
    result = {
        f"{prefix}.input_weight": rng.normal(0, 0.1, (width, 12)),
        f"{prefix}.input_bias": rng.normal(0, 0.1, width),
        f"{prefix}.identity": rng.normal(0, 0.1, (identities, width)),
        f"{prefix}.norm1_weight": np.ones(width),
        f"{prefix}.norm1_bias": np.zeros(width),
        f"{prefix}.q_weight": rng.normal(0, 0.1, (width, width)),
        f"{prefix}.k_weight": rng.normal(0, 0.1, (width, width)),
        f"{prefix}.v_weight": rng.normal(0, 0.1, (width, width)),
        f"{prefix}.o_weight": rng.normal(0, 0.1, (width, width)),
        f"{prefix}.norm2_weight": np.ones(width),
        f"{prefix}.norm2_bias": np.zeros(width),
        f"{prefix}.ff1_weight": rng.normal(0, 0.1, (width * 2, width)),
        f"{prefix}.ff1_bias": np.zeros(width * 2),
        f"{prefix}.ff2_weight": rng.normal(0, 0.1, (width, width * 2)),
        f"{prefix}.ff2_bias": np.zeros(width),
        f"{prefix}.norm3_weight": np.ones(width),
        f"{prefix}.norm3_bias": np.zeros(width),
        f"{prefix}.score_weight": rng.normal(0, 0.1, width),
        f"{prefix}.score_bias": np.zeros(1),
        f"{prefix}.reliability_bias": np.zeros(identities),
        f"{prefix}.temperature_raw": np.full(identities, 0.5),
    }
    return result


def rotate_key(values, semitones):
    values = np.asarray(values)
    return np.concatenate((np.roll(values[:12], semitones), np.roll(values[12:], semitones)))


def test_key_gate_is_rotation_invariant_and_output_is_equivariant():
    rng = np.random.default_rng(9)
    likelihoods = [rng.dirichlet(np.ones(24)) for _ in range(3)]
    rotated = [rotate_key(values, 5) for values in likelihoods]
    tokens = invariant_token_features(likelihoods, "key", 3, (0.7, 0.8))
    rotated_tokens = invariant_token_features(rotated, "key", 3, (0.7, 0.8))
    assert rotated_tokens == pytest.approx(tokens, abs=1e-12)
    params = parameters("key", 4)
    posterior, alpha = pool_likelihoods(params, "key", likelihoods, [0, 1, 2], tokens)
    rotated_posterior, rotated_alpha = pool_likelihoods(
        params, "key", rotated, [0, 1, 2], rotated_tokens
    )
    assert rotated_alpha == pytest.approx(alpha, abs=1e-12)
    assert rotated_posterior == pytest.approx(rotate_key(posterior, 5), abs=1e-12)


def test_masked_voter_subset_is_finite_and_normalized():
    rng = np.random.default_rng(4)
    likelihood = rng.dirichlet(np.ones(24))
    tokens = invariant_token_features([likelihood], "key", 1, (0.2, 0.3))
    posterior, alpha = pool_likelihoods(parameters("key", 4), "key", [likelihood], [2], tokens)
    assert np.isfinite(posterior).all()
    assert posterior.sum() == pytest.approx(1.0)
    assert alpha.tolist() == pytest.approx([1.0])


def test_warm_start_matches_patch4_key_pool_with_base_voters():
    rng = np.random.default_rng(14)
    likelihoods = [rng.dirichlet(np.ones(24)) for _ in range(3)]
    tokens = invariant_token_features(likelihoods, "key", 3, (0.7, 0.8))
    params = parameters("key", 4)
    params["key.score_weight"][:] = 0.0
    params["key.reliability_bias"] = np.log(np.asarray([0.7, 1.0, 1.0, 0.9]))
    target_temperature = 1.0 / 2.7
    params["key.temperature_raw"][:] = np.log(np.expm1(target_temperature - 0.05))
    learned, _ = pool_likelihoods(params, "key", likelihoods, [0, 1, 2], tokens)
    hand = log_linear_pool(likelihoods, [0.7, 1.0, 1.0])
    assert learned == pytest.approx(hand, abs=1e-12)


def test_tempo_circle_projection_and_smoothing_wrap():
    grid = np.geomspace(60, 120, 500)
    likelihood = np.exp(-0.5 * ((grid - 119.8) / 0.2) ** 2)
    circle = tempo_circle_likelihood(grid, likelihood, 72)
    assert circle.sum() == pytest.approx(1.0)
    assert circle[0] > 0.1 or circle[-1] > 0.1
    target = tempo_circular_target(0, bins=72)
    assert target.sum() == pytest.approx(1.0)
    assert target[71] == pytest.approx(target[1])


def test_harmonic_similarity_snapshot():
    assert harmonic_similarity(Key(0, "major"), Key(9, "minor")) == pytest.approx(0.93)
    assert harmonic_similarity(Key(0, "major"), Key(7, "major")) == pytest.approx(0.61, abs=0.04)
    assert harmonic_similarity(Key(0, "major"), Key(1, "minor")) == pytest.approx(0.14, abs=0.05)


def test_parameter_artifact_loads_without_torch(tmp_path):
    params = parameters("key", 4) | parameters("tempo", 3)
    path = tmp_path / "fusion_params.npz"
    np.savez(path, **params)
    path.with_suffix(".json").write_text(json.dumps({"heads": 2}))
    fusion = LearnedFusion.load(path)
    assert fusion.manifest["heads"] == 2
    assert "key.q_weight" in fusion.params


def test_score_key_activates_artifact_when_present(tmp_path):
    path = tmp_path / "fusion_params.npz"
    np.savez(path, **parameters("key", 4))
    config = load_config()
    config["fusion"]["learned"]["params_path"] = str(path)
    result = score_key(
        [
            KeyVote("libkeyfinder", Key(0, "major")),
            KeyVote("essentia", Key(0, "major"), strength=0.8, margin=0.7),
        ],
        0.8,
        config,
        active_duration_s=8.0,
        harmonic_ratio=0.8,
    )
    assert result.signals["learned_fusion"]["active"] is True
    assert set(result.signals["learned_fusion"]["weights"]) == {"libkeyfinder", "essentia"}
