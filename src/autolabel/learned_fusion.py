from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

KEY_VOTERS = ("libkeyfinder", "essentia", "skey", "bass_root")
TEMPO_VOTERS = ("tempocnn", "essentia", "beat_this")
TOKEN_FEATURES = 12


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("likelihood must contain positive mass")
    return values / total


def _softmax(values: np.ndarray, mask: np.ndarray | None = None, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if mask is not None:
        values = np.where(mask, values, -np.inf)
    maximum = np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(values - maximum)
    if mask is not None:
        exponent = np.where(mask, exponent, 0.0)
    return exponent / np.maximum(exponent.sum(axis=axis, keepdims=True), 1e-15)


def _gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (values + 0.044715 * values**3)))


def _layer_norm(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    variance = np.square(values - mean).mean(axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + 1e-5) * weight + bias


def tempo_circle_likelihood(
    grid: np.ndarray, likelihood: np.ndarray, bins: int = 72, reference_bpm: float = 60.0
) -> np.ndarray:
    """Fold a fine BPM likelihood onto one log2 octave without moving fine-grid mass."""
    grid = np.asarray(grid, dtype=float)
    likelihood = _normalize(likelihood)
    positions = np.mod(np.log2(grid / float(reference_bpm)), 1.0) * bins
    lower = np.floor(positions).astype(int) % bins
    fraction = positions - np.floor(positions)
    result = np.zeros(bins, dtype=float)
    np.add.at(result, lower, likelihood * (1.0 - fraction))
    np.add.at(result, (lower + 1) % bins, likelihood * fraction)
    return _normalize(result)


def _dft_magnitudes(likelihood: np.ndarray, field: str) -> np.ndarray:
    likelihood = _normalize(likelihood)
    if field == "key":
        if likelihood.size != 24:
            raise ValueError("key likelihoods must contain 24 classes")
        major = np.abs(np.fft.rfft(likelihood[:12]))[1:4]
        minor = np.abs(np.fft.rfft(likelihood[12:]))[1:4]
        values = np.concatenate((major, minor))
    else:
        values = np.abs(np.fft.rfft(likelihood))[1:7]
    return values / max(float(likelihood.sum()), 1e-12)


def invariant_token_features(
    likelihoods: Sequence[np.ndarray],
    field: str,
    n_effective: float,
    materiality: tuple[float, float],
) -> np.ndarray:
    """Create rotation-invariant gate inputs; likelihood phases never enter the gate."""
    rows = []
    for raw in likelihoods:
        likelihood = _normalize(raw)
        order = np.sort(likelihood)[::-1]
        entropy = -float(np.sum(likelihood * np.log(np.maximum(likelihood, 1e-15)))) / math.log(likelihood.size)
        rows.append(np.concatenate((
            np.asarray([
                entropy,
                float(order[0]),
                float(order[0] - order[1]),
            ]),
            _dft_magnitudes(likelihood, field),
            np.asarray([
                min(float(n_effective) / 6.0, 1.0),
                float(np.clip(materiality[0], 0.0, 1.0)),
                float(np.clip(materiality[1], 0.0, 1.0)),
            ]),
        )))
    result = np.asarray(rows, dtype=float)
    if result.shape[1] != TOKEN_FEATURES:
        raise AssertionError(f"token feature width is {result.shape[1]}, expected {TOKEN_FEATURES}")
    return result


def _attention_weights(
    params: dict[str, np.ndarray], prefix: str, tokens: np.ndarray, identity_indices: np.ndarray,
    mask: np.ndarray, heads: int,
) -> np.ndarray:
    def value(name: str) -> np.ndarray:
        return params[f"{prefix}.{name}"]

    hidden = tokens @ value("input_weight").T + value("input_bias")
    hidden += value("identity")[identity_indices]
    hidden = _layer_norm(hidden, value("norm1_weight"), value("norm1_bias"))
    if f"{prefix}.q_weight" in params:
        width = hidden.shape[-1]
        if width % heads:
            raise ValueError("d_model must be divisible by attention heads")
        head_width = width // heads
        query = (hidden @ value("q_weight").T).reshape(-1, heads, head_width)
        key = (hidden @ value("k_weight").T).reshape(-1, heads, head_width)
        val = (hidden @ value("v_weight").T).reshape(-1, heads, head_width)
        scores = np.einsum("ihd,jhd->hij", query, key) / math.sqrt(head_width)
        key_mask = np.broadcast_to(mask[None, None, :], scores.shape)
        attention = _softmax(scores, key_mask, axis=-1)
        mixed = np.einsum("hij,jhd->ihd", attention, val).reshape(-1, width)
        hidden = _layer_norm(
            hidden + mixed @ value("o_weight").T,
            value("norm2_weight"), value("norm2_bias"),
        )
        feedforward = _gelu(hidden @ value("ff1_weight").T + value("ff1_bias"))
        feedforward = feedforward @ value("ff2_weight").T + value("ff2_bias")
        hidden = _layer_norm(
            hidden + feedforward, value("norm3_weight"), value("norm3_bias")
        )
    scores = (
        hidden @ value("score_weight")
        + float(value("score_bias").reshape(-1)[0])
        + value("reliability_bias")[identity_indices]
    )
    return _softmax(scores, mask, axis=0)


def pool_likelihoods(
    params: dict[str, np.ndarray], prefix: str, likelihoods: Sequence[np.ndarray],
    identity_indices: Sequence[int], tokens: np.ndarray, gate: str = "attention", heads: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if not likelihoods:
        raise ValueError("learned pooling requires at least one voter")
    mask = np.ones(len(likelihoods), dtype=bool)
    local = dict(params)
    if gate == "mlp":
        for name in (
            "q_weight", "k_weight", "v_weight", "o_weight", "norm2_weight", "norm2_bias",
            "ff1_weight", "ff1_bias", "ff2_weight", "ff2_bias", "norm3_weight", "norm3_bias",
        ):
            local.pop(f"{prefix}.{name}", None)
    weights = _attention_weights(
        local, prefix, np.asarray(tokens, dtype=float), np.asarray(identity_indices, dtype=int), mask, heads
    )
    temperature = np.log1p(np.exp(params[f"{prefix}.temperature_raw"])) + 0.05
    logits = np.zeros_like(np.asarray(likelihoods[0], dtype=float))
    for alpha, likelihood, identity in zip(weights, likelihoods, identity_indices):
        logits += float(alpha) * np.log(np.maximum(_normalize(likelihood), 1e-15)) / temperature[int(identity)]
    return _softmax(logits), weights


@dataclass(frozen=True)
class LearnedFusion:
    params: dict[str, np.ndarray]
    manifest: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "LearnedFusion":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            params = {name: np.asarray(archive[name]) for name in archive.files}
        manifest_path = path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        return cls(params, manifest)

    def pool(
        self, field: str, likelihoods: Sequence[np.ndarray], identities: Sequence[str],
        n_effective: float, materiality: tuple[float, float], gate: str,
        token_likelihoods: Sequence[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        vocabulary = KEY_VOTERS if field == "key" else TEMPO_VOTERS
        identity_indices = [vocabulary.index(name) for name in identities]
        tokens = invariant_token_features(
            token_likelihoods or likelihoods, field, n_effective, materiality
        )
        posterior, weights = pool_likelihoods(
            self.params, field, likelihoods, identity_indices, tokens, gate,
            int(self.manifest.get("heads", 2)),
        )
        return posterior, {name: float(weight) for name, weight in zip(identities, weights)}


@lru_cache(maxsize=4)
def load_if_available(path: str) -> LearnedFusion | None:
    candidate = Path(path)
    return LearnedFusion.load(candidate) if candidate.is_file() else None
