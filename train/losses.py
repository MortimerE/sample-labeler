from __future__ import annotations

import numpy as np
import torch

from autolabel.objectives import harmonic_similarity
from autolabel.scoring import KEY_LABELS


def similarity_matrix(**weights: float) -> np.ndarray:
    return np.asarray([
        [harmonic_similarity(truth, candidate, **weights) for candidate in KEY_LABELS]
        for truth in KEY_LABELS
    ])


def key_soft_targets(indices: torch.Tensor, eta: float, matrix: torch.Tensor) -> torch.Tensor:
    return torch.softmax(float(eta) * matrix[indices], dim=-1)


def tempo_circular_targets(
    indices: torch.Tensor, bins: int = 72, sigma_bins: float = 1.0, three_two_weight: float = 0.15,
) -> torch.Tensor:
    positions = torch.arange(bins, device=indices.device, dtype=torch.float32)
    center = indices.to(torch.float32).unsqueeze(-1)

    def bump(offset: float) -> torch.Tensor:
        target = torch.remainder(center + offset, bins)
        distance = torch.abs(positions - target)
        distance = torch.minimum(distance, bins - distance)
        return torch.exp(-0.5 * (distance / float(sigma_bins)) ** 2)

    three_two = bins * float(np.log2(1.5))
    values = bump(0.0) + float(three_two_weight) * (bump(three_two) + bump(-three_two))
    return values / values.sum(dim=-1, keepdim=True)


def semantic_fusion_loss(
    outputs: dict[str, torch.Tensor], key_indices: torch.Tensor, tempo_indices: torch.Tensor,
    similarity: torch.Tensor, eta: float = 4.0, tempo_weight: float = 1.0,
    expected_cost_weight: float = 0.1, sigma_bins: float = 1.0, three_two_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    key_target = key_soft_targets(key_indices, eta, similarity)
    tempo_target = tempo_circular_targets(tempo_indices, outputs["tempo_logits"].shape[-1], sigma_bins, three_two_weight)
    key_logprob = torch.log_softmax(outputs["key_logits"], dim=-1)
    tempo_logprob = torch.log_softmax(outputs["tempo_logits"], dim=-1)
    key_ce = -(key_target * key_logprob).sum(dim=-1).mean()
    tempo_ce = -(tempo_target * tempo_logprob).sum(dim=-1).mean()
    key_probability = torch.softmax(outputs["key_logits"], dim=-1)
    expected_cost = (key_probability * (1.0 - similarity[key_indices])).sum(dim=-1).mean()
    total = key_ce + float(tempo_weight) * tempo_ce + float(expected_cost_weight) * expected_cost
    return total, {"key_ce": key_ce, "tempo_ce": tempo_ce, "expected_cost": expected_cost}
