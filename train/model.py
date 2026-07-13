from __future__ import annotations

import math

import torch
from torch import nn


class SetAttentionGate(nn.Module):
    def __init__(
        self, voters: int, warm_reliability: tuple[float, ...], warm_total: float,
        token_features: int = 12, d_model: int = 48, heads: int = 2,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.input_weight = nn.Parameter(torch.empty(d_model, token_features))
        self.input_bias = nn.Parameter(torch.zeros(d_model))
        self.identity = nn.Parameter(torch.empty(voters, d_model))
        self.norm1_weight = nn.Parameter(torch.ones(d_model))
        self.norm1_bias = nn.Parameter(torch.zeros(d_model))
        self.q_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.k_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.v_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.o_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.norm2_weight = nn.Parameter(torch.ones(d_model))
        self.norm2_bias = nn.Parameter(torch.zeros(d_model))
        self.ff1_weight = nn.Parameter(torch.empty(d_model * 2, d_model))
        self.ff1_bias = nn.Parameter(torch.zeros(d_model * 2))
        self.ff2_weight = nn.Parameter(torch.empty(d_model, d_model * 2))
        self.ff2_bias = nn.Parameter(torch.zeros(d_model))
        self.norm3_weight = nn.Parameter(torch.ones(d_model))
        self.norm3_bias = nn.Parameter(torch.zeros(d_model))
        self.score_weight = nn.Parameter(torch.zeros(d_model))
        self.score_bias = nn.Parameter(torch.zeros(1))
        self.reliability_bias = nn.Parameter(torch.log(torch.tensor(warm_reliability)))
        target_temperature = 1.0 / warm_total
        raw_temperature = math.log(math.expm1(target_temperature - 0.05))
        self.temperature_raw = nn.Parameter(torch.full((voters,), raw_temperature))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_weight)
        nn.init.normal_(self.identity, std=0.02)
        for parameter in (self.q_weight, self.k_weight, self.v_weight, self.o_weight):
            nn.init.xavier_uniform_(parameter)
        nn.init.xavier_uniform_(self.ff1_weight)
        nn.init.xavier_uniform_(self.ff2_weight)

    @staticmethod
    def _norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, 1e-5)

    def reliability(
        self, tokens: torch.Tensor, identities: torch.Tensor, mask: torch.Tensor, gate: str = "attention"
    ) -> torch.Tensor:
        hidden = tokens @ self.input_weight.T + self.input_bias + self.identity[identities]
        hidden = self._norm(hidden, self.norm1_weight, self.norm1_bias)
        if gate == "attention":
            batch, voters, width = hidden.shape
            head_width = width // self.heads
            query = (hidden @ self.q_weight.T).reshape(batch, voters, self.heads, head_width)
            key = (hidden @ self.k_weight.T).reshape(batch, voters, self.heads, head_width)
            value = (hidden @ self.v_weight.T).reshape(batch, voters, self.heads, head_width)
            scores = torch.einsum("bihd,bjhd->bhij", query, key) / math.sqrt(head_width)
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            mixed = torch.einsum("bhij,bjhd->bihd", attention, value).reshape(batch, voters, width)
            hidden = self._norm(hidden + mixed @ self.o_weight.T, self.norm2_weight, self.norm2_bias)
            feedforward = torch.nn.functional.gelu(hidden @ self.ff1_weight.T + self.ff1_bias, approximate="tanh")
            feedforward = feedforward @ self.ff2_weight.T + self.ff2_bias
            hidden = self._norm(hidden + feedforward, self.norm3_weight, self.norm3_bias)
        elif gate != "mlp":
            raise ValueError(f"unsupported gate: {gate}")
        scores = hidden @ self.score_weight + self.score_bias + self.reliability_bias[identities]
        scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1)

    def forward(
        self, tokens: torch.Tensor, log_likelihoods: torch.Tensor, identities: torch.Tensor,
        mask: torch.Tensor, gate: str = "attention",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.reliability(tokens, identities, mask, gate)
        temperature = torch.nn.functional.softplus(self.temperature_raw) + 0.05
        scaled = log_likelihoods / temperature[identities].unsqueeze(-1)
        logits = torch.sum(alpha.unsqueeze(-1) * scaled, dim=1)
        return logits, alpha


class FusionModel(nn.Module):
    """Two musically structured output heads: 24 keys and 72 tempo-circle bins."""

    def __init__(self, d_model: int = 48, heads: int = 2, gate: str = "attention") -> None:
        super().__init__()
        self.key = SetAttentionGate(4, (0.7, 1.0, 1.0, 0.9), 2.7, d_model=d_model, heads=heads)
        self.tempo = SetAttentionGate(3, (1.0, 1.0, 0.8), 2.8, d_model=d_model, heads=heads)
        self.gate = gate

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        key_logits, key_alpha = self.key(
            batch["key_tokens"], batch["key_log_likelihoods"], batch["key_identities"],
            batch["key_mask"], self.gate,
        )
        tempo_logits, tempo_alpha = self.tempo(
            batch["tempo_tokens"], batch["tempo_log_likelihoods"], batch["tempo_identities"],
            batch["tempo_mask"], self.gate,
        )
        return {
            "key_logits": key_logits,
            "tempo_logits": tempo_logits,
            "key_alpha": key_alpha,
            "tempo_alpha": tempo_alpha,
        }
