from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .domain import FieldResult, Key
from .music import key_dict, relation, relative, short_name

REVIEW_FLAGS = {
    "KEY_MODEL_DISAGREEMENT",
    "KEY_LOW_CONFIDENCE",
    "TEMPO_MODEL_DISAGREEMENT",
    "TEMPO_LOW_CONFIDENCE",
}


@dataclass(frozen=True, slots=True)
class KeyVote:
    detector: str
    key: Key
    strength: float = 0.0
    margin: float | None = None
    runner_up: Key | None = None
    probabilities: tuple[float, ...] | None = None
    margin_ratio_raw: float | None = None


@dataclass(frozen=True, slots=True)
class TempoEvidence:
    tempocnn_hypotheses: tuple[tuple[float, float], ...]
    tempocnn_peakedness: float
    beat_this_bpm: float | None
    beat_this_n_beats: int
    beat_this_stability: float
    essentia_bpm: float
    essentia_confidence: float
    pulse_clarity: float
    activation_flatness: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def key_agreement(votes: Sequence[KeyVote]) -> tuple[float, bool]:
    credits = {"exact": 1.0, "relative": 0.85, "fifth": 0.4, None: 0.0}
    pairs = [credits[relation(a.key, b.key)] for a, b in combinations(votes, 2)]
    return (sum(pairs) / len(pairs) if pairs else 0.0, any(x == 0 for x in pairs))


def _consensus_key(votes: Sequence[KeyVote], chroma: np.ndarray | None) -> Key:
    def score(candidate: Key) -> float:
        credits = {"exact": 1.0, "relative": 0.85, "fifth": 0.4, None: 0.0}
        return sum(credits[relation(candidate, vote.key)] for vote in votes)

    best = max((vote.key for vote in votes), key=score)
    modes = [vote.key.mode for vote in votes if vote.key.pitch_class in (best.pitch_class, relative(best).pitch_class)]
    if modes.count("major") != modes.count("minor"):
        desired = "major" if modes.count("major") > modes.count("minor") else "minor"
        options = [vote.key for vote in votes if vote.key.mode == desired]
        return max(options, key=score)
    if chroma is not None and chroma.size == 12:
        major_third = chroma[(best.pitch_class + 4) % 12]
        minor_third = chroma[(best.pitch_class + 3) % 12]
        return Key(best.pitch_class, "major" if major_third >= minor_third else "minor")
    return best


def score_key(votes: Sequence[KeyVote], tonalness: float, config: dict[str, Any], chroma: np.ndarray | None = None) -> FieldResult:
    if len(votes) != 3:
        raise ValueError("key ensemble requires exactly three votes")
    by_name = {vote.detector: vote for vote in votes}
    essentia = by_name["essentia"]
    skey = by_name["skey"]
    margin_values = []
    relative_ambiguity = False
    signal_votes: dict[str, Any] = {}
    for vote in votes:
        rel = relation(vote.key, vote.runner_up) if vote.runner_up else None
        margin = clamp01(vote.margin) if vote.margin is not None else None
        effective = margin
        if margin is not None and rel in config["neighbor_set"]:
            effective = max(margin, config["neighbor_floor"])
        if effective is not None:
            margin_values.append(effective)
        relative_ambiguity |= rel == "relative" and margin is not None and margin < config["neighbor_floor"]
        data: dict[str, Any] = {"key": short_name(vote.key)}
        if margin is not None and effective is not None:
            data.update({"margin_raw": margin, "margin_eff": effective})
        if vote.margin_ratio_raw is not None:
            data["margin_ratio_raw"] = float(vote.margin_ratio_raw)
        if vote.runner_up:
            data.update({"runner_up": short_name(vote.runner_up), "neighbor_relation": rel})
        if vote.detector == "essentia":
            data["strength"] = clamp01(vote.strength)
        signal_votes[vote.detector] = data

    probs = np.asarray(skey.probabilities or (), dtype=float)
    if probs.size != 24 or probs.sum() <= 0:
        raise ValueError("skey vote must include 24 probabilities")
    probs = probs / probs.sum()
    maxprob = float(probs.max())
    entropy_norm = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))) / math.log(24))
    peakedness = 1.0 - entropy_norm
    agreement, has_zero_pair = key_agreement(votes)
    values = {
        "strength": clamp01(essentia.strength),
        "margin": sum(margin_values) / len(margin_values) if margin_values else 0.0,
        "maxprob": maxprob,
        "entropy": peakedness,
        "agree": agreement,
        "tonalness": clamp01(tonalness),
    }
    confidence = clamp01(sum(config["weights"][name] * value for name, value in values.items()))
    flags: list[str] = []
    if relative_ambiguity:
        flags.append("KEY_MODE_AMBIGUOUS")
    if has_zero_pair and confidence >= config["thresholds"]["atonal"]:
        flags.append("KEY_MODEL_DISAGREEMENT")
    best = _consensus_key(votes, chroma)
    value: dict[str, Any] | None = key_dict(best)
    if confidence < config["thresholds"]["atonal"]:
        status = "atonal"
        value = None
        flags = [flag for flag in flags if flag != "KEY_MODEL_DISAGREEMENT"]
    elif confidence < config["thresholds"]["accept"]:
        status = "review"
        flags.append("KEY_LOW_CONFIDENCE")
    else:
        status = "review" if "KEY_MODEL_DISAGREEMENT" in flags else "detected"
    if value and relative_ambiguity and config.get("dual_mode_output"):
        other = relative(best)
        value = {
            "rendering": "dual",
            "primary": key_dict(best),
            "relative": key_dict(other),
            "display": f"{TONIC_DISPLAY(best)} / {TONIC_DISPLAY(other)}",
        }
    signal_votes["skey"].update({"max_prob": maxprob, "entropy_norm": entropy_norm})
    signal_votes.update({"agreement": agreement, "tonalness": clamp01(tonalness)})
    return FieldResult(status, value, confidence, signal_votes, flags)


def TONIC_DISPLAY(key: Key) -> str:
    from .music import TONICS
    return f"{TONICS[key.pitch_class]} {key.mode}"


_TEMPO_RELATIONS = ((1, 1, "1:1"), (2, 1, "2:1"), (1, 2, "1:2"), (3, 2, "3:2"), (2, 3, "2:3"), (3, 4, "3:4"), (4, 3, "4:3"))


def tempo_relation(a: float, b: float, tolerance: float = 0.02) -> tuple[str | None, float]:
    ratio = a / b
    for numerator, denominator, name in _TEMPO_RELATIONS:
        target = numerator / denominator
        if abs(ratio - target) / target <= tolerance:
            return name, 1.0 if name == "1:1" else 0.7
    return None, 0.0


def bar_snap(bpm: float, duration_s: float, settings: dict[str, Any]) -> dict[str, float]:
    best: dict[str, float] | None = None
    for meter in settings["beats_per_bar"]:
        bars = duration_s * bpm / (60.0 * meter)
        score = math.exp(-((bars - round(bars)) ** 2) / (settings["sigma"] ** 2))
        candidate = {"bpm": bpm, "bars": bars, "score": score, "beats_per_bar": meter}
        if best is None or score > best["score"]:
            best = candidate
    assert best is not None
    return best


def score_tempo(evidence: TempoEvidence, active_duration_s: float, config: dict[str, Any]) -> FieldResult:
    low, high = config["bpm_range"]
    hypotheses = [
        (float(bpm), float(probability))
        for bpm, probability in evidence.tempocnn_hypotheses
        if low <= float(bpm) <= high and float(probability) >= 0.0
    ]
    hypotheses.sort(key=lambda item: item[1], reverse=True)
    if not hypotheses:
        raise ValueError("tempocnn must return at least one hypothesis in configured BPM range")

    flags: list[str] = []
    tempocnn_top_bpm, tempocnn_top_prob = hypotheses[0]
    tempocnn_signal = 0.5 * clamp01(tempocnn_top_prob) + 0.5 * clamp01(evidence.tempocnn_peakedness)
    essentia_norm = clamp01(evidence.essentia_confidence / config["essentia_confidence_ceiling"])

    voters = [
        ("tempocnn", tempocnn_top_bpm),
        ("essentia", float(evidence.essentia_bpm)),
    ]
    beat_this_bpm = float(evidence.beat_this_bpm) if evidence.beat_this_bpm is not None else None
    if beat_this_bpm is not None and low <= beat_this_bpm <= high:
        voters.append(("beat_this", beat_this_bpm))
    elif beat_this_bpm is None:
        flags.append("BEAT_TRACKING_SPARSE")

    pair_relations: list[tuple[str | None, float]] = []
    for (_, bpm_a), (_, bpm_b) in combinations(voters, 2):
        pair_relations.append(tempo_relation(float(bpm_a), float(bpm_b), config["relation_tolerance"]))
    octagree = sum(credit for _, credit in pair_relations) / len(pair_relations) if pair_relations else 1.0
    if any(credit == 0.0 for _, credit in pair_relations):
        flags.append("TEMPO_MODEL_DISAGREEMENT")

    pulse_score = clamp01(evidence.pulse_clarity)
    activation_score = 1.0 - clamp01(evidence.activation_flatness)
    stability_score = clamp01(evidence.beat_this_stability)

    weights = dict(config["weights"])
    if beat_this_bpm is None:
        redistributed = weights["stability"]
        weights["stability"] = 0.0
        weights["pulse_clarity"] += redistributed

    values = {
        "essentia": essentia_norm,
        "tempocnn": tempocnn_signal,
        "stability": stability_score,
        "pulse_clarity": pulse_score,
        "activation": activation_score,
        "octagree": clamp01(octagree),
    }
    confidence = clamp01(sum(weights[name] * values[name] for name in weights))
    if evidence.pulse_clarity < 0.1 and evidence.activation_flatness > 0.9:
        confidence = min(confidence, config["thresholds"]["arhythmic"] - 1e-9)

    candidate_strengths: list[tuple[float, float]] = []
    for bpm, probability in hypotheses[:3]:
        candidate_strengths.append((float(bpm), clamp01(float(probability))))
    if beat_this_bpm is not None and low <= beat_this_bpm <= high:
        candidate_strengths.append((beat_this_bpm, stability_score))
    if low <= evidence.essentia_bpm <= high:
        candidate_strengths.append((float(evidence.essentia_bpm), essentia_norm))

    if not candidate_strengths:
        raise ValueError("no tempo candidates available inside configured BPM range")
    snap_scores = [bar_snap(bpm, active_duration_s, config["bar_snap"]) for bpm, _ in candidate_strengths]
    winner = candidate_strengths[0][0]
    top_strength = max(strength for _, strength in candidate_strengths)
    eligible = [
        snap
        for snap, (_, strength) in zip(snap_scores, candidate_strengths)
        if strength >= top_strength * config["bar_snap"]["strength_ratio"]
    ]
    if eligible:
        best_snap = max(eligible, key=lambda item: item["score"])
        if best_snap["score"] > 0.5:
            winner = best_snap["bpm"]
        else:
            flags.append("NO_BAR_SNAP")

    if evidence.pulse_clarity < 0.2:
        flags.append("LOW_PULSE_CLARITY")

    bpm: float | None = float(winner)
    if confidence < config["thresholds"]["arhythmic"]:
        status = "tempoless"
        bpm = None
        flags = [flag for flag in flags if flag != "TEMPO_MODEL_DISAGREEMENT"]
    elif confidence < config["thresholds"]["accept"]:
        status = "review"
        flags.append("TEMPO_LOW_CONFIDENCE")
    else:
        status = "review" if "TEMPO_MODEL_DISAGREEMENT" in flags else "detected"

    winner_snap = next((item for item in snap_scores if item["bpm"] == winner), max(snap_scores, key=lambda item: item["score"]))
    signals = {
        "tempocnn": {
            "hypotheses": [[float(bpm), float(probability)] for bpm, probability in hypotheses],
            "maxprob": float(tempocnn_top_prob),
            "peakedness": clamp01(evidence.tempocnn_peakedness),
        },
        "beat_this": {
            "bpm": beat_this_bpm,
            "stability": stability_score,
            "n_beats": int(evidence.beat_this_n_beats),
        },
        "essentia": {
            "bpm": float(evidence.essentia_bpm),
            "confidence_raw": float(evidence.essentia_confidence),
            "confidence_norm": essentia_norm,
        },
        "pulse_clarity": pulse_score,
        "activation_flatness": clamp01(evidence.activation_flatness),
        "octagree": clamp01(octagree),
        "bar_snap": {**winner_snap, "winner": winner},
    }
    return FieldResult(status, bpm, confidence, signals, flags)
