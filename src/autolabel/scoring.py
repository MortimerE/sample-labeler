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
    margin: float = 0.0
    runner_up: Key | None = None
    probabilities: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class TempoEvidence:
    madmom_hypotheses: tuple[tuple[float, float], ...]
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
        margin = clamp01(vote.margin)
        effective = max(margin, config["neighbor_floor"]) if rel in config["neighbor_set"] else margin
        if vote.detector in ("libkeyfinder", "essentia"):
            margin_values.append(effective)
        relative_ambiguity |= rel == "relative" and margin < config["neighbor_floor"]
        data: dict[str, Any] = {"key": short_name(vote.key)}
        if vote.detector in ("libkeyfinder", "essentia"):
            data.update({"margin_raw": margin, "margin_eff": effective})
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
        "margin": sum(margin_values) / len(margin_values),
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
    if not evidence.madmom_hypotheses:
        raise ValueError("madmom must return at least one hypothesis")
    low, high = config["bpm_range"]
    hypotheses = [(b, s) for b, s in evidence.madmom_hypotheses if low <= b <= high and s >= 0]
    if not hypotheses:
        raise ValueError("no madmom hypothesis inside configured BPM range")
    hypotheses.sort(key=lambda item: item[1], reverse=True)
    mm_bpm, mm_strength = hypotheses[0]
    relation_name, octagree = tempo_relation(mm_bpm, evidence.essentia_bpm, config["relation_tolerance"])
    flags: list[str] = []
    if relation_name is None:
        flags.append("TEMPO_MODEL_DISAGREEMENT")

    total_strength = sum(strength for _, strength in hypotheses[:5])
    madmom_norm = clamp01(mm_strength / total_strength) if total_strength else 0.0
    essentia_norm = clamp01(evidence.essentia_confidence / config["essentia_confidence_ceiling"])
    values = {
        "essentia": essentia_norm,
        "madmom": madmom_norm,
        "pulse_clarity": clamp01(evidence.pulse_clarity),
        "activation": 1.0 - clamp01(evidence.activation_flatness),
        "octagree": octagree,
    }
    confidence = clamp01(sum(config["weights"][name] * value for name, value in values.items()))
    # Two independent no-pulse signals agreeing is stronger negative evidence
    # than tempo estimators' octave-related hallucinations on sustained audio.
    if evidence.pulse_clarity < 0.1 and evidence.activation_flatness > 0.9:
        confidence = min(confidence, config["thresholds"]["arhythmic"] - 1e-9)

    candidates = [(mm_bpm, mm_strength)]
    if low <= evidence.essentia_bpm <= high:
        candidates.append((evidence.essentia_bpm, essentia_norm))
    snap_scores = [bar_snap(bpm, active_duration_s, config["bar_snap"]) for bpm, _ in candidates]
    winner = mm_bpm
    top_strength = max(strength for _, strength in candidates)
    eligible = [snap for snap, (_, strength) in zip(snap_scores, candidates) if strength >= top_strength * config["bar_snap"]["strength_ratio"]]
    if eligible:
        best_snap = max(eligible, key=lambda item: item["score"])
        if best_snap["score"] > 0.5:
            winner = best_snap["bpm"]
        else:
            flags.append("NO_BAR_SNAP")
    if relation_name not in (None, "1:1") and (not eligible or max(x["score"] for x in eligible) <= 0.5):
        flags.append("TEMPO_OCTAVE_AMBIGUOUS")
    if evidence.pulse_clarity < 0.2:
        flags.append("LOW_PULSE_CLARITY")

    bpm: float | None = float(winner)
    if confidence < config["thresholds"]["arhythmic"]:
        status = "tempoless"
        bpm = None
        flags = [f for f in flags if f != "TEMPO_MODEL_DISAGREEMENT"]
    elif confidence < config["thresholds"]["accept"]:
        status = "review"
        flags.append("TEMPO_LOW_CONFIDENCE")
    else:
        status = "review" if "TEMPO_MODEL_DISAGREEMENT" in flags else "detected"
    winner_snap = next((item for item in snap_scores if item["bpm"] == winner), max(snap_scores, key=lambda item: item["score"]))
    signals = {
        "madmom": {"hypotheses": [[float(b), float(s)] for b, s in hypotheses]},
        "essentia": {"bpm": evidence.essentia_bpm, "confidence_raw": evidence.essentia_confidence, "confidence_norm": essentia_norm},
        "pulse_clarity": clamp01(evidence.pulse_clarity),
        "activation_flatness": clamp01(evidence.activation_flatness),
        "octave_relation": relation_name,
        "bar_snap": {**winner_snap, "winner": winner},
    }
    return FieldResult(status, bpm, confidence, signals, flags)
