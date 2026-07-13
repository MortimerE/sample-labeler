from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .domain import FieldResult, Key
from .learned_fusion import load_if_available, tempo_circle_likelihood
from .music import key_dict, relation, relative, short_name

REVIEW_FLAGS = {
    "KEY_MODEL_DISAGREEMENT",
    "KEY_LOW_CONFIDENCE",
    "TEMPO_MODEL_DISAGREEMENT",
    "TEMPO_LOW_CONFIDENCE",
    "TEMPO_OCTAVE_AMBIGUOUS",
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
    flags: tuple[str, ...] = ()
    downbeats: tuple[float, ...] = ()
    onset_events: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class TempoLobe:
    bpm: float
    mass: float
    low: float
    high: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("likelihood must contain positive mass")
    return values / total


def log_linear_pool(likelihoods: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Pool normalized likelihoods with a weighted product of experts."""
    if not likelihoods or len(likelihoods) != len(weights):
        raise ValueError("pooling requires one weight per likelihood")
    normalized = [_normalize(values) for values in likelihoods]
    width = normalized[0].size
    if any(values.size != width for values in normalized):
        raise ValueError("all likelihoods must use the same label space")
    log_posterior = np.zeros(width, dtype=float)
    for values, weight in zip(normalized, weights):
        log_posterior += float(weight) * np.log(np.maximum(values, 1e-15))
    log_posterior -= float(np.max(log_posterior))
    return _normalize(np.exp(log_posterior))


def temper_posterior(posterior: np.ndarray, evidence_weight: float, background_weight: float) -> np.ndarray:
    """Flatten a posterior according to missing/degraded and short-input evidence.

    A literal uniform product-of-experts term cancels during normalization. Its
    intended confidence effect is therefore represented as posterior temperature.
    """
    total_weight = max(float(evidence_weight + background_weight), 1e-9)
    exponent = float(evidence_weight) / total_weight
    flattened = _normalize(np.maximum(posterior, 1e-15) ** exponent)
    uniform = np.full(flattened.size, 1.0 / flattened.size)
    return _normalize(float(evidence_weight) * flattened + float(background_weight) * uniform)


def bass_root_likelihood(histogram: Sequence[float], config: dict[str, float]) -> np.ndarray:
    hist = _normalize(np.asarray(histogram, dtype=float))
    if hist.size != 12:
        raise ValueError("bass-root histogram must contain 12 pitch classes")
    major = {0, 2, 4, 5, 7, 9, 11}
    minor = {0, 2, 3, 5, 7, 8, 10}
    values = np.zeros(24, dtype=float)
    for index, key in enumerate(KEY_LABELS):
        scale = major if key.mode == "major" else minor
        for root, mass in enumerate(hist):
            interval = (root - key.pitch_class) % 12
            if interval == 0:
                kernel = config["tonic"]
            elif interval == 7:
                kernel = config["fifth"]
            elif interval in scale:
                kernel = config["scale"]
            else:
                kernel = config["floor"]
            values[index] += mass * float(kernel)
    return _normalize(values)


def _all_keys() -> tuple[Key, ...]:
    return tuple(Key(pc, mode) for mode in ("major", "minor") for pc in range(12))


KEY_LABELS = _all_keys()


def _key_index(key: Key) -> int:
    return key.pitch_class + (12 if key.mode == "minor" else 0)


def harmonic_kernel(center: Key, config: dict[str, float]) -> np.ndarray:
    values = np.empty(24, dtype=float)
    for index, candidate in enumerate(KEY_LABELS):
        kind = relation(center, candidate)
        values[index] = float(config.get(kind or "floor", config["floor"]))
    return _normalize(values)


def point_voter_likelihood(vote: KeyVote, config: dict[str, Any]) -> np.ndarray:
    likelihood = harmonic_kernel(vote.key, config["kernel"])
    if vote.runner_up is None:
        return likelihood
    if vote.detector == "libkeyfinder":
        mix = float(config["runner_mix"]["libkeyfinder_lambda"])
    else:
        margin = clamp01(vote.margin or 0.0)
        mix = float(config["runner_mix"]["lambda_base"]) * (1.0 - margin)
    runner = harmonic_kernel(vote.runner_up, config["kernel"])
    return _normalize((1.0 - mix) * likelihood + mix * runner)


def skey_likelihood(vote: KeyVote, temperature: float) -> np.ndarray:
    probabilities = np.asarray(vote.probabilities or (), dtype=float)
    if probabilities.size != 24 or float(probabilities.sum()) <= 0:
        raise ValueError("skey vote must include 24 probabilities")
    return _normalize(np.maximum(probabilities, 1e-15) ** (1.0 / max(float(temperature), 1e-6)))


def _posterior_status(p1: float, p2: float, related: bool, detect: dict[str, float]) -> str:
    if p1 >= float(detect["p1"]):
        return "detected"
    if related and p1 >= float(detect["p1_related"]) and p1 + p2 >= float(detect["p12_related"]):
        return "detected"
    return "review"


def key_agreement(votes: Sequence[KeyVote]) -> tuple[float, bool]:
    credits = {"exact": 1.0, "relative": 0.85, "parallel": 0.6, "fifth": 0.4, None: 0.0}
    pairs = [credits[relation(a.key, b.key)] for a, b in combinations(votes, 2)]
    return (sum(pairs) / len(pairs) if pairs else 0.0, any(value == 0 for value in pairs))


def _legacy_key_confidence(votes: Sequence[KeyVote], tonalness: float) -> float | None:
    by_name = {vote.detector: vote for vote in votes}
    if "essentia" not in by_name or "skey" not in by_name or len(votes) != 3:
        return None
    skey = by_name["skey"]
    probabilities = _normalize(np.asarray(skey.probabilities or (), dtype=float))
    margins = [clamp01(vote.margin) for vote in votes if vote.margin is not None]
    agreement, _ = key_agreement(votes)
    entropy_signal = 1.0 - float(
        -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))) / math.log(24)
    )
    values = {
        "strength": clamp01(by_name["essentia"].strength),
        "margin": sum(margins) / len(margins) if margins else 0.0,
        "maxprob": float(probabilities.max()),
        "entropy": entropy_signal,
        "agree": agreement,
        "tonalness": clamp01(tonalness),
    }
    weights = {"strength": .20, "margin": .15, "maxprob": .20, "entropy": .10, "agree": .25, "tonalness": .10}
    return clamp01(sum(weights[name] * value for name, value in values.items()))


def _key_candidate(key: Key, probability: float) -> dict[str, Any]:
    value = key_dict(key)
    return {
        "tonic": value["tonic"],
        "mode": value["mode"],
        "camelot": value["camelot"],
        "p": float(probability),
    }


def score_key(
    votes: Sequence[KeyVote],
    tonalness: float,
    config: dict[str, Any],
    chroma: np.ndarray | None = None,
    *,
    evidence_flags: Sequence[str] = (),
    active_duration_s: float = 0.0,
    harmonic_ratio: float = 0.0,
    bass_root_histogram: Sequence[float] | None = None,
    sub_prominence: float = 0.0,
    sub_coverage: float = 0.0,
    bass_segments: int = 0,
    emit_legacy_confidence: bool = False,
) -> FieldResult:
    del chroma  # posterior argmax is the deterministic label policy in schema 1.2
    fusion = config["fusion"]["key"]
    likelihoods: list[np.ndarray] = []
    weights: list[float] = []
    learned_identities: list[str] = []
    signal_votes: dict[str, Any] = {}
    skey_probabilities: np.ndarray | None = None
    essentia_strength = 0.0
    effective_voters = 0.0
    for vote in votes:
        if vote.detector == "skey":
            likelihood = skey_likelihood(vote, fusion["skey_temperature"])
            skey_probabilities = likelihood
        else:
            likelihood = point_voter_likelihood(vote, fusion)
        likelihoods.append(likelihood)
        learned_identities.append(vote.detector)
        weight = float(fusion["reliability"][vote.detector])
        effective = 1.0
        if vote.detector == "skey" and "SKEY_INPUT_TILED" in evidence_flags:
            discount = float(fusion["skey_tiled_discount"])
            weight *= discount
            effective = discount
        weights.append(weight)
        effective_voters += effective
        data: dict[str, Any] = {"key": short_name(vote.key)}
        if vote.margin is not None:
            data["margin"] = clamp01(vote.margin)
        if vote.runner_up is not None:
            data.update({"runner_up": short_name(vote.runner_up), "neighbor_relation": relation(vote.key, vote.runner_up)})
        if vote.margin_ratio_raw is not None:
            data["margin_ratio_raw"] = float(vote.margin_ratio_raw)
        if vote.detector == "essentia":
            essentia_strength = clamp01(vote.strength)
            data["strength"] = essentia_strength
        data["likelihood"] = [float(value) for value in likelihood]
        signal_votes[vote.detector] = data
    bass_config = fusion["bass_root"]
    bass_signal: dict[str, Any] | None = None
    if bass_root_histogram is not None:
        bass_likelihood = bass_root_likelihood(bass_root_histogram, bass_config["kernel"])
        likelihoods.append(bass_likelihood)
        learned_identities.append("bass_root")
        weights.append(float(bass_config["reliability"]) * clamp01(sub_prominence))
        effective_voters += 1.0
        bass_signal = {
            "histogram": [float(value) for value in bass_root_histogram],
            "sub_prominence": clamp01(sub_prominence),
            "sub_coverage": clamp01(sub_coverage),
            "segments": int(bass_segments),
            "likelihood": [float(value) for value in bass_likelihood],
        }
    expected_weight = sum(float(value) for value in fusion["reliability"].values()) + float(bass_config["reliability"])
    active_weight = sum(weights)
    background_config = config["fusion"]["background"]
    shortness = float(background_config["k_short"]) * max(
        0.0,
        (float(background_config["min_full_evidence_s"]) - active_duration_s)
        / float(background_config["min_full_evidence_s"]),
    )
    background_weight = max(0.0, expected_weight - active_weight) + shortness
    learned_config = config["fusion"].get("learned", {})
    learned = load_if_available(str(learned_config.get("params_path", "")))
    learned_weights: dict[str, float] | None = None
    if learned is None:
        pooled = log_linear_pool(likelihoods, weights)
    else:
        pooled, learned_weights = learned.pool(
            "key", likelihoods, learned_identities, effective_voters,
            (clamp01(tonalness), clamp01(harmonic_ratio)),
            str(learned_config.get("gate", "attention")),
        )
    posterior = temper_posterior(pooled, active_weight, background_weight)
    order = np.argsort(posterior)[::-1]
    top1 = KEY_LABELS[int(order[0])]
    top2 = KEY_LABELS[int(order[1])]
    p1 = float(posterior[order[0]])
    p2 = float(posterior[order[1]])
    report_count = int(config["report"]["top_k"])
    top_k = [_key_candidate(KEY_LABELS[int(index)], float(posterior[index])) for index in order[:report_count]]

    neighborhood = np.asarray([relation(top1, key) is not None for key in KEY_LABELS], dtype=bool)
    coherent_source = skey_probabilities if skey_probabilities is not None else posterior
    coherent_mass = float(coherent_source[neighborhood].sum())
    axis = config["axes"]["tonality"]
    axis_values = {
        "harmonic_ratio": max(
            clamp01(harmonic_ratio),
            clamp01(sub_prominence) if bass_root_histogram is not None else 0.0,
        ),
        "tonalness_h": clamp01(tonalness),
        "coherent_mass": clamp01(coherent_mass),
        "strength": essentia_strength,
    }
    tonality_score = clamp01(sum(float(axis["weights"][name]) * value for name, value in axis_values.items()))

    flags = list(evidence_flags)
    related = relation(top1, top2) is not None
    label_status = _posterior_status(p1, p2, related, fusion["detect"])
    if effective_voters < 2.0:
        label_status = "review"
    if (p2 >= 0.25 and not related) or key_agreement(votes)[1]:
        flags.append("KEY_MODEL_DISAGREEMENT")
    if tonality_score < float(axis["atonal"]):
        status = "atonal"
        value: dict[str, Any] | None = None
    else:
        status = label_status
        value = key_dict(top1)
        if status == "review":
            flags.append("KEY_LOW_CONFIDENCE")
        if (
            config["key"].get("dual_mode_output")
            and relation(top1, top2) == "relative"
            and p1 - p2 < float(fusion["detect"]["dual_gap"])
        ):
            value = {
                "rendering": "dual",
                "primary": key_dict(top1),
                "relative": key_dict(top2),
                "display": f"{short_name(top1)} / {short_name(top2)}",
            }

    entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1e-15))) / math.log(24))
    signals: dict[str, Any] = {
        **signal_votes,
        "bass_root": bass_signal,
        "sub_prominence": clamp01(sub_prominence),
        "sub_coverage": clamp01(sub_coverage),
        "bass_segments": int(bass_segments),
        "harmonic_ratio": clamp01(harmonic_ratio),
        "n_effective_voters": effective_voters,
        "background_weight": background_weight,
        "learned_fusion": {
            "active": learned is not None,
            "weights": learned_weights,
        },
        "posterior": [float(value) for value in posterior],
        "posterior_entropy": entropy,
        "tonality": {**axis_values, "score": tonality_score},
    }
    if emit_legacy_confidence:
        signals["legacy_confidence"] = _legacy_key_confidence(votes, tonalness)
    return FieldResult(status, value, p1, top_k, signals, list(dict.fromkeys(flags)))


_TEMPO_RELATIONS = (
    (1, 1, "1:1"), (2, 1, "2:1"), (1, 2, "1:2"),
    (3, 2, "3:2"), (2, 3, "2:3"), (3, 1, "3:1"), (1, 3, "1:3"),
)


def tempo_relation(a: float, b: float, tolerance: float = 0.02) -> tuple[str | None, float]:
    ratio = float(a) / float(b)
    for numerator, denominator, name in _TEMPO_RELATIONS:
        target = numerator / denominator
        if abs(ratio - target) / target <= tolerance:
            return name, 1.0 if name == "1:1" else 0.7
    return None, 0.0


def log_gaussian(grid: np.ndarray, bpm: float, sigma: float) -> np.ndarray:
    if bpm <= 0:
        return np.zeros_like(grid, dtype=float)
    distance = np.log(np.asarray(grid, dtype=float) / float(bpm))
    return _normalize(np.exp(-0.5 * np.square(distance / max(float(sigma), 1e-6))))


def fold_tempo_likelihood(grid: np.ndarray, likelihood: np.ndarray, config: dict[str, float]) -> np.ndarray:
    ratios = (
        (1.0, 1.0),
        (2.0, float(config["octave"])),
        (0.5, float(config["octave"])),
        (1.5, float(config["threehalves"])),
        (2.0 / 3.0, float(config["threehalves"])),
        (3.0, float(config["triple"])),
        (1.0 / 3.0, float(config["triple"])),
    )
    folded = np.zeros_like(likelihood, dtype=float)
    for ratio, discount in ratios:
        sampled = np.interp(ratio * grid, grid, likelihood, left=0.0, right=0.0)
        folded = np.maximum(folded, discount * sampled)
    return _normalize(folded)


def extract_lobes(grid: np.ndarray, posterior: np.ndarray, separation: float) -> list[TempoLobe]:
    remaining = np.asarray(posterior, dtype=float).copy()
    lobes: list[TempoLobe] = []
    while float(remaining.sum()) > 1e-12:
        peak_index = int(np.argmax(remaining))
        peak = float(grid[peak_index])
        low = peak * (1.0 - separation)
        high = peak * (1.0 + separation)
        mask = (grid >= low) & (grid <= high)
        mass = float(remaining[mask].sum())
        if mass <= 1e-12:
            break
        lobes.append(TempoLobe(peak, mass, low, high))
        remaining[mask] = 0.0
    return lobes


def centroid_lobes(
    lobes: Sequence[TempoLobe], grid: np.ndarray, reference_posterior: np.ndarray
) -> list[TempoLobe]:
    """Set lobe locations from pre-prior posterior mass, preserving ranked masses."""
    refined = []
    for lobe in lobes:
        if lobe.mass < 1e-6:
            refined.append(lobe)
            continue
        mask = (grid >= lobe.low) & (grid <= lobe.high)
        mass = float(reference_posterior[mask].sum())
        bpm = (
            float(np.sum(grid[mask] * reference_posterior[mask]) / mass)
            if mass > 1e-15 else lobe.bpm
        )
        refined.append(TempoLobe(bpm, lobe.mass, lobe.low, lobe.high))
    return refined


def _discount_metrical_lobes(
    lobes: Sequence[TempoLobe],
    folding: dict[str, float],
    tolerance: float,
) -> list[TempoLobe]:
    if not lobes:
        return []
    discounts = {
        "2:1": float(folding["octave"]),
        "1:2": float(folding["octave"]),
        "3:2": float(folding["threehalves"]),
        "2:3": float(folding["threehalves"]),
        "3:1": float(folding["triple"]),
        "1:3": float(folding["triple"]),
    }
    weighted = []
    for index, lobe in enumerate(lobes):
        kind = tempo_relation(lobes[0].bpm, lobe.bpm, tolerance)[0] if index else "1:1"
        weighted.append(lobe.mass * discounts.get(kind, 1.0))
    total = sum(weighted)
    return [
        TempoLobe(lobe.bpm, mass / total, lobe.low, lobe.high)
        for lobe, mass in zip(lobes, weighted)
    ]


def metrical_support(lobes: Sequence[TempoLobe], folding: dict[str, float], tolerance: float) -> list[float]:
    discounts = {
        "2:1": folding["octave"], "1:2": folding["octave"],
        "3:2": folding["threehalves"], "2:3": folding["threehalves"],
        "3:1": folding["triple"], "1:3": folding["triple"],
    }
    return [
        clamp01(sum(
            other.mass * float(discounts.get(tempo_relation(lobe.bpm, other.bpm, tolerance)[0], 0.0))
            for other in lobes if other is not lobe
        ))
        for lobe in lobes
    ]


def onset_grid_fit(events: Sequence[float], bpm: float, tolerance: float) -> float:
    if len(events) < 2 or bpm <= 0:
        return 0.0
    period = 60.0 / bpm
    iois = np.diff(np.asarray(events, dtype=float))
    iois = iois[iois > 0]
    if not iois.size:
        return 0.0
    multiples = np.asarray([0.5, 1.0, 2.0, 3.0, 4.0])
    weights = np.asarray([0.5, 1.0, 0.8, 0.5, 0.3])
    relative = np.abs(iois[:, None] / period - multiples) / multiples
    matched = np.where(relative <= tolerance, weights, 0.0).max(axis=1)
    return clamp01(float(np.mean(matched)))


def _beat_count_fit(n_beats: int, active_duration_s: float, bpm: float) -> float:
    if n_beats <= 1 or active_duration_s <= 0 or bpm <= 0:
        return 0.0
    estimate = 60.0 * n_beats / active_duration_s
    return clamp01(math.exp(-0.5 * (math.log(estimate / bpm) / 0.12) ** 2))


def _bar_fit(active_duration_s: float, bpm: float, bar_config: dict[str, Any]) -> float:
    if active_duration_s <= 0:
        return 0.0
    sigma = max(float(bar_config["sigma"]), 1e-6)
    values = []
    for meter in bar_config["beats_per_bar"]:
        bars = active_duration_s * bpm / (60.0 * meter)
        values.append(math.exp(-((bars - round(bars)) / sigma) ** 2))
    return max(values)


def octave_decision(
    lobes: list[TempoLobe], evidence: TempoEvidence, active_duration_s: float,
    config: dict[str, Any], bar_config: dict[str, Any], relation_tolerance: float,
    candidate_support_ratio: float = 0.0,
) -> tuple[list[TempoLobe], float | None, bool, list[dict[str, float]]]:
    if len(lobes) < 2:
        return lobes, None, False, []
    first, second = lobes[:2]
    related = tempo_relation(first.bpm, second.bpm, relation_tolerance)[0]
    mass_ratio = second.mass / max(first.mass, 1e-12)
    if related not in {"2:1", "1:2"} or max(mass_ratio, candidate_support_ratio) < float(config["mass_ratio"]):
        return lobes, None, False, []
    details = []
    for lobe in (first, second):
        onset = onset_grid_fit(evidence.onset_events, lobe.bpm, float(config["ioi_tolerance"]))
        beats = _beat_count_fit(evidence.beat_this_n_beats, active_duration_s, lobe.bpm)
        bar = _bar_fit(active_duration_s, lobe.bpm, bar_config)
        details.append({"bpm": lobe.bpm, "onset_grid": onset, "beat_count": beats, "bar": bar,
                        "score": 0.6 * onset + 0.3 * beats + 0.1 * bar})
    max_onset = max(item["onset_grid"] for item in details)
    max_beats = max(item["beat_count"] for item in details)
    for item in details:
        onset = item["onset_grid"] / max(max_onset, 1e-9)
        beats = item["beat_count"] / max(max_beats, 1e-9)
        item["score"] = 0.6 * onset + 0.3 * beats + 0.1 * item["bar"]
    if details[1]["score"] > details[0]["score"]:
        lobes = [second, first, *lobes[2:]]
        details = [details[1], details[0]]
    margin = abs(details[0]["score"] - details[1]["score"]) / max(details[0]["score"], 1e-9)
    ambiguous = margin < float(config["margin"])
    return lobes, margin, ambiguous, details


def reported_bpm(lobe: TempoLobe, essentia_bpm: float, config: dict[str, float]) -> float:
    centroid = float(lobe.bpm)
    if lobe.low <= essentia_bpm <= lobe.high and abs(essentia_bpm - centroid) / centroid <= float(config["essentia_substitution_pct"]):
        return float(essentia_bpm)
    integer = float(round(centroid))
    if lobe.low <= integer <= lobe.high and abs(integer - centroid) / centroid <= float(config["integer_snap_pct"]):
        return integer
    return centroid


def _bar_snap_prior(grid: np.ndarray, duration_s: float, config: dict[str, Any]) -> np.ndarray:
    if duration_s <= 0:
        return np.ones_like(grid)
    prior = np.zeros_like(grid)
    sigma = max(float(config["sigma"]), 1e-6)
    for meter in config["beats_per_bar"]:
        bars = duration_s * grid / (60.0 * float(meter))
        prior = np.maximum(prior, np.exp(-np.square(bars - np.round(bars)) / (sigma * sigma)))
    # This is deliberately a weak prior: the bar alignment may nudge a close
    # posterior, but it must not overwhelm three agreeing tempo estimators.
    return 0.5 + 0.5 * prior


def _tempo_status(m1: float, m2: float, related: bool, detect: dict[str, float]) -> str:
    mapped = {
        "p1": detect["m1"],
        "p1_related": detect["m1_related"],
        "p12_related": detect["m12_related"],
    }
    return _posterior_status(m1, m2, related, mapped)


def _legacy_tempo_confidence(evidence: TempoEvidence, octagree: float, essentia_norm: float) -> float:
    tempocnn_top = evidence.tempocnn_hypotheses[0][1] if evidence.tempocnn_hypotheses else 0.0
    values = {
        "essentia": essentia_norm,
        "tempocnn": 0.5 * clamp01(tempocnn_top) + 0.5 * clamp01(evidence.tempocnn_peakedness),
        "stability": clamp01(evidence.beat_this_stability),
        "pulse_clarity": clamp01(evidence.pulse_clarity),
        "activation": 1.0 - clamp01(evidence.activation_flatness),
        "octagree": clamp01(octagree),
    }
    weights = {"essentia": .15, "tempocnn": .20, "stability": .15, "pulse_clarity": .15, "activation": .10, "octagree": .25}
    return clamp01(sum(weights[name] * value for name, value in values.items()))


def score_tempo(
    evidence: TempoEvidence,
    active_duration_s: float,
    config: dict[str, Any],
    *,
    emit_legacy_confidence: bool = False,
) -> FieldResult:
    fusion = config["fusion"]["tempo"]
    grid_config = fusion["grid"]
    grid = np.geomspace(float(grid_config["low"]), float(grid_config["high"]), int(grid_config["points"]))
    likelihoods: list[np.ndarray] = []
    weights: list[float] = []
    voter_likelihoods: list[np.ndarray] = []
    learned_identities: list[str] = []
    sigma = fusion["sigma"]

    valid_hypotheses = [(float(bpm), max(0.0, float(probability))) for bpm, probability in evidence.tempocnn_hypotheses if bpm > 0]
    if valid_hypotheses:
        residual = max(0.0, 1.0 - sum(probability for _, probability in valid_hypotheses))
        tempocnn = np.full(grid.size, residual / grid.size, dtype=float)
        for bpm, probability in valid_hypotheses:
            tempocnn += probability * log_gaussian(grid, bpm, float(sigma["tempocnn"]))
        likelihoods.append(_normalize(tempocnn))
        voter_likelihoods.append(likelihoods[-1])
        learned_identities.append("tempocnn")
        weights.append(float(fusion["reliability"]["tempocnn"]))

    essentia_norm = clamp01(evidence.essentia_confidence / float(config["tempo"]["essentia_confidence_ceiling"]))
    if evidence.essentia_bpm > 0:
        essentia_sigma = float(sigma["essentia_base"]) + float(sigma["essentia_spread"]) * (1.0 - essentia_norm)
        essentia = log_gaussian(grid, float(evidence.essentia_bpm), essentia_sigma)
        likelihoods.append(essentia)
        voter_likelihoods.append(likelihoods[-1])
        learned_identities.append("essentia")
        weights.append(float(fusion["reliability"]["essentia"]))

    beat_this_bpm = float(evidence.beat_this_bpm) if evidence.beat_this_bpm is not None else None
    if beat_this_bpm is not None and beat_this_bpm > 0:
        beat_sigma = float(sigma["beat_this_base"]) + float(sigma["beat_this_spread"]) * (1.0 - clamp01(evidence.beat_this_stability))
        beat = log_gaussian(grid, beat_this_bpm, beat_sigma)
        likelihoods.append(beat)
        voter_likelihoods.append(likelihoods[-1])
        learned_identities.append("beat_this")
        weights.append(float(fusion["reliability"]["beat_this"]))

    if not likelihoods:
        raise ValueError("tempo scoring requires at least one voter")
    active_weight = sum(weights)
    expected_weight = sum(float(value) for value in fusion["reliability"].values())
    background_config = config["fusion"]["background"]
    shortness = float(background_config["k_short"]) * max(
        0.0,
        (float(background_config["min_full_evidence_s"]) - active_duration_s)
        / float(background_config["min_full_evidence_s"]),
    )
    background_weight = max(0.0, expected_weight - active_weight) + shortness
    circle_bins = int(config["fusion"].get("learned", {}).get("tempo_loss", {}).get("bins", 72))
    circle_likelihoods = [tempo_circle_likelihood(grid, likelihood, circle_bins) for likelihood in likelihoods]
    learned_config = config["fusion"].get("learned", {})
    learned = load_if_available(str(learned_config.get("params_path", "")))
    learned_weights: dict[str, float] | None = None
    if learned is None:
        pooled = log_linear_pool(likelihoods, weights)
    else:
        pooled, learned_weights = learned.pool(
            "tempo", likelihoods, learned_identities, float(len(likelihoods)),
            (clamp01(evidence.pulse_clarity), 1.0 - clamp01(evidence.activation_flatness)),
            str(learned_config.get("gate", "attention")), circle_likelihoods,
        )
    unadjusted_posterior = temper_posterior(pooled, active_weight, background_weight)
    posterior = _normalize(
        unadjusted_posterior
        * _bar_snap_prior(grid, active_duration_s, config["tempo"]["bar_snap"])
        ** float(fusion["snap_eta"])
    )
    lobes = extract_lobes(grid, posterior, float(fusion["lobe_separation"]))
    lobes = centroid_lobes(lobes, grid, unadjusted_posterior)
    if not lobes:
        raise ValueError("tempo posterior contains no lobes")
    relation_tolerance = float(config["tempo"]["relation_tolerance"])
    candidate_support_ratio = 0.0
    if valid_hypotheses:
        first_bpm = lobes[0].bpm
        base_support = sum(probability for bpm, probability in valid_hypotheses if abs(bpm - first_bpm) / first_bpm <= relation_tolerance)
        alternatives = [value for value in (first_bpm * 2.0, first_bpm / 2.0) if grid[0] <= value <= grid[-1]]
        if alternatives:
            target = max(
                alternatives,
                key=lambda value: sum(probability for bpm, probability in valid_hypotheses if abs(bpm - value) / value <= relation_tolerance),
            )
            alternate_support = sum(
                probability for bpm, probability in valid_hypotheses
                if abs(bpm - target) / target <= relation_tolerance
            )
            alternate_center = (
                sum(
                    bpm * probability for bpm, probability in valid_hypotheses
                    if abs(bpm - target) / target <= relation_tolerance
                ) / max(alternate_support, 1e-12)
            )
            alternate_hypotheses = sum(
                abs(bpm - target) / target <= relation_tolerance
                for bpm, _ in valid_hypotheses
            )
            candidate_support_ratio = (
                alternate_support / max(base_support, 1e-12)
                if alternate_hypotheses >= 2 else 0.0
            )
            if candidate_support_ratio >= float(fusion["octave_decision"]["mass_ratio"]):
                sibling_index = next(
                    (index for index, lobe in enumerate(lobes[1:], 1)
                     if tempo_relation(lobes[0].bpm, lobe.bpm, relation_tolerance)[0] in {"2:1", "1:2"}),
                    None,
                )
                if sibling_index is None:
                    peak_index = int(np.argmin(np.abs(grid - target)))
                    low = target * (1.0 - float(fusion["lobe_separation"]))
                    high = target * (1.0 + float(fusion["lobe_separation"]))
                    mass = float(posterior[(grid >= low) & (grid <= high)].sum())
                    sibling = TempoLobe(float(grid[peak_index]), mass, low, high)
                else:
                    sibling = lobes.pop(sibling_index)
                sibling = TempoLobe(
                    float(alternate_center), sibling.mass, sibling.low, sibling.high
                )
                lobes.insert(1, sibling)
    lobes, octave_margin, octave_ambiguous, octave_details = octave_decision(
        lobes, evidence, active_duration_s, fusion["octave_decision"],
        config["tempo"]["bar_snap"], relation_tolerance, candidate_support_ratio,
    )
    supports = metrical_support(lobes, fusion["folding"], relation_tolerance)

    first = lobes[0]
    precise_bpm = reported_bpm(first, float(evidence.essentia_bpm), config["reporting"])
    report_count = int(config["report"]["top_k"])
    top_k = [
        {"bpm": precise_bpm if index == 0 else reported_bpm(lobe, float(evidence.essentia_bpm), config["reporting"]), "p": float(lobe.mass)}
        for index, lobe in enumerate(lobes[:report_count])
    ]
    m1 = float(first.mass)
    m2 = float(lobes[1].mass) if len(lobes) > 1 else 0.0
    related = len(lobes) > 1 and tempo_relation(first.bpm, lobes[1].bpm, relation_tolerance)[0] is not None

    activation = 1.0 - clamp01(evidence.activation_flatness)
    axis = config["axes"]["rhythmicity"]
    corroborating_voters = sum(
        first.low <= float(grid[int(np.argmax(likelihood))]) <= first.high
        for likelihood in voter_likelihoods
    )
    concentration = clamp01(m1) if corroborating_voters >= 2 else 0.0
    axis_values = {
        "pulse_clarity": clamp01(evidence.pulse_clarity),
        "activation": activation,
        "stability": clamp01(evidence.beat_this_stability),
        "concentration": concentration,
    }
    rhythmicity_score = clamp01(sum(float(axis["weights"][name]) * value for name, value in axis_values.items()))
    if corroborating_voters < 2:
        rhythmicity_score = clamp01(rhythmicity_score + float(axis["weights"]["concentration"]) * axis_values["pulse_clarity"])
    flags = list(evidence.flags)
    if beat_this_bpm is None:
        flags.append("BEAT_TRACKING_SPARSE")
    if evidence.pulse_clarity < 0.2:
        flags.append("LOW_PULSE_CLARITY")
    if m2 >= 0.25 and not related:
        flags.append("TEMPO_MODEL_DISAGREEMENT")
    if octave_ambiguous:
        flags.append("TEMPO_OCTAVE_AMBIGUOUS")

    if rhythmicity_score < float(axis["arhythmic"]):
        status = "tempoless"
        bpm: float | None = None
    else:
        status = _tempo_status(m1, m2, related, fusion["detect"])
        if len(likelihoods) < 2 or octave_ambiguous:
            status = "review"
        bpm = precise_bpm
        if status == "review":
            flags.append("TEMPO_LOW_CONFIDENCE")

    voter_bpms = [bpm for bpm in (evidence.essentia_bpm, beat_this_bpm, valid_hypotheses[0][0] if valid_hypotheses else None) if bpm]
    relations = [tempo_relation(a, b, float(config["tempo"]["relation_tolerance"]))[1] for a, b in combinations(voter_bpms, 2)]
    octagree = sum(relations) / len(relations) if relations else 1.0
    signals: dict[str, Any] = {
        "tempocnn": {"hypotheses": [[bpm, probability] for bpm, probability in valid_hypotheses], "peakedness": clamp01(evidence.tempocnn_peakedness)},
        "beat_this": {"bpm": beat_this_bpm, "stability": clamp01(evidence.beat_this_stability), "n_beats": int(evidence.beat_this_n_beats)},
        "essentia": {"bpm": float(evidence.essentia_bpm), "confidence_raw": float(evidence.essentia_confidence), "confidence_norm": essentia_norm},
        "posterior_lobes": [
            {"bpm": float(lobe.bpm), "p": float(lobe.mass), "metrical_support": supports[index]}
            for index, lobe in enumerate(lobes)
        ],
        "octave_decision": {"margin": octave_margin, "candidates": octave_details},
        "octave_candidate_support_ratio": candidate_support_ratio,
        "n_effective_voters": float(len(likelihoods)),
        "background_weight": background_weight,
        "corroborating_voters": int(corroborating_voters),
        "learned_fusion": {"active": learned is not None, "weights": learned_weights},
        "posterior_entropy": float(-np.sum(posterior * np.log(np.maximum(posterior, 1e-15))) / math.log(posterior.size)),
        "rhythmicity": {**axis_values, "score": rhythmicity_score},
        "pulse_clarity": axis_values["pulse_clarity"],
        "activation_flatness": clamp01(evidence.activation_flatness),
    }
    for identity, circle in zip(learned_identities, circle_likelihoods):
        signals[identity]["circle_likelihood"] = [float(value) for value in circle]
    if emit_legacy_confidence:
        signals["legacy_confidence"] = _legacy_tempo_confidence(evidence, octagree, essentia_norm)
    return FieldResult(status, bpm, m1, top_k, signals, list(dict.fromkeys(flags)))
