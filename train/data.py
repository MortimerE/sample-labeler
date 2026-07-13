from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf

from autolabel.backends import parse_key
from autolabel.learned_fusion import KEY_VOTERS, TEMPO_VOTERS, invariant_token_features


@dataclass(frozen=True)
class LabelRow:
    path: str
    source_id: str
    key_index: int | None
    bpm: float | None
    atonal: bool
    tempoless: bool
    split: str | None = None


@dataclass(frozen=True)
class Augmentation:
    semitones: int = 0
    reverse: bool = False
    stretch: float = 1.0


def read_index(path: str | Path) -> list[LabelRow]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "source_id", "key", "bpm"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"index is missing columns: {', '.join(sorted(missing))}")
        for number, row in enumerate(reader, 2):
            atonal = str(row.get("atonal", "")).strip().lower() in {"1", "true", "yes"}
            tempoless = str(row.get("tempoless", "")).strip().lower() in {"1", "true", "yes"}
            key_text = str(row.get("key", "")).strip()
            bpm_text = str(row.get("bpm", "")).strip()
            if not atonal and not key_text:
                raise ValueError(f"row {number}: key is required unless atonal=true")
            if not tempoless and not bpm_text:
                raise ValueError(f"row {number}: bpm is required unless tempoless=true")
            key_index = None
            if key_text:
                key = parse_key(key_text)
                key_index = key.pitch_class + (12 if key.mode == "minor" else 0)
            bpm = float(bpm_text) if bpm_text else None
            if bpm is not None and bpm <= 0:
                raise ValueError(f"row {number}: bpm must be positive")
            rows.append(LabelRow(
                str(row["path"]), str(row["source_id"]), key_index, bpm, atonal, tempoless,
                str(row.get("split", "")).strip() or None,
            ))
    if not rows:
        raise ValueError("index contains no samples")
    validate_source_splits(rows)
    return rows


def assigned_split(source_id: str, seed: int = 7, validation_fraction: float = 0.2) -> str:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if fraction < validation_fraction else "train"


def resolved_source_splits(rows: Sequence[LabelRow], seed: int = 7) -> dict[str, str]:
    explicit = {row.source_id: row.split for row in rows if row.split is not None}
    return {
        row.source_id: explicit.get(row.source_id) or assigned_split(row.source_id, seed)
        for row in rows
    }


def training_split(
    row: LabelRow, seed: int = 7, resolved: dict[str, str] | None = None,
) -> str | None:
    split = (resolved or {}).get(row.source_id) or row.split or assigned_split(row.source_id, seed)
    return None if split == "test" else split


def validate_source_splits(rows: Sequence[LabelRow]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        if row.split is None:
            continue
        if row.split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split {row.split!r} for source {row.source_id}")
        previous = seen.setdefault(row.source_id, row.split)
        if previous != row.split:
            raise ValueError(
                f"source leakage: {row.source_id!r} occurs in both {previous!r} and {row.split!r}"
            )


def augmentation_grid(
    transpositions: Iterable[int] = range(12), reversals: Iterable[bool] = (False, True),
    stretches: Iterable[float] = (0.9, 1.0, 1.1),
) -> list[Augmentation]:
    result = [
        Augmentation(int(semitones), bool(reverse), float(stretch))
        for semitones in transpositions
        for reverse in reversals
        for stretch in stretches
    ]
    if any(item.stretch <= 0 for item in result):
        raise ValueError("time-stretch factors must be positive")
    return result


def dataset_hash(
    index_path: str | Path, samples: str | Path, augmentations: Sequence[Augmentation],
    config: dict[str, Any],
) -> str:
    """Hash every input that can change cached voter evidence."""
    digest = hashlib.sha256()

    def update_file(path: Path, label: str) -> None:
        digest.update(label.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

    index_path = Path(index_path)
    samples = Path(samples)
    update_file(index_path, "index")
    for path in sorted(samples.rglob("*")):
        if path.is_file() and path != index_path:
            update_file(path, f"sample:{path.relative_to(samples)}")
    digest.update(repr(tuple(augmentations)).encode())
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    repository = Path(__file__).resolve().parents[1]
    evidence_sources = sorted((repository / "src" / "autolabel").glob("*.py"))
    evidence_sources.append(repository / "src" / "autolabel" / "default.yaml")
    evidence_sources.append(Path(__file__))
    for path in evidence_sources:
        update_file(path, f"source:{path.relative_to(repository)}")
    checksum_path = Path(os.environ.get("MODEL_SHA256SUMS", ""))
    if checksum_path.is_file():
        update_file(checksum_path, "model-checksums")
    return digest.hexdigest()


def transform_labels(row: LabelRow, augmentation: Augmentation, tempo_bins: int = 72) -> tuple[int | None, int | None]:
    key_index = row.key_index
    if key_index is not None:
        key_index = (key_index % 12 + augmentation.semitones) % 12 + (12 if key_index >= 12 else 0)
    tempo_index = None
    if row.bpm is not None:
        bpm = row.bpm * augmentation.stretch
        tempo_index = int(round((math.log2(bpm / 60.0) % 1.0) * tempo_bins)) % tempo_bins
    return key_index, tempo_index


def augment_audio(samples: np.ndarray, sample_rate: int, augmentation: Augmentation) -> np.ndarray:
    """Training-only pitch/time transforms. Torchaudio is imported lazily."""
    import torch
    import torchaudio

    waveform = torch.as_tensor(np.asarray(samples, dtype=np.float32)).reshape(1, -1)
    if augmentation.semitones:
        waveform = torchaudio.functional.pitch_shift(
            waveform, sample_rate, n_steps=float(augmentation.semitones)
        )
    if augmentation.stretch != 1.0:
        n_fft = 2048
        hop = 512
        spectrogram = torch.stft(
            waveform, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), return_complex=True
        )
        phase_advance = torch.linspace(0, math.pi * hop, spectrogram.shape[-2])[..., None]
        stretched = torchaudio.functional.phase_vocoder(
            spectrogram, rate=float(augmentation.stretch), phase_advance=phase_advance
        )
        length = max(1, round(waveform.shape[-1] / augmentation.stretch))
        waveform = torch.istft(
            stretched, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), length=length
        )
    if augmentation.reverse:
        waveform = torch.flip(waveform, dims=(-1,))
    return waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _pad_field(
    signals: dict[str, Any], identities: Sequence[str], field: str, width: int,
    n_effective: float, materiality: tuple[float, float], likelihood_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    likelihoods = []
    indices = []
    for index, identity in enumerate(identities):
        data = signals.get(identity)
        if isinstance(data, dict) and data.get(likelihood_name) is not None:
            likelihoods.append(np.asarray(data[likelihood_name], dtype=np.float32))
            indices.append(index)
    if not likelihoods:
        raise ValueError(f"analysis record has no {field} voter likelihoods")
    tokens = invariant_token_features(likelihoods, field, n_effective, materiality).astype(np.float32)
    padded_tokens = np.zeros((len(identities), tokens.shape[-1]), dtype=np.float32)
    padded_logs = np.zeros((len(identities), width), dtype=np.float32)
    mask = np.zeros(len(identities), dtype=bool)
    padded_identities = np.arange(len(identities), dtype=np.int64)
    for local, identity in enumerate(indices):
        padded_tokens[identity] = tokens[local]
        likelihood = np.maximum(likelihoods[local], 1e-15)
        padded_logs[identity] = np.log(likelihood / likelihood.sum())
        mask[identity] = True
    return padded_tokens, padded_logs, padded_identities, mask


def example_from_record(payload: dict[str, Any], key_index: int, tempo_index: int) -> dict[str, np.ndarray | int]:
    key_signals = payload["key"]["signals"]
    tempo_signals = payload["tempo"]["signals"]
    key = _pad_field(
        key_signals, KEY_VOTERS, "key", 24, float(key_signals["n_effective_voters"]),
        (float(key_signals["tonality"]["tonalness_h"]), float(key_signals["harmonic_ratio"])),
        "likelihood",
    )
    tempo = _pad_field(
        tempo_signals, TEMPO_VOTERS, "tempo", 72, float(tempo_signals["n_effective_voters"]),
        (float(tempo_signals["pulse_clarity"]), 1.0 - float(tempo_signals["activation_flatness"])),
        "circle_likelihood",
    )
    return {
        "key_tokens": key[0], "key_log_likelihoods": key[1], "key_identities": key[2], "key_mask": key[3],
        "tempo_tokens": tempo[0], "tempo_log_likelihoods": tempo[1], "tempo_identities": tempo[2], "tempo_mask": tempo[3],
        "key_index": int(key_index), "tempo_index": int(tempo_index),
    }


def analyze_augmented(
    source: Path, augmentation: Augmentation, config_path: str | Path | None = None
) -> dict[str, Any]:
    samples, sample_rate = sf.read(source, always_2d=False)
    if np.asarray(samples).ndim == 2:
        samples = np.asarray(samples).mean(axis=1)
    transformed = augment_audio(np.asarray(samples), int(sample_rate), augmentation)
    with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
        sf.write(temporary.name, transformed, sample_rate, subtype="PCM_16")
        command = [os.environ.get("AUTOLABEL_COMMAND", "autolabel"), "analyze", temporary.name]
        if config_path is not None:
            command.extend(("--config", str(config_path)))
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "autolabel failed")
        return json.loads(process.stdout)
