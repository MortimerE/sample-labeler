import csv

import pytest

from train.data import (
    Augmentation, LabelRow, assigned_split, dataset_hash, read_index,
    resolved_source_splits, training_split, transform_labels,
)


def test_transposition_and_stretch_rotate_labels():
    row = LabelRow("a.wav", "source-a", 9 + 12, 100.0, False, False)
    key, tempo = transform_labels(row, Augmentation(semitones=4, stretch=1.1), 72)
    assert key == 1 + 12
    assert tempo is not None


def test_explicit_source_leakage_fails(tmp_path):
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "source_id", "key", "bpm", "split"))
        writer.writeheader()
        writer.writerow({"path": "a.wav", "source_id": "same", "key": "C", "bpm": 120, "split": "train"})
        writer.writerow({"path": "b.wav", "source_id": "same", "key": "C", "bpm": 120, "split": "val"})
    with pytest.raises(ValueError, match="source leakage"):
        read_index(path)


def test_assigned_split_is_stable_by_source():
    assert assigned_split("source-a", seed=7) == assigned_split("source-a", seed=7)


def test_explicit_test_split_is_excluded_from_training():
    row = LabelRow("a.wav", "source-a", 0, 120.0, False, False, split="test")
    assert training_split(row, seed=7) is None


def test_blank_split_inherits_explicit_split_for_same_source():
    rows = [
        LabelRow("a.wav", "source-a", 0, 120.0, False, False, split="train"),
        LabelRow("b.wav", "source-a", 0, 120.0, False, False),
    ]
    splits = resolved_source_splits(rows, seed=7)
    assert training_split(rows[1], resolved=splits) == "train"


def test_dataset_hash_tracks_same_size_audio_content(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    index = tmp_path / "labels.csv"
    index.write_text("path,source_id,key,bpm\na.wav,a,C,120\n")
    audio = samples / "a.wav"
    audio.write_bytes(b"first")
    first = dataset_hash(index, samples, [Augmentation()], {})
    audio.write_bytes(b"other")
    second = dataset_hash(index, samples, [Augmentation()], {})
    assert first != second
