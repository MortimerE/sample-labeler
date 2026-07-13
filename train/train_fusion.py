from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autolabel.config import load_config
from train.data import (
    analyze_augmented, augmentation_grid, dataset_hash, example_from_record, read_index,
    resolved_source_splits, training_split, transform_labels,
)
from train.export_params import export_params
from train.losses import semantic_fusion_loss, similarity_matrix
from train.model import FusionModel


class EvidenceDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}
        for name, value in self.examples[index].items():
            if name in {"key_index", "tempo_index"}:
                result[name] = torch.tensor(value, dtype=torch.long)
            elif name.endswith("_mask"):
                result[name] = torch.as_tensor(value, dtype=torch.bool)
            elif name.endswith("_identities"):
                result[name] = torch.as_tensor(value, dtype=torch.long)
            else:
                result[name] = torch.as_tensor(value, dtype=torch.float32)
        return result


def _drop_voters(batch: dict[str, torch.Tensor], probability: float) -> None:
    if probability <= 0:
        return
    for field in ("key", "tempo"):
        mask = batch[f"{field}_mask"]
        dropped = mask & (torch.rand_like(mask, dtype=torch.float32) < probability)
        candidate = mask & ~dropped
        empty = ~candidate.any(dim=1)
        if empty.any():
            first = mask.to(torch.int64).argmax(dim=1)
            candidate[empty, first[empty]] = True
        batch[f"{field}_mask"] = candidate


def _metrics(
    model: FusionModel, loader: DataLoader, similarity: torch.Tensor, config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    key_loss = config["fusion"]["learned"]["key_loss"]
    tempo_loss = config["fusion"]["learned"]["tempo_loss"]
    key_score = []
    tempo_correct = []
    confidences = []
    correctness = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            loss, _ = semantic_fusion_loss(
                output, batch["key_index"], batch["tempo_index"], similarity,
                eta=float(key_loss["eta"]),
                expected_cost_weight=float(key_loss["expected_cost"]),
                sigma_bins=float(tempo_loss["sigma_bins"]),
                three_two_weight=float(tempo_loss["three_two_weight"]),
            )
            losses.append(float(loss))
            key_probability = torch.softmax(output["key_logits"], dim=-1)
            key_prediction = key_probability.argmax(dim=-1)
            key_score.extend(similarity[batch["key_index"], key_prediction].tolist())
            tempo_prediction = output["tempo_logits"].argmax(dim=-1)
            circular = torch.abs(tempo_prediction - batch["tempo_index"])
            circular = torch.minimum(circular, output["tempo_logits"].shape[-1] - circular)
            tempo_correct.extend((circular <= 1).to(torch.float32).tolist())
            confidence, prediction = key_probability.max(dim=-1)
            confidences.extend(confidence.tolist())
            correctness.extend((prediction == batch["key_index"]).to(torch.float32).tolist())
    ece = 0.0
    confidence_array = np.asarray(confidences)
    correctness_array = np.asarray(correctness)
    for low in np.linspace(0.0, 0.9, 10):
        selected = (confidence_array >= low) & (confidence_array < low + 0.1)
        if selected.any():
            ece += selected.mean() * abs(confidence_array[selected].mean() - correctness_array[selected].mean())
    return {
        "loss": float(np.mean(losses)),
        "weighted_key_accuracy": float(np.mean(key_score)),
        "tempo_circle_accuracy": float(np.mean(tempo_correct)),
        "key_ece": float(ece),
    }


def _build_examples(args: argparse.Namespace, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    rows = read_index(args.index)
    unique_sources = {row.source_id for row in rows}
    if len(unique_sources) < 150 and not args.allow_small_dataset:
        raise ValueError(
            f"training requires at least 150 source IDs; found {len(unique_sources)}. "
            "Use --allow-small-dataset only for pipeline smoke tests."
        )
    transpositions = range(12) if args.transpositions == "all" else [int(value) for value in args.transpositions.split(",")]
    stretches = [float(value) for value in args.stretches.split(",")]
    reversals = (False, True) if args.reverse else (False,)
    augmentations = augmentation_grid(transpositions, reversals, stretches)
    if args.max_augmentations:
        augmentations = augmentations[: args.max_augmentations]
    evidence_hash = dataset_hash(args.index, args.samples, augmentations, config)
    cache_path = args.cache / f"{evidence_hash}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=True) as archive:
            train = list(archive["train"])
            val = list(archive["val"])
        return train, val, evidence_hash
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    tempo_bins = int(config["fusion"]["learned"]["tempo_loss"]["bins"])
    splits = resolved_source_splits(rows, int(config["fusion"]["learned"]["train"]["seed"]))
    for row in rows:
        split = training_split(row, resolved=splits)
        if split is None:
            continue
        destination = val if split == "val" else train
        source = args.samples / row.path
        if not source.is_file():
            raise FileNotFoundError(source)
        for augmentation in augmentations:
            key_index, tempo_index = transform_labels(row, augmentation, tempo_bins)
            if key_index is None or tempo_index is None:
                continue  # materiality-only rows are reserved until the binary heads ship
            payload = analyze_augmented(source, augmentation, args.config)
            destination.append(example_from_record(payload, key_index, tempo_index))
    if not train or not val:
        raise ValueError("source split must produce at least one train and one validation example")
    args.cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, train=np.asarray(train, dtype=object), val=np.asarray(val, dtype=object))
    return train, val, evidence_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Train equivariant key/tempo voter fusion from labeled audio.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="CSV: path,source_id,key,bpm[,atonal,tempoless,split]")
    parser.add_argument("--out", type=Path, default=Path("artifacts/fusion_params.npz"))
    parser.add_argument("--cache", type=Path, default=Path(".fusion-cache"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--transpositions", default="all", help="all or comma-separated semitone shifts")
    parser.add_argument("--stretches", default="0.9,1.0,1.1")
    parser.add_argument("--reverse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-augmentations", type=int)
    parser.add_argument("--allow-small-dataset", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    training = config["fusion"]["learned"]["train"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_examples, val_examples, evidence_hash = _build_examples(args, config)
    train_loader = DataLoader(EvidenceDataset(train_examples), batch_size=int(training["batch"]), shuffle=True)
    val_loader = DataLoader(EvidenceDataset(val_examples), batch_size=int(training["batch"]), shuffle=False)
    model = FusionModel(gate=str(config["fusion"]["learned"]["gate"]))
    key_loss = config["fusion"]["learned"]["key_loss"]
    similarity = torch.as_tensor(similarity_matrix(
        ring_weight=float(key_loss["ring"]),
        overlap_weight=float(key_loss["overlap"]),
        root_weight=float(key_loss["root"]),
    ), dtype=torch.float32)
    baseline = _metrics(model, val_loader, similarity, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["lr"]), weight_decay=float(training["weight_decay"])
    )
    best_state = None
    best_loss = float("inf")
    patience = 0
    for epoch in range(int(training["epochs"])):
        model.train()
        for batch in train_loader:
            _drop_voters(batch, float(training["voter_dropout"]))
            optimizer.zero_grad()
            output = model(batch)
            loss, _ = semantic_fusion_loss(
                output, batch["key_index"], batch["tempo_index"], similarity,
                eta=float(config["fusion"]["learned"]["key_loss"]["eta"]),
                expected_cost_weight=float(config["fusion"]["learned"]["key_loss"]["expected_cost"]),
                sigma_bins=float(config["fusion"]["learned"]["tempo_loss"]["sigma_bins"]),
                three_two_weight=float(config["fusion"]["learned"]["tempo_loss"]["three_two_weight"]),
            )
            loss.backward()
            optimizer.step()
        metrics = _metrics(model, val_loader, similarity, config)
        print(json.dumps({"epoch": epoch + 1, **metrics}))
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(training["early_stop_patience"]):
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    learned = _metrics(model, val_loader, similarity, config)
    if (
        learned["weighted_key_accuracy"] < baseline["weighted_key_accuracy"]
        or learned["tempo_circle_accuracy"] < baseline["tempo_circle_accuracy"]
        or learned["key_ece"] > baseline["key_ece"]
    ):
        raise RuntimeError(f"validation gate failed: baseline={baseline}, learned={learned}")
    export_params(
        model, args.out, evidence_hash, {f"baseline_{k}": v for k, v in baseline.items()} | learned,
        config["fusion"]["learned"],
    )


if __name__ == "__main__":
    main()
