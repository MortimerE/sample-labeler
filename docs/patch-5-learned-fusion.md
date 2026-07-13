# Patch 5 — Equivariant Learned Fusion and Augmented Training

## 1. Scope and invariants

Patch 5 learns how much to trust existing key/tempo voters; it does not replace the DSP/model voters.

1. If `fusion.learned.params_path` is absent, inference is exactly Patch 4 hand fusion.
2. Production code imports NumPy only. Torch and torchaudio live under `train/`.
3. Key fusion is exactly transposition equivariant. Gate inputs are invariant; likelihood values carry orientation.
4. Tempo learning is octave-free on a 72-bin log-BPM circle. Patch 4 retains octave placement and fine BPM reporting.
5. Parameter artifacts ship only after source-isolated validation beats the warm baseline without worse key ECE.

Temporal attention is deferred. The bass/downbeat voter already captures the highest-value temporal prior.

## 2. Dataset contract and leakage prevention

The index is CSV with:

```text
path,source_id,key,bpm,atonal,tempoless,split
```

- `path` is relative to `--samples`.
- `source_id` identifies the original recording before crops/encodes/augmentations.
- `key` accepts the analyzer's normal key syntax; blank is allowed only for `atonal=true`.
- `bpm` is positive; blank is allowed only for `tempoless=true`.
- optional `split` is `train|val|test`. Without it, SHA-256 of `(seed, source_id)` assigns the split.
- one source ID appearing in multiple explicit splits is a hard error.

At least 150 distinct sources are required for a shippable run. `--allow-small-dataset` exists only for smoke tests.

## 3. Augmentation and learning surface

Default augmentation is the Cartesian product of:

- 12 root rotations (0–11 semitones), with key pitch class rotated identically;
- forward/reversed audio (labels unchanged);
- time stretch 0.9/1.0/1.1, with BPM multiplied by the stretch rate.

Pitch shift preserves duration; phase-vocoder stretch preserves pitch. Every transformed file is analyzed by the same production voters and cached by dataset/augmentation hash.

Each voter token contains invariant scalars: normalized entropy, p1, margin, a compact invariant-DFT embedding (six magnitudes), effective-voter context, and two field materiality signals. Learned voter identity is added after projection. Raw normalized log-likelihoods are values, never gate features. Larger clip embeddings such as EffNet are reserved until their artifacts and transposition-invariance audit can ship together.

The default gate is one two-head attention block (`d_model=48`) plus a scalar reliability projection. The `mlp` ablation uses the same token projection without Q/K/V/O. Missing voters are masked; training voter dropout is 0.15 and always preserves one token.

Two output heads are produced:

- key: 24-way `(12 pitch classes × major/minor)` posterior;
- tempo: 72 circular bins over one log2 octave.

The runtime applies learned tempo reliability weights to the fine DSP grid, preserving Patch 4 precision.

## 4. Heads and losses

### Key

Use softmax cross-entropy against harmonic soft targets:

```text
q(k') ∝ exp(η S(k_true,k')), η=4
```

The fixed similarity objective combines circle-of-fifths distance (0.55), seven-note scale overlap (0.30), and tonic/fifth relation (0.15). Exact is 1.0 and relative major/minor is explicitly 0.93. A small expected-cost term (0.1) optimizes the same metric without replacing calibrated CE.

### Tempo

Fold `log2(BPM/60) mod 1` into 72 bins. Targets are circular Gaussians (`σ=1` bin), including wraparound, plus 0.15 secondary bumps at `±log2(3/2)`. Octave placement remains policy.

### Total

```text
L = CE_soft(key) + CE_circular(tempo) + 0.1 expected_key_cost
```

Materiality stays under the proven Patch 4 axes for this two-head patch. Independent BCE materiality heads are reserved until the index includes enough atonal/tempoless sources.

## 5. Training and export

```bash
python -m train.train_fusion \
  --samples /data/samples \
  --index /data/labels.csv \
  --out artifacts/fusion_params.npz
```

Training uses AdamW, early stopping, source-isolated validation, and deterministic seed 7. The initial model is evaluated as the warm baseline. Export is refused if learned weighted-key accuracy or octave-folded tempo accuracy is worse, or key ECE is higher.

`train/export_params.py` writes plain arrays plus a JSON manifest containing schema, Git SHA, dataset hash, validation metrics, configuration, and attention-head count. Runtime activation is entirely artifact-driven.

## 6. Acceptance

- NumPy/Torch forward parity `<1e-5`.
- Exact key equivariance under all 12 rotations.
- Attention weights invariant under rotations to `1e-6`.
- Circular tempo smoothing wraps bin 71↔0 and normalizes.
- Missing-voter masking is finite and normalized.
- Explicit source leakage fails tests.
- No-parameter fallback matches Patch 4 outputs.
- Learned artifacts cannot export unless both task metrics meet baseline and ECE does not regress.
