# InBloom Sample Labeler

Milestone 1 is a single-file analysis core that combines independent musical-key
and tempo estimates, retains all decision signals, and abstains when evidence is
weak. Its output is a validated, versioned JSON record; atonal, tempoless, and
review outcomes are successful analyses rather than process errors.

## What is implemented

- shared decode, mono downmix, resampling, peak normalization, SHA-1, and
  active-duration tail trimming;
- concurrent key and tempo analyzer interfaces;
- exact/relative/fifth key agreement, symmetric relative-key margin adjustment,
  majority/chroma mode resolution, Camelot rendering, and optional dual output;
- metrical tempo reconciliation, confidence-gated bar-snap prior, pulse clarity,
  and activation-flatness evidence;
- configurable weighted confidence and calibrated abstention decisions;
- strict schema 1.0 validation and the requested CLI shape;
- unit tests for scoring invariants plus end-to-end tests with deterministic
  detector doubles.

## Install and run

Python 3.11 is the deployment target. Install the lightweight core and tests:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

Run a production analysis after installing the native/model backends:

```bash
autolabel analyze sample.wav --pretty
autolabel analyze sample.wav --config my-config.yaml --out result.json
```

The CLI exits zero for `detected`, `review`, `atonal`, and `tempoless`; decode,
backend, configuration, and schema failures are non-zero.

## Backend packaging note

The scoring pipeline uses a narrow `DetectorSuite` interface, keeping native
dependencies out of unit tests. `ProductionDetectors` lazily integrates Essentia,
madmom, libkeyfinder, and S-KEY. The upstream S-KEY repository currently says its
PyPI release is forthcoming, so `requirements-analysis.txt` deliberately does not
pretend an unpublished wheel exists. Deployment must pin a reviewed S-KEY source
revision and the project-owned libkeyfinder binding that exposes the top two
candidates. The Docker build fails at this boundary until those two immutable
artifacts are supplied; it never silently drops an ensemble voter.

This is important for calibration: changing a detector build changes the evidence
distribution and invalidates tuned confidence thresholds.

## Configuration

Defaults are in [`src/autolabel/default.yaml`](src/autolabel/default.yaml). A supplied YAML file
is deep-merged over the defaults. Weight groups must sum to 1.0 and decision
thresholds must be ordered.

Every signal used by scoring is written to the record, allowing threshold and
weight searches to run against saved results without decoding audio again.

## Calibration gate

Before calling M1 production-ready, collect 60–100 hand-labeled samples across
melodic loops, vocals, one-shots, drums, and atmospheres. Tune on a train split,
freeze configuration, then report the four-way confusion matrix and acceptance
criteria on a held-out split. Synthetic correctness tests do not substitute for
that calibration set.

The production image additionally requires immutable wheel inputs:

```bash
docker build \
  --build-arg LIBKEYFINDER_WHEEL=artifacts/libkeyfinder.whl \
  --build-arg SKEY_WHEEL=artifacts/skey.whl \
  .
```

For the default Docker Compose path, the image is built and run as `linux/amd64` so
`pip` can install the published manylinux wheels for Essentia and the other
binary dependencies without manual intervention.
