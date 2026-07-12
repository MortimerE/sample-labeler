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
- strict schema 1.1 validation and the requested CLI shape;
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
dependencies out of unit tests. `ProductionDetectors` lazily integrates Essentia
(including TempoCNN), Beat This, libkeyfinder, and S-KEY. The ML model runners are
installed into an isolated virtual environment in the image so torch-backed
dependencies do not interfere with the main analyzer environment.

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

## Reproducible model packaging

The production image contains every model needed for offline analysis. Model
downloads happen only during `docker build`, and the build verifies these pins:

- S-KEY commit: `918b83d273568d5041569bb8068843d19a335726`
- Beat This: `1.1.0`; `final0` checkpoint from
  `https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt`,
  SHA-256 `8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`
- TempoCNN graph `https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.pb`, SHA-256
  `21c328332a221695dd6e8572728c617373064df882e8f81da6d88dc3a821e3b3`
- TempoCNN metadata `https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.json`, SHA-256
  `c0c62a52aa4a05f197208133906775c1e87077a520cdec53598b67ea9d625998`

The image records all artifact hashes in `/app/artifacts/SHA256SUMS`, and output
records identify TempoCNN with a digest-qualified version. Verify a fresh image
without network access using:

```bash
docker run --rm --network=none \
  -v "$PWD/inputs:/inputs:ro" \
  inbloom-sample-labeler:latest analyze "/inputs/100 Am.wav" >/dev/null
```

The default Docker Compose path remains `linux/amd64`: the pinned
`essentia-tensorflow==2.1b6.dev1389` release publishes x86-64 Linux wheels but no
Linux aarch64 wheel. On ARM hosts Docker therefore uses emulation. Remove the
compose platform pin once the pinned Essentia dependency has Linux aarch64
wheel support; building Essentia and TensorFlow from source is outside this
milestone.
