import json
import types

import numpy as np
import soundfile as sf

from autolabel import backends
from autolabel.domain import Key
from autolabel.pipeline import analyze_file
from autolabel.scoring import KeyVote, TempoEvidence


class FakeDetectors:
    def key_votes(self, audio):
        probs = np.full(24, 0.005)
        probs[19] = 0.885
        return backends.KeyEvidence(
            votes=(
                KeyVote("libkeyfinder", Key(7, "minor"), runner_up=Key(10, "major")),
                KeyVote("essentia", Key(7, "minor"), strength=0.9, margin=0.3, runner_up=Key(10, "major")),
                KeyVote("skey", Key(7, "minor"), margin=0.5, runner_up=Key(10, "major"), probabilities=tuple(probs)),
            ),
            chroma=np.ones(12, dtype=float) / 12.0,
            tonalness=0.9,
        )

    def tempo_evidence(self, audio):
        return TempoEvidence(((120, 0.8), (60, 0.2)), 0.7, 120.0, 16, 0.9, 120, 4.8, 0.9, 0.1)

    def versions(self):
        return {"fake": "1"}


def test_end_to_end_record_validates_and_serializes(tmp_path):
    path = tmp_path / "tone.wav"
    time = np.arange(44100 * 2) / 44100
    sf.write(path, 0.5 * np.sin(2 * np.pi * 196 * time), 44100, subtype="FLOAT")
    record = analyze_file(path, detectors=FakeDetectors())
    payload = json.loads(record.model_dump_json())
    assert payload["schema_version"] == "1.1"
    assert payload["key"]["status"] == "detected"
    assert payload["tempo"]["status"] == "detected"
    assert payload["review_required"] is False
    assert payload["file"]["sha1"]


def test_degenerate_input_does_not_call_models(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(44100), 44100, subtype="FLOAT")
    record = analyze_file(path, detectors=FakeDetectors())
    assert record.key.status == "atonal"
    assert record.tempo.status == "tempoless"
    assert record.flags == ["SILENT_FILE"]


def test_keyfinder_temporary_file_is_pcm16(monkeypatch, tmp_path):
    captured = {}

    class FakeWindowing:
        def __init__(self, type):
            self.type = type

        def __call__(self, frame):
            return frame

    class FakeSpectrum:
        def __call__(self, frame):
            return frame

    class FakeSpectralPeaks:
        def __call__(self, spectrum):
            return np.asarray([440.0]), np.asarray([1.0])

    class FakeHPCP:
        def __init__(self, size):
            self.size = size

        def __call__(self, frequencies, magnitudes):
            return np.ones(self.size)

    class FakeKey:
        def __init__(self, profileType):
            self.profileType = profileType

        def __call__(self, hpcp):
            return "C", "major", 0.9, 0.2

    fake_essentia = types.SimpleNamespace(
        Windowing=FakeWindowing,
        Spectrum=FakeSpectrum,
        SpectralPeaks=FakeSpectralPeaks,
        HPCP=FakeHPCP,
        Key=FakeKey,
        FrameGenerator=lambda samples, frameSize, hopSize, startFromZero: [samples[:4096]],
    )

    detectors = backends.ProductionDetectors()
    monkeypatch.setattr(backends.ProductionDetectors, "_imports", lambda self: fake_essentia)
    monkeypatch.setattr(backends, "_profile_candidates", lambda hpcp: [(1.0, Key(0, "major")), (0.5, Key(7, "minor"))])
    monkeypatch.setattr(backends, "parse_key", lambda tonic, mode=None: Key(0, "major"))
    skey_probs = [0.001] * 24
    skey_probs[0] = 0.7
    skey_probs[13] = 0.2
    monkeypatch.setattr(
        backends,
        "_run",
        lambda command, backend: json.dumps(skey_probs) if backend == "S-KEY" else "C major",
    )

    def fake_write(path, samples, sample_rate, subtype):
        captured["subtype"] = subtype
        captured["sample_rate"] = sample_rate

    monkeypatch.setattr(backends.sf, "write", fake_write)

    audio = backends.AudioBuffer(np.ones(8192, dtype=np.float32), 44100, 44100, 1, 1.0, 1.0)
    evidence = detectors.key_votes(audio)

    assert captured["subtype"] == "PCM_16"
    assert len(evidence.votes) == 3
    assert evidence.votes[0].margin is None
