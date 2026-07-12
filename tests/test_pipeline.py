import json

import numpy as np
import soundfile as sf

from autolabel.domain import Key
from autolabel.pipeline import analyze_file
from autolabel.scoring import KeyVote, TempoEvidence


class FakeDetectors:
    def key_votes(self, audio):
        probs = np.full(24, 0.005)
        probs[19] = 0.885
        return [
            KeyVote("libkeyfinder", Key(7, "minor"), margin=0.4, runner_up=Key(10, "major")),
            KeyVote("essentia", Key(7, "minor"), strength=0.9, margin=0.3, runner_up=Key(10, "major")),
            KeyVote("skey", Key(7, "minor"), probabilities=tuple(probs)),
        ]

    def tempo_evidence(self, audio):
        return TempoEvidence(((120, 0.8), (60, 0.2)), 120, 4.8, 0.9, 0.1)

    def versions(self):
        return {"fake": "1"}


def test_end_to_end_record_validates_and_serializes(tmp_path):
    path = tmp_path / "tone.wav"
    time = np.arange(44100 * 2) / 44100
    sf.write(path, 0.5 * np.sin(2 * np.pi * 196 * time), 44100, subtype="FLOAT")
    record = analyze_file(path, detectors=FakeDetectors())
    payload = json.loads(record.model_dump_json())
    assert payload["schema_version"] == "1.0"
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

