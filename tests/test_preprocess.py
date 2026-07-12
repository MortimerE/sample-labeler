import numpy as np
import soundfile as sf

from autolabel.preprocess import active_duration, decode


def test_active_duration_trims_tail():
    sample_rate = 1000
    samples = np.concatenate((np.ones(1000) * 0.5, np.zeros(500)))
    duration = active_duration(samples, sample_rate, -45, 100)
    assert 0.99 <= duration <= 1.1


def test_decode_downmixes_normalizes_and_preserves_source_metadata(tmp_path):
    sample_rate = 22050
    stereo = np.column_stack((np.ones(sample_rate), np.ones(sample_rate) * 0.5)).astype("float32")
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, sample_rate, subtype="FLOAT")
    config = {"sample_rate": 44100, "trim_db": -45, "trim_hysteresis_ms": 100, "min_duration_s": 0.3}
    audio, context, flags = decode(path, config)
    assert audio.channels == 2
    assert audio.source_sample_rate == sample_rate
    assert audio.sample_rate == 44100
    assert len(audio.samples) == 44100
    assert np.max(np.abs(audio.samples)) == 1
    assert context.sha1
    assert flags == []


def test_decode_marks_short_silent_file(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(100), 44100, subtype="FLOAT")
    config = {"sample_rate": 44100, "trim_db": -45, "trim_hysteresis_ms": 100, "min_duration_s": 0.3}
    _, _, flags = decode(path, config)
    assert flags == ["SILENT_FILE", "SHORT_FILE"]

