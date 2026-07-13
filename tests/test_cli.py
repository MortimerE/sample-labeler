import json
from pathlib import Path

from click.testing import CliRunner

import autolabel.cli as cli_module
from autolabel.backends import BackendUnavailable
from autolabel.report import render_report


class DummyRecord:
    def model_dump_json(self, indent=None):
        return json.dumps({"schema_version": "1.2", "ok": True}, indent=indent)


def test_batch_isolates_file_failures_and_succeeds_if_any_file_succeeds(monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem():
        inputs = Path("inputs")
        inputs.mkdir()
        (inputs / "good.wav").touch()
        (inputs / "bad.mp3").touch()

        def fake_analyze(path, config_path, emit_legacy_confidence=False):
            if Path(path).stem == "bad":
                raise BackendUnavailable("model failed")
            return DummyRecord()

        monkeypatch.setattr(cli_module, "analyze_file", fake_analyze)
        result = runner.invoke(cli_module.cli, ["batch", "inputs", "--out-dir", "results"])
        assert result.exit_code == 0
        assert Path("results/good.json").is_file()
        error = json.loads(Path("results/bad._error.json").read_text())
        assert error["error"]["stage"] == "backend"


def test_batch_exits_nonzero_only_when_every_file_fails(monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("inputs").mkdir()
        Path("inputs/bad.wav").touch()
        monkeypatch.setattr(cli_module, "analyze_file", lambda *args, **kwargs: (_ for _ in ()).throw(BackendUnavailable("nope")))
        result = runner.invoke(cli_module.cli, ["batch", "inputs", "--out-dir", "results"])
        assert result.exit_code != 0
        assert Path("results/bad._error.json").is_file()


def test_report_renders_results_and_error_stubs(tmp_path):
    payload = {
        "file": {"path": "/inputs/bass.wav"},
        "key": {"status": "detected", "top_k": [{"tonic": "Ab", "mode": "minor", "p": .72}], "flags": []},
        "tempo": {"status": "review", "top_k": [{"bpm": 174, "p": .82}], "flags": ["TEMPO_LOW_CONFIDENCE"]},
        "flags": [],
    }
    (tmp_path / "bass.json").write_text(json.dumps(payload))
    (tmp_path / "broken._error.json").write_text(json.dumps({"error": {"stage": "decode", "message": "bad data"}}))
    report = render_report(tmp_path)
    assert "Abm 0.72" in report
    assert "174.00 0.82" in report
    assert "error:decode" in report
