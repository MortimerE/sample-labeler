from __future__ import annotations

import json
from pathlib import Path

import click

from .backends import BackendUnavailable
from .pipeline import analyze_file
from .preprocess import DecodeError
from .report import render_report

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a"}
_ANALYSIS_ERRORS = (DecodeError, BackendUnavailable, ValueError, OSError)


@click.group()
def cli() -> None:
    """Analyze audio samples with posterior-fused key and tempo models."""


@cli.command("analyze")
@click.argument("audio_file", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--config", "config_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--out", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--pretty", is_flag=True, help="Indent JSON output.")
@click.option("--emit-legacy-confidence", is_flag=True, help="Include the schema 1.1 composite for comparison.")
def analyze_command(
    audio_file: Path,
    config_path: Path | None,
    out: Path | None,
    pretty: bool,
    emit_legacy_confidence: bool,
) -> None:
    """Analyze one AUDIO_FILE and emit a schema 1.2 JSON record."""
    try:
        record = analyze_file(audio_file, config_path, emit_legacy_confidence=emit_legacy_confidence)
        payload = record.model_dump_json(indent=2 if pretty else None)
        if out:
            out.write_text(payload + "\n", encoding="utf-8")
        else:
            click.echo(payload)
    except _ANALYSIS_ERRORS as error:
        raise click.ClickException(str(error)) from error


@cli.command("batch")
@click.argument("input_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--out-dir", required=True, type=click.Path(path_type=Path, file_okay=False))
@click.option("--config", "config_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--emit-legacy-confidence", is_flag=True)
def batch_command(
    input_dir: Path,
    out_dir: Path,
    config_path: Path | None,
    emit_legacy_confidence: bool,
) -> None:
    """Analyze every supported audio file in INPUT_DIR with per-file isolation."""
    audio_files = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )
    if not audio_files:
        raise click.ClickException(f"no supported audio files found in {input_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    for audio_file in audio_files:
        try:
            record = analyze_file(audio_file, config_path, emit_legacy_confidence=emit_legacy_confidence)
            destination = out_dir / f"{audio_file.stem}.json"
            destination.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
            succeeded += 1
        except _ANALYSIS_ERRORS as error:
            stage = "decode" if isinstance(error, DecodeError) else "backend" if isinstance(error, BackendUnavailable) else "analysis"
            destination = out_dir / f"{audio_file.stem}._error.json"
            destination.write_text(
                json.dumps({"error": {"stage": stage, "message": str(error)}}, indent=2) + "\n",
                encoding="utf-8",
            )
    failed = len(audio_files) - succeeded
    click.echo(f"analyzed {succeeded}/{len(audio_files)} files; {failed} failed")
    if succeeded == 0:
        raise click.ClickException("every file failed")


@cli.command("report")
@click.argument("results_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--top-k", default=3, show_default=True, type=click.IntRange(min=1))
@click.option("--format", "output_format", type=click.Choice(("md", "csv")), default="md", show_default=True)
@click.option("--out", type=click.Path(path_type=Path, dir_okay=False))
def report_command(results_dir: Path, top_k: int, output_format: str, out: Path | None) -> None:
    """Render result JSON and error stubs as a compact table."""
    payload = render_report(results_dir, top_k, output_format)
    if out:
        out.write_text(payload, encoding="utf-8")
    else:
        click.echo(payload, nl=False)


if __name__ == "__main__":
    cli()
