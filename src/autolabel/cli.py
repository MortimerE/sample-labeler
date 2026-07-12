from __future__ import annotations

from pathlib import Path

import click

from .backends import BackendUnavailable
from .pipeline import analyze_file
from .preprocess import DecodeError


@click.group()
def cli() -> None:
    """Analyze audio samples with calibrated abstention."""


@cli.command("analyze")
@click.argument("audio_file", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--config", "config_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--out", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--pretty", is_flag=True, help="Indent JSON output.")
def analyze_command(audio_file: Path, config_path: Path | None, out: Path | None, pretty: bool) -> None:
    """Analyze one AUDIO_FILE and emit a schema 1.0 JSON record."""
    try:
        record = analyze_file(audio_file, config_path)
        payload = record.model_dump_json(indent=2 if pretty else None)
        if out:
            out.write_text(payload + "\n", encoding="utf-8")
        else:
            click.echo(payload)
    except (DecodeError, BackendUnavailable, ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    cli()

