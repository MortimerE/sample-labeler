from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


def _key_text(candidates: list[dict[str, Any]], top_k: int) -> str:
    parts = []
    for candidate in candidates[:top_k]:
        suffix = "m" if candidate["mode"] == "minor" else ""
        parts.append(f"{candidate['tonic']}{suffix} {float(candidate['p']):.2f}")
    return " · ".join(parts) or "—"


def _tempo_text(candidates: list[dict[str, Any]], top_k: int) -> str:
    return " · ".join(
        f"{float(candidate['bpm']):.2f} {float(candidate['p']):.2f}"
        for candidate in candidates[:top_k]
    ) or "—"


def report_rows(results_dir: Path, top_k: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(results_dir.glob("*.json"), key=lambda item: item.name.lower()):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            rows.append({"file": path.stem, "key": "—", "bpm": "—", "status": "error:report", "flags": str(error)})
            continue
        if "error" in payload:
            error = payload["error"]
            name = path.name.removesuffix("._error.json")
            rows.append({
                "file": name,
                "key": "—",
                "bpm": "—",
                "status": f"error:{error.get('stage', 'analysis')}",
                "flags": str(error.get("message", "unknown error")),
            })
            continue
        key = payload.get("key", {})
        tempo = payload.get("tempo", {})
        rows.append({
            "file": Path(payload.get("file", {}).get("path", path.stem)).stem,
            "key": _key_text(key.get("top_k", []), top_k),
            "bpm": _tempo_text(tempo.get("top_k", []), top_k),
            "status": f"key:{key.get('status', '?')} · tempo:{tempo.get('status', '?')}",
            "flags": ", ".join([*key.get("flags", []), *tempo.get("flags", []), *payload.get("flags", [])]) or "—",
        })
    return rows


def render_report(results_dir: Path, top_k: int = 3, output_format: str = "md") -> str:
    rows = report_rows(results_dir, top_k)
    fields = ("file", "key", "bpm", "status", "flags")
    if output_format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()
    header = "| file | key (top-k) | bpm (top-k) | status | flags |"
    separator = "|---|---|---|---|---|"
    lines = [header, separator]
    for row in rows:
        values = [row[field].replace("|", "\\|").replace("\n", " ") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"
