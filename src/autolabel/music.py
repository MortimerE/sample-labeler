from __future__ import annotations

from .domain import Key

TONICS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

# pitch class -> Camelot number for major/minor
_CAMELOT_MAJOR = (8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1)
_CAMELOT_MINOR = (5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10)


def relative(key: Key) -> Key:
    if key.mode == "major":
        return Key((key.pitch_class - 3) % 12, "minor")
    return Key((key.pitch_class + 3) % 12, "major")


def relation(a: Key, b: Key) -> str | None:
    if a == b:
        return "exact"
    if relative(a) == b:
        return "relative"
    if a.mode == b.mode and (a.pitch_class - b.pitch_class) % 12 in (5, 7):
        return "fifth"
    return None


def key_dict(key: Key) -> dict[str, object]:
    number = (_CAMELOT_MAJOR if key.mode == "major" else _CAMELOT_MINOR)[key.pitch_class]
    suffix = "B" if key.mode == "major" else "A"
    return {
        "tonic": TONICS[key.pitch_class],
        "mode": key.mode,
        "pitch_class": key.pitch_class,
        "camelot": f"{number}{suffix}",
        "rendering": "single",
    }


def short_name(key: Key) -> str:
    return TONICS[key.pitch_class] + ("m" if key.mode == "minor" else "")

