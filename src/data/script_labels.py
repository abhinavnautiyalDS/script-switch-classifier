"""Unicode-based dominant-script labels for Devanagari vs Latin."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScriptCounts:
    devanagari: int
    latin: int
    other_letters: int
    label: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def count_scripts(text: str) -> ScriptCounts:
    dev = latin = other = 0
    for char in text:
        if not char.isalpha():
            continue
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            dev += 1
        elif ("A" <= char <= "Z") or ("a" <= char <= "z"):
            latin += 1
        else:
            other += 1

    if dev == 0 and latin == 0:
        label = "OUT_OF_SCOPE" if other else "NO_LETTERS"
    elif dev > latin:
        label = "DEVANAGARI"
    elif latin > dev:
        label = "LATIN"
    else:
        label = "TIE"
    return ScriptCounts(devanagari=dev, latin=latin, other_letters=other, label=label)


def label_text(text: str) -> str:
    return count_scripts(text).label
