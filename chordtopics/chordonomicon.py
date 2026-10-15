"""Loader for the Chordonomicon dataset (Kantarelis et al. 2024).

Rows look like
    <intro_1> C <verse_1> F C E7 Amin C ... <chorus_1> F C F C G C ...
with no durations, metre or key. Each chord symbol therefore becomes one
"bar" token with unit duration, and the key is estimated from the chord
tones with a Krumhansl-Schmuckler profile so roots can be made relative.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .chords import Chord, pitch_class, relative_root, RELATIVE_NAMES

# Krumhansl & Kessler (1982) key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_SECTION = re.compile(r"^<([a-z_]+?)(?:_(\d+))?>$")
# Sharps are spelled "s" (Fs = F#), which collides with "sus": a lookahead
# stops the root from swallowing the s of Csus4.
_SYMBOL = re.compile(r"^([A-G](?:b|s(?!us))*)([^/]*)(?:/(.*))?$")

# Chordonomicon suffix -> (coarse quality, chord-tone intervals for key finding)
_QUALITIES: dict[str, tuple[str, tuple[int, ...]]] = {
    "": ("maj", (0, 4, 7)),
    "maj": ("maj", (0, 4, 7)),
    "5": ("maj", (0, 7)),
    "sus2": ("maj", (0, 2, 7)),
    "sus4": ("maj", (0, 5, 7)),
    "sus": ("maj", (0, 5, 7)),
    "6": ("maj", (0, 4, 7, 9)),
    "add9": ("maj", (0, 2, 4, 7)),
    "maj7": ("maj7", (0, 4, 7, 11)),
    "maj9": ("maj7", (0, 2, 4, 7, 11)),
    "min": ("min", (0, 3, 7)),
    "m": ("min", (0, 3, 7)),
    "min6": ("min", (0, 3, 7, 9)),
    "m6": ("min", (0, 3, 7, 9)),
    "min7": ("min7", (0, 3, 7, 10)),
    "m7": ("min7", (0, 3, 7, 10)),
    "min9": ("min7", (0, 2, 3, 7, 10)),
    "m9": ("min7", (0, 2, 3, 7, 10)),
    "min11": ("min7", (0, 3, 5, 7, 10)),
    "7": ("dom7", (0, 4, 7, 10)),
    "9": ("dom7", (0, 2, 4, 7, 10)),
    "11": ("dom7", (0, 4, 5, 7, 10)),
    "13": ("dom7", (0, 4, 7, 9, 10)),
    "7sus4": ("dom7", (0, 5, 7, 10)),
    "dim": ("dim", (0, 3, 6)),
    "dim7": ("dim", (0, 3, 6, 9)),
    "m7b5": ("dim", (0, 3, 6, 10)),
    "hdim7": ("dim", (0, 3, 6, 10)),
    "aug": ("aug", (0, 4, 8)),
    "minmaj7": ("minmaj7", (0, 3, 7, 11)),
    "minmaj9": ("minmaj7", (0, 2, 3, 7, 11)),
    # Chordonomicon-specific spellings
    "no3d": ("maj", (0, 7)),            # power chord
    "add11": ("maj", (0, 4, 5, 7)),
    "add13": ("maj", (0, 4, 7, 9)),
    "majs9": ("maj", (0, 3, 4, 7)),      # maj(#9)
    "maj911s": ("maj7", (0, 2, 4, 6, 7, 11)),
    "maj1311s": ("maj7", (0, 2, 4, 6, 7, 9, 11)),
    "maj7sus2": ("maj7", (0, 2, 7, 11)),
    "maj7sus4": ("maj7", (0, 5, 7, 11)),
    "maj11": ("maj7", (0, 2, 4, 5, 7, 11)),
    "maj13": ("maj7", (0, 2, 4, 7, 9, 11)),
    "minadd9": ("min", (0, 2, 3, 7)),
    "minadd11": ("min", (0, 3, 5, 7)),
    "minadd13": ("min", (0, 3, 7, 9)),
    "min13": ("min7", (0, 3, 7, 9, 10)),
    "minb9": ("min", (0, 1, 3, 7)),
    "7sus2": ("dom7", (0, 2, 7, 10)),
    "7b9": ("dom7", (0, 1, 4, 7, 10)),
    "11b9": ("dom7", (0, 1, 4, 5, 7, 10)),
    "13b9": ("dom7", (0, 1, 4, 7, 9, 10)),
    "11s": ("dom7", (0, 4, 6, 7, 10)),
    "13b": ("dom7", (0, 4, 7, 8, 10)),
    "augmaj7": ("aug", (0, 4, 8, 11)),
    "augmaj9": ("aug", (0, 2, 4, 8, 11)),
    "dim9": ("dim", (0, 2, 3, 6, 9)),
    "dimb9": ("dim", (0, 1, 3, 6, 9)),
    "dimb7": ("dim", (0, 3, 6, 10)),
    "dimadd13": ("dim", (0, 3, 6, 9)),
    "dim13b9": ("dim", (0, 1, 3, 6, 9)),
}


@dataclass
class CSong:
    id: int
    tonic: str
    mode: str
    sections: list[tuple[str, list[Chord]]]   # (label, chords)
    release_year: int | None
    decade: int | None
    main_genre: str
    genres: list[str] = field(default_factory=list)


def parse_symbol(sym: str) -> tuple[int, str, tuple[int, ...]] | None:
    m = _SYMBOL.match(sym)
    if not m:
        return None
    root, suffix, _bass = m.groups()
    q = _QUALITIES.get(suffix)
    if q is None:
        # unknown suffix: fall back on whether it reads as minor
        q = ("min", (0, 3, 7)) if suffix.startswith(("m", "min")) and not suffix.startswith("maj") else ("other", (0, 7))
    return pitch_class(root), q[0], q[1]


def estimate_key(pcs_weighted: np.ndarray) -> tuple[str, str]:
    best, best_r = None, -2.0
    for shift in range(12):
        hist = np.roll(pcs_weighted, -shift)
        for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
            if hist.std() == 0:
                r = 0.0
            else:
                r = np.corrcoef(hist, prof)[0, 1]
            if r > best_r:
                best, best_r = (RELATIVE_NAMES[shift], mode), r
    return best


def parse_row(row: dict) -> CSong | None:
    toks = row["chords"].split()
    sections: list[tuple[str, list[tuple[int, str, tuple[int, ...]]]]] = []
    label = "unlabelled"
    cur: list = []
    for t in toks:
        sm = _SECTION.match(t)
        if sm:
            if cur:
                sections.append((label, cur))
            label, cur = sm.group(1), []
            continue
        p = parse_symbol(t)
        if p is not None:
            cur.append(p)
    if cur:
        sections.append((label, cur))
    if not sections:
        return None

    hist = np.zeros(12)
    for _, chords in sections:
        for root, _q, tones in chords:
            for iv in tones:
                hist[(root + iv) % 12] += 1.0
    tonic, mode = estimate_key(hist)

    out = []
    for lab, chords in sections:
        out.append((lab, [Chord(Fraction(0), Fraction(1), relative_root(RELATIVE_NAMES[root], tonic), q)
                          for root, q, _ in chords]))

    year = None
    if row.get("release_date"):
        try:
            year = int(row["release_date"][:4])
        except ValueError:
            year = None
    decade = int(float(row["decade"])) if row.get("decade") else None
    genres = re.findall(r"'([^']+)'", row.get("genres", "") or "")
    return CSong(int(row["id"]), tonic, mode, out, year, decade, row.get("main_genre", "") or "", genres)


def load_corpus(csv_path: Path, limit: int | None = None, seed: int = 0,
                require_year: bool = False) -> list[CSong]:
    """Load the CSV; with `limit`, keep a uniform random subsample of songs."""
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["id", "chords", "release_date", "genres", "decade", "main_genre"],
                     dtype=str, keep_default_na=False)
    if require_year:
        df = df[df["release_date"] != ""]
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed)
    kept: list[CSong] = []
    for row in df.to_dict("records"):
        s = parse_row(row)
        if s:
            kept.append(s)
    return kept
