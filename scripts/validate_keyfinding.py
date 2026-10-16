"""Measure chord-based key estimation against the McGill Billboard tonic
annotations, at song level and at section level (82 songs change key).

    python scripts/validate_keyfinding.py --data data/McGill-Billboard --out results/billboard/keyfinding.csv

Billboard annotates the tonic pitch class only (no mode), so accuracy is
"estimated tonic == annotated tonic". The relative-major/minor confusion shows
up as a tonic error (A minor read as C major).
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chordtopics import billboard, keyfinding  # noqa: E402
from chordtopics.chords import pitch_class  # noqa: E402


def song_sections(song):
    """Per section: list of (absolute root pc, quality, duration) and the annotated tonic pc."""
    out = []
    for sec in song.sections:
        chords, tonics = [], []
        for bar in sec.bars:
            tpc = pitch_class(bar.tonic)
            tonics.append(tpc)
            for c in bar.chords:
                if c.root == "N":
                    continue
                chords.append(((pitch_class(c.root) + tpc) % 12, c.quality, float(c.dur)))
        if chords:
            out.append((chords, max(set(tonics), key=tonics.count)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    songs, _ = billboard.load_corpus(args.data, None, "coarse")
    data = [(s, song_sections(s)) for s in songs]
    data = [(s, secs) for s, secs in data if secs]

    grid = {"profile": ["krumhansl", "temperley"], "anchor_bonus": [0.0, 0.1, 0.2, 0.3],
            "section_prior": [np.inf, 0.0, 0.5, 1.0, 2.0]}
    rows = []
    for vals in itertools.product(*grid.values()):
        cfg = keyfinding.KeyConfig(**dict(zip(grid.keys(), vals)))
        song_ok = sec_ok = sec_n = 0
        mod_ok = mod_n = 0   # sections of songs that change key
        for s, secs in data:
            song_key, keys = keyfinding.estimate_song_and_sections([c for c, _ in secs], cfg)
            song_ok += song_key[0] == pitch_class(s.tonic)
            for (chords, tonic), k in zip(secs, keys):
                sec_n += 1; sec_ok += k[0] == tonic
                if s.key_changes:
                    mod_n += 1; mod_ok += k[0] == tonic
        rows.append({**dict(zip(grid.keys(), vals)), "song_tonic_acc": song_ok / len(data),
                     "section_tonic_acc": sec_ok / sec_n, "modulating_section_acc": mod_ok / mod_n})
    df = pd.DataFrame(rows).sort_values("section_tonic_acc", ascending=False)
    print(df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
