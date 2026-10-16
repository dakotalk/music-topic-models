"""Harmonic features of song sections, for the question: what makes a
beginning sound like a beginning, a middle like a middle, an end like an end?

Every feature is computed from the section's own harmony (and its relation
to the rest of the song); none encodes where the section sits in the song,
so a classifier on these features answers "can harmony alone tell?".
Position features are computed separately as a baseline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .chords import Chord

LABELS6 = ["intro", "verse", "prechorus", "chorus", "bridge", "outro"]
COARSE = {"intro": "beginning", "verse": "middle", "prechorus": "middle", "chorus": "middle",
          "bridge": "middle", "outro": "end"}
FLAT_SIDE = {"Bb", "Eb", "Ab", "Db", "Gb"}
SHARP_SIDE = {"D", "E", "A", "B"}
SEVENTHS = {"maj7", "min7", "dom7", "minmaj7"}


def chord_key(c: Chord) -> str:
    return "N" if c.root == "N" else f"{c.root}:{c.quality}"


def chord_weights(bars: list[list[Chord]]) -> Counter:
    """Duration-weighted chord counts for a list of bars."""
    w: Counter = Counter()
    for bar in bars:
        for c in bar:
            if c.root != "N":
                w[chord_key(c)] += float(c.dur)
    return w


def _dist(w: Counter, keys: list[str]) -> np.ndarray:
    v = np.array([w.get(k, 0.0) for k in keys], dtype=float)
    return v / v.sum() if v.sum() else v


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def loop_period(bar_tokens: list[str], max_period: int = 8) -> int:
    """Smallest p such that most bars equal the bar p earlier; 0 if none."""
    n = len(bar_tokens)
    for p in range(1, min(max_period, n - 1) + 1):
        matches = sum(bar_tokens[i] == bar_tokens[i - p] for i in range(p, n))
        if matches >= 0.75 * (n - p):
            return p
    return 0


@dataclass
class SectionFeatures:
    names: list[str]
    values: np.ndarray


FEATURE_NAMES = [
    # size and pace
    "n_bars", "chords_per_bar", "n_distinct_chords", "chord_entropy_bits",
    # tonal centre
    "share_I", "share_IV", "share_V", "share_flat_side", "share_sharp_side",
    "share_minor_chords", "share_seventh_chords", "share_no_chord",
    "starts_on_I", "ends_on_I", "ends_on_V", "first_last_same",
    # repetition
    "frac_repeated_bars", "loop_period", "is_loop",
    # relation to the rest of the song
    "jsd_vs_rest", "frac_chords_unique_to_section", "key_differs_from_song",
    "frac_bars_recurring_elsewhere", "max_jaccard_with_other_section", "n_sections_sharing_material",
]


def section_features(bars: list[list[Chord]], rest_bars: list[list[Chord]],
                     other_sections: list[list[list[Chord]]], key_differs: bool) -> np.ndarray:
    """`other_sections` are the song's other sections, unlabelled: recurrence
    features ask how much of this section's material returns elsewhere,
    never what the other sections are called."""
    nb = len(bars)
    chords = [c for b in bars for c in b]
    if nb == 0 or not any(c.root != "N" for c in chords):
        return np.full(len(FEATURE_NAMES), np.nan)
    w = chord_weights(bars)
    total = sum(w.values())
    n_no_chord = sum(float(c.dur) for c in chords if c.root == "N")
    all_dur = total + n_no_chord
    keys = sorted(set(w) | set(chord_weights(rest_bars)))
    p_sec = _dist(w, keys)
    p_rest = _dist(chord_weights(rest_bars), keys)
    probs = p_sec[p_sec > 0]
    entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0

    def share(pred):
        return sum(v for k, v in w.items() if pred(k)) / total if total else 0.0

    root = lambda k: k.split(":")[0]
    qual = lambda k: k.split(":")[1]
    sounding = [c for c in chords if c.root != "N"]
    first, last = sounding[0], sounding[-1]
    bar_tokens = ["---".join(chord_key(c) for c in b) for b in bars]
    repeated = sum(bar_tokens[i] == bar_tokens[i - 1] for i in range(1, nb)) / (nb - 1) if nb > 1 else 0.0
    period = loop_period(bar_tokens)
    rest_keys = set(chord_weights(rest_bars))
    unique = sum(v for k, v in w.items() if k not in rest_keys) / total if total else 0.0
    my_tokens = set(bar_tokens)
    other_token_sets = [set("---".join(chord_key(c) for c in b) for b in sec) for sec in other_sections]
    all_other = set().union(*other_token_sets) if other_token_sets else set()
    recurring = sum(t in all_other for t in bar_tokens) / nb
    jaccards = [len(my_tokens & o) / len(my_tokens | o) for o in other_token_sets if my_tokens | o]
    max_jac = max(jaccards) if jaccards else 0.0
    n_sharing = sum(j >= 0.5 for j in jaccards)
    return np.array([
        nb, len(sounding) / nb, len(w), entropy,
        share(lambda k: root(k) == "C"), share(lambda k: root(k) == "F"), share(lambda k: root(k) == "G"),
        share(lambda k: root(k) in FLAT_SIDE), share(lambda k: root(k) in SHARP_SIDE),
        share(lambda k: qual(k) in ("min", "min7", "minmaj7")), share(lambda k: qual(k) in SEVENTHS),
        n_no_chord / all_dur if all_dur else 0.0,
        float(first.root == "C"), float(last.root == "C"), float(last.root == "G"),
        float(chord_key(first) == chord_key(last)),
        repeated, period, float(period > 0),
        jsd(p_sec, p_rest), unique, float(key_differs), recurring, max_jac, n_sharing,
    ])


POSITION_NAMES = ["section_index", "section_index_from_end", "relative_position", "n_sections_in_song"]


def position_features(idx: int, n_sections: int) -> np.ndarray:
    return np.array([idx, n_sections - 1 - idx, idx / max(n_sections - 1, 1), n_sections], dtype=float)


def boundary_chord_profiles(sections: list[tuple[str, list[list[Chord]]]], labels: list[str]) -> dict:
    """For each label: duration-weighted chord distribution of the first bar,
    the last bar, and the interior bars of its sections (relative roots)."""
    out = {}
    for lab in labels:
        first, last, middle = Counter(), Counter(), Counter()
        n = 0
        for l, bars in sections:
            if l != lab or len(bars) < 3:
                continue
            n += 1
            first.update(chord_weights([bars[0]]))
            last.update(chord_weights([bars[-1]]))
            middle.update(chord_weights(bars[1:-1]))
        out[lab] = {"n": n, "first": first, "last": last, "middle": middle}
    return out
