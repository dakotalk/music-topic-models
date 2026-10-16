"""Chord-spelling utilities shared by the corpus loaders."""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

PITCH_CLASS = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

# Relative roots are spelled as if the tonic were C, so a flat-side chord
# (bVII in a major key) always reads "Bb" and a sharp-side one (iii) "E".
RELATIVE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def pitch_class(name: str) -> int:
    """'Ab' -> 8, 'F#' -> 6, 'Cs' -> 1 (Chordonomicon spells sharps with 's')."""
    m = re.fullmatch(r"([A-Ga-g])([b#s]*)", name)
    if not m:
        raise ValueError(f"not a note name: {name!r}")
    pc = PITCH_CLASS[m.group(1).upper()]
    for acc in m.group(2):
        pc += -1 if acc == "b" else 1
    return pc % 12


def relative_root(root: str, tonic: str) -> str:
    """Spell `root` relative to `tonic`, as if the tonic were C."""
    return RELATIVE_NAMES[(pitch_class(root) - pitch_class(tonic)) % 12]


# Coarse quality buckets. The Billboard vocabulary has ~70 qualities, most
# of them rare; collapsing to these keeps topics about harmony rather than
# about voicing.
_QUALITY_BUCKETS = [  # tested in order: most specific first
    (re.compile(r"^minmaj"), "minmaj7"),
    (re.compile(r"^(maj7|maj9|maj13)"), "maj7"),
    (re.compile(r"^(7|9|11|13|sus4\(b7|sus2\(b7|5\(b7\)|aug\(b7\))"), "dom7"),
    (re.compile(r"^(dim|hdim)"), "dim"),
    (re.compile(r"^aug"), "aug"),
    (re.compile(r"^(min7|min9|min11|min13|min\(9\)|min\(11\)|min\(b13\))"), "min7"),
    (re.compile(r"^min"), "min"),
    (re.compile(r"^(maj|1|5|sus2|sus4)"), "maj"),
]


def simplify_quality(quality: str, level: str = "coarse") -> str:
    """Reduce a Billboard chord quality string.

    level="full"   keep everything (incl. extensions, bass notes are handled by caller)
    level="noext"  strip parenthesised extensions:  'sus4(b7,9)' -> 'sus4'
    level="coarse" bucket into maj/maj7/min/min7/dom7/dim/aug/minmaj7
    """
    if level == "full":
        return quality
    if level == "noext":
        return re.sub(r"\(.*", "", quality)
    for pat, name in _QUALITY_BUCKETS:
        if pat.match(quality):
            return name
    return "other"


@dataclass(frozen=True)
class Chord:
    """A chord event inside a bar.

    onset and dur are in beats (Fractions keep 3-chords-in-4/4 exact).
    root is already spelled relative to the local tonic.
    """
    onset: Fraction
    dur: Fraction
    root: str          # relative root, "C".."B"; "N" for no-chord
    quality: str
    bass: str = ""     # scale-degree bass from the annotation, e.g. "5", "b7"

    def token(self, with_rhythm: bool = True, with_bass: bool = False) -> str:
        if self.root == "N":
            core = "N"
        else:
            core = f"{self.root}:{self.quality}"
            if with_bass and self.bass:
                core += "/" + self.bass
        if not with_rhythm:
            return core
        return f"{_fmt(self.onset)},{_fmt(self.dur)},{core}"


def _fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{float(x):.3g}"
