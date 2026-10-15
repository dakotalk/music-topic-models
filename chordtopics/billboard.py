"""Parser for the McGill Billboard "salami_chords.txt" annotations.

Format reference: Burgoyne, Wild & Fujinaga (ISMIR 2011) and the project's
annotation guidelines. The parts that matter here:

    # title / # artist / # metre / # tonic     header; metre and tonic lines
                                               can recur mid-song (key change)
    <time>\t[A', ][label, ]| bar | bar |[ xN][, (comments)]
    <time>\tsilence | end | Z | fadeout        structural markers, no chords

Inside a bar, tokens are separated by spaces:
    Ab:maj7/5     chord, root:quality[/bass-degree]
    .             the previous chord continues for another beat slot
    N             no chord;  *  unknown chord
    &pause        a pause;   (3/4) at the start of a bar overrides the metre
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .chords import Chord, relative_root, simplify_quality

LABEL_ALIASES = {
    "pre-chorus": "prechorus",
    "pre-verse": "preverse",
    "instrumental": "instrumental",
    "solo": "solo",
    "verse": "verse",
    "chorus": "chorus",
    "bridge": "bridge",
    "intro": "intro",
    "outro": "outro",
    "interlude": "interlude",
    "trans": "transition",
    "transition": "transition",
    "fadeout": "fadeout",
    "coda": "outro",
    "ending": "outro",
    "refrain": "chorus",
    "secondary theme": "secondary_theme",
    "main theme": "main_theme",
}


@dataclass
class Bar:
    beats: Fraction
    chords: list[Chord]
    tonic: str
    metre: str


@dataclass
class Section:
    letter: str          # SALAMI letter, e.g. "A", "B'"
    label: str           # normalised function label, e.g. "verse"
    start: float         # seconds
    bars: list[Bar] = field(default_factory=list)


@dataclass
class Song:
    id: int
    title: str
    artist: str
    metre: str
    tonic: str
    sections: list[Section]
    key_changes: int = 0
    metre_changes: int = 0

    @property
    def bars(self) -> list[Bar]:
        return [b for s in self.sections for b in s.bars]


def beats_in_metre(metre: str) -> Fraction:
    """Beats per bar for a time signature. Compound metres (6/8, 12/8) count
    dotted-quarter beats, which matches how the annotators split bars."""
    num, den = (int(x) for x in metre.split("/"))
    if den == 8 and num % 3 == 0:
        return Fraction(num // 3)
    return Fraction(num)


_HEADER = re.compile(r"^#\s*(\w+):\s*(.*)$")
_CHORD = re.compile(r"^([A-G][b#]*):([^/]+)(?:/(.+))?$")
_METRE_OVERRIDE = re.compile(r"^\((\d+/\d+)\)$")
_REPEAT = re.compile(r"^x(\d+)$")


def parse_bar(text: str, metre: str, tonic: str, quality_level: str) -> Bar | None:
    toks = text.split()
    if not toks:
        return None
    local_metre = metre
    if _METRE_OVERRIDE.match(toks[0]):
        local_metre = _METRE_OVERRIDE.match(toks[0]).group(1)
        toks = toks[1:]
    if not toks:
        return None
    beats = beats_in_metre(local_metre)
    # every token (chords and dots) occupies an equal slot of the bar
    slot = beats / len(toks)
    chords: list[Chord] = []
    pos = Fraction(0)
    for t in toks:
        if t == ".":
            if chords:
                prev = chords[-1]
                chords[-1] = Chord(prev.onset, prev.dur + slot, prev.root, prev.quality, prev.bass)
            pos += slot
            continue
        if t in ("N", "*", "&pause"):
            chords.append(Chord(pos, slot, "N", "", ""))
            pos += slot
            continue
        m = _CHORD.match(t)
        if not m:
            raise ValueError(f"unparseable chord token {t!r} in bar {text!r}")
        root, quality, bass = m.group(1), m.group(2), m.group(3) or ""
        chords.append(Chord(pos, slot, relative_root(root, tonic),
                            simplify_quality(quality, quality_level), bass))
        pos += slot
    return Bar(beats, chords, tonic, local_metre)


def normalise_label(label: str) -> str:
    label = label.strip().lower()
    return LABEL_ALIASES.get(label, label.replace(" ", "_").replace("-", ""))


def parse_song(path: Path, song_id: int, quality_level: str = "coarse") -> Song:
    title = artist = metre = tonic = None
    sections: list[Section] = []
    key_changes = metre_changes = 0
    current: Section | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        h = _HEADER.match(line)
        if h:
            key, val = h.group(1), h.group(2).strip()
            if key == "title":
                title = val
            elif key == "artist":
                artist = val
            elif key == "metre":
                if metre is not None and val != metre:
                    metre_changes += 1
                metre = val
            elif key == "tonic":
                if tonic is not None and val != tonic:
                    key_changes += 1
                tonic = val
            continue

        time_str, _, body = line.partition("\t")
        start = float(time_str)
        # strip trailing comments / instrumentation after the last bar
        body = body.strip()
        if "|" not in body:
            continue  # silence, end, Z, fadeout-only lines
        head, bars_txt = body.split("|", 1)
        bars_txt = "|" + bars_txt
        # head: optional "A', label, " ; may also be just "fadeout, "
        head_parts = [p.strip() for p in head.split(",") if p.strip()]
        letter = label = None
        for p in head_parts:
            if re.fullmatch(r"[A-Z]'*", p):
                letter = p
            elif p not in ("fadeout",) and not p.startswith("("):
                label = p
        if letter is not None or label is not None:
            current = Section(letter or "", normalise_label(label or "unlabelled"), start)
            sections.append(current)
        if current is None:
            current = Section("", "unlabelled", start)
            sections.append(current)

        # tail after the last '|': repeat count and comments
        last_bar = bars_txt.rfind("|")
        tail = bars_txt[last_bar + 1:]
        repeat = 1
        for piece in tail.replace(",", " ").split():
            r = _REPEAT.match(piece)
            if r:
                repeat = int(r.group(1))
        bars_txt = bars_txt[: last_bar + 1]
        bars = []
        for seg in bars_txt.split("|")[1:-1]:
            seg = seg.replace("->", "").strip()
            b = parse_bar(seg, metre, tonic, quality_level)
            if b is not None:
                bars.append(b)
        current.bars.extend(bars * repeat)

    return Song(song_id, title or "", artist or "", metre or "", tonic or "",
                sections, key_changes, metre_changes)


def load_index(index_csv: Path) -> dict[int, dict]:
    rows = {}
    with open(index_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[int(r["id"])] = r
    return rows


def load_corpus(root: Path, index_csv: Path | None = None,
                quality_level: str = "coarse") -> tuple[list[Song], dict[int, dict]]:
    """Load every annotated song under `root` (the McGill-Billboard folder)."""
    songs = []
    for d in sorted(Path(root).iterdir()):
        f = d / "salami_chords.txt"
        if f.exists():
            songs.append(parse_song(f, int(d.name), quality_level))
    index = load_index(index_csv) if index_csv else {}
    return songs, index
