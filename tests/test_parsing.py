"""Run with:  python -m pytest tests/   (or plain  python tests/test_parsing.py)"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chordtopics.billboard import parse_bar, parse_song, beats_in_metre  # noqa: E402
from chordtopics.chords import relative_root, simplify_quality  # noqa: E402
from chordtopics.chordonomicon import parse_symbol, parse_row  # noqa: E402
from chordtopics.tokenize import TokenizerConfig, bargrams  # noqa: E402


def test_relative_root():
    assert relative_root("Ab", "Ab") == "C"
    assert relative_root("Db", "Ab") == "F"      # IV
    assert relative_root("Gb", "Ab") == "Bb"     # bVII
    assert relative_root("F#", "D") == "E"       # iii


def test_quality_buckets():
    assert simplify_quality("sus4(b7,9)") == "dom7"
    assert simplify_quality("maj(9)") == "maj"
    assert simplify_quality("maj7") == "maj7"
    assert simplify_quality("hdim7") == "dim"
    assert simplify_quality("min7(11)") == "min7"
    assert simplify_quality("sus4(b7)", "noext") == "sus4"


def test_compound_metre():
    assert beats_in_metre("12/8") == 4
    assert beats_in_metre("6/8") == 2
    assert beats_in_metre("3/4") == 3


def test_bar_dots_extend_previous_chord():
    bar = parse_bar("F:min/b3 C:7/5 F:min .", "4/4", "Ab", "coarse")
    assert [c.dur for c in bar.chords] == [Fraction(1), Fraction(1), Fraction(2)]
    assert [c.onset for c in bar.chords] == [Fraction(0), Fraction(1), Fraction(2)]
    assert bar.chords[0].root == "A"   # F in Ab major is vi: (5 - 8) % 12 = 9 -> "A"
    assert sum(c.dur for c in bar.chords) == bar.beats


def test_bar_metre_override_and_no_chord():
    bar = parse_bar("(2/4) N", "4/4", "C", "coarse")
    assert bar.beats == 2 and bar.chords[0].root == "N"
    bar = parse_bar("G:maj C:maj .", "3/4", "C", "coarse")
    assert [c.dur for c in bar.chords] == [Fraction(1), Fraction(2)]


def test_token_merges_chords_identical_at_token_resolution():
    bar = parse_bar("C:maj C:maj/5 F:maj", "4/4", "C", "coarse")
    assert len(bar.chords) == 3  # the parser keeps the events
    assert bargrams([bar.chords], TokenizerConfig()) == ["0,2.67,C:maj---2.67,1.33,F:maj"]
    assert bargrams([bar.chords], TokenizerConfig(with_bass=True)) == ["0,1.33,C:maj---1.33,1.33,C:maj/5---2.67,1.33,F:maj"]


def test_song_repeats_key_change_and_last_bar(tmp_path):
    txt = """# title: t
# artist: a
# metre: 4/4
# tonic: G

0.0\tsilence
1.0\tA, verse, | G:maj | D:maj/3 |, (voice
5.0\t| C:maj | x2
# tonic: A
9.0\tB, chorus, | A:maj | E:maj |
13.0\tend
"""
    p = tmp_path / "salami_chords.txt"
    p.write_text(txt)
    s = parse_song(p, 1)
    assert [sec.label for sec in s.sections] == ["verse", "chorus"]
    verse, chorus = s.sections
    assert len(verse.bars) == 4            # 2 bars + "| C:maj | x2"
    assert [b.chords[0].root for b in verse.bars] == ["C", "G", "F", "F"]
    assert s.key_changes == 1
    # after the key change A:maj is the new tonic
    assert [b.chords[0].root for b in chorus.bars] == ["C", "G"]
    # the final bar of the song is retained by the tokenizer
    cfg = TokenizerConfig(collapse_repeats=False)
    toks = bargrams([b.chords for b in s.bars], cfg)
    assert len(toks) == 6 and toks[-1] == "0,4,G:maj"


def test_repeat_collapse():
    bars = [[c] for c in []]
    cfg = TokenizerConfig(collapse_repeats=True, repeat_counts=True, max_repeat=4)
    s = parse_bar("C:maj", "4/4", "C", "coarse").chords
    g = parse_bar("G:maj", "4/4", "C", "coarse").chords
    assert bargrams([s, s, s, s, s, g], cfg) == ["0,4,C:majx4", "0,4,G:maj"]
    cfg.repeat_counts = False
    assert bargrams([s, s, s, g, g], cfg) == ["0,4,C:maj", "0,4,G:maj"]


def test_chordonomicon_symbols():
    assert parse_symbol("Csus4")[0] == 0 and parse_symbol("Csus4")[1] == "maj"
    assert parse_symbol("Csmin")[0] == 1 and parse_symbol("Csmin")[1] == "min"
    assert parse_symbol("Fs7")[1] == "dom7"
    assert parse_symbol("Gno3d")[1] == "maj"
    assert parse_symbol("A/Cs")[0] == 9


def test_chordonomicon_row_key_estimate():
    row = {"id": "1", "chords": "<verse_1> G D Emin C G D Emin C <chorus_1> C D G G",
           "release_date": "1999-05-01", "decade": "1990.0", "genres": "'pop' 'rock'", "main_genre": "pop"}
    s = parse_row(row)
    assert s.tonic == "G" and s.mode == "major"
    assert [lab for lab, _ in s.sections] == ["verse", "chorus"]
    assert [c.root for c in s.sections[0][1][:4]] == ["C", "G", "A", "F"]
    assert s.release_year == 1999 and s.genres == ["pop", "rock"]
    assert s.section_keys == [("G", "major"), ("G", "major")] and s.key_changes == 0


def test_chordonomicon_modulating_section():
    # verse in G, chorus a semitone up in Ab (no shared chords, major chord on the tonic)
    verse, chorus = "G D Emin C " * 6, "Ab Eb Fmin Db Ab " * 6
    row = {"id": "2", "chords": f"<verse_1> {verse}<chorus_1> {chorus}".strip(),
           "release_date": "", "decade": "", "genres": "", "main_genre": ""}
    s = parse_row(row)
    assert s.section_keys == [("G", "major"), ("Ab", "major")] and s.key_changes == 1
    # both sections read as I V vi IV relative to their own key
    assert [c.root for c in s.sections[1][1][:4]] == ["C", "G", "A", "F"]


def test_learned_key_prefers_minor_tonic_for_aeolian_loop():
    row = {"id": "3", "chords": "<verse_1> Amin F C G Amin F C G Amin F C G Amin",
           "release_date": "", "decade": "", "genres": "", "main_genre": ""}
    s = parse_row(row)
    assert (s.tonic, s.mode) == ("A", "minor")
    assert [c.token(with_rhythm=False) for c in s.sections[0][1][:4]] == ["C:min", "Ab:maj", "Eb:maj", "Bb:maj"]


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames:
                fn(Path(tempfile.mkdtemp()))
            else:
                fn()
            print("ok", name)
