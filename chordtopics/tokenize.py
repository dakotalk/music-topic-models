"""Bargram tokenization and collocation detection.

A *bargram* is one bar of music rendered as a single token: every chord in
the bar with its onset beat and duration, joined with "---". Collocation
detection (gensim Phrases, the Mikolov et al. 2013 scoring the original
notebook used) then glues frequently co-occurring bars into multi-bar
progressions, joined with "___".

Differences from the 2015 notebook, all deliberate:
  * the final bar of a song is kept (the old loop never flushed it);
  * runs of identical bars can be collapsed to a single token carrying the
    run length, so a 16-bar tonic vamp is one word, not 16 words that form
    four different collocations;
  * documents that end up empty after thresholding are removed and counted,
    instead of silently feeding the model their prior.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby

from gensim.models.phrases import Phrases, FrozenPhrases

from .chords import Chord


@dataclass
class TokenizerConfig:
    with_rhythm: bool = True      # include onset,duration in chord tokens
    with_bass: bool = False
    collapse_repeats: bool = True  # one token for a run of identical bars
    repeat_counts: bool = False    # append "xN" with the (clipped) run length
    max_repeat: int = 4            # run lengths above this are clipped
    # Collocation rounds default to 0: on Billboard every round lowers held-out
    # coherence and seed stability (see scripts/experiment_tokenization.py),
    # because multi-bar tokens are near-unique to a few songs.
    collocation_rounds: int = 0
    collocation_threshold: float = 0.1   # Phrases(scoring="default") threshold
    collocation_min_count: int = 5
    min_doc_freq: int = 10         # keep tokens in >= this many documents
    min_term_freq: int = 30        # and with >= this many occurrences overall
    drop_no_chord: bool = True     # drop bars that are entirely N / silence


def bar_token(chords: list[Chord], cfg: TokenizerConfig) -> str | None:
    if cfg.drop_no_chord and all(c.root == "N" for c in chords):
        return None
    # Adjacent chords that read the same at this token resolution (e.g.
    # "C:maj C:maj/5" without bass, or "C:maj C:maj(9)" after quality
    # bucketing) are one harmony: merge them so the bar is "C:maj", not
    # "C:maj---C:maj".
    merged: list[Chord] = []
    for c in chords:
        if merged and c.token(False, cfg.with_bass) == merged[-1].token(False, cfg.with_bass):
            p = merged[-1]
            merged[-1] = Chord(p.onset, p.dur + c.dur, p.root, p.quality, p.bass)
        else:
            merged.append(c)
    return "---".join(c.token(cfg.with_rhythm, cfg.with_bass) for c in merged)


def bargrams(bars: list[list[Chord]], cfg: TokenizerConfig) -> list[str]:
    toks = [t for b in bars if (t := bar_token(b, cfg)) is not None]
    if not cfg.collapse_repeats:
        return toks
    out = []
    for tok, run in groupby(toks):
        n = min(len(list(run)), cfg.max_repeat)
        out.append(f"{tok}x{n}" if (cfg.repeat_counts and n > 1) else tok)
    return out


@dataclass
class Corpus:
    docs: list[list[str]]
    ids: list            # one id per document (song id, or (song id, section idx))
    meta: list[dict]     # per-document metadata
    dropped: int = 0     # documents removed because empty after thresholding
    vocab: Counter = field(default_factory=Counter)
    phrasers: list[FrozenPhrases] = field(default_factory=list)


def build_corpus(docs: list[list[str]], ids: list, meta: list[dict],
                 cfg: TokenizerConfig,
                 phrasers: list[FrozenPhrases] | None = None) -> Corpus:
    """Apply collocation detection + thresholding.

    Pass `phrasers` (from a training corpus) to tokenize held-out documents
    with the same vocabulary; otherwise phrasers are fit on `docs`.
    """
    fit = phrasers is None
    phrasers = list(phrasers or [])
    for r in range(cfg.collocation_rounds):
        if fit:
            p = Phrases(docs, min_count=cfg.collocation_min_count,
                        threshold=cfg.collocation_threshold, delimiter="___")
            phrasers.append(p.freeze())
        docs = [phrasers[r][d] for d in docs]

    if fit:
        df = Counter(t for d in docs for t in set(d))
        tf = Counter(t for d in docs for t in d)
        keep = {t for t in tf if df[t] >= cfg.min_doc_freq and tf[t] >= cfg.min_term_freq}
    else:
        keep = None  # held-out docs: vocabulary restriction happens in the Dictionary

    out_docs, out_ids, out_meta, dropped = [], [], [], 0
    for d, i, m in zip(docs, ids, meta):
        if keep is not None:
            d = [t for t in d if t in keep]
        if not d:
            dropped += 1
            continue
        out_docs.append(d); out_ids.append(i); out_meta.append(m)
    return Corpus(out_docs, out_ids, out_meta, dropped,
                  Counter(t for d in out_docs for t in d), phrasers)
