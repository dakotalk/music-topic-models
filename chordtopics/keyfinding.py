"""Key estimation from chord symbols, for corpora without annotated tonics.

A pitch-class histogram of chord tones (weighted by duration) is correlated
with major and minor key profiles for all 12 transpositions. Two refinements
matter for pop, where Aeolian progressions (Am F C G) correlate about equally
with A minor and C major:

* an **anchor bonus** for keys whose tonic is the most frequent chord root
  with a matching mode (a minor chord that dominates the song is more likely
  i than vi);
* **per-section estimation** with a song-level prior, so a modulated chorus
  gets its own key while short or ambiguous sections fall back to the song.

`scripts/validate_keyfinding.py` measures each option against the McGill
Billboard tonic annotations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chords import RELATIVE_NAMES

# Krumhansl & Kessler (1982)
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Temperley (2001), "The Cognition of Basic Musical Structures"
T_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
T_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
PROFILES = {"krumhansl": (KK_MAJOR, KK_MINOR), "temperley": (T_MAJOR, T_MINOR)}

# chord tones for the coarse quality buckets used across the package
QUALITY_TONES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7), "maj7": (0, 4, 7, 11), "min": (0, 3, 7), "min7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10), "dim": (0, 3, 6), "aug": (0, 4, 8), "minmaj7": (0, 3, 7, 11),
    "other": (0, 7), "": (),
}


@dataclass
class KeyConfig:
    profile: str = "krumhansl"
    anchor_bonus: float = 0.0     # added to r for the key anchored on the dominant chord root
    section_prior: float = 1.0    # weight of the (normalised) song histogram added to a section's
    min_section_chords: int = 4   # sections shorter than this take the song key


def chord_histogram(chords: list[tuple[int, str, float]]) -> np.ndarray:
    """chords: (absolute root pitch class, coarse quality, weight)."""
    hist = np.zeros(12)
    for root, quality, w in chords:
        for iv in QUALITY_TONES.get(quality, (0, 7)):
            hist[(root + iv) % 12] += w
    return hist


def dominant_root(chords: list[tuple[int, str, float]]) -> tuple[int, str] | None:
    """Most frequent (root, mode-of-chord) by weight; mode is 'minor' for minor-type chords."""
    acc: dict[tuple[int, str], float] = {}
    for root, quality, w in chords:
        mode = "minor" if quality in ("min", "min7", "minmaj7", "dim") else "major"
        acc[(root, mode)] = acc.get((root, mode), 0.0) + w
    return max(acc, key=acc.get) if acc else None


def estimate_key(hist: np.ndarray, cfg: KeyConfig = KeyConfig(),
                 anchor: tuple[int, str] | None = None) -> tuple[int, str, float]:
    """Returns (tonic pitch class, mode, score)."""
    major, minor = PROFILES[cfg.profile]
    best, best_s = (0, "major"), -np.inf
    if hist.sum() == 0:
        return 0, "major", 0.0
    for shift in range(12):
        h = np.roll(hist, -shift)
        for mode, prof in (("major", major), ("minor", minor)):
            r = 0.0 if h.std() == 0 else float(np.corrcoef(h, prof)[0, 1])
            if anchor is not None and anchor == (shift, mode):
                r += cfg.anchor_bonus
            if r > best_s:
                best, best_s = (shift, mode), r
    return best[0], best[1], best_s


def estimate_song_and_sections(sections: list[list[tuple[int, str, float]]],
                               cfg: KeyConfig = KeyConfig()) -> tuple[tuple[int, str], list[tuple[int, str]]]:
    """Song key plus one key per section (list of (tonic pc, mode))."""
    all_chords = [c for sec in sections for c in sec]
    song_hist = chord_histogram(all_chords)
    song_key = estimate_key(song_hist, cfg, dominant_root(all_chords))[:2]
    prior = song_hist / song_hist.sum() if song_hist.sum() else song_hist
    keys = []
    for sec in sections:
        n = sum(w for _, _, w in sec)
        if n < cfg.min_section_chords or cfg.section_prior == np.inf:
            keys.append(song_key)
            continue
        hist = chord_histogram(sec)
        hist = hist / hist.sum() * n if hist.sum() else hist
        hist = hist + cfg.section_prior * prior * n   # prior scaled to the section's own mass
        keys.append(estimate_key(hist, cfg, dominant_root(sec))[:2])
    return song_key, keys


def key_name(tonic: int) -> str:
    return RELATIVE_NAMES[tonic % 12]


# ---------------------------------------------------------------------------
# Learned chord-root profile: how often each (relative root, chord type) occurs
# in annotated songs. Scoring a candidate tonic is then a multinomial
# log-likelihood of the song's chord histogram under the shifted profile.
# ---------------------------------------------------------------------------

def _is_minor_type(quality: str) -> bool:
    return quality in ("min", "min7", "minmaj7", "dim")


def chord_histogram24(chords: list[tuple[int, str, float]]) -> np.ndarray:
    """12 roots x {major-type, minor-type} weighted by duration."""
    h = np.zeros((12, 2))
    for root, quality, w in chords:
        h[root % 12, int(_is_minor_type(quality))] += w
    return h


def learn_profile(songs: list[tuple[list[tuple[int, str, float]], int]], smoothing: float = 1.0) -> np.ndarray:
    """songs: (chords with absolute roots, annotated tonic pc).

    Returns a 2x12x2 table: for major-key and minor-key songs separately (a
    song counts as minor when the chord on its annotated tonic is mostly
    minor-type), the probability of each (root relative to tonic, chord type).
    The third slot of the array's last row is not used; mode priors are stored
    separately via `mode_prior`.
    """
    prof = np.full((2, 12, 2), smoothing)
    for chords, tonic in songs:
        h = chord_histogram24(chords)
        m = int(h[tonic, 1] > h[tonic, 0])
        prof[m] += np.roll(h, -tonic, axis=0) / max(h.sum(), 1e-9)
    return prof / prof.sum(axis=(1, 2), keepdims=True)


def mode_prior(songs: list[tuple[list[tuple[int, str, float]], int]]) -> np.ndarray:
    """P(major), P(minor) over annotated songs."""
    minor = 0
    for chords, tonic in songs:
        h = chord_histogram24(chords)
        minor += h[tonic, 1] > h[tonic, 0]
    p = (len(songs) - minor + 1) / (len(songs) + 2)
    return np.array([p, 1 - p])


@dataclass
class LearnedKeyConfig:
    first_chord_bonus: float = 0.0   # added (in nats, per unit of song mass) to the first chord's root
    section_prior: float = 2.0
    min_section_chords: int = 4
    change_margin: float = 0.0       # nats per chord a section key must beat the song key by


MODES = ("major", "minor")


def key_scores(chords: list[tuple[int, str, float]], profile: np.ndarray,
               cfg: LearnedKeyConfig, prior_hist: np.ndarray | None = None,
               mode_log_prior: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Log-likelihood score for each of the 24 (tonic, mode) candidates as a
    2x12 array [mode, tonic], and the (prior-augmented) histogram used."""
    h = chord_histogram24(chords)
    mass = h.sum()
    if mass == 0:
        return np.zeros((2, 12)), h
    if prior_hist is not None and prior_hist.sum() > 0:
        h = h + cfg.section_prior * prior_hist / prior_hist.sum() * mass
    logp = np.log(profile)   # 2 x 12 x 2
    scores = np.array([[(np.roll(h, -t, axis=0) * logp[m]).sum() for t in range(12)] for m in range(2)])
    if mode_log_prior is not None:
        scores += mode_log_prior[:, None]
    if cfg.first_chord_bonus:
        scores[:, chords[0][0] % 12] += cfg.first_chord_bonus * mass
    return scores, h


def _argmax_key(scores: np.ndarray) -> tuple[int, str]:
    m, t = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(t), MODES[m]


def estimate_key_learned(chords: list[tuple[int, str, float]], profile: np.ndarray,
                         cfg: LearnedKeyConfig = LearnedKeyConfig(),
                         prior_hist: np.ndarray | None = None,
                         mode_log_prior: np.ndarray | None = None) -> tuple[int, str]:
    scores, h = key_scores(chords, profile, cfg, prior_hist, mode_log_prior)
    if h.sum() == 0:
        return 0, "major"
    return _argmax_key(scores)


def estimate_song_and_sections_learned(sections: list[list[tuple[int, str, float]]], profile: np.ndarray,
                                       cfg: LearnedKeyConfig = LearnedKeyConfig(),
                                       mode_log_prior: np.ndarray | None = None):
    all_chords = [c for sec in sections for c in sec]
    song_key = estimate_key_learned(all_chords, profile, cfg, mode_log_prior=mode_log_prior)
    song_hist = chord_histogram24(all_chords)
    keys = []
    for sec in sections:
        n = sum(w for _, _, w in sec)
        if n < cfg.min_section_chords or cfg.section_prior == np.inf:
            keys.append(song_key)
            continue
        scores, h = key_scores(sec, profile, cfg, prior_hist=song_hist, mode_log_prior=mode_log_prior)
        tonic, mode = _argmax_key(scores)
        # keep the song key unless the section's own key wins by a clear margin
        song_score = scores[MODES.index(song_key[1]), song_key[0]]
        if (tonic, mode) != song_key and scores.max() - song_score < cfg.change_margin * n:
            tonic, mode = song_key
        keys.append((tonic, mode))
    return song_key, keys
