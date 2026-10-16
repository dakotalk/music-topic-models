"""5-fold cross-validated tonic accuracy of the learned chord-root profile on
Billboard, compared with Krumhansl. Writes the profile learned on all songs
to chordtopics/data/billboard_key_profile.npy for use on other corpora.

    python scripts/validate_learned_keyfinding.py --data data/McGill-Billboard
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
from scripts.validate_keyfinding import song_sections  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--profile-out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "chordtopics/data/billboard_key_profile.npy")
    args = ap.parse_args()
    songs, _ = billboard.load_corpus(args.data, None, "coarse")
    data = [(s, song_sections(s)) for s in songs]
    data = [(s, secs) for s, secs in data if secs]
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(data))

    grid = {"first_chord_bonus": [0.0, 0.2, 0.4], "section_prior": [np.inf, 1.0, 2.0],
            "change_margin": [0.0, 0.3, 0.5, 0.8]}
    rows = []
    n_mod_songs = sum(1 for s, _ in data if s.key_changes)
    for vals in itertools.product(*grid.values()):
        if vals[1] == np.inf and vals[2] > 0:
            continue  # margin is meaningless without per-section estimation
        cfg = keyfinding.LearnedKeyConfig(**dict(zip(grid.keys(), vals)), min_section_chords=8)
        song_ok = sec_ok = sec_n = mod_ok = mod_n = detected = false_alarm = 0
        for f in range(5):
            train = [([c for cs, _ in secs for c in cs], pitch_class(s.tonic))
                     for (s, secs), ff in zip(data, fold) if ff != f]
            profile = keyfinding.learn_profile(train)
            mode_lp = np.log(keyfinding.mode_prior(train))
            for (s, secs), ff in zip(data, fold):
                if ff != f:
                    continue
                song_key, keys = keyfinding.estimate_song_and_sections_learned([c for c, _ in secs], profile, cfg,
                                                                                mode_log_prior=mode_lp)
                song_ok += song_key[0] == pitch_class(s.tonic)
                flagged = any(a[0] != b[0] for a, b in zip(keys, keys[1:]))
                if s.key_changes:
                    detected += flagged
                else:
                    false_alarm += flagged
                for (chords, tonic), k in zip(secs, keys):
                    sec_n += 1; sec_ok += k[0] == tonic
                    if s.key_changes:
                        mod_n += 1; mod_ok += k[0] == tonic
        rows.append({**dict(zip(grid.keys(), vals)), "song_tonic_acc": song_ok / len(data),
                     "section_tonic_acc": sec_ok / sec_n, "modulating_section_acc": mod_ok / mod_n,
                     "key_change_detected": detected / n_mod_songs,
                     "key_change_false_alarm": false_alarm / (len(data) - n_mod_songs)})
    df = pd.DataFrame(rows).sort_values("section_tonic_acc", ascending=False)
    print(df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)

    full = [([c for cs, _ in secs for c in cs], pitch_class(s.tonic)) for s, secs in data]
    profile = keyfinding.learn_profile(full)
    prior = keyfinding.mode_prior(full)
    args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.profile_out, profile)
    np.save(args.profile_out.with_name("billboard_mode_prior.npy"), prior)
    names = keyfinding.RELATIVE_NAMES
    print(f"\nmode prior: major {prior[0]:.3f}, minor {prior[1]:.3f}")
    for m, name in enumerate(keyfinding.MODES):
        print(f"learned {name}-key profile (relative root: P(major-type chord), P(minor-type chord)):")
        for i in np.argsort(-profile[m].sum(axis=1))[:8]:
            print(f"  {names[i]:>2}  {profile[m, i, 0]:.3f}  {profile[m, i, 1]:.3f}")


if __name__ == "__main__":
    main()
