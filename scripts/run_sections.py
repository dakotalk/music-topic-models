"""What makes a beginning sound like a beginning? Harmonic features of song
sections on the McGill Billboard corpus.

    python scripts/run_sections.py --data data/McGill-Billboard --out results/sections

Outputs
  features.csv                one row per labelled section
  descriptives.md             mean of each feature per label, with bootstrap CIs
  classification.txt          GroupKFold-by-song classifiers: harmony only, position
                              only, both; 6 labels and beginning/middle/end
  importance.csv / .png       permutation importance of the harmony features
  boundary_profiles.md/.png   first-bar vs last-bar vs interior chord distributions per label
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chordtopics import billboard, plotting, sections as S  # noqa: E402
from chordtopics.evaluate import bootstrap_mean_ci  # noqa: E402

log = logging.getLogger("run_sections")


def build_table(song_records, labels):
    """song_records: iterable of (song_id, title, artist, sections) where sections is a
    list of (label, bars, tonic) and bars is a list of chord lists."""
    rows, meta, boundary_input = [], [], []
    for song_id, title, artist, secs in song_records:
        secs = [(lab, bars, tonic) for lab, bars, tonic in secs if bars]
        if not secs:
            continue
        song_tonic = Counter(t for _, _, t in secs).most_common(1)[0][0]
        for i, (lab, bars, tonic) in enumerate(secs):
            if lab not in labels:
                continue
            others = [b for j, (_, b, _) in enumerate(secs) if j != i]
            rest = [bar for sec_bars in others for bar in sec_bars]
            f = S.section_features(bars, rest, others, tonic != song_tonic)
            if np.isnan(f[0]):
                continue
            rows.append(np.concatenate([f, S.position_features(i, len(secs))]))
            meta.append({"song_id": song_id, "label": lab, "coarse": S.COARSE[lab], "title": title, "artist": artist})
            boundary_input.append((lab, bars))
    df = pd.DataFrame(rows, columns=S.FEATURE_NAMES + S.POSITION_NAMES)
    return pd.concat([pd.DataFrame(meta), df], axis=1), boundary_input


def billboard_records(songs):
    for song in songs:
        yield song.id, song.title, song.artist, [
            (sec.label, [b.chords for b in sec.bars], Counter(b.tonic for b in sec.bars).most_common(1)[0][0] if sec.bars else "")
            for sec in song.sections]


def chordonomicon_records(csongs):
    # no bars in this corpus: every chord is its own unit-length "bar"
    for s in csongs:
        yield s.id, str(s.id), s.main_genre, [
            (lab, [[c] for c in chords], key[0]) for (lab, chords), key in zip(s.sections, s.section_keys)]


def cv_classify(X, y, groups, labels, seed=0):
    gkf = GroupKFold(n_splits=5)
    accs, bal, maj = [], [], []
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for tr, te in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed,
                                             class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append((pred == y[te]).mean())
        bal.append(balanced_accuracy_score(y[te], pred))
        cm += confusion_matrix(y[te], pred, labels=labels)
        maj.append(DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr]).score(X[te], y[te]))
    return {"acc": np.mean(accs), "acc_sd": np.std(accs), "bal": np.mean(bal), "majority": np.mean(maj),
            "chance_balanced": 1 / len(labels), "cm": cm}


def fmt_cm(cm, labels):
    w = max(len(l) for l in labels)
    lines = [" " * (w + 1) + " ".join(f"{l[:5]:>5}" for l in labels)]
    for l, row in zip(labels, cm):
        lines.append(f"{l:>{w}} " + " ".join(f"{v:5d}" for v in row))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="McGill-Billboard folder")
    ap.add_argument("--chordonomicon", type=Path, help="chordonomicon_v2.csv instead of Billboard")
    ap.add_argument("--n", type=int, default=40000, help="Chordonomicon songs to sample")
    ap.add_argument("--out", type=Path, default=Path("results/sections"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    plotting.style()
    import matplotlib.pyplot as plt

    if args.chordonomicon:
        from chordtopics import chordonomicon
        csongs = chordonomicon.load_corpus(args.chordonomicon, limit=args.n, seed=0, require_year=True)
        labels = [l for l in S.LABELS6 if l != "prechorus"]   # Chordonomicon has no prechorus tag
        rhythm_free = True
        df, boundary_input = build_table(chordonomicon_records(csongs), labels)
    else:
        songs, _ = billboard.load_corpus(args.data, None, "coarse")
        labels = S.LABELS6
        rhythm_free = False
        df, boundary_input = build_table(billboard_records(songs), labels)
    feature_names = [f for f in S.FEATURE_NAMES if not (rhythm_free and f in ("chords_per_bar", "share_no_chord"))]
    df.to_csv(out / "features.csv", index=False)
    log.info("%d labelled sections from %d songs; labels %s", len(df), df.song_id.nunique(),
             dict(Counter(df.label)))

    # ---- descriptives --------------------------------------------------------
    with open(out / "descriptives.md", "w") as f:
        f.write("# Harmonic features by section label (mean, 95% bootstrap CI)\n\n")
        f.write("| feature | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels) + "\n")
        for feat in feature_names:
            cells = []
            for lab in labels:
                v = df.loc[df.label == lab, feat].dropna().to_numpy()
                m, lo, hi = bootstrap_mean_ci(v, n_boot=500)
                cells.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
            f.write(f"| {feat} | " + " | ".join(cells) + " |\n")
        f.write(f"\nn per label: {dict(Counter(df.label))}\n")

    # ---- classification ---------------------------------------------------------
    Xh = df[feature_names].to_numpy()
    Xp = df[S.POSITION_NAMES].to_numpy()
    Xb = np.hstack([Xh, Xp])
    groups = df.song_id.to_numpy()
    results = {}
    with open(out / "classification.txt", "w") as f:
        for task, y, labs in [("6 labels", df.label.to_numpy(), labels),
                              ("beginning / middle / end", df.coarse.to_numpy(), ["beginning", "middle", "end"]),
                              ("intro vs outro", None, ["intro", "outro"]),
                              ("verse vs chorus", None, ["verse", "chorus"]),
                              ("bridge vs chorus", None, ["bridge", "chorus"])]:
            if y is None:
                mask = df.label.isin(labs).to_numpy()
                y_t, g_t = df.label.to_numpy()[mask], groups[mask]
                sets = {"harmony only": Xh[mask], "position only": Xp[mask], "harmony + position": Xb[mask]}
            else:
                y_t, g_t = y, groups
                sets = {"harmony only": Xh, "position only": Xp, "harmony + position": Xb}
                if len(y_t) == 0:
                continue
            f.write(f"== {task} (n={len(y_t)}, GroupKFold by song, gradient boosting, balanced class weights) ==\n")
            for name, X in sets.items():
                r = cv_classify(X, y_t, g_t, labs)
                results[(task, name)] = r
                f.write(f"{name:>20}: accuracy {r['acc']:.3f} ± {r['acc_sd']:.3f}   balanced {r['bal']:.3f}"
                        f"   (majority {r['majority']:.3f}, balanced chance {r['chance_balanced']:.3f})\n")
            f.write("\nconfusion, harmony only (rows=true, cols=pred):\n")
            f.write(fmt_cm(results[(task, 'harmony only')]['cm'], labs) + "\n\n")
    log.info("\n%s", (out / "classification.txt").read_text())

    # ---- permutation importance (harmony only, 6 labels, held-out folds) ------
    y6 = df.label.to_numpy()
    imp = np.zeros(len(feature_names))
    for tr, te in GroupKFold(n_splits=5).split(Xh, y6, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0,
                                             class_weight="balanced").fit(Xh[tr], y6[tr])
        pi = permutation_importance(clf, Xh[te], y6[te], scoring="balanced_accuracy", n_repeats=5, random_state=0)
        imp += pi.importances_mean / 5
    imp_df = pd.DataFrame({"feature": feature_names, "drop_in_balanced_accuracy": imp}).sort_values(
        "drop_in_balanced_accuracy", ascending=False)
    imp_df.to_csv(out / "importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    top = imp_df.head(15)[::-1]
    ax.barh(top.feature, top.drop_in_balanced_accuracy, color=plotting.SERIES[0], height=0.6)
    ax.set_xlabel("drop in held-out balanced accuracy when shuffled")
    ax.set_title("Which harmonic cues tell section types apart")
    fig.tight_layout(); fig.savefig(out / "importance.png", dpi=150); plt.close(fig)

    # ---- within-section: first bar vs last bar vs interior ----------------------
    prof = S.boundary_chord_profiles(boundary_input, labels)
    top_chords = ["C:maj", "F:maj", "G:maj", "A:min", "C:min", "D:min", "Bb:maj", "G:dom7", "E:min", "Ab:maj",
                  "Eb:maj", "D:maj"]
    with open(out / "boundary_profiles.md", "w") as f:
        f.write("# How sections begin and end\n\nDuration-weighted share of each chord in the first bar, "
                "the interior bars and the last bar of sections with >= 3 bars (roots relative to the local tonic).\n\n")
        for lab in labels:
            p = prof[lab]
            f.write(f"## {lab} (n={p['n']})\n\n| chord | first bar | interior | last bar |\n|---|---|---|---|\n")
            tots = {k: sum(p[k].values()) or 1 for k in ("first", "middle", "last")}
            for ch in top_chords:
                f.write(f"| {ch} | {p['first'][ch]/tots['first']:.2f} | {p['middle'][ch]/tots['middle']:.2f} | "
                        f"{p['last'][ch]/tots['last']:.2f} |\n")
            f.write("\n")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    for ax, lab in zip(axes.flat, labels):
        p = prof[lab]
        tots = {k: sum(p[k].values()) or 1 for k in ("first", "middle", "last")}
        x = np.arange(3)
        for j, (ch, color) in enumerate(zip(["C:maj", "G:maj", "F:maj", "C:min"], plotting.SERIES[:4])):
            vals = [p[k][ch] / tots[k] for k in ("first", "middle", "last")]
            ax.plot(x, vals, marker="o", color=color, label=ch if lab == labels[0] else None)
        ax.set_xticks(x, ["first bar", "interior", "last bar"])
        ax.set_title(f"{lab} (n={p['n']})")
    axes[0, 0].legend(title="relative chord", fontsize=8)
    fig.suptitle("Share of the main chords at the start, middle and end of each section type", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "boundary_profiles.png", dpi=150); plt.close(fig)
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
