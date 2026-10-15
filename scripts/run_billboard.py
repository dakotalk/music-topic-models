"""End-to-end run on the McGill Billboard corpus.

    python scripts/run_billboard.py --data /path/to/McGill-Billboard \
        --index /path/to/billboard-2.0-index.csv --out results/billboard

Steps
 1. parse songs; song-level bargram documents; 80/20 song split
 2. collocations + vocabulary fit on the training split only
 3. K sweep x seeds: held-out NPMI coherence, held-out perplexity, seed stability
 4. final model at the chosen K; topic listing
 5. section-type classification (GroupKFold by song) vs. baselines
 6. year regression vs. predict-the-mean baseline
 7. topic share by period with bootstrap CIs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chordtopics import billboard, tokenize, model, evaluate, plotting  # noqa: E402
from gensim.corpora import Dictionary  # noqa: E402

log = logging.getLogger("run_billboard")
TARGET_LABELS = ["verse", "chorus", "bridge", "prechorus"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--out", default=Path("results/billboard"), type=Path)
    ap.add_argument("--ks", default="6,8,10,12,16,20,24,32")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--k", type=int, default=None, help="override chosen K")
    ap.add_argument("--quality", default="coarse", choices=["coarse", "noext", "full"])
    ap.add_argument("--no-collapse", action="store_true", help="keep every repeated bar as its own token")
    ap.add_argument("--repeat-counts", action="store_true", help="append xN run length to collapsed tokens")
    ap.add_argument("--doc-unit", default="song", choices=["section", "song"],
                    help="what counts as a document for LDA (held-out split is always by song)")
    ap.add_argument("--collocations", type=int, default=0,
                    help="rounds of multi-bar collocation detection (0 = bars only)")
    ap.add_argument("--min-doc-freq", type=int, default=10)
    ap.add_argument("--passes", type=int, default=30)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    plotting.style()
    import matplotlib.pyplot as plt

    cfg = tokenize.TokenizerConfig(collapse_repeats=not args.no_collapse, repeat_counts=args.repeat_counts,
                                   collocation_rounds=args.collocations, min_doc_freq=args.min_doc_freq,
                                   min_term_freq=args.min_doc_freq * 3)
    songs, index = billboard.load_corpus(args.data, args.index, args.quality)
    log.info("parsed %d songs, %d bars, %d with key changes",
             len(songs), sum(len(s.bars) for s in songs), sum(1 for s in songs if s.key_changes))

    def year_of(song):
        r = index.get(song.id)
        return int(r["chart_date"][:4]) if r and r.get("chart_date") else None

    # ---- 1. documents + split by song ------------------------------------------
    raw_docs = [tokenize.bargrams([b.chords for b in s.bars], cfg) for s in songs]   # song level
    ids = [s.id for s in songs]
    meta = [{"title": s.title, "artist": s.artist, "year": year_of(s), "metre": s.metre,
             "tonic": s.tonic} for s in songs]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(songs))
    n_held = len(songs) // 5
    held_idx, train_idx = sorted(perm[:n_held].tolist()), perm[n_held:].tolist()
    sel = lambda xs, ix: [xs[i] for i in ix]

    if args.doc_unit == "song":
        tr_docs, tr_ids, tr_meta = sel(raw_docs, train_idx), sel(ids, train_idx), sel(meta, train_idx)
    else:  # every labelled section of a training song is a document
        tr_docs, tr_ids, tr_meta = [], [], []
        for i in train_idx:
            for j, sec in enumerate(songs[i].sections):
                tr_docs.append(tokenize.bargrams([b.chords for b in sec.bars], cfg))
                tr_ids.append((songs[i].id, j))
                tr_meta.append({**meta[i], "label": sec.label})
    train = tokenize.build_corpus(tr_docs, tr_ids, tr_meta, cfg)
    # held-out coherence is always measured on whole held-out songs
    held = tokenize.build_corpus(sel(raw_docs, held_idx), sel(ids, held_idx),
                                 sel(meta, held_idx), cfg, phrasers=train.phrasers)
    dictionary = Dictionary(train.docs)
    held_docs = [[t for t in d if t in dictionary.token2id] for d in held.docs]
    held_docs = [d for d in held_docs if d]
    log.info("doc unit=%s: train docs %d (dropped %d empty), held-out songs %d, vocab %d",
             args.doc_unit, len(train.docs), train.dropped, len(held_docs), len(dictionary))

    stats = {
        "config": vars(args) | {"tokenizer": vars(cfg)},
        "songs": len(songs), "bars": sum(len(s.bars) for s in songs),
        "songs_with_key_change": sum(1 for s in songs if s.key_changes),
        "train_docs": len(train.docs), "train_dropped_empty": train.dropped,
        "heldout_docs": len(held_docs), "vocab": len(dictionary),
        "tokens_train": sum(len(d) for d in train.docs),
        "top_tokens": train.vocab.most_common(30),
    }

    # ---- 2. K sweep ----------------------------------------------------------
    ks = [int(k) for k in args.ks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = model.sweep(train.docs, held_docs, ks, seeds, dictionary, passes=args.passes)
    rows = []
    for k, ts in runs.items():
        st = model.stability(ts)
        rows.append({"k": k, "npmi_mean": np.mean([t.coherence for t in ts]),
                     "npmi_std": np.std([t.coherence for t in ts]),
                     "heldout_log_perplexity": np.mean([t.log_perplexity for t in ts]),
                     "stability_cosine": st["mean_matched_cosine"],
                     "frac_topics_stable": st["frac_topics_stable"]})
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(out / "k_sweep.csv", index=False)
    log.info("\n%s", sweep_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].errorbar(sweep_df.k, sweep_df.npmi_mean, yerr=sweep_df.npmi_std, color=plotting.SERIES[0],
                     marker="o", capsize=3)
    axes[0].set_title("Held-out topic coherence (NPMI)"); axes[0].set_xlabel("K")
    axes[1].plot(sweep_df.k, sweep_df.heldout_log_perplexity, color=plotting.SERIES[0], marker="o")
    axes[1].set_title("Held-out per-word log likelihood"); axes[1].set_xlabel("K")
    axes[2].plot(sweep_df.k, sweep_df.stability_cosine, color=plotting.SERIES[0], marker="o")
    axes[2].set_ylim(0, 1); axes[2].set_title("Seed stability (matched cosine)"); axes[2].set_xlabel("K")
    fig.tight_layout(); fig.savefig(out / "k_sweep.png", dpi=150); plt.close(fig)

    # choose K: best NPMI among K whose stability is within 0.05 of the max
    # Choose K by reproducibility: the fraction of topics that re-appear (cosine
    # >= 0.7) in every other seed. NPMI and held-out likelihood both keep
    # improving as K grows and topics degenerate to 2-3 tokens each, so they
    # cannot pick K on their own; NPMI breaks ties.
    chosen_k = args.k or int(sweep_df.sort_values(["frac_topics_stable", "npmi_mean"],
                                                  ascending=False).iloc[0].k)
    best = max(runs[chosen_k], key=lambda t: t.coherence)
    stats["chosen_k"] = chosen_k
    stats["chosen_seed"] = best.seed
    log.info("chosen K=%d seed=%d NPMI=%.3f", chosen_k, best.seed, best.coherence)

    # ---- 3. topic listing ------------------------------------------------------
    lda = best.lda
    # song-level topic proportions for every song (used for examples, years, trends)
    all_docs = []
    for d in raw_docs:
        for p in train.phrasers:
            d = p[d]
        all_docs.append([t for t in d if t in dictionary.token2id])
    song_keep = [i for i, d in enumerate(all_docs) if d]
    X_songs = evaluate.doc_topic_matrix(lda, dictionary, sel(all_docs, song_keep))
    share = evaluate.doc_topic_matrix(lda, dictionary, train.docs).mean(axis=0)
    order = np.argsort(-share)
    with open(out / "topics.md", "w") as f:
        f.write(f"# Billboard topics (K={chosen_k}, seed {best.seed})\n\n")
        f.write(f"Held-out mean NPMI {best.coherence:.3f}. Topics sorted by corpus share. ")
        f.write("Roots are relative to the local tonic (C = I). Token format: `onset,duration,root:quality` "
                "per chord, `---` joins chords within a bar, `___` joins collocated bars, `xN` = bar repeated N times.\n\n")
        for rank, t in enumerate(order, 1):
            f.write(f"## Topic {t}  (share {share[t]:.3f}, NPMI {best.per_topic_coherence[t]:.2f})\n\n")
            for w, p in lda.show_topic(int(t), topn=12):
                f.write(f"- `{w}`  {p:.3f}\n")
            # example songs (the corpus has duplicate entries, so dedupe on title)
            examples, seen = [], set()
            for i in np.argsort(-X_songs[:, t]):
                m = meta[song_keep[i]]
                key = (m["artist"].lower(), m["title"].lower())
                if key in seen:
                    continue
                seen.add(key)
                examples.append(f"{m['artist']} – {m['title']} ({m['year']}, {X_songs[i, t]:.2f})")
                if len(examples) == 5:
                    break
            f.write("\nExamples: " + "; ".join(examples) + "\n\n")

    # ---- 4. section classification -------------------------------------------
    sec_docs, sec_y, sec_groups = [], [], []
    for s in songs:
        for sec in s.sections:
            if sec.label not in TARGET_LABELS:
                continue
            d = tokenize.bargrams([b.chords for b in sec.bars], cfg)
            for p in train.phrasers:
                d = p[d]
            d = [t for t in d if t in dictionary.token2id]
            if not d:
                continue  # empty section: would only carry the prior
            sec_docs.append(d); sec_y.append(sec.label); sec_groups.append(s.id)
    n_sections_total = sum(1 for s in songs for sec in s.sections if sec.label in TARGET_LABELS)
    vc = [i for i, l in enumerate(sec_y) if l in ("verse", "chorus")]
    keep_rows = [r for r, i in enumerate(song_keep) if meta[i]["year"]]
    years = [meta[song_keep[r]]["year"] for r in keep_rows]

    # Downstream scores depend on which seed's model is used, so evaluate every
    # seed at the chosen K and report mean ± sd; detailed tables use `best`.
    per_seed = []
    reports = {}
    for t in runs[chosen_k]:
        X_sec_t = evaluate.doc_topic_matrix(t.lda, dictionary, sec_docs)
        r4 = evaluate.section_classification(X_sec_t, sec_y, sec_groups, TARGET_LABELS, model="rf")
        rvc = evaluate.section_classification(X_sec_t[vc], [sec_y[i] for i in vc], [sec_groups[i] for i in vc],
                                              ["verse", "chorus"], model="rf")
        X_songs_t = evaluate.doc_topic_matrix(t.lda, dictionary, sel(all_docs, song_keep))
        ry = evaluate.year_regression(X_songs_t[keep_rows], years, model="rf")
        per_seed.append({"seed": t.seed, "npmi": t.coherence, "section4_acc": r4.accuracy,
                         "section4_bal_acc": r4.balanced_accuracy, "section4_majority": r4.majority_baseline,
                         "verse_chorus_acc": rvc.accuracy, "verse_chorus_majority": rvc.majority_baseline,
                         "year_mae": ry.mae, "year_baseline_mae": ry.baseline_mae})
        if t.seed == best.seed:
            reports = {"rf": r4, "vc": rvc, "year": ry}
            X_sec = X_sec_t
    per_seed_df = pd.DataFrame(per_seed)
    per_seed_df.to_csv(out / "downstream_by_seed.csv", index=False)
    rep_rf, rep_vc, rep_year = reports["rf"], reports["vc"], reports["year"]
    rep_lr = evaluate.section_classification(X_sec, sec_y, sec_groups, TARGET_LABELS, model="lr")

    def ms(col):
        return f"{per_seed_df[col].mean():.3f} ± {per_seed_df[col].std():.3f}"

    with open(out / "section_classification.txt", "w") as f:
        f.write(f"{n_sections_total} labelled sections, {len(sec_docs)} non-empty after thresholding "
                f"({n_sections_total - len(sec_docs)} dropped)\n")
        f.write("Features: topic proportions of the section. CV: GroupKFold by song.\n\n")
        f.write(f"== Across {len(per_seed)} seeds at K={chosen_k} (mean ± sd over LDA seeds) ==\n")
        f.write(f"4-class accuracy      {ms('section4_acc')}   majority {ms('section4_majority')}\n")
        f.write(f"4-class balanced acc  {ms('section4_bal_acc')}\n")
        f.write(f"verse-vs-chorus acc   {ms('verse_chorus_acc')}   majority {ms('verse_chorus_majority')}\n\n")
        f.write(f"== Random forest, 4 classes (seed {best.seed}) ==\n" + rep_rf.text() + "\n\n")
        f.write(f"== Logistic regression, 4 classes (seed {best.seed}) ==\n" + rep_lr.text() + "\n\n")
        f.write(f"== Random forest, verse vs chorus (seed {best.seed}) ==\n" + rep_vc.text() + "\n")
    log.info("\n%s", rep_rf.text())

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cmn = rep_rf.confusion / rep_rf.confusion.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap=plotting.SEQ_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(4), TARGET_LABELS, rotation=30, ha="right"); ax.set_yticks(range(4), TARGET_LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({rep_rf.confusion[i, j]})", ha="center", va="center",
                    fontsize=8, color=plotting.INK if cmn[i, j] < 0.5 else plotting.SURFACE)
    ax.set_title(f"Section type from topic shares\nacc {rep_rf.accuracy:.2f} vs majority {rep_rf.majority_baseline:.2f}",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out / "section_confusion.png", dpi=150); plt.close(fig)

    # ---- 5. year regression ----------------------------------------------------
    X_all = X_songs[keep_rows]
    with open(out / "year_regression.txt", "w") as f:
        f.write("Chart year from song topic proportions, 5-fold CV, random forest.\n")
        f.write(f"Across {len(per_seed)} seeds: MAE {ms('year_mae')} years, predict-mean baseline "
                f"{ms('year_baseline_mae')}\n\n== seed {best.seed} ==\n" + rep_year.text() + "\n")
    log.info("\n%s", rep_year.text())

    # ---- 6. topic share by period with bootstrap CI ---------------------------
    periods = [(1958, 1964), (1965, 1970), (1971, 1976), (1977, 1982), (1983, 1991)]
    plabels = [f"{a}–{b}" for a, b in periods]
    yrs = np.array(years)
    trend = []
    for t in range(chosen_k):
        for (a, b), pl in zip(periods, plabels):
            m, lo, hi = evaluate.bootstrap_mean_ci(X_all[(yrs >= a) & (yrs <= b), t])
            trend.append({"topic": t, "period": pl, "mean": m, "lo": lo, "hi": hi,
                          "n": int(((yrs >= a) & (yrs <= b)).sum())})
    trend_df = pd.DataFrame(trend)
    trend_df.to_csv(out / "topic_share_by_period.csv", index=False)

    ncol = 4
    nrow = int(np.ceil(chosen_k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.3 * nrow), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, order):
        d = trend_df[trend_df.topic == t]
        x = np.arange(len(periods))
        ax.fill_between(x, d.lo, d.hi, color=plotting.SERIES[0], alpha=0.18, linewidth=0)
        ax.plot(x, d["mean"], color=plotting.SERIES[0], marker="o", markersize=4)
        top = lda.show_topic(int(t), topn=1)[0][0]
        ax.set_title(f"T{t}: {top[:34]}", fontsize=8)
        ax.set_xticks(x, plabels, rotation=45, ha="right", fontsize=7)
    for ax in axes.flat[chosen_k:]:
        ax.axis("off")
    fig.suptitle("Mean topic share by chart period (95% bootstrap CI)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "topic_share_by_period.png", dpi=150); plt.close(fig)

    stats["downstream_mean_over_seeds"] = per_seed_df.drop(columns="seed").mean().round(4).to_dict()
    stats["downstream_sd_over_seeds"] = per_seed_df.drop(columns="seed").std().round(4).to_dict()
    stats["heldout_npmi_best"] = best.coherence
    with open(out / "summary.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    lda.save(str(out / "lda.model"))
    dictionary.save(str(out / "dictionary.dict"))
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
