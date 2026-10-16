"""Replicate the Billboard topic analysis on a Chordonomicon subsample.

    python scripts/run_chordonomicon.py --csv chordonomicon_v2.csv --out results/chordonomicon \
        --n 40000

Chordonomicon has no durations or metre, so each chord is one unit-length
"bar" and tokens carry no rhythm. Keys are estimated (Krumhansl profiles)
before transposing. Metadata gives genre and release decade, so the
downstream tasks become: section type (verse/chorus/bridge), main genre,
and release decade, each against baselines.
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
from chordtopics import chordonomicon, tokenize, model, evaluate, plotting  # noqa: E402
from gensim.corpora import Dictionary  # noqa: E402

log = logging.getLogger("run_chordonomicon")
SECTION_LABELS = ["verse", "chorus", "bridge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", default=Path("results/chordonomicon"), type=Path)
    ap.add_argument("--n", type=int, default=40000, help="songs to sample (reservoir)")
    ap.add_argument("--ks", default="8,12,16,24,32")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--max-sections", type=int, default=20000)
    ap.add_argument("--collocations", type=int, default=1,
                    help="rounds of collocation detection; chords carry no rhythm here, so one round "
                         "turns single chords into two-chord progressions")
    ap.add_argument("--min-doc-frac", type=float, default=0.005, help="doc-frequency floor as a fraction of songs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    plotting.style()
    import matplotlib.pyplot as plt

    mdf = max(5, int(args.n * 0.8 * args.min_doc_frac))
    cfg = tokenize.TokenizerConfig(with_rhythm=False, collapse_repeats=True,
                                   collocation_rounds=args.collocations,
                                   min_doc_freq=mdf, min_term_freq=mdf * 3,
                                   collocation_min_count=mdf, collocation_threshold=0.1)
    songs = chordonomicon.load_corpus(args.csv, limit=args.n, seed=0, require_year=True)
    log.info("loaded %d songs; modes %s", len(songs), Counter(s.mode for s in songs))

    raw_docs = [tokenize.bargrams([[c] for _, ch in s.sections for c in ch], cfg) for s in songs]
    ids = [s.id for s in songs]
    meta = [{"year": s.release_year, "decade": s.decade, "genre": s.main_genre, "mode": s.mode}
            for s in songs]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(songs))
    n_held = len(songs) // 5
    held_ix, train_ix = sorted(perm[:n_held].tolist()), perm[n_held:].tolist()
    sel = lambda xs, ix: [xs[i] for i in ix]
    train = tokenize.build_corpus(sel(raw_docs, train_ix), sel(ids, train_ix), sel(meta, train_ix), cfg)
    held = tokenize.build_corpus(sel(raw_docs, held_ix), sel(ids, held_ix), sel(meta, held_ix), cfg,
                                 phrasers=train.phrasers)
    dictionary = Dictionary(train.docs)
    held_docs = [[t for t in d if t in dictionary.token2id] for d in held.docs]
    held_docs = [d for d in held_docs if d]
    log.info("train %d (dropped %d), held-out %d, vocab %d", len(train.docs), train.dropped,
             len(held_docs), len(dictionary))
    stats = {"config": vars(args) | {"tokenizer": vars(cfg)},
             "songs": len(songs), "train_docs": len(train.docs), "train_dropped_empty": train.dropped,
             "heldout_docs": len(held_docs), "vocab": len(dictionary),
             "top_tokens": train.vocab.most_common(30),
             "modes": dict(Counter(s.mode for s in songs)),
             "genres": Counter(s.main_genre for s in songs).most_common(15)}

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
                     "frac_topics_stable": st["frac_topics_stable"],
                     "frac_topics_real": st["frac_topics_real"]})
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

    # Choose K by reproducibility: the fraction of topics that re-appear (cosine
    # >= 0.7) in every other seed. NPMI and held-out likelihood both keep
    # improving as K grows and topics degenerate to 2-3 tokens each, so they
    # cannot pick K on their own; ties go to the smaller model.
    chosen_k = args.k or int(sweep_df.sort_values(["frac_topics_stable", "k"],
                                                  ascending=[False, True]).iloc[0].k)
    best = max(runs[chosen_k], key=lambda t: t.coherence)
    lda = best.lda
    stats.update(chosen_k=chosen_k, chosen_seed=best.seed, heldout_npmi=best.coherence)

    X_train = evaluate.doc_topic_matrix(lda, dictionary, train.docs)
    share = X_train.mean(axis=0)
    order = np.argsort(-share)
    genres = np.array([m["genre"] or "unknown" for m in train.meta])
    modes = np.array([m["mode"] for m in train.meta])
    with open(out / "topics.md", "w") as f:
        f.write(f"# Chordonomicon topics (K={chosen_k}, seed {best.seed}, {len(songs)} songs)\n\n")
        f.write(f"Held-out mean NPMI {best.coherence:.3f}. Roots relative to the estimated tonic (C = I); "
                "no rhythm information in this corpus. `___` joins collocated chords, `xN` = repeated.\n\n")
        for t in order:
            f.write(f"## Topic {t}  (share {share[t]:.3f}, NPMI {best.per_topic_coherence[t]:.2f})\n\n")
            for w, p in lda.show_topic(int(t), topn=12):
                f.write(f"- `{w}`  {p:.3f}\n")
            # which genres / modes over-index on this topic
            w = X_train[:, t]
            g_lift = {g: float(w[genres == g].mean() / share[t]) for g in set(genres)
                      if (genres == g).sum() >= 200}
            top_g = sorted(g_lift.items(), key=lambda x: -x[1])[:4]
            m_lift = {m: float(w[modes == m].mean() / share[t]) for m in ("major", "minor")}
            f.write("\nGenre lift: " + ", ".join(f"{g} {l:.2f}" for g, l in top_g))
            f.write(f"  |  mode lift: major {m_lift['major']:.2f}, minor {m_lift['minor']:.2f}\n\n")

    # ---- downstream: section type (GroupKFold by song), genre, decade -------
    sec_docs, sec_y, sec_groups = [], [], []
    for s in songs:
        for lab, ch in s.sections:
            if lab not in SECTION_LABELS:
                continue
            d = tokenize.bargrams([[c] for c in ch], cfg)
            for p in train.phrasers:
                d = p[d]
            d = [t for t in d if t in dictionary.token2id]
            if d:
                sec_docs.append(d); sec_y.append(lab); sec_groups.append(s.id)
    if len(sec_docs) > args.max_sections:
        pick = sorted(rng.choice(len(sec_docs), args.max_sections, replace=False))
        sec_docs, sec_y, sec_groups = sel(sec_docs, pick), sel(sec_y, pick), sel(sec_groups, pick)
    vc = [i for i, l in enumerate(sec_y) if l != "bridge"]
    all_docs = []
    for d in raw_docs:
        for p in train.phrasers:
            d = p[d]
        all_docs.append([t for t in d if t in dictionary.token2id])
    keep = [i for i, d in enumerate(all_docs) if d]
    years = np.array([meta[i]["year"] for i in keep], dtype=float)
    genre_all = np.array([meta[i]["genre"] or "unknown" for i in keep])
    top_genres = [str(g) for g, c in Counter(genre_all.tolist()).most_common(9) if g != "unknown"][:8]
    gi = [i for i, g in enumerate(genre_all) if g in top_genres]

    # evaluate every seed at the chosen K; detailed tables use the best seed
    per_seed, reports = [], {}
    for t in runs[chosen_k]:
        X_sec_t = evaluate.doc_topic_matrix(t.lda, dictionary, sec_docs)
        r_sec = evaluate.section_classification(X_sec_t, sec_y, sec_groups, SECTION_LABELS, model="rf")
        r_vc = evaluate.section_classification(X_sec_t[vc], sel(sec_y, vc), sel(sec_groups, vc),
                                               ["verse", "chorus"], model="rf")
        X_all_t = evaluate.doc_topic_matrix(t.lda, dictionary, sel(all_docs, keep))
        r_year = evaluate.year_regression(X_all_t, years, model="rf")
        r_genre = evaluate.section_classification(X_all_t[gi], genre_all[gi].tolist(), list(range(len(gi))),
                                                  top_genres, model="rf")
        per_seed.append({"seed": t.seed, "npmi": t.coherence,
                         "section3_acc": r_sec.accuracy, "section3_majority": r_sec.majority_baseline,
                         "verse_chorus_acc": r_vc.accuracy, "verse_chorus_majority": r_vc.majority_baseline,
                         "year_mae": r_year.mae, "year_baseline_mae": r_year.baseline_mae,
                         "genre_acc": r_genre.accuracy, "genre_bal_acc": r_genre.balanced_accuracy,
                         "genre_majority": r_genre.majority_baseline})
        if t.seed == best.seed:
            reports = {"sec": r_sec, "vc": r_vc, "year": r_year, "genre": r_genre}
            X_all = X_all_t
    per_seed_df = pd.DataFrame(per_seed)
    per_seed_df.to_csv(out / "downstream_by_seed.csv", index=False)
    rep_sec, rep_vc, rep_year, rep_genre = reports["sec"], reports["vc"], reports["year"], reports["genre"]

    def ms(col):
        return f"{per_seed_df[col].mean():.3f} ± {per_seed_df[col].std():.3f}"

    with open(out / "downstream.txt", "w") as f:
        f.write(f"== Across {len(per_seed)} seeds at K={chosen_k} (mean ± sd over LDA seeds) ==\n")
        f.write(f"section (3-class) acc {ms('section3_acc')}   majority {ms('section3_majority')}\n")
        f.write(f"verse-vs-chorus acc   {ms('verse_chorus_acc')}   majority {ms('verse_chorus_majority')}\n")
        f.write(f"year MAE              {ms('year_mae')}   predict-mean {ms('year_baseline_mae')}\n")
        f.write(f"genre acc             {ms('genre_acc')}   majority {ms('genre_majority')}   "
                f"balanced {ms('genre_bal_acc')}\n\n")
        f.write(f"== Section type ({len(sec_docs)} sections, GroupKFold by song, seed {best.seed}) ==\n"
                + rep_sec.text() + "\n\n")
        f.write("== Verse vs chorus ==\n" + rep_vc.text() + "\n\n")
        f.write("== Release year (random forest, 5-fold) ==\n" + rep_year.text() + "\n\n")
        f.write(f"== Main genre, top {len(top_genres)} genres ==\n" + rep_genre.text() + "\n")
    log.info("\n%s\n%s\n%s", rep_sec.text(), rep_year.text(), rep_genre.text())

    # ---- topic share by decade with CI -----------------------------------------
    decades = [1960, 1970, 1980, 1990, 2000, 2010, 2020]
    dec = (years // 10 * 10).astype(int)
    trend = []
    for t in range(chosen_k):
        for d in decades:
            m, lo, hi = evaluate.bootstrap_mean_ci(X_all[dec == d, t])
            trend.append({"topic": t, "decade": d, "mean": m, "lo": lo, "hi": hi, "n": int((dec == d).sum())})
    trend_df = pd.DataFrame(trend)
    trend_df.to_csv(out / "topic_share_by_decade.csv", index=False)
    ncol = 4
    nrow = int(np.ceil(chosen_k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.3 * nrow), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, order):
        d = trend_df[trend_df.topic == t]
        ax.fill_between(d.decade, d.lo, d.hi, color=plotting.SERIES[0], alpha=0.18, linewidth=0)
        ax.plot(d.decade, d["mean"], color=plotting.SERIES[0], marker="o", markersize=4)
        ax.set_title(f"T{t}: {lda.show_topic(int(t), topn=1)[0][0][:34]}", fontsize=8)
    for ax in axes.flat[chosen_k:]:
        ax.axis("off")
    fig.suptitle("Mean topic share by release decade (95% bootstrap CI)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "topic_share_by_decade.png", dpi=150); plt.close(fig)

    # genre x topic heatmap (lift)
    G = np.array([[X_all[genre_all == g, t].mean() / X_all[:, t].mean() for t in order] for g in top_genres])
    fig, ax = plt.subplots(figsize=(max(6, chosen_k * 0.45), 3.6))
    im = ax.imshow(np.log2(G), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(chosen_k), [f"T{t}" for t in order], fontsize=7)
    ax.set_yticks(range(len(top_genres)), top_genres); ax.grid(False)
    fig.colorbar(im, ax=ax, label="log2 lift vs corpus share", shrink=0.8)
    ax.set_title("Which genres over-use which topics")
    fig.tight_layout(); fig.savefig(out / "genre_topic_lift.png", dpi=150); plt.close(fig)

    stats["downstream_mean_over_seeds"] = per_seed_df.drop(columns="seed").mean().round(4).to_dict()
    stats["downstream_sd_over_seeds"] = per_seed_df.drop(columns="seed").std().round(4).to_dict()
    stats["genre_classes"] = top_genres
    with open(out / "summary.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    lda.save(str(out / "lda.model")); dictionary.save(str(out / "dictionary.dict"))
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
