"""Grid over tokenizer / LDA settings, reporting held-out coherence, held-out
likelihood and seed stability. Used to choose the defaults in run_billboard.py.

    python scripts/experiment_tokenization.py --data ... --index ... > results/billboard/tokenizer_grid.txt
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chordtopics import billboard, tokenize, model  # noqa: E402
from gensim.corpora import Dictionary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--ks", default="8,16")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--passes", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    songs, _ = billboard.load_corpus(args.data, args.index, "coarse")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(songs))
    n_held = len(songs) // 5
    held_idx, train_idx = sorted(perm[:n_held].tolist()), perm[n_held:].tolist()

    grid = {
        "with_rhythm": [True, False],
        "collocation_rounds": [0, 1, 2],
        "min_doc_freq": [3, 10],
        "alpha": ["auto", "symmetric"],
    }
    ks = [int(k) for k in args.ks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for vals in itertools.product(*grid.values()):
        g = dict(zip(grid.keys(), vals))
        cfg = tokenize.TokenizerConfig(with_rhythm=g["with_rhythm"], collapse_repeats=True,
                                       collocation_rounds=g["collocation_rounds"],
                                       min_doc_freq=g["min_doc_freq"], min_term_freq=g["min_doc_freq"] * 3)
        raw = [tokenize.bargrams([b.chords for b in s.bars], cfg) for s in songs]
        train = tokenize.build_corpus([raw[i] for i in train_idx], train_idx, [{}] * len(train_idx), cfg)
        held = tokenize.build_corpus([raw[i] for i in held_idx], held_idx, [{}] * len(held_idx), cfg,
                                     phrasers=train.phrasers)
        dictionary = Dictionary(train.docs)
        held_docs = [[t for t in d if t in dictionary.token2id] for d in held.docs]
        held_docs = [d for d in held_docs if d]
        npmi = model.NPMI(held_docs)
        for k in ks:
            runs = [model.train(train.docs, held_docs, k, s, dictionary, passes=args.passes,
                                npmi=npmi, alpha=g["alpha"], topn=10) for s in seeds]
            top5 = [float(np.nanmean([npmi.topic_coherence(model.top_words(t.lda, i, 5)) for i in range(k)]))
                    for t in runs]
            # how many top-10 pairs ever co-occur in held-out songs at all
            seen = []
            for t in runs:
                pairs = 0; hit = 0
                for i in range(k):
                    ws = model.top_words(t.lda, i, 10)
                    for a, b in itertools.combinations(ws, 2):
                        pairs += 1; hit += npmi.pair[frozenset((a, b))] > 0
                seen.append(hit / pairs)
            st = model.stability(runs)
            rows.append({**g, "k": k, "vocab": len(dictionary), "tokens_per_doc": np.mean([len(d) for d in train.docs]),
                         "npmi10": np.mean([t.coherence for t in runs]), "npmi5": np.mean(top5),
                         "pairs_seen": np.mean(seen), "heldout_ll": np.mean([t.log_perplexity for t in runs]),
                         "stability": st["mean_matched_cosine"]})
            print(pd.DataFrame(rows[-1:]).to_string(index=False, header=False), flush=True)
    df = pd.DataFrame(rows)
    print("\n" + df.sort_values("npmi10", ascending=False).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
