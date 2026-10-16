"""LDA training and evaluation.

Coherence is measured as mean pairwise NPMI of each topic's top words,
computed on *held-out* songs that the model never saw, using sliding
windows over the bar sequence. Measuring PMI on the training corpus
rewards any model that is pushed toward high PMI, which was the circularity
in the original "Quad-Reg vs LDA" plot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)


@dataclass
class Topics:
    k: int
    seed: int
    lda: LdaModel
    dictionary: Dictionary
    coherence: float            # held-out mean NPMI
    per_topic_coherence: list[float]
    log_perplexity: float       # held-out
    share: np.ndarray | None = None   # mean topic proportion over (a sample of) training docs


def top_words(lda: LdaModel, topic: int, n: int = 10) -> list[str]:
    return [w for w, _ in lda.show_topic(topic, topn=n)]


class NPMI:
    """Sliding-window co-occurrence statistics over a list of token sequences."""

    def __init__(self, docs: list[list[str]], window: int | None = None):
        """window=None uses whole documents as the co-occurrence unit (Newman
        et al. 2010); an int gives a sliding window over the bar sequence."""
        from collections import Counter
        self.n_windows = 0
        self.uni: Counter = Counter()
        self.pair: Counter = Counter()
        for d in docs:
            if window is None or len(d) <= window:
                windows = [d]
            else:
                windows = [d[i:i + window] for i in range(len(d) - window + 1)]
            for w in windows:
                s = set(w)
                self.n_windows += 1
                self.uni.update(s)
                self.pair.update(frozenset(p) for p in combinations(sorted(s), 2))

    def npmi(self, a: str, b: str, eps: float = 1e-12) -> float:
        pa = self.uni[a] / self.n_windows
        pb = self.uni[b] / self.n_windows
        pab = self.pair[frozenset((a, b))] / self.n_windows
        if pab == 0 or pa == 0 or pb == 0:
            return -1.0
        return np.log(pab / (pa * pb) + eps) / -np.log(pab + eps)

    def topic_coherence(self, words: list[str]) -> float:
        pairs = list(combinations(words, 2))
        return float(np.mean([self.npmi(a, b) for a, b in pairs])) if pairs else float("nan")


def train(train_docs: list[list[str]], heldout_docs: list[list[str]],
          k: int, seed: int, dictionary: Dictionary | None = None,
          passes: int = 20, iterations: int = 200, topn: int = 10,
          npmi: NPMI | None = None, alpha="auto", eta="auto") -> Topics:
    dictionary = dictionary or Dictionary(train_docs)
    bow = [dictionary.doc2bow(d) for d in train_docs]
    lda = LdaModel(bow, num_topics=k, id2word=dictionary, random_state=seed,
                   passes=passes, iterations=iterations, alpha=alpha, eta=eta,
                   eval_every=None, chunksize=256)
    npmi = npmi or NPMI(heldout_docs)
    per_topic = [npmi.topic_coherence(top_words(lda, t, topn)) for t in range(k)]
    held_bow = [dictionary.doc2bow(d) for d in heldout_docs]
    held_bow = [b for b in held_bow if b]
    lp = lda.log_perplexity(held_bow) if held_bow else float("nan")
    # corpus share of each topic, from a fixed sample of training documents
    sample = bow if len(bow) <= 2000 else [bow[i] for i in np.random.default_rng(0).choice(len(bow), 2000, replace=False)]
    share = np.zeros(k)
    for b in sample:
        for t, p in lda.get_document_topics(b, minimum_probability=0.0):
            share[t] += p
    share /= max(len(sample), 1)
    return Topics(k, seed, lda, dictionary, float(np.nanmean(per_topic)), per_topic, lp, share)


def topic_word_matrix(t: Topics) -> np.ndarray:
    return t.lda.get_topics()  # k x V, rows sum to 1


def match_topics(a: Topics, b: Topics) -> tuple[np.ndarray, float]:
    """Hungarian matching of topics between two runs by cosine similarity.
    Returns (matched similarities, mean similarity)."""
    A, B = topic_word_matrix(a), topic_word_matrix(b)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    sim = A @ B.T
    r, c = linear_sum_assignment(-sim)
    return sim[r, c], float(sim[r, c].mean())


def stability(runs: list[Topics], min_share_factor: float = 0.25, min_effective_vocab: float = 3.0) -> dict:
    """Mean matched cosine similarity over all pairs of seeds, and the
    fraction of topics that (a) carry at least `min_share_factor`/K of the
    corpus in the reference run, (b) spread over at least `min_effective_vocab`
    tokens and (c) find a match >= 0.7 in every other run. Near-empty and
    one-chord topics are excluded because they match each other trivially and
    would otherwise reward degenerate large-K models."""
    k = runs[0].k
    pair_means, per_topic_min = [], np.ones(k)
    for i, j in combinations(range(len(runs)), 2):
        sims, m = match_topics(runs[i], runs[j])
        pair_means.append(m)
        if i == 0:
            per_topic_min = np.minimum(per_topic_min, sims)
    real = runs[0].share >= min_share_factor / k if runs[0].share is not None else np.ones(k, bool)
    # a topic that is effectively one or two chords is not a "style": require
    # an effective vocabulary (exp entropy of the word distribution) >= 3
    real &= effective_vocabulary(runs[0]) >= min_effective_vocab
    return {"mean_matched_cosine": float(np.mean(pair_means)) if pair_means else float("nan"),
            "frac_topics_stable": float(np.mean((per_topic_min >= 0.7) & real)),
            "frac_topics_real": float(real.mean()),
            "per_topic_min_sim": per_topic_min.tolist()}


def effective_vocabulary(t: Topics) -> np.ndarray:
    """exp(entropy) of each topic's word distribution: how many tokens it is 'about'."""
    W = topic_word_matrix(t)
    return np.exp(-(W * np.log(W + 1e-12)).sum(axis=1))


def sweep(train_docs, heldout_docs, ks: list[int], seeds: list[int],
          dictionary: Dictionary, **kw) -> dict[int, list[Topics]]:
    npmi = NPMI(heldout_docs)
    out = {}
    for k in ks:
        out[k] = [train(train_docs, heldout_docs, k, s, dictionary, npmi=npmi, **kw) for s in seeds]
        log.info("K=%d  NPMI=%.3f±%.3f  stability=%.2f", k,
                 np.mean([t.coherence for t in out[k]]), np.std([t.coherence for t in out[k]]),
                 stability(out[k])["mean_matched_cosine"])
    return out
