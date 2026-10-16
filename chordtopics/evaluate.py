"""Downstream evaluations with the baselines the original project lacked.

* Section-type classification: topic proportions of a song section ->
  verse / chorus / bridge / prechorus. Cross-validated with GroupKFold so
  sections of one song never straddle train and test. Reported next to the
  majority-class baseline, with a confusion matrix whose axes are labelled
  from the same `labels` list used to build it.
* Year prediction: regression, reported as mean absolute error in years
  against the predict-the-training-mean baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, mean_absolute_error, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def doc_topic_matrix(lda, dictionary, docs: list[list[str]]) -> np.ndarray:
    k = lda.num_topics
    X = np.zeros((len(docs), k))
    for i, d in enumerate(docs):
        for t, p in lda.get_document_topics(dictionary.doc2bow(d), minimum_probability=0.0):
            X[i, t] = p
    return X


@dataclass
class ClassifierReport:
    labels: list[str]
    accuracy: float
    accuracy_std: float
    balanced_accuracy: float
    majority_baseline: float
    stratified_baseline: float
    confusion: np.ndarray      # rows = true, cols = predicted
    n: int

    def text(self) -> str:
        lines = [f"n={self.n}  classes={self.labels}",
                 f"accuracy          {self.accuracy:.3f} ± {self.accuracy_std:.3f}",
                 f"balanced accuracy {self.balanced_accuracy:.3f}",
                 f"majority baseline {self.majority_baseline:.3f}",
                 f"stratified-random baseline {self.stratified_baseline:.3f}",
                 "confusion (rows=true, cols=pred):"]
        w = max(len(l) for l in self.labels)
        lines.append(" " * (w + 1) + " ".join(f"{l[:6]:>6}" for l in self.labels))
        for l, row in zip(self.labels, self.confusion):
            lines.append(f"{l:>{w}} " + " ".join(f"{v:6d}" for v in row))
        return "\n".join(lines)


def section_classification(X: np.ndarray, y: list[str], groups: list,
                           labels: list[str], seed: int = 0, n_splits: int = 5,
                           model: str = "rf") -> ClassifierReport:
    y = np.asarray(y)
    groups = np.asarray(groups)
    gkf = GroupKFold(n_splits=n_splits)
    accs, cm = [], np.zeros((len(labels), len(labels)), dtype=int)
    y_true_all, y_pred_all = [], []
    maj, strat = [], []
    for tr, te in gkf.split(X, y, groups):
        if model == "rf":
            clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                         class_weight="balanced", min_samples_leaf=3)
        else:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append((pred == y[te]).mean())
        cm += confusion_matrix(y[te], pred, labels=labels)
        y_true_all.extend(y[te]); y_pred_all.extend(pred)
        maj.append(DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr]).score(X[te], y[te]))
        strat.append(DummyClassifier(strategy="stratified", random_state=seed).fit(X[tr], y[tr]).score(X[te], y[te]))
    return ClassifierReport(labels, float(np.mean(accs)), float(np.std(accs)),
                            float(balanced_accuracy_score(y_true_all, y_pred_all)),
                            float(np.mean(maj)), float(np.mean(strat)), cm, len(y))


@dataclass
class RegressionReport:
    mae: float
    mae_std: float
    baseline_mae: float
    n: int

    def text(self) -> str:
        return (f"n={self.n}\nMAE (years)        {self.mae:.2f} ± {self.mae_std:.2f}\n"
                f"predict-mean MAE   {self.baseline_mae:.2f}")


def year_regression(X: np.ndarray, years: list[float], seed: int = 0,
                    n_splits: int = 5, model: str = "rf") -> RegressionReport:
    y = np.asarray(years, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    maes, base = [], []
    for tr, te in kf.split(X):
        reg = (RandomForestRegressor(n_estimators=300, random_state=seed, min_samples_leaf=3, n_jobs=-1)
               if model == "rf" else Ridge(alpha=1.0))
        reg.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[te], reg.predict(X[te])))
        dummy = DummyRegressor(strategy="mean").fit(X[tr], y[tr])
        base.append(mean_absolute_error(y[te], dummy.predict(X[te])))
    return RegressionReport(float(np.mean(maes)), float(np.std(maes)), float(np.mean(base)), len(y))


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean with percentile bootstrap CI, for topic share per year/decade."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
