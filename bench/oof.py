"""Out-of-fold scores under the same claim-grouped split the published numbers use.

The published AUROCs come from `cross_val_score(..., scoring="roc_auc").mean()`, which is the
MEAN OF PER-FOLD AUROCs. A bootstrap needs one score per trial instead, so we refit the same
pipeline under the same splitter with `cross_val_predict` and keep the probabilities.

Those two numbers are not identical. The mean of per-fold AUROCs and the pooled AUROC over all
out-of-fold scores differ whenever folds have different class balance, which grouped folds
always do. `check()` reports both so the gap is visible rather than assumed away: the per-fold
mean is what reproduces the paper, the pooled figure is what the interval is built on.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline

N_SPLITS = 5


def text_pipeline():
    """The reader used throughout the paper: word and bigram counts, rarer terms up-weighted."""
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                         LogisticRegression(max_iter=2000))


def oof_scores(X, y, groups, pipeline=None, n_splits: int = N_SPLITS):
    """One out-of-fold probability per trial, from folds that hold out whole claims."""
    pipeline = pipeline or text_pipeline()
    proba = cross_val_predict(pipeline, X, y, cv=GroupKFold(n_splits=n_splits),
                              groups=groups, method="predict_proba")
    return np.asarray(proba)[:, 1]


def per_fold_mean(X, y, groups, pipeline=None, n_splits: int = N_SPLITS) -> float:
    """Reproduce the published statistic exactly: mean of per-fold AUROCs."""
    pipeline = pipeline or text_pipeline()
    return float(cross_val_score(pipeline, X, y, cv=GroupKFold(n_splits=n_splits),
                                 groups=groups, scoring="roc_auc").mean())


def check(X, y, groups, published: float | None = None, pipeline=None,
          n_splits: int = N_SPLITS) -> dict:
    """Both statistics side by side, plus the gap against the published figure."""
    y = np.asarray(y)
    s = oof_scores(X, y, groups, pipeline, n_splits)
    fold_mean = per_fold_mean(X, y, groups, pipeline, n_splits)
    pooled = float(roc_auc_score(y, s))
    out = {"per_fold_mean": fold_mean, "pooled_oof": pooled,
           "gap_pooled_minus_fold": pooled - fold_mean, "scores": s}
    if published is not None:
        out["published"] = float(published)
        out["delta_vs_published"] = fold_mean - float(published)
    return out
