"""Claim-clustered bootstrap confidence intervals for the AUROC statistic the paper reports.

WHY CLAIMS AND NOT TRIALS
  Every trial from one claim shares the question, its wording, and its base rate. Resampling
  trials independently treats correlated observations as independent evidence and returns
  intervals far too narrow — on sycophancy the effective sample size is 32 claims, not 318
  trials. We draw claims with replacement and take every trial belonging to each drawn claim.

WHICH STATISTIC, AND WHY IT IS THE PER-FOLD MEAN
  Two estimators are available from a grouped cross-validation and they are not the same
  number. The paper reports the MEAN OF PER-FOLD AUROCs, which is what cross_val_score
  returns. Pooling all out-of-fold scores into one AUROC gives a different figure, lower by
  up to 0.058 on the weak-signal conditions.

  That gap is not per-fold bias. Each fold fits its own logistic regression and those five
  models share no calibration: on the few-shot run their scores span [0.09, 0.20], [0.07,
  0.16], [0.07, 0.18], [0.08, 0.16] and [0.09, 0.20] against fold positive rates from 0.06 to
  0.19. Pooling ranks one fold's maximum against another's midpoint, which injects noise that
  is a fixed cost — negligible where the signal is strong, dominant where it is weak.
  Rank-normalising within fold before pooling closes the gap to ±0.01 with no consistent sign,
  which is the diagnostic that settles it. So the pooled figure is the biased one here, and
  everything below bootstraps the per-fold mean.

DEGENERATE DRAWS
  A resample must yield a usable AUROC in every fold, since the reported statistic is a mean
  over five of them. Draws that leave any fold single-class are discarded and redrawn, and the
  rate is returned rather than logged: with five folds the redraw probability is roughly five
  times the single-fold rate, and above ~20% the resampling is no longer covering the space
  evenly and the interval should be described as approximate.

REPRODUCIBILITY
  GroupKFold is deterministic and unshuffled, so claim-to-fold assignment is a pure function
  of the claim ordering in the trials file. Every stochastic step takes an explicit seed and
  returns it in its result dict.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def fast_auc(y, s):
    """Mann-Whitney AUROC with tie-averaged ranks.

    Matches sklearn to machine precision (max deviation 2e-16 over tie-heavy inputs) and runs
    ~70x faster. That matters because one interval evaluates this 25,000 times and the full
    sweep runs roughly fifty intervals.
    """
    y = np.asarray(y, dtype=bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="stable")
    sv = np.asarray(s)[order]
    r = np.empty(len(s), dtype=float)
    r[order] = np.arange(1, len(s) + 1)
    i = 0                                   # average ranks within ties, as sklearn does
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

SEED = 0
N_BOOT = 5000
N_SPLITS = 5
REDRAW_MAX = 0.20
"""Above this rejection rate the interval is NOT REPORTED, only the point estimate.

Rejection is not neutral. Accepted draws are those that keep all five folds populated, which
correlates with the class balance being estimated. Measured directly: at 0-8% rejection the
accepted subsample's positive rate matches the full data to within 0.0005, but at 49%
rejection few-shot/Gemma's accepted rate runs 0.0598 against a true 0.0550 — 8.7% high in
relative terms. A caveat does not repair that, so the interval is withheld and the cell reads
'not estimable at this sample size'.
"""


def fold_assignments(y, claim_ids, n_splits: int = N_SPLITS) -> np.ndarray:
    """Fold id per trial under the same splitter the published numbers use."""
    y = np.asarray(y)
    claim_ids = np.asarray(claim_ids)
    fold = np.full(len(y), -1, dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y,
                                                                    claim_ids)):
        fold[te] = k
    assert (fold >= 0).all(), "GroupKFold left trials unassigned"
    return fold


def _per_fold_mean(y, s, fold, n_splits):
    """Mean of per-fold AUROCs, or None if any fold is unusable."""
    vals = []
    for k in range(n_splits):
        m = fold == k
        if not m.any():
            return None
        yk = y[m]
        if yk.all() or not yk.any():
            return None
        a = fast_auc(yk, s[m])
        if a is None:
            return None
        vals.append(a)
    return float(np.mean(vals))


def _claim_index(claim_ids):
    claims, inv = np.unique(claim_ids, return_inverse=True)
    return claims, [np.flatnonzero(inv == i) for i in range(len(claims))]


def bootstrap_auroc_ci(y_true, y_score, claim_ids, fold_ids=None, n_boot: int = N_BOOT,
                       alpha: float = 0.05, seed: int = SEED,
                       n_splits: int = N_SPLITS) -> dict:
    """Percentile CI on the per-fold-mean AUROC, resampling claims with replacement.

    y_true      binary labels, one per trial
    y_score     out-of-fold scores, one per trial (higher = more likely positive)
    claim_ids   cluster id per trial; trials sharing an id move together
    fold_ids    fold id per trial; computed from claim_ids if not supplied
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=float)
    claim_ids = np.asarray(claim_ids)
    if not (len(y_true) == len(y_score) == len(claim_ids)):
        raise ValueError("y_true, y_score and claim_ids must be the same length")
    fold = fold_assignments(y_true, claim_ids, n_splits) if fold_ids is None \
        else np.asarray(fold_ids, dtype=int)

    point = _per_fold_mean(y_true, y_score, fold, n_splits)
    if point is None:
        raise ValueError("a fold is single-class on the observed data; AUROC undefined")

    claims, by_claim = _claim_index(claim_ids)
    rng = np.random.default_rng(seed)

    boots, redraws, attempts = [], 0, 0
    max_attempts = n_boot * 200
    while len(boots) < n_boot and attempts < max_attempts:
        attempts += 1
        pick = rng.integers(0, len(claims), len(claims))
        idx = np.concatenate([by_claim[p] for p in pick])
        v = _per_fold_mean(y_true[idx], y_score[idx], fold[idx], n_splits)
        if v is None:
            redraws += 1
            continue
        boots.append(v)

    if len(boots) < n_boot:
        raise RuntimeError(
            f"only {len(boots)}/{n_boot} usable draws after {attempts} attempts; "
            f"the positive class ({int(y_true.sum())} of {len(y_true)}) cannot support "
            f"a five-fold statistic")

    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    rate = redraws / attempts if attempts else 0.0
    return {
        "point": float(point), "lo": float(lo), "hi": float(hi),
        "n_claims": int(len(claims)), "n_trials": int(len(y_true)),
        "n_positives": int(y_true.sum()), "n_boot": int(n_boot),
        "redraws": int(redraws), "redraw_rate": float(rate),
        "estimable": bool(rate <= REDRAW_MAX), "seed": int(seed),
        "estimator": "per_fold_mean",
    }


def permute_labels_by_claim(y_true, claim_ids, rng) -> np.ndarray:
    """Reassign whole label-blocks between claims of equal size.

    NOT USED FOR THE FLOOR, and kept only to document why. The idea was to preserve
    within-claim label clustering that a trial-level permutation would destroy. Measuring it
    found none DETECTABLE: across every target the spread of per-claim positive rates is at or
    below what random assignment gives, and on sycophancy it is under-dispersed (0.071 against
    an expected 0.158) because the labels follow a fixed positional pattern inside each claim —
    every claim reads 1111100000.

    That test has limited power. With 22 to 92 positives spread over 200 claims it would not
    detect weak over-dispersion, and claims differing in how swayable they are is a plausible
    mechanism. The direction of the error is safe: residual clustering would make this floor
    band slightly too NARROW, which makes observed values look more distinguishable from chance
    rather than less. Since the paper's claim is a null, that biases against our own
    conclusion.

    With identical blocks, permuting them between claims is close to a no-op: it left the
    answer-target floor at 0.97 instead of 0.5, because it never broke the score-label
    association it was supposed to destroy.
    """
    y_true = np.asarray(y_true).astype(bool)
    claim_ids = np.asarray(claim_ids)
    _, by_claim = _claim_index(claim_ids)
    out = np.empty_like(y_true)
    sizes = np.array([len(ix) for ix in by_claim])
    for size in np.unique(sizes):
        which = np.flatnonzero(sizes == size)
        order = rng.permutation(len(which))
        for dst, src in zip(which, which[order]):
            out[by_claim[dst]] = y_true[by_claim[src]]
    return out


def shuffled_floor_ci(y_true, y_score, claim_ids, fold_ids=None, n_boot: int = N_BOOT,
                      alpha: float = 0.05, seed: int = SEED,
                      n_splits: int = N_SPLITS) -> dict:
    """The empirical floor, built with the same estimator as the observed values.

    A floor computed as a pooled AUROC cannot be compared against a per-fold-mean observation:
    the two statistics differ by up to 0.058 on exactly the conditions where the comparison
    matters. This permutes label blocks between claims, keeps fold assignments fixed, and
    recomputes the per-fold mean — so observed and floor differ only in whether the readout
    carries anything.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=float)
    claim_ids = np.asarray(claim_ids)
    fold = fold_assignments(y_true, claim_ids, n_splits) if fold_ids is None \
        else np.asarray(fold_ids, dtype=int)
    rng = np.random.default_rng(seed)

    draws, redraws, attempts = [], 0, 0
    max_attempts = n_boot * 200
    while len(draws) < n_boot and attempts < max_attempts:
        attempts += 1
        # trial-level, because the measurement above found no claim-level label clustering
        # to preserve. Fold assignments stay fixed, so the only thing destroyed is the
        # association between a readout's score and its own label.
        yp = rng.permutation(y_true)
        v = _per_fold_mean(yp, y_score, fold, n_splits)
        if v is None:
            redraws += 1
            continue
        draws.append(v)

    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    rate = redraws / attempts if attempts else 0.0
    return {"mean": float(draws.mean()), "lo": float(lo), "hi": float(hi),
            "n_boot": int(len(draws)), "redraws": int(redraws), "redraw_rate": float(rate),
            "seed": int(seed), "estimator": "per_fold_mean"}


def paired_difference_ci(y_true, score_a, score_b, claim_ids, fold_ids=None,
                         n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = SEED,
                         n_splits: int = N_SPLITS) -> dict:
    """CI on AUROC(a) - AUROC(b), drawing claims once and scoring both readers on that draw.

    Two marginal intervals cannot answer whether two readers differ: they are computed on the
    same trials, so their errors are correlated. Differencing inside the draw keeps that
    correlation, which is why a paired interval is narrower than the overlap of two separate
    ones.
    """
    y_true = np.asarray(y_true).astype(bool)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    claim_ids = np.asarray(claim_ids)
    fold = fold_assignments(y_true, claim_ids, n_splits) if fold_ids is None \
        else np.asarray(fold_ids, dtype=int)

    pa = _per_fold_mean(y_true, score_a, fold, n_splits)
    pb = _per_fold_mean(y_true, score_b, fold, n_splits)
    if pa is None or pb is None:
        raise ValueError("a fold is single-class on the observed data")

    claims, by_claim = _claim_index(claim_ids)
    rng = np.random.default_rng(seed)

    diffs, redraws, attempts = [], 0, 0
    max_attempts = n_boot * 200
    while len(diffs) < n_boot and attempts < max_attempts:
        attempts += 1
        pick = rng.integers(0, len(claims), len(claims))
        idx = np.concatenate([by_claim[p] for p in pick])
        va = _per_fold_mean(y_true[idx], score_a[idx], fold[idx], n_splits)
        vb = _per_fold_mean(y_true[idx], score_b[idx], fold[idx], n_splits)
        if va is None or vb is None:
            redraws += 1
            continue
        diffs.append(va - vb)

    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    rate = redraws / attempts if attempts else 0.0
    return {"point": float(pa - pb), "lo": float(lo), "hi": float(hi),
            "contains_zero": bool(lo <= 0.0 <= hi), "auroc_a": float(pa), "auroc_b": float(pb),
            "n_boot": int(len(diffs)), "redraws": int(redraws), "redraw_rate": float(rate),
            "estimable": bool(rate <= REDRAW_MAX), "seed": int(seed),
            "estimator": "per_fold_mean"}


def fmt(ci: dict, places: int = 3) -> str:
    """`0.601 [0.54, 0.66]` — the format the tables use."""
    return f"{ci['point']:.{places}f} [{ci['lo']:.2f}, {ci['hi']:.2f}]"
