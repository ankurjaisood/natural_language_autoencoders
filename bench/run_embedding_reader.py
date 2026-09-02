"""A non-lexical reader on the same readouts, to test whether bag-of-words was the limit.

THE OBJECTION THIS ANSWERS
  Our reader counts words and word pairs, so it can only find lexical patterns. A reviewer's
  natural reading of a null is that a stronger reader would have found the signal.

  The probe does not answer this. It bounds what is recoverable from the ACTIVATION; the
  objection is about what is recoverable from the READOUT TEXT. Those are different objects,
  and only a second reader on the same text addresses the second.

WHY SYCOPHANCY IS INCLUDED
  It is the positive control and it does more work than it looks. If the embedding reader
  recovers sycophancy near its bag-of-words value while landing at the floor on social proof,
  the reader demonstrably extracts non-lexical structure from readout text when the structure
  is there — which turns a null into a discriminating result. Without it, a floor-level social
  proof number is ambiguous between "the readout lacks it" and "embeddings read this text
  badly".

TWO INSTRUMENT GATES, NOT ONE
  The answer target (~0.98) checks the reader works at all. The direction-of-influence target
  (~0.81 on persona) is the more informative check: a harder task with less lexical signal. A
  reader that passes the first and fails the second is telling us about the reader rather than
  about the readout, and its sway number would not be interpretable.

OVERFITTING CONTROL
  With 384-dimensional embeddings against 50-92 positives an unregularised model can score
  WORSE than bag-of-words purely from variance, and that result would be uninterpretable while
  looking like evidence. So: PCA fit on training folds only, and the penalty chosen by grouped
  cross-validation INSIDE the training folds. The outer loop is written out rather than handed
  to cross_val_predict so the nesting is visible.

USAGE
  python -m bench.run_embedding_reader --out-dir results
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from bench.bootstrap import N_BOOT, SEED, bootstrap_auroc_ci, fast_auc, fold_assignments
from bench.oof import oof_scores, text_pipeline
from bench.run_bootstrap_cis import _norm, _top2

warnings.filterwarnings("ignore")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"      # 384-d; named because it is a free parameter
N_PCA = 50
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
INNER_SPLITS = 3
RUNS = [
    ("sycophancy", "sycophancy_influence_raw"),        # positive control
    ("social proof", "influence_socialproof_raw"),
    ("persona", "influence_persona_raw"),
    ("few-shot", "influence_fewshot_raw"),
]


def embed(texts, model_name=MODEL, batch_size=64):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    return np.asarray(m.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                               convert_to_numpy=True), dtype=np.float64)


def _select_C(Xtr, ytr, gtr, seed):
    """Grouped inner CV over the penalty, on training claims only."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_groups = len(np.unique(gtr))
    splits = min(INNER_SPLITS, n_groups)
    if splits < 2 or len(np.unique(ytr)) < 2:
        return 1.0
    best, best_auc = 1.0, -1.0
    for C in C_GRID:
        scores = []
        for itr, ite in GroupKFold(n_splits=splits).split(Xtr, ytr, gtr):
            if len(np.unique(ytr[ite])) < 2 or len(np.unique(ytr[itr])) < 2:
                continue
            pipe = make_pipeline(StandardScaler(),
                                 PCA(n_components=min(N_PCA, len(itr) - 1, Xtr.shape[1]),
                                     random_state=seed),
                                 LogisticRegression(C=C, max_iter=3000))
            pipe.fit(Xtr[itr], ytr[itr])
            a = fast_auc(ytr[ite], pipe.predict_proba(Xtr[ite])[:, 1])
            if a is not None:
                scores.append(a)
        if scores and np.mean(scores) > best_auc:
            best, best_auc = C, float(np.mean(scores))
    return best


def oof_embedding_scores(X, y, g, seed=SEED, n_splits=5):
    """Out-of-fold scores with PCA and penalty selection confined to the training folds."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    g = np.asarray(g)
    out = np.full(len(y), np.nan)
    chosen = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        C = _select_C(X[tr], y[tr], g[tr], seed)
        chosen.append(C)
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=min(N_PCA, len(tr) - 1, X.shape[1]),
                                 random_state=seed),
                             LogisticRegression(C=C, max_iter=3000))
        pipe.fit(X[tr], y[tr])                    # PCA sees training folds only
        out[te] = pipe.predict_proba(X[te])[:, 1]
    assert not np.isnan(out).any(), "some trial was never in a test fold"
    return out, chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials-dir", default="results/trials")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rows, gate_fail = [], []
    for dataset, stem in RUNS:
        path = Path(args.trials_dir) / f"{stem}.json"
        if not path.exists():
            continue
        trials = json.loads(path.read_text())["trials"]
        txt = [_norm(t.get("readout")) for t in trials]
        g = np.array([t["claim"] for t in trials])
        y_sway = np.array([bool(t["swayed"]) for t in trials])
        ans = np.array([str(t.get("answer") or t.get("answer_opt")) for t in trials])
        E = embed(txt, args.model)
        print(f"\n  {dataset}: {len(txt)} readouts -> {E.shape[1]}-d embeddings")

        # targets carry INDICES, not text: two trials can share a readout string and
        # matching by value would silently mis-align the embedding rows.
        def top2_idx(values):
            v = np.array([str(x) for x in values])
            common = [c for c, _ in sorted(((c, int((v == c).sum())) for c in set(v)),
                                           key=lambda z: -z[1])[:2]]
            if len(common) < 2:
                return None
            m = np.flatnonzero(np.isin(v, common))
            return (m, v[m] == common[0]) if len(m) >= 40 else None

        targets = [("influence changed answer", np.arange(len(txt)), y_sway)]
        for field in ("target", "stance"):
            if any(t.get(field) is not None for t in trials):
                sel = top2_idx([t.get(field) for t in trials])
                if sel:
                    targets.append(("direction of influence", *sel))
                break
        sel = top2_idx(ans)
        if sel:
            targets.append(("answer chosen", *sel))

        for name, idx, yy in targets:
            gg, sub_txt, Xs = g[idx], [txt[i] for i in idx], E[idx]
            if len(np.unique(yy)) < 2 or len(np.unique(gg)) < 6:
                print(f"    {name:26s} not measurable")
                continue
            fold = fold_assignments(yy, gg)
            try:
                s_bow = oof_scores(sub_txt, yy, gg, text_pipeline())
                ci_bow = bootstrap_auroc_ci(yy, s_bow, gg, fold_ids=fold,
                                            n_boot=args.n_boot, seed=args.seed)
                s_emb, cs = oof_embedding_scores(Xs, yy, gg, args.seed)
                ci_emb = bootstrap_auroc_ci(yy, s_emb, gg, fold_ids=fold,
                                            n_boot=args.n_boot, seed=args.seed)
            except (ValueError, RuntimeError) as e:
                # a fold single-class on the observed data — the same cells Table 4.1 already
                # reports as not measurable. Substituting a number would invent precision.
                print(f"    {name:26s} not measurable ({e})")
                rows.append({"dataset": dataset, "target": name, "status": "not measurable",
                             "reason": str(e)})
                continue
            rows.append({"dataset": dataset, "target": name, "n": int(len(yy)),
                         "positives": int(yy.sum()), "bow": ci_bow, "embedding": ci_emb,
                         "penalties": cs, "embed_model": args.model, "n_pca": N_PCA})
            def show(ci):
                return (f"{ci['point']:.3f} [{ci['lo']:.2f}, {ci['hi']:.2f}]"
                        if ci["estimable"] else f"{ci['point']:.3f} (not estimable)")
            print(f"    {name:26s} bag-of-words {show(ci_bow):22s} "
                  f"embedding {show(ci_emb):22s} C={sorted(set(cs))}")
            if name == "answer chosen" and ci_emb["point"] < 0.90:
                gate_fail.append(f"{dataset}/answer ({ci_emb['point']:.3f})")
            if name == "direction of influence" and ci_emb["point"] < 0.70:
                gate_fail.append(f"{dataset}/direction ({ci_emb['point']:.3f})")

    out = Path(args.out_dir) / "embedding_reader.json"
    out.write_text(json.dumps({"embed_model": args.model, "n_pca": N_PCA,
                               "c_grid": C_GRID, "seed": args.seed,
                               "gate_failures": gate_fail, "rows": rows}, indent=2))
    print(f"\n  wrote {out}")
    if gate_fail:
        print(f"  INSTRUMENT GATE FAILURES: {gate_fail}")
        print("  the sway numbers above are not interpretable for those conditions")
    else:
        print("  instrument gates passed: the reader recovers answer and direction")


if __name__ == "__main__":
    main()
