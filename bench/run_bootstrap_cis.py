"""Every AUROC in the paper, with a claim-clustered interval and a like-for-like floor.

WHAT THIS COVERS
  The text-classifier numbers: all recovery targets from run_readout_capacity, and every arm
  of run_prompt_baseline, on both checkpoints and all four influence conditions. Each gets a
  point estimate (the per-fold mean the paper reports), a 95% percentile interval from
  resampling claims, and a shuffled floor computed with the SAME estimator so the comparison
  is like for like.

WHAT THIS DOES NOT COVER
  The probe numbers. Those need activations, which are not stored with the trials and have to
  be recaptured on a GPU. They are produced by the layer sweep instead, where the read-site
  layer is one slice of a capture that has to happen anyway.

PAIRED DIFFERENCES
  Two marginal intervals cannot say whether two readers differ, because both are computed on
  the same trials and their errors are strongly correlated. Each difference below resamples
  claims ONCE and evaluates both readers on that draw, so the correlation is retained.

USAGE
  python -m bench.run_bootstrap_cis --out-dir results
  python -m bench.run_bootstrap_cis --only sway --n-boot 1000     # quick pass
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from bench.bootstrap import (N_BOOT, SEED, bootstrap_auroc_ci, fold_assignments,
                             paired_difference_ci, shuffled_floor_ci)
from bench.oof import oof_scores, per_fold_mean, text_pipeline

warnings.filterwarnings("ignore")

RUNS = [
    ("sycophancy",   "qwen",  "sycophancy_influence_raw"),
    ("sycophancy",   "gemma", "sycophancy_influence_raw_gemma"),
    ("persona",      "qwen",  "influence_persona_raw"),
    ("persona",      "gemma", "influence_persona_gemma_raw"),
    ("social proof", "qwen",  "influence_socialproof_raw"),
    ("social proof", "gemma", "influence_socialproof_gemma_raw"),
    ("few-shot",     "qwen",  "influence_fewshot_raw"),
    ("few-shot",     "gemma", "influence_fewshot_gemma_raw"),
]
MIN_TRIALS = 40          # run_readout_capacity's threshold for "not measurable"


def _norm(s):
    return " ".join((s or "").split())


def _top2(values, txt, groups):
    """Restrict a multi-valued target to its two most common values.

    Mirrors run_readout_capacity: binarising against the single most common value leaves the
    positive class too small for grouped folds, which produced undefined AUROCs.
    """
    v = np.array([str(x) for x in values])
    common = [a for a, _ in sorted(((a, int((v == a).sum())) for a in set(v)),
                                   key=lambda z: -z[1])[:2]]
    if len(common) < 2:
        return None
    m = np.isin(v, common)
    if m.sum() < MIN_TRIALS:
        return None
    return ([t for t, k in zip(txt, m) if k], v[m] == common[0], groups[m])


def _categorical_pipeline():
    """The answer as its own arm: one token per answer value, not a bag of its words.

    Sycophancy answers are bare letters, which the word tokenizer discards entirely; on the
    other corpora a word model would read the option's wording, which is more than an auditor
    knowing what the model answered.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(analyzer=lambda s: [s]),
                         LogisticRegression(max_iter=2000))


def targets_for(trials):
    """(name, X, y, groups, pipeline) for every AUROC the paper reports on this run."""
    txt = [_norm(t.get("readout")) for t in trials]
    groups = np.array([t["claim"] for t in trials])
    y = np.array([bool(t["swayed"]) for t in trials])
    ans = np.array([str(t.get("answer") or t.get("answer_opt")) for t in trials])
    out = [("influence changed answer", txt, y, groups, None)]

    for name, field in (("direction of influence", "target"),
                        ("direction of influence", "stance")):
        if any(t.get(field) is not None for t in trials):
            sel = _top2([t.get(field) for t in trials], txt, groups)
            if sel:
                out.append((name, *sel, None))

    sel = _top2(ans, txt, groups)
    if sel:
        out.append(("answer chosen", *sel, None))

    # run_prompt_baseline's arms, on the sway target
    for name, field in (("question alone", "prompt"), ("model's own reasoning", "reasoning")):
        col = [_norm(t.get(field)) for t in trials]
        if any(col):
            out.append((name, col, y, groups, None))
    out.append(("answer alone", [str(a) for a in ans], y, groups, _categorical_pipeline()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials-dir", default="results/trials")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only", default=None, help="substring filter on target name")
    args = ap.parse_args()

    trials_dir, out_dir = Path(args.trials_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, pairs = [], []

    for dataset, model, stem in RUNS:
        path = trials_dir / f"{stem}.json"
        if not path.exists():
            print(f"  {dataset}/{model}: {path.name} missing, skipped")
            continue
        trials = json.loads(path.read_text())["trials"]
        cache = {}
        for name, X, y, g, pipe in targets_for(trials):
            if args.only and args.only not in name:
                continue
            y = np.asarray(y)
            if len(set(y.tolist())) < 2 or len(set(g.tolist())) < 6:
                print(f"  {dataset}/{model} {name}: not measurable")
                results.append({"dataset": dataset, "model": model, "target": name,
                                "status": "not measurable"})
                continue
            s = oof_scores(X, y, g, pipe or text_pipeline())
            fold = fold_assignments(y, g)
            try:
                ci = bootstrap_auroc_ci(y, s, g, fold_ids=fold, n_boot=args.n_boot,
                                        seed=args.seed)
            except (ValueError, RuntimeError) as e:
                # a fold with one class on the OBSERVED data, or a positive class too small to
                # support a five-fold statistic. Both mean the same thing for the paper: this
                # cell is not measurable, and substituting a number would invent precision.
                print(f"  {dataset:12s} {model:5s} {name:26s} not measurable ({e})")
                results.append({"dataset": dataset, "model": model, "target": name,
                                "status": "not measurable", "reason": str(e)})
                continue
            if not ci["estimable"]:
                # withhold the interval rather than caveat it; see REDRAW_MAX in bootstrap.py
                ci.update({"lo": None, "hi": None})
            fl = shuffled_floor_ci(y, s, g, fold_ids=fold, n_boot=args.n_boot, seed=args.seed)
            ci.update({"dataset": dataset, "model": model, "target": name, "status": "ok",
                       "floor_mean": fl["mean"], "floor_lo": fl["lo"], "floor_hi": fl["hi"],
                       "clears_floor": (None if ci["lo"] is None
                                        else bool(ci["lo"] > fl["hi"])),
                       "published_per_fold": per_fold_mean(X, y, g, pipe or text_pipeline())})
            results.append(ci)
            cache[name] = (y, s, g, fold)
            if ci["estimable"]:
                print(f"  {dataset:12s} {model:5s} {name:26s} {ci['point']:.3f} "
                      f"[{ci['lo']:.2f}, {ci['hi']:.2f}]  floor [{fl['lo']:.2f}, {fl['hi']:.2f}]"
                      f"  {'clears' if ci['clears_floor'] else 'OVERLAPS':8s} "
                      f"redraw {100*ci['redraw_rate']:.1f}%")
            else:
                print(f"  {dataset:12s} {model:5s} {name:26s} {ci['point']:.3f} "
                      f"  INTERVAL NOT ESTIMABLE (redraw {100*ci['redraw_rate']:.1f}%)")

        # paired differences, on one resample each
        base = cache.get("influence changed answer")
        for other in ("answer alone", "question alone", "model's own reasoning"):
            if base and other in cache:
                y, s_a, g, fold = base
                d = paired_difference_ci(y, s_a, cache[other][1], g, fold_ids=fold,
                                         n_boot=args.n_boot, seed=args.seed)
                d.update({"dataset": dataset, "model": model,
                          "comparison": f"readout minus {other}"})
                if not d["estimable"]:
                    d.update({"lo": None, "hi": None, "contains_zero": None})
                pairs.append(d)
                if d["estimable"]:
                    print(f"  {dataset:12s} {model:5s} readout - {other:22s} {d['point']:+.3f} "
                          f"[{d['lo']:+.2f}, {d['hi']:+.2f}]  "
                          f"{'CONTAINS ZERO' if d['contains_zero'] else 'excludes zero'}")
                else:
                    print(f"  {dataset:12s} {model:5s} readout - {other:22s} {d['point']:+.3f} "
                          f"  NOT ESTIMABLE (redraw {100*d['redraw_rate']:.1f}%)")

    payload = {"estimator": "per_fold_mean", "n_boot": args.n_boot, "seed": args.seed,
               "note": "probe AUROCs are produced by the layer sweep, which captures "
                       "activations; they are not in this file",
               "aurocs": results, "paired_differences": pairs}
    (out_dir / "bootstrap_cis.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out_dir / 'bootstrap_cis.json'} "
          f"({len(results)} AUROCs, {len(pairs)} paired differences)")


if __name__ == "__main__":
    main()
