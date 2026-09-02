"""Minimum detectable effect: the smallest true AUROC this design would have caught.

WHAT IT ANSWERS
  A null result has two readings — the readout lacks the signal, or we could not have seen it.
  Only a power calculation separates them. For each condition we find the smallest true AUROC
  that would clear the shuffled floor with 80% power at this sample size.

WHY THE SIMULATION CARRIES A CLAIM RANDOM EFFECT
  Simulating independent scores would answer an easier question than the one we face. Trials
  from one claim share a question and produce near-identical readouts, so their scores are
  strongly correlated: the measured intraclass correlation runs 0.44 to 0.83 on the
  GlobalOpinionQA conditions. Scores are therefore drawn as a claim-level effect plus trial
  noise, with the ICC estimated from that condition's own out-of-fold scores.

  Labels, claim sizes, the distribution of positives across claims, and fold assignments are
  all taken from the real data rather than a nominal rate. That matters: few-shot/Gemma's 22
  positives sit in 14 of 200 claims, so its effective sample for a cluster bootstrap is far
  smaller than 22 suggests.

TWO KINDS OF STATEMENT
  Where a condition has adequate positives the MDE bounds the null: we would have detected
  0.68 or above and we observed 0.577. Where it does not, the MDE does not bound anything —
  it documents that the condition cannot, and should not anchor a claim. The output separates
  these rather than presenting one table of numbers.

USAGE
  python -m bench.run_mde --out-dir results
  python -m bench.run_mde --conditions persona/qwen --n-sims 100      # quick pass
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from bench.bootstrap import (SEED, _claim_index, _per_fold_mean, fast_auc, fold_assignments)
from bench.oof import oof_scores, text_pipeline

warnings.filterwarnings("ignore")

GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
POWER_TARGET = 0.80
N_SIMS = 100
N_BOOT_SIM = 600         # enough for a 2.5th percentile; the full sweep uses 5000
CONDITIONS = [
    ("persona", "qwen", "influence_persona_raw"),
    ("social proof", "qwen", "influence_socialproof_raw"),
    ("few-shot", "qwen", "influence_fewshot_raw"),
    ("persona", "gemma", "influence_persona_gemma_raw"),
    ("social proof", "gemma", "influence_socialproof_gemma_raw"),
    ("few-shot", "gemma", "influence_fewshot_gemma_raw"),
]


def score_icc(s, claim_ids) -> float:
    """One-way intraclass correlation: the between-claim share of score variance."""
    cl = np.unique(claim_ids)
    mb = np.array([s[claim_ids == c].mean() for c in cl])
    nb = np.array([(claim_ids == c).sum() for c in cl])
    msb = (nb * (mb - s.mean()) ** 2).sum() / (len(cl) - 1)
    msw = sum(((s[claim_ids == c] - mb[i]) ** 2).sum()
              for i, c in enumerate(cl)) / (len(s) - len(cl))
    k = nb.mean()
    return float(np.clip((msb - msw) / (msb + (k - 1) * msw), 0.0, 0.95))


def simulate(y, inv, n_claims, icc, d, rng):
    """Scores with the condition's own claim-level correlation and a label shift of d."""
    u = rng.normal(0.0, np.sqrt(icc), n_claims)[inv]
    e = rng.normal(0.0, np.sqrt(1.0 - icc), len(y))
    return u + e + d * y


def calibrate(y, inv, n_claims, icc, fold, target, seed, n=150):
    """Find the label shift d whose expected per-fold-mean AUROC equals target."""
    def achieved(d):
        rng = np.random.default_rng(seed)
        vals = [_per_fold_mean(y, simulate(y, inv, n_claims, icc, d, rng), fold, 5)
                for _ in range(n)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 0.5
    lo, hi = 0.0, 6.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if achieved(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, achieved((lo + hi) / 2)


def bootstrap_draws(y, claim_ids, fold, n_boot, seed):
    """Precompute the resample index sets once, with their labels and per-fold masks.

    The draws depend only on the claim structure, not on the simulated scores, so redrawing
    them inside every simulation repeats identical work n_sims times over. Hoisting them out
    is the same computation, reused, and it is what makes the power calculation affordable.

    Whether a draw is usable also depends only on the labels, so degenerate draws are dropped
    here rather than re-tested in the inner loop.
    """
    claims, by_claim = _claim_index(claim_ids)
    rng = np.random.default_rng(seed)
    keep = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(claims), len(claims))
        idx = np.concatenate([by_claim[p] for p in pick])
        yb, fb = y[idx], fold[idx]
        masks = [np.flatnonzero(fb == k) for k in range(5)]
        if all(len(m) and yb[m].any() and not yb[m].all() for m in masks):
            keep.append((idx, yb, masks))
    return keep


def power_at(y, claim_ids, fold, icc, d, floor_hi, n_sims, n_boot, seed, draws=None):
    """Fraction of simulated datasets whose bootstrap lower bound clears the floor."""
    _, inv = np.unique(claim_ids, return_inverse=True)
    n_claims = len(np.unique(claim_ids))
    draws = draws if draws is not None else bootstrap_draws(y, claim_ids, fold, n_boot, seed)
    if len(draws) < n_boot * 0.5:          # this condition cannot support the statistic
        return float("nan"), len(draws)
    rng = np.random.default_rng(seed)
    cleared = 0
    for _ in range(n_sims):
        s = simulate(y, inv, n_claims, icc, d, rng)
        vals = np.empty(len(draws))
        for i, (idx, yb, masks) in enumerate(draws):
            sb = s[idx]
            vals[i] = np.mean([fast_auc(yb[m], sb[m]) for m in masks])
        if float(np.percentile(vals, 2.5)) > floor_hi:
            cleared += 1
    return cleared / n_sims, len(draws)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials-dir", default="results/trials")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cis", default="results/bootstrap_cis.json")
    ap.add_argument("--n-sims", type=int, default=N_SIMS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_SIM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--conditions", default=None, help="comma list, e.g. persona/qwen")
    args = ap.parse_args()

    cis = json.loads(Path(args.cis).read_text())
    observed = {(r["dataset"], r["model"]): r for r in cis["aurocs"]
                if r.get("target") == "influence changed answer" and r.get("status") == "ok"}
    want = set(args.conditions.split(",")) if args.conditions else None
    out = []

    for dataset, model, stem in CONDITIONS:
        if want and f"{dataset}/{model}" not in want and f"{stem}" not in want:
            continue
        obs = observed.get((dataset, model))
        if obs is None:
            continue
        trials = json.loads((Path(args.trials_dir) / f"{stem}.json").read_text())["trials"]
        X = [" ".join((t.get("readout") or "").split()) for t in trials]
        y = np.array([bool(t["swayed"]) for t in trials])
        g = np.array([t["claim"] for t in trials])
        s_real = oof_scores(X, y, g, text_pipeline())
        fold = fold_assignments(y, g)
        icc = score_icc(s_real, g)
        _, inv = np.unique(g, return_inverse=True)
        n_claims = len(np.unique(g))
        claims_with_pos = int(sum(1 for c in np.unique(g) if y[g == c].any()))
        floor_hi = obs["floor_hi"]

        print(f"\n  {dataset}/{model}: {int(y.sum())} positives in {claims_with_pos}/"
              f"{n_claims} claims, score ICC {icc:.2f}, floor top {floor_hi:.3f}")
        print(f"    observed {obs['point']:.3f}"
              + (f" [{obs['lo']:.2f}, {obs['hi']:.2f}]" if obs["estimable"]
                 else "  (interval not estimable)"))

        curve, mde = [], None
        for t in GRID:
            d, ach = calibrate(y, inv, n_claims, icc, fold, t, args.seed)
            p, usable = power_at(y, g, fold, icc, d, floor_hi, args.n_sims, args.n_boot,
                                 args.seed + 1)
            curve.append({"true_auroc": t, "achieved": ach, "d": d, "power": p,
                          "usable_sims": usable})
            print(f"    true {t:.2f} (achieved {ach:.3f})  power {p:.2f}"
                  f"{'   <-- MDE' if p >= POWER_TARGET and mde is None else ''}")
            if p >= POWER_TARGET and mde is None:
                mde = t
            if mde is not None:
                break

        # the 0.05 grid is too coarse to bound a null usefully: power ran 0.45 at 0.65 and
        # 0.97 at 0.70 on persona/qwen, so "0.70" understates what the design would catch.
        # Bisect the bracketing interval to report the crossing rather than the grid step.
        if mde is not None and len(curve) >= 2:
            lo_t, hi_t = curve[-2]["true_auroc"], curve[-1]["true_auroc"]
            for _ in range(3):
                mid = (lo_t + hi_t) / 2
                d, ach = calibrate(y, inv, n_claims, icc, fold, mid, args.seed)
                p, usable = power_at(y, g, fold, icc, d, floor_hi, args.n_sims, args.n_boot,
                                     args.seed + 1)
                curve.append({"true_auroc": mid, "achieved": ach, "d": d, "power": p,
                              "usable_sims": usable, "refinement": True})
                print(f"    refine {mid:.3f} (achieved {ach:.3f})  power {p:.2f}")
                if p >= POWER_TARGET:
                    hi_t = mid
                else:
                    lo_t = mid
            mde = hi_t

        out.append({"dataset": dataset, "model": model, "n_positives": int(y.sum()),
                    "n_claims": n_claims, "claims_with_positives": claims_with_pos,
                    "score_icc": icc, "floor_hi": floor_hi,
                    "observed": obs["point"], "observed_estimable": obs["estimable"],
                    "mde": mde, "bounds_the_null": bool(mde is not None and mde <= 0.80),
                    "curve": curve, "n_sims": args.n_sims, "n_boot": args.n_boot,
                    "seed": args.seed})

    payload = {"power_target": POWER_TARGET, "grid": GRID, "seed": args.seed,
               "note": "scores simulated with each condition's own claim-level ICC; labels, "
                       "claim sizes and fold assignments taken from the real data",
               "conditions": out}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "mde.json").write_text(json.dumps(payload, indent=2))

    print("\n  ── summary ──")
    for r in out:
        if r["mde"] is None:
            worst = max(r["curve"], key=lambda c: c["true_auroc"])
            if not np.isfinite(worst["power"]):
                # not "power was low" but "the statistic does not exist here": more than half
                # of claim resamples leave some fold with no positives, so a cluster bootstrap
                # cannot be formed at any effect size.
                print(f"  {r['dataset']}/{r['model']}: a claim-clustered bootstrap cannot be "
                      f"formed at all -- {r['n_positives']} positives in "
                      f"{r['claims_with_positives']} of {r['n_claims']} claims leaves most "
                      f"resamples with an empty fold. No effect size is detectable; this "
                      f"condition constrains nothing.")
            else:
                print(f"  {r['dataset']}/{r['model']}: even a true AUROC of "
                      f"{worst['true_auroc']:.2f} would have been detected only "
                      f"{worst['power']:.0%} of the time. This condition constrains nothing.")
        else:
            print(f"  {r['dataset']}/{r['model']}: would have detected a true AUROC of "
                  f"{r['mde']:.2f} or above with 80% power; observed {r['observed']:.3f}.")
    print(f"\n  wrote {Path(args.out_dir) / 'mde.json'}")


if __name__ == "__main__":
    main()
