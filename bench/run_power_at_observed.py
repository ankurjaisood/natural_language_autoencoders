"""Power at the value we actually observed, and a check that MDEs track sensitivity.

WHY POWER AT THE OBSERVED VALUE
  An MDE alone is easy to over-read. "80% power to detect 0.68" does not say the true effect is
  below 0.68 — it says effects of that size or larger would usually have been caught. What
  makes the bound legible is the power at the observed value: if an effect the size of what we
  measured would usually have been MISSED, that is the honest characterisation of the evidence,
  and it stops the MDE from being read as a claim that no effect exists.

WHY THE FLOOR-WIDTH CHECK
  The power criterion is "the bootstrap lower bound clears the floor's upper bound", and floor
  width varies by condition. If MDEs tracked floor width rather than positive count and score
  correlation, the differences between conditions would be floor estimation noise dressed up as
  design sensitivity. This reports all four so the reader can see which one the MDE follows.

USAGE
  python -m bench.run_power_at_observed --out-dir results
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from bench.bootstrap import SEED, fold_assignments
from bench.oof import oof_scores, text_pipeline
from bench.run_mde import CONDITIONS, calibrate, power_at, score_icc

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials-dir", default="results/trials")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--mde", default="results/mde.json")
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    mde = {(c["dataset"], c["model"]): c
           for c in json.loads(Path(args.mde).read_text())["conditions"]}
    rows = []

    for dataset, model, stem in CONDITIONS:
        c = mde.get((dataset, model))
        if c is None:
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

        d, ach = calibrate(y, inv, n_claims, icc, fold, c["observed"], args.seed)
        p, usable = power_at(y, g, fold, icc, d, c["floor_hi"], args.n_sims, args.n_boot,
                             args.seed + 1)
        row = {"dataset": dataset, "model": model, "observed": c["observed"],
               "power_at_observed": p, "usable_sims": usable, "mde": c["mde"],
               "n_positives": c["n_positives"],
               "claims_with_positives": c["claims_with_positives"],
               "score_icc": icc, "floor_hi": c["floor_hi"],
               "floor_width": None, "seed": args.seed}
        rows.append(row)
        print(f"  {dataset:12s} {model:5s} observed {c['observed']:.3f}  "
              f"power at observed {p:.2f}  MDE "
              + (f"{c['mde']:.3f}" if c["mde"] is not None else "none"))

    # does the MDE follow sensitivity, or floor width?
    cis = json.loads(Path("results/bootstrap_cis.json").read_text())
    fl = {(r["dataset"], r["model"]): r["floor_hi"] - r["floor_lo"] for r in cis["aurocs"]
          if r.get("target") == "influence changed answer" and r.get("status") == "ok"}
    for r in rows:
        r["floor_width"] = fl.get((r["dataset"], r["model"]))

    have = [r for r in rows if r["mde"] is not None and r["floor_width"] is not None]
    if len(have) >= 3:
        m = np.array([r["mde"] for r in have])
        print("\n  does the MDE track sensitivity or floor width?")
        for name, v in (("positives", [r["n_positives"] for r in have]),
                        ("claims with positives", [r["claims_with_positives"] for r in have]),
                        ("score ICC", [r["score_icc"] for r in have]),
                        ("floor width", [r["floor_width"] for r in have])):
            v = np.array(v, dtype=float)
            corr = float(np.corrcoef(m, v)[0, 1]) if v.std() > 0 else float("nan")
            print(f"    corr(MDE, {name:22s}) = {corr:+.2f}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "power_at_observed.json").write_text(
        json.dumps({"seed": args.seed, "n_sims": args.n_sims, "rows": rows}, indent=2))
    print(f"\n  wrote {Path(args.out_dir) / 'power_at_observed.json'}")


if __name__ == "__main__":
    main()
