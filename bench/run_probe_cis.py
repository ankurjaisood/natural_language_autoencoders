"""Probe confidence intervals at the layer the NLA reads from.

WHY THIS IS THE LOAD-BEARING RUN
  The paper's strongest form is a joint claim: the counterfactual signal is linearly present
  in the activation the verbaliser was given, at an effect size this design could have
  detected in the readout, and the readout does not carry it. That argument compares a probe
  AUROC against a minimum detectable effect, and on persona those two numbers sit 0.002 apart.
  A comparison that close is meaningless without an interval on the probe.

GATE BEFORE INTERVALS
  Activations are not stored with the trials, so this recaptures them. If recapture does not
  reproduce the published probe numbers, something differs between this run and the original
  and no interval built on top would mean anything. The published values are asserted first.

WHAT IT SAVES
  Every layer, not just the read site. The forward pass computes them anyway, so saving them
  turns the layer sweep into a CPU job over a file instead of a second GPU run.

USAGE
  python -m bench.run_probe_cis --model qwen --out-dir results
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from bench.bootstrap import N_BOOT, SEED, bootstrap_auroc_ci, fold_assignments, fmt
from bench.capture_all_layers import (capture_all_layers, read_site_from_checkpoint,
                                      verify_matches_single_layer)

warnings.filterwarnings("ignore")

AV_REPO = {"qwen": "kitft/nla-qwen2.5-7b-L20-av", "gemma": "kitft/nla-gemma3-12b-L32-av"}
# published probe values, from results/t5_bandwidth/*.json — the gate
PUBLISHED = {
    ("persona", "qwen"):      {"1": 0.490, "full": 0.683, "stem": "influence_persona_raw"},
    ("social proof", "qwen"): {"1": 0.765, "full": 0.859, "stem": "influence_socialproof_raw"},
    ("sycophancy", "qwen"):   {"1": 0.847, "full": 0.893, "stem": "sycophancy_influence_raw"},
    ("few-shot", "qwen"):     {"1": None,  "full": None,  "stem": "influence_fewshot_raw"},
}
GATE_TOL = 0.01
DIMS = [1, 2, 5, 10, 25, 100, 500]


def probe_pipeline(k):
    """The published probe: scale, select k dimensions, fit a linear model.

    SelectKBest sits INSIDE the pipeline so it refits per fold. Hoisting it out would be
    faster and would leak the test fold's labels into feature selection, inflating exactly
    the number this run exists to bound.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k),
                         LogisticRegression(max_iter=3000))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen", choices=["qwen", "gemma"])
    ap.add_argument("--trials-dir", default="results/trials")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--acts-dir", default="results/activations")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-gate", action="store_true",
                    help="only for conditions with no published value to gate against")
    args = ap.parse_args()

    from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

    from bench.capture import load_model
    from bench.roundtrip import free_model
    from bench.run_controlled_claims import MODELS

    block = read_site_from_checkpoint(AV_REPO[args.model])
    cfg = MODELS[args.model]
    if block != cfg["block"]:
        raise SystemExit(f"sidecar says block {block}, code constant says {cfg['block']}")
    print(f"  read site from {AV_REPO[args.model]}: block {block}\n")

    acts_dir = Path(args.acts_dir)
    acts_dir.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(cfg["base"])
    results, failed = [], []

    try:
        for (dataset, mdl), spec in PUBLISHED.items():
            if mdl != args.model:
                continue
            path = Path(args.trials_dir) / f"{spec['stem']}.json"
            if not path.exists():
                continue
            trials = json.loads(path.read_text())["trials"]
            y = np.array([bool(t["swayed"]) for t in trials])
            g = np.array([t["claim"] for t in trials])
            prompts = [tok.apply_chat_template([{"role": "user", "content": t["prompt"]}],
                                               add_generation_prompt=True, tokenize=False)
                       for t in trials]

            print(f"  {dataset}/{mdl}: capturing {len(prompts)} prompts x all layers")
            allacts = capture_all_layers(model, tok, prompts)
            dev = verify_matches_single_layer(model, tok, prompts, allacts, block)
            np.save(acts_dir / f"{spec['stem']}_alllayers.npy", allacts)
            print(f"    saved {allacts.shape} -> {spec['stem']}_alllayers.npy "
                  f"(block-{block} slice matches single-layer capture, max dev {dev:.1e})")

            acts = allacts[:, block, :].astype(np.float64)
            cv = GroupKFold(n_splits=5)

            # ── gate: reproduce the published numbers before anything is built on them ──
            got = {}
            for k in (1, acts.shape[1]):
                got[k] = float(cross_val_score(probe_pipeline(min(k, acts.shape[1])), acts, y,
                                               cv=cv, groups=g, scoring="roc_auc").mean())
            pub1, pubf = spec["1"], spec["full"]
            if pub1 is not None:
                d1, df = abs(got[1] - pub1), abs(got[acts.shape[1]] - pubf)
                ok = d1 < GATE_TOL and df < GATE_TOL
                print(f"    GATE 1-D {got[1]:.3f} vs published {pub1:.3f} (d {d1:+.3f}); "
                      f"full {got[acts.shape[1]]:.3f} vs {pubf:.3f} (d {df:+.3f}) -> "
                      f"{'PASS' if ok else 'FAIL'}")
                if not ok and not args.skip_gate:
                    failed.append(f"{dataset}/{mdl}")
                    continue
            else:
                print(f"    no published value; 1-D {got[1]:.3f}, "
                      f"full {got[acts.shape[1]]:.3f}")

            # ── intervals, on out-of-fold scores from the same folds ──
            for k in DIMS + [acts.shape[1]]:
                if k > acts.shape[1]:
                    continue
                proba = cross_val_predict(probe_pipeline(k), acts, y, cv=cv, groups=g,
                                          method="predict_proba")[:, 1]
                fold = fold_assignments(y, g)
                ci = bootstrap_auroc_ci(y, proba, g, fold_ids=fold, n_boot=args.n_boot,
                                        seed=args.seed)
                ci.update({"dataset": dataset, "model": mdl, "k": int(k), "block": block,
                           "published": (pubf if k == acts.shape[1] else
                                         (pub1 if k == 1 else None))})
                results.append(ci)
                tag = "  <- read-site full probe" if k == acts.shape[1] else ""
                shown = fmt(ci) if ci["estimable"] else f"{ci['point']:.3f}  (not estimable)"
                print(f"    probe k={k:<5} {shown}{tag}")
    finally:
        free_model(model)
        del model

    payload = {"model": args.model, "block": block, "n_boot": args.n_boot, "seed": args.seed,
               "gate_tolerance": GATE_TOL, "gate_failures": failed, "probes": results}
    out = Path(args.out_dir) / f"probe_cis_{args.model}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out} ({len(results)} probe intervals)"
          + (f"  GATE FAILURES: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
