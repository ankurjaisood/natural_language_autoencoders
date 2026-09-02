"""Probe AUROC across every layer, from saved activations. CPU only.

WHAT IT ANSWERS
  A reviewer objection: the analysis reads one layer at one token position, so a null could
  mean the counterfactual signal lives elsewhere rather than being absent from the readout.
  This converts the objection into a curve.

WHAT IT DOES NOT DO
  Replace the read-site number. The pre-registered claim is the probe at the layer the NLA
  actually reads from, which run_probe_cis established with a gate against the published
  values. Everything here is descriptive context around that point.

MULTIPLE COMPARISONS
  Sweeping 28 layers and reporting the best one would be selection on the test set. The
  deliverable is the full curve with intervals. The summary prints the read-site value as the
  claim, and prints any maximum only alongside the fact that it was chosen post hoc across all
  layers — a number that needs that sentence attached is not a number the paper can lean on.

USAGE
  python -m bench.run_layer_sweep --model qwen --out-dir results
  python -m bench.run_layer_sweep --stride 4        # every 4th layer, if time is short
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from bench.bootstrap import N_BOOT, SEED, bootstrap_auroc_ci, fold_assignments
from bench.capture_all_layers import read_site_from_checkpoint
from bench.run_probe_cis import AV_REPO, PUBLISHED, probe_pipeline

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen", choices=["qwen", "gemma"])
    ap.add_argument("--trials-dir", default="results/trials")
    ap.add_argument("--acts-dir", default="results/activations")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, -1],
                    help="-1 means all dimensions")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from sklearn.model_selection import GroupKFold, cross_val_predict

    block = read_site_from_checkpoint(AV_REPO[args.model])
    print(f"  read site: block {block} (the pre-registered claim; the rest is context)\n")
    rows, t_start = [], time.time()

    for (dataset, mdl), spec in PUBLISHED.items():
        if mdl != args.model:
            continue
        acts_path = Path(args.acts_dir) / f"{spec['stem']}_alllayers.npy"
        trials_path = Path(args.trials_dir) / f"{spec['stem']}.json"
        if not acts_path.exists() or not trials_path.exists():
            print(f"  {dataset}/{mdl}: activations not saved, skipped")
            continue
        allacts = np.load(acts_path, mmap_mode="r")
        trials = json.loads(trials_path.read_text())["trials"]
        y = np.array([bool(t["swayed"]) for t in trials])
        g = np.array([t["claim"] for t in trials])
        cv = GroupKFold(n_splits=5)
        fold = fold_assignments(y, g)
        n_layers, d_model = allacts.shape[1], allacts.shape[2]
        layers = list(range(0, n_layers, args.stride))
        if block not in layers:                      # never skip the claim
            layers = sorted(layers + [block])
        print(f"  {dataset}/{mdl}: {n_layers} layers, sweeping {len(layers)}")

        for li in layers:
            X = np.asarray(allacts[:, li, :], dtype=np.float64)
            for k in args.ks:
                kk = d_model if k == -1 else k
                proba = cross_val_predict(probe_pipeline(kk), X, y, cv=cv, groups=g,
                                          method="predict_proba")[:, 1]
                ci = bootstrap_auroc_ci(y, proba, g, fold_ids=fold, n_boot=args.n_boot,
                                        seed=args.seed)
                ci.update({"dataset": dataset, "model": mdl, "layer": int(li), "k": int(kk),
                           "is_read_site": bool(li == block)})
                rows.append(ci)
            got = [r for r in rows if r["layer"] == li and r["dataset"] == dataset]
            marks = "  <- READ SITE" if li == block else ""
            print(f"    layer {li:2d}  " + "  ".join(
                f"k={r['k']}: {r['point']:.3f} [{r['lo']:.2f}, {r['hi']:.2f}]" for r in got)
                + marks, flush=True)

    out = Path(args.out_dir) / f"layer_sweep_{args.model}.json"
    out.write_text(json.dumps({"model": args.model, "read_site": block, "stride": args.stride,
                               "n_boot": args.n_boot, "seed": args.seed, "points": rows},
                              indent=2))

    print(f"\n  ── summary ── ({time.time()-t_start:.0f}s)")
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        full = [r for r in rows if r["dataset"] == dataset and r["k"] > 1]
        if not full:
            continue
        site = next((r for r in full if r["is_read_site"]), None)
        best = max(full, key=lambda r: r["point"])
        print(f"  {dataset}:")
        if site:
            print(f"    read site (layer {site['layer']}): {site['point']:.3f} "
                  f"[{site['lo']:.2f}, {site['hi']:.2f}]   <- the claim")
        print(f"    maximum over {len(full)} swept layers: {best['point']:.3f} at layer "
              f"{best['layer']} — selected post hoc, not a claim")
    print(f"\n  wrote {out} ({len(rows)} points)")


if __name__ == "__main__":
    main()
