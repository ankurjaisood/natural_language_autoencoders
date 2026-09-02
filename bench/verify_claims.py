"""Recompute every derived claim from the artifacts, and optionally check a draft against them.

WHY THIS EXISTS
  A derived number — "five of six", "89% of the full probe", "3 dashes of 24 cells" — is
  correct when written and can stop being correct when scope shifts. Nothing about it looks
  wrong afterwards. One such number survived in the outline: the not-estimable count was 7
  across all seven targets in the full sweep, and Table 1 carries three of those targets, so
  the right figure there is 5. Only recomputing it against the actual table exposed it.

  Compression is when scope shifts most — rows move to the appendix, conditions get dropped,
  a table loses a column. So these need recomputing at the end, not checking by eye.

USAGE
  python -m bench.verify_claims                          # print canonical values
  python -m bench.verify_claims --check paper.tex        # and flag any that the draft
                                                         # states differently
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

R = lambda f: json.loads(Path(f).read_text())          # noqa: E731


def claims(results="results") -> dict:
    """Every derived number the paper asserts, recomputed from artifacts."""
    cis = R(f"{results}/bootstrap_cis.json")
    mde = R(f"{results}/mde.json")["conditions"]
    pao = R(f"{results}/power_at_observed.json")["rows"]
    prb = R(f"{results}/probe_cis_qwen.json")["probes"]
    emb = R(f"{results}/embedding_reader.json")["rows"]
    lay = R(f"{results}/layer_sweep_qwen.json")
    A = cis["aurocs"]
    c = {}

    # ── counts that move when a row moves ──
    goq = [r for r in A if r.get("target") == "influence changed answer"
           and r.get("dataset") != "sycophancy" and r.get("status") == "ok"]
    overlap = [r for r in goq if r["estimable"] and not r["clears_floor"]]
    c["sway: GlobalOpinionQA conditions overlapping their floor"] = f"{len(overlap)} of {len(goq)}"

    full = [p for p in lay["points"] if p["k"] > 1]
    inside = 0
    for d in {p["dataset"] for p in full}:
        s = [p for p in full if p["dataset"] == d]
        site = next(p for p in s if p["is_read_site"])
        best = max(s, key=lambda p: p["point"])
        inside += site["lo"] <= best["point"] <= site["hi"]
    c["layer sweep: maxima inside the read-site interval"] = \
        f"{inside} of {len({p['dataset'] for p in full})}"

    t1 = [r for r in A if r.get("target") in ("influence changed answer",
                                              "direction of influence", "answer chosen")]
    c["Table 1 cells / dashes / daggers"] = (
        f"{len(t1)} / {sum(1 for r in t1 if r.get('status') == 'not measurable')} / "
        f"{sum(1 for r in t1 if r.get('status') == 'ok' and not r['estimable'])}")
    c["Table 2 dashes (no MDE)"] = str(sum(1 for x in mde if x["mde"] is None))
    c["Table 3 embedding dashes / daggers"] = (
        f"{sum(1 for r in emb if r.get('status') == 'not measurable')} / "
        f"{sum(1 for r in emb if r.get('status') != 'not measurable' and not r['embedding']['estimable'])}")

    for x in mde:
        c[f"positives / claims carrying them, {x['dataset']}/{x['model']}"] = (
            f"{x['n_positives']} in {x['claims_with_positives']} of {x['n_claims']}")

    # ── ranges, which narrow silently when a condition is dropped ──
    ans = [r["point"] for r in A if r.get("target") == "answer chosen" and r.get("status") == "ok"]
    c["answer recovery range"] = f"{min(ans):.3f}-{max(ans):.3f}"
    qm = [x["mde"] for x in mde if x["model"] == "qwen" and x["mde"]]
    c["Qwen MDE range"] = f"{min(qm):.3f}-{max(qm):.3f}"
    qp = [r["power_at_observed"] for r in pao if r["model"] == "qwen"]
    c["Qwen power-at-observed range"] = f"{min(qp):.2f}-{max(qp):.2f}"
    icc = [x["score_icc"] for x in mde]
    icc_m = [x["score_icc"] for x in mde if x["mde"]]
    c["ICC range, all six"] = f"{min(icc):.2f}-{max(icc):.2f}"
    c["ICC range, five with an MDE"] = f"{min(icc_m):.2f}-{max(icc_m):.2f}"

    # ── percentages of other numbers ──
    for d in ("social proof", "sycophancy", "few-shot", "persona"):
        o = [p for p in prb if p["dataset"] == d]
        one = next(p["point"] for p in o if p["k"] == 1)
        allk = next(p["point"] for p in o if p["k"] == 3584)
        c[f"1-D share of full probe, {d}"] = f"{one/allk:.0%}"

    # ── differences ──
    for r in emb:
        if r.get("target") == "influence changed answer":
            c[f"embedding minus bag-of-words, {r['dataset']}"] = \
                f"{r['embedding']['point'] - r['bow']['point']:+.3f}"
    pd = [p for p in cis["paired_differences"] if p.get("estimable")]
    c["paired differences containing zero"] = \
        f"{sum(1 for p in pd if p['contains_zero'])} of {len(pd)}"
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--check", default=None, help="draft to scan for contradicting values")
    args = ap.parse_args()

    c = claims(args.results)
    w = max(len(k) for k in c)
    for k, v in c.items():
        print(f"  {k:<{w}}  {v}")

    if args.check:
        text = Path(args.check).read_text()
        print(f"\n  scanning {args.check} for 'N of M' claims that disagree\n")
        # substring, not equality: a canonical value may embed the phrase, as in
        # "22 in 14 of 200", and an exact match would flag it as unrecognised
        canon = " || ".join(v for v in c.values() if " of " in v)
        found = set(re.findall(r"\b(\d+) of (\d+)\b", text))
        suspect = [f"{a} of {b}" for a, b in found if f"{a} of {b}" not in canon]
        if suspect:
            print("  present in the draft but not among the recomputed values —")
            print("  each needs checking by hand, since it may be a different quantity:")
            for s in sorted(set(suspect)):
                print(f"    {s}")
        else:
            print("  every 'N of M' in the draft matches a recomputed value")


if __name__ == "__main__":
    main()
