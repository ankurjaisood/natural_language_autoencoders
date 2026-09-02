"""Appendix Table A1: reader and probe specifications.

WHY THIS IS GENERATED
  Two numbers in this table are data-derived and had already drifted once when they lived in
  prose: the TF-IDF vocabulary size and the readout token length. Both were stated for the Qwen
  runs only and read as if they covered all eight. Computing them here means the range in the
  paper is the range in the artifacts.

USAGE
  python scripts/gen_appendix_tables.py >> tables.tex
"""

from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # first scripts/ file to use bench/

C_GRID = r"\{0.01, 0.03, 0.1, 0.3, 1, 3, 10\}"
K_GRID = r"1, 2, 5, 10, 25, 100, 500, $d_\text{model}$"


def _runs():
    from bench.run_bootstrap_cis import _norm
    for f in sorted(glob.glob("results/trials/influence_*_raw.json")) + \
             sorted(glob.glob("results/trials/sycophancy_influence_raw*.json")):
        t = json.load(open(f))["trials"]
        yield os.path.basename(f), [_norm(x.get("readout")) for x in t], \
            sum(bool(x.get("swayed")) for x in t)


def vocab_range():
    """TF-IDF vocabulary and features-per-positive, across every run in the paper."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    v, per = [], []
    for _, txt, pos in _runs():
        n = len(TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(txt).vocabulary_)
        v.append(n)
        if pos:
            per.append(n / pos)
    return min(v), max(v), min(per), max(per)


def token_range(model="sentence-transformers/all-MiniLM-L6-v2"):
    """Readout length against the limit that actually truncates.

    Loaded through SentenceTransformer, not AutoTokenizer: the raw tokenizer reports BERT's
    512 default, while the wrapper the reader actually calls truncates at max_seq_length
    (256 here). Quoting 512 would be checking a limit nothing in this pipeline enforces.
    """
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model)
    except Exception as e:                       # offline: say so rather than invent a range
        print(f"% token range not computed ({type(e).__name__}); "
              f"rerun with the encoder available", flush=True)
        return None
    lens = [len(m.tokenizer(t)["input_ids"]) for _, txt, _ in _runs() for t in txt]
    return min(lens), max(lens), m.max_seq_length



JUDGE = "results/t1_t2_judge_arms/tier3_replicates.json"


def judge_table() -> str:
    """Judge accuracy per arm, averaged over the five replicates that vary pairs and sampling.

    Means over replicates, so these drift if the arms are rerun; generated for the same reason
    as the vocabulary counts above.
    """
    d = json.load(open(JUDGE))
    def cell(v):
        return f"{sum(v)/len(v)*100:.1f} [{min(v)*100:.0f}--{max(v)*100:.0f}]"
    rows = []
    for key, label in (("signposted", "Signposted"), ("cued", "Cued"), ("subtle", "Subtle")):
        rows.append(f"{label} & {cell(d['judge_qwen'][key])} & {cell(d['judge_gemma'][key])} \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{3}{l}{\emph{Planted word, by share of swayed readouts carrying it}} \\")
    for k, label in (("1", "all"), ("0.6", "60\\%"), ("0.3", "30\\%"), ("0", "none")):
        rows.append(f"\\quad {label} & {cell(d['sensitivity_qwen'][k])} & "
                    f"{cell(d['sensitivity_gemma'][k])} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{\textbf{{LLM judge accuracy, per arm.}} Percentage of answer-matched pairs the judge
assigns correctly, where 50 is chance. Each arm runs five times with the seed changed, which
varies both the sampled pairs and the judge's own sampling; we report the mean and the range
across those five runs. The planted word appears in the stated share of swayed readouts and in no
unswayed one, so a judge that notices it scores 100. On Qwen the judge scores 43.0 with the word in every swayed
readout and 41.0 with it in none, so it is not responding to the word at all.}}
\label{{tab:judge}}
\begin{{tabular}}{{lcc}}
\toprule
Arm & Qwen & Gemma \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def main() -> None:
    lo, hi, plo, phi = vocab_range()
    tr = token_range()
    tokrow = (rf"& Input length & {tr[0]}--{tr[1]} tokens against a {tr[2]}-token limit; "
              r"nothing truncates \\" if tr else
              r"& Input length & Below the encoder's token limit; nothing truncates \\")

    print(rf"""\begin{{table}}[t]
\centering
\small
\caption{{\textbf{{Reader and probe specifications.}} Both readers see the same readouts,
labels and claim-grouped folds; only the text representation differs. Every fit listed
here happens inside the training folds. The bag-of-words reader is deliberately
overparameterized and untuned --- {lo:,}--{hi:,} features against as few as
{plo:.0f} per positive --- which under claim-held-out folds and an empirical floor shows up
as failure to generalize rather than as an inflated score. Only the embedding reader has
its penalty tuned; that asymmetry is deliberate and runs against our own conclusion, since
the tuned reader is the one given every advantage.}}
\label{{tab:hyper}}
\begin{{tabular}}{{lll}}
\toprule
& Setting & Value \\
\midrule
\multicolumn{{3}}{{l}}{{\emph{{Bag-of-words reader}}}} \\
& Features & TF-IDF, 1--2 grams, minimum document frequency 2 \\
& Vocabulary & {lo:,}--{hi:,} terms ({plo:.0f}--{phi:.0f} per positive) \\
& Classifier & Logistic regression, L2 penalty, $C=1$, 2000 iterations \\
\midrule
\multicolumn{{3}}{{l}}{{\emph{{Embedding reader}}}} \\
& Encoder & \texttt{{all-MiniLM-L6-v2}}, frozen, 384-d \\
{tokrow}
& Reduction & PCA, 50 components, fit on training folds only \\
& Classifier & Logistic regression, $C$ from ${C_GRID}$ by 3-fold \\
&            & grouped inner CV on training claims, 3000 iterations \\
\midrule
\multicolumn{{3}}{{l}}{{\emph{{Probe on the activation}}}} \\
& Features & Standardized, then top $k$ by univariate $F$-statistic, \\
&          & refit within each training fold \\
& $k$ grid & {K_GRID} \\
& Classifier & Logistic regression, L2 penalty, $C=1$, 3000 iterations \\
\midrule
\multicolumn{{3}}{{l}}{{\emph{{Shared}}}} \\
& Folds & 5-fold cross-validation, grouped by claim \\
& Class weighting & None; AUROC is threshold-free \\
& Penalty & $C$ is the inverse regularization strength: the objective is \\
&         & $\tfrac{1}{2}\lVert w\rVert^2 + C\sum_i \ell_i$, summed over trials, so \\
&         & $C=1$ is weak regularization at these sample sizes \\
& Statistic & Per-fold mean AUROC \\
\bottomrule
\end{{tabular}}
\end{{table}}""")
    print()
    print(judge_table())


if __name__ == "__main__":
    main()
