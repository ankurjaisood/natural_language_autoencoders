"""Appendix floats: the chat templates, and the four influence prompts in full.

The main text shows only what each condition prepends, because four full prompts run 386-1298
characters and would consume most of a page. The appendix has no such constraint, and a reader
checking whether the manipulation is what we say it is needs the exact strings.

Verbatim rather than a table: these contain special tokens, curly quotes and option letters in
parentheses, and the point is that they are reproduced exactly.
"""
import json, pathlib, textwrap

R = lambda f: json.loads(pathlib.Path(f"results/trials/{f}.json").read_text())["trials"]

def wrap(s, width=84):
    """Hard-wrap for verbatim, which does not break lines itself."""
    out = []
    for para in s.split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return "\n".join(out)

# ── the chat template wrapper ──────────────────────────────────────────────────
print(r"""\begin{figure}[t]
\centering
\begin{minipage}{\linewidth}
\footnotesize
\begin{verbatim}
Qwen2.5-7B-Instruct
  <|im_start|>system
  You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
  <|im_start|>user
  {PROMPT}<|im_end|>
  <|im_start|>assistant
                        <-- activation read here, at the final token

Gemma-3-12B-IT
  <bos><start_of_turn>user
  {PROMPT}<end_of_turn>
  <start_of_turn>model
                        <-- activation read here, at the final token
\end{verbatim}
\end{minipage}
\caption{\textbf{Chat template and read position.} Each influenced prompt is wrapped in its
model's own chat template with a generation prompt appended, exactly as the model would receive
it at inference. \texttt{\{PROMPT\}} is the influenced prompt shown in
Figure~\ref{fig:prompts-full}. We take the residual stream at the \emph{final token} of this
formatted string --- the last token before generation begins --- from the block each NLA was
trained to read. Qwen's template inserts a default system message; Gemma's does not, and prefixes
a \texttt{<bos>}. Neither difference is under our control: both are the templates shipped with
the released tokenizers.}
\label{fig:template}
\end{figure}
""")

# ── the prompts in full, split by length and by corpus ────────────────────────
# All four in one float runs ~106 wrapped lines, about two pages. Few-shot alone is 53 because
# it carries four exemplar questions with their options. Split into three: the pair that differs
# minimally, the long structural outlier, and the other corpus.
def block(name, stem, note=""):
    t = R(stem)[0]
    body = wrap(t["prompt"].replace("’", "'").replace("“", '"').replace("”", '"'))
    return rf"""\textbf{{{name}}}{note} \\[2pt]
\begin{{verbatim}}
{body}
\end{{verbatim}}"""

print(rf"""\begin{{figure}}[t]
\centering
\begin{{minipage}}{{\linewidth}}
\scriptsize
{block("persona", "influence_persona_raw")}
\vspace{{4pt}}
{block("social proof", "influence_socialproof_raw")}
\end{{minipage}}
\caption{{\textbf{{Persona and social proof, in full.}} Both applied to the same GlobalOpinionQA
question, so the only difference between them is the first line: whether a person is attached to
the stated preference. That pair carries the paper's central comparison. The endorsed option is
drawn at random per trial, and one of six biographies is sampled per trial, so neither the
option nor the wording is fixed across the run.}}
\label{{fig:prompts-goqa}}
\end{{figure}}

\begin{{figure}}[p]
\centering
\begin{{minipage}}{{\linewidth}}
\scriptsize
{block("few-shot", "influence_fewshot_raw")}
\end{{minipage}}
\caption{{\textbf{{Few-shot, in full.}} Four questions drawn from elsewhere in the corpus, never
the one under test, each shown answered \texttt{{(A)}}, followed by the question under test. The
push is toward a \emph{{position}} rather than any option's meaning, so the pushed option is
whichever the test question happens to list first. Because GlobalOpinionQA lists options in a
consistent order, slot (A) is not content-neutral --- it holds the strongest affirmative choice
almost throughout --- which is the limitation discussed in \S\ref{{sec:fewshot}}. Shown separately
because it runs four times the length of the other conditions.}}
\label{{fig:prompts-fewshot}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\begin{{minipage}}{{\linewidth}}
\scriptsize
{block("stated opinion", "sycophancy_influence_raw")}
\end{{minipage}}
\caption{{\textbf{{Stated opinion, in full.}} From the sycophancy corpus, whose items ask for
agreement with a \emph{{claim}} rather than a choice among several options. The speaker therefore
endorses the claim itself, and the biography and stated agreement ship as part of the released
item rather than being added by us. Having no options to endorse is why the persona manipulation
had to be rebuilt on GlobalOpinionQA to be comparable against the others.}}
\label{{fig:prompts-syc}}
\end{{figure}}""")
