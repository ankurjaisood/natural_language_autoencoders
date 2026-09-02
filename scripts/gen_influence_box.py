"""The four influence conditions as one compact float, with no package dependencies.

Full prompts run 386-1298 characters; four verbatim would consume most of a page against a
1.0-page Methods budget. Persona and social proof are applied to the SAME question, so showing
the question once and then only what each condition prepends fits in about a third of the space
and makes the contrast visible, which four full prompts do not.

Uses only tabular + booktabs, already required by the other tables. No tcolorbox, no listings.
"""
import json, pathlib, re

R = lambda f: json.loads(pathlib.Path(f"results/trials/{f}.json").read_text())["trials"]

def tex(s, n=None):
    """LaTeX-safe: corpus text carries curly quotes, %, & and _."""
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("“", "``").replace("”", "''").replace("—", "---"))
    for a, b in (("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    s = " ".join(s.split())
    return s if n is None or len(s) <= n else s[:n].rstrip() + r"\ldots"

per, soc = R("influence_persona_raw")[0], R("influence_socialproof_raw")[0]
syc, few = R("sycophancy_influence_raw")[0], R("influence_fewshot_raw")[0]
q_full = per["prompt"].split("\n\n", 1)[1].strip().splitlines()[0]
# sycophancy's persona is built INTO the item and endorses the claim, so it has no separable
# "prefix" and applies to a different corpus. Showing it under the GlobalOpinionQA question
# would imply the four conditions share one question. They do not.
# show the part that DOES the influencing -- the stated agreement -- not the bio opening
_p = syc["prompt"]
_bio = _p.split(".")[0] + "."                                  # "Hello, my name is ..."
_stance = "I agree with the claim" if "I agree with the claim" in _p else "I disagree with the claim"
_i = _p.index(_stance)
_span = _p[_i:_i + 96]
_span = _span[:_span.rfind(" ")]                               # do not cut mid-word
syc_open = tex(_bio + " [...] " + _span, 200) + r"\ldots"
# the claim itself sits after the question phrasing, before Choices
syc_claim = tex(_p.split("?", 1)[-1].split("Choices")[0], 120)

# the sycophancy base question is what remains once the persona and its stated agreement are
# stripped -- the codebase's own strip_persona does exactly this, so the corpus is separable in
# the same way the GlobalOpinionQA conditions are
syc_q = tex(_p.split("?", 1)[-1].split("Choices")[0], 130)

print(rf"""\begin{{table}}[t]
\centering
\footnotesize
\begin{{tabular}}{{@{{}}p{{0.20\linewidth}}p{{0.76\linewidth}}@{{}}}}
\toprule
\multicolumn{{2}}{{@{{}}l}}{{\textbf{{GlobalOpinionQA}} --- 200 claims and three influence conditions}} \\
\midrule
\emph{{base question}} & {tex(q_full, 150)} \\
& (A) Has too much influence \quad (B) Has too little influence \\
& (C) Has about the right amount of influence \\
\addlinespace[4pt]
\emph{{prepended influence}} & \\
\quad persona & ``{tex(per['prompt'].split(chr(10)+chr(10))[0], 120)}'' \\
\addlinespace[2pt]
\quad social proof & ``{tex(soc['prompt'].split(chr(10)+chr(10))[0], 120)}'' \\
\addlinespace[2pt]
\quad few-shot & Four questions from elsewhere in the corpus, each shown with
\texttt{{Answer: (A)}} \\
\midrule
\multicolumn{{2}}{{@{{}}l}}{{\textbf{{Sycophancy}} --- 32 claims and one influence condition}} \\
\midrule
\emph{{base question}} & Do you agree or disagree with this claim? {syc_q} \\
& (A) Agree \quad (B) Disagree \\
\addlinespace[4pt]
\emph{{prepended influence}} & \\
\quad stated opinion & ``{syc_open}'' \\
\bottomrule
\end{{tabular}}
\caption{{\textbf{{The four influence conditions.}} Each corpus is shown as a base question and
what the influence prepends to it. The three GlobalOpinionQA conditions share a question, which
is what lets them be compared directly: persona and social proof differ only in whether a person
is attached to the stated preference, and few-shot pushes toward a \emph{{letter}} rather than any
option's meaning, so the pushed option is whichever the question lists first. Sycophancy items
ask for agreement with a \emph{{claim}} rather than a choice among several options, so its speaker
endorses the claim itself; having no options to endorse is why the persona manipulation had to be
rebuilt on GlobalOpinionQA in order to be compared against the others. Full example prompts are
in Appendix~\ref{{app:prompts}}.}}
\label{{tab:conditions}}
\end{{table}}""")
