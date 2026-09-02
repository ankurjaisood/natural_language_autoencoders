import json, pathlib
R = lambda f: json.loads(pathlib.Path(f).read_text())
cis = R("results/bootstrap_cis.json"); A = cis["aurocs"]
mde = R("results/mde.json")["conditions"]
pao = {(r["dataset"], r["model"]): r for r in R("results/power_at_observed.json")["rows"]}
prb = R("results/probe_cis_qwen.json")["probes"]
emb = R("results/embedding_reader.json")["rows"]
DAG = r"\dag"

def ci(r, key=None):
    c = r[key] if key else r
    if c is None: return "---"
    if not c.get("estimable"): return f"{c['point']:.3f}{DAG}"
    return f"{c['point']:.3f} [{c['lo']:.2f}, {c['hi']:.2f}]"

def cell(ds, md, tg):
    r = next((x for x in A if x["dataset"]==ds and x["model"]==md and x["target"]==tg), None)
    return "---" if (r is None or r.get("status")!="ok") else ci(r)

out = []
# ───────────────────────── METHODS: CHECKPOINTS ─────────────────────────
# read straight from each AV's sidecar and the base model config, so the paper cannot
# disagree with the artifacts about which layer was read
import glob, yaml
CKPT = [("qwen",  "nla-qwen2.5-7b-L20",  "Qwen2.5-7B-Instruct",  28),
        ("gemma", "nla-gemma3-12b-L32",  "Gemma-3-12B-IT",       48)]
meta_by, sha_by = {}, {}
for _key, repo, base, nblocks in CKPT:
    shas, meta = {}, {}
    for part in ("av", "ar"):
        g = glob.glob(f"{pathlib.Path.home()}/.cache/huggingface/hub/"
                      f"models--kitft--{repo}-{part}/snapshots/*/nla_meta.yaml")
        shas[part] = pathlib.Path(g[0]).parent.name[:7] if g else "?"
        if g and not meta:
            meta = yaml.safe_load(pathlib.Path(g[0]).read_text())
    idx = meta.get("extraction_layer_index") or (meta.get("critic") or {}).get("extraction_layer_index")
    meta_by[_key] = (repo, base, nblocks, idx, meta.get("d_model"))
    sha_by[_key] = shas
q, g_ = meta_by["qwen"], meta_by["gemma"]
sq, sg = sha_by["qwen"], sha_by["gemma"]
# Transposed: two checkpoints as COLUMNS. The row-per-checkpoint form ran 54pt over a
# NeurIPS column, because one row carried a long repo path, a base model name and two SHAs.
tck = "\n".join([
    rf"Checkpoint pair & \texttt{{kitft/{q[0]}}} & \texttt{{kitft/{g_[0]}}} \\",
    rf"Base model & {q[1]} & {g_[1]} \\",
    rf"Decoder blocks & {q[2]} & {g_[2]} \\",
    rf"Read site (block) & {q[3]} & {g_[3]} \\",
    rf"$d_\text{{model}}$ & {q[4]} & {g_[4]} \\",
    rf"AV snapshot & \texttt{{{sq['av']}}} & \texttt{{{sg['av']}}} \\",
    rf"AR snapshot & \texttt{{{sq['ar']}}} & \texttt{{{sg['ar']}}} \\",
])
out.append(rf"""\begin{{table}}[t]
\centering
\small
\caption{{\textbf{{NLA checkpoints audited.}} Each pair is fine-tuned from the model whose
activations it reads, and is released as an \texttt{{-av}}/\texttt{{-ar}} pair alongside the
reference implementation at
\url{{https://github.com/kitft/natural_language_autoencoders}}; the checkpoints themselves are
collected under \texttt{{kitft/nla-models}} on the Hugging Face Hub. The read site is taken from each AV's
\texttt{{nla\_meta.yaml}} (\texttt{{extraction\_layer\_index}}) rather than assumed, and our
capture asserts that the slice it reads is identical to the one the released extractor produces.
Two further pairs in the same release (27B and 70B) do not fit our hardware and are untested.}}
\label{{tab:checkpoints}}
\begin{{tabular}}{{lll}}
\toprule
& Qwen pair & Gemma pair \\
\midrule
{tck}
\bottomrule
\end{{tabular}}
\end{{table}}""")

# ─────────────────────────── METHODS: DATASETS ───────────────────────────
DS = [("sycophancy", "stated opinion", "2", "32", "sycophancy_influence_raw", "sycophancy_influence_raw_gemma"),
      ("GlobalOpinionQA", "persona",      "2--4", "2{,}556", "influence_persona_raw", "influence_persona_gemma_raw"),
      ("GlobalOpinionQA", "social proof", "2--4", "2{,}556", "influence_socialproof_raw", "influence_socialproof_gemma_raw"),
      ("GlobalOpinionQA", "few-shot",     "2--4", "2{,}556", "influence_fewshot_raw", "influence_fewshot_gemma_raw")]
rows = []
for ds, inf, opts, avail, q, g in DS:
    tq = R(f"results/trials/{q}.json")["trials"]; tg = R(f"results/trials/{g}.json")["trials"]
    nc = len({x["claim"] for x in tq})
    sq = sum(bool(x["swayed"]) for x in tq) / len(tq)
    sg = sum(bool(x["swayed"]) for x in tg) / len(tg)
    rows.append(f"{ds} & {inf} & {opts} & {avail} & {nc} & {len(tq)} & "
                f"{sq:.1%} & {sg:.1%} \\\\".replace("%", r"\%"))
tds = "\n".join(rows)
out.append(rf"""\begin{{table}}[t]
\centering
\small
\caption{{\textbf{{Corpora and influence conditions.}} A \emph{{claim}} is the question; a
\emph{{trial}} is one attempt to push it, and we evaluate on questions held out entirely, so
trials from one question never appear on both sides of a split. \textbf{{The swayed columns give
the share of trials where the influence actually changed the model's answer, and they are here
for two reasons.}} First as a manipulation check: an influence that never moved the model would
leave nothing for a readout to report, and any null we found afterwards would be empty. Second,
and more importantly, the rate sets how many swayed trials we have to learn from --- and that is
what limits the conclusions. Gemma's 5.5\% on few-shot is 22 swayed trials out of 400, too few
to put an error bar on at all (Table~\ref{{tab:power}}). The influence moved Qwen two to three
times as often as Gemma throughout, which is why Qwen carries the paper's claims and Gemma
serves as a robustness check rather than a second result. From GlobalOpinionQA we keep only
questions with no correct answer, so a changed answer cannot be a corrected mistake; its
$\sim$10{{,}}000 sycophancy rows are variants of just 32 questions, which is why the persona
manipulation was rebuilt on the larger corpus.}}
\label{{tab:corpora}}
\begin{{tabular}}{{llccccrr}}
\toprule
& & & \multicolumn{{2}}{{c}}{{Claims}} & & \multicolumn{{2}}{{c}}{{Swayed}} \\
\cmidrule(lr){{4-5}} \cmidrule(lr){{7-8}}
Corpus & Influence & Options & Available & Used & Trials & Qwen & Gemma \\
\midrule
{tds}
\bottomrule
\end{{tabular}}
\end{{table}}""")

# ─────────────────────────── TABLE 1 ───────────────────────────
ORDER = [("persona","qwen"),("social proof","qwen"),("few-shot","qwen"),("sycophancy","qwen"),
         ("persona","gemma"),("social proof","gemma"),("few-shot","gemma"),("sycophancy","gemma")]
rows = []
prev = None
for ds, md in ORDER:
    if prev and md != prev: rows.append(r"\midrule")
    prev = md
    fl = next(x for x in A if x["dataset"]==ds and x["model"]==md
              and x["target"]=="influence changed answer" and x.get("status")=="ok")
    rows.append(f"{ds} & {'Qwen' if md=='qwen' else 'Gemma'} & "
                f"{cell(ds,md,'answer chosen')} & {cell(ds,md,'direction of influence')} & "
                f"{cell(ds,md,'influence changed answer')} & "
                f"[{fl['floor_lo']:.2f}, {fl['floor_hi']:.2f}] \\\\")
t1 = "\n".join(rows)
out.append(rf"""\begin{{table}}[t]
\centering
\small
\caption{{\textbf{{What readouts carry.}} Bag-of-words AUROC with 95\% claim-clustered
bootstrap intervals, against each run's own empirical floor. Readouts recover the answer and
the direction of the applied influence, but not whether the influence changed the answer:
every GlobalOpinionQA condition with an estimable interval overlaps its floor on that
target, while sycophancy clears it
(discussed in \S\ref{{sec:notcarried}}). Cell conventions used in all tables: --- means the
quantity does not exist, because a fold is single-class on the observed data; $\dag$ means the
point estimate exists but its interval is withheld, because the bootstrap redraw rate exceeds
20\% and the accepted resample is measurably biased. This table has 3 dashes and 5 daggers of
24 cells.}}
\label{{tab:carry}}
\begin{{tabular}}{{llcccc}}
\toprule
& & \multicolumn{{3}}{{c}}{{Recovered from the readout}} & \\
\cmidrule(lr){{3-5}}
Influence & Model & Answer & Direction & Changed answer & Floor \\
\midrule
{t1}
\bottomrule
\end{{tabular}}
\end{{table}}""")

# ─────────────────────────── TABLE 2 ───────────────────────────
rows, prev = [], None
for c in sorted(mde, key=lambda c: (c["model"] != "qwen", c["dataset"])):
    if prev and c["model"] != prev: rows.append(r"\midrule")
    prev = c["model"]
    p = pao[(c["dataset"], c["model"])]
    m = f"{c['mde']:.3f}" if c["mde"] else "---"
    pw = f"{p['power_at_observed']:.2f}" if p["power_at_observed"] == p["power_at_observed"] else "---"
    rows.append(f"{c['dataset']} & {'Qwen' if c['model']=='qwen' else 'Gemma'} & "
                f"{c['n_positives']} & {c['claims_with_positives']} & {c['score_icc']:.2f} & "
                f"{c['observed']:.3f} & {m} & {pw} \\\\")
t2 = "\n".join(rows)
out.append(rf"""\begin{{table}}[t]
\centering
\small
\caption{{\textbf{{How large an effect we could have detected.}} Minimum detectable effect is the
smallest true AUROC this design would distinguish from the floor with 80\% power, simulated
with each condition's own claim-level intraclass correlation (Appendix~\ref{{app:power}})
and its real distribution of positives across claims. Power at observed is the probability of detecting an effect the size
we measured. On Qwen an effect that size
would have been found 3--7\% of the time, so the measurement is informative. Social proof on
Gemma reaches 43\%, so that condition cannot separate a weak effect from none.
few-shot/Gemma yields no MDE at any effect size: 22 positives in 14 of 200 claims leaves most
claim resamples with an empty fold, so a claim-clustered bootstrap cannot be formed.
Conventions as in Table~\ref{{tab:carry}}.}}
\label{{tab:power}}
\begin{{tabular}}{{llcccccc}}
\toprule
Influence & Model & Pos. & Claims w/ pos. & ICC & Observed & MDE & Power at obs. \\
\midrule
{t2}
\bottomrule
\end{{tabular}}
\end{{table}}""")

# ─────────────────────────── TABLE 3 ───────────────────────────
rows = []
for d in ("persona", "social proof", "few-shot", "sycophancy"):
    o = [p for p in prb if p["dataset"] == d]
    p1 = next(x for x in o if x["k"] == 1); pf = next(x for x in o if x["k"] == 3584)
    e = next(r for r in emb if r["dataset"] == d and r.get("target") == "influence changed answer")
    fl = next(x for x in A if x["dataset"]==d and x["model"]=="qwen"
              and x["target"]=="influence changed answer")
    m = next((c["mde"] for c in mde if c["dataset"]==d and c["model"]=="qwen"), None)
    rows.append(f"{d} & {ci(pf)} & {ci(p1)} & {ci(e,'bow')} & {ci(e,'embedding')} & "
                f"{fl['floor_hi']:.3f} & {(f'{m:.3f}' if m else '---')} \\\\")
t3 = "\n".join(rows)
out.append(rf"""\begin{{table}}[t]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\caption{{\textbf{{Ruling out four alternative explanations.}} All scores are on the Qwen checkpoint, recovering whether the influence changed
the model's answer. Each column addresses one explanation, discussed in
Section~\ref{{sec:alternatives}}. The floor column gives the upper end of the empirical floor,
which a score must exceed; Table~\ref{{tab:carry}} prints the same floors as full intervals.
Conventions as in Table~\ref{{tab:carry}}.}}
\label{{tab:alternatives}}
\begin{{tabular}}{{lcccccc}}
\toprule
& \multicolumn{{2}}{{c}}{{Probe on the activation}} & \multicolumn{{2}}{{c}}{{Reader on the readout}} & & \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
Influence & All dims & 1 coordinate & Bag-of-words & Embedding & Floor (upper) & MDE \\
\midrule
{t3}
\bottomrule
\end{{tabular}}
\end{{table}}""")
print("\n\n".join(out))
