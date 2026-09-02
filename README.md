# Natural Language Autoencoders (NLA) — faithfulness audit

This branch is an audit of two released NLA checkpoints. We influence a model, verbalize the
activation behind its answer, and ask whether the readout reports three things: which way the
influence pushed, which option the model chose, and whether the influence changed that choice.
The readouts answer the first two and not the third.

Every reported number is reproduced by the code and artifacts on this branch.
[REPRODUCE.md](REPRODUCE.md) covers the environment, pinned checkpoint SHAs, and what each tier
does or does not reproduce exactly.

## Reproduce everything

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-bench.txt
bash reproduce.sh 1                # analysis only, no GPU and no API key
bash reproduce.sh 2                # + activations: GPU and the released checkpoints
bash reproduce.sh 3                # + judge arms: an authenticated `claude` CLI
python compare_repro.py results_repro
```

`compare_repro.py` diffs every number produced against the committed `results/` and names
anything that moved further than its tier allows. Tier 1 should be exact.

## Reproduce one result at a time

Each row regenerates the numbers behind one part of the audit. All of them write to
`results_repro/` and leave the committed `results/` untouched, so you can diff afterwards.
`$V` is `.venv/bin/python`.

| Result | Tier | Command |
|---|---|---|
| Both models moved by influence | 1 | rates are counted from the committed trial sets in `results/trials/` |
| The readout carries influence and answer but not cause | 1 | `$V -m bench.run_readout_capacity --raw results/trials/sycophancy_influence_raw.json --out-dir results_repro` (repeat for the other four trial files) |
| What the readout adds over knowing the answer | 1 | `$V -m bench.run_prompt_baseline --model qwen --raw results/trials/influence_persona_raw.json --out-dir results_repro --label prompt_baseline_persona` |
| The activation held what the readout left out | 2 | `$V -m bench.run_bandwidth_control --model qwen --raw results/trials/influence_socialproof_raw.json --out-dir results_repro` |
| What the readout carries that the question does not | 1 | same `run_prompt_baseline` call as above, once per trial file and checkpoint |
| Why the information is missing | 2 | `$V -m bench.run_locate_loss --model qwen --raw results/trials/sycophancy_influence_raw.json --out-dir results_repro` |
| The judge misses a signal we planted | 3 | `$V -m bench.run_judge_competence --model qwen --raw results/trials/sycophancy_influence_raw.json --out-dir results_repro` then `$V -m bench.run_monitor_sensitivity --model qwen --raw <same> --out-dir results_repro` |

Two supporting controls, both tier 1:

```bash
$V -m bench.run_classifier_scrutiny --trials-dir results/trials --out-dir results_repro
$V -m bench.run_readout_signal      --trials-dir results/trials --out-dir results_repro
```

## Reproduce the paper's tables and figures

The workshop paper adds intervals, an empirical floor, and a power analysis on top of the point
estimates above. These write to `results/` by default; pass `--out-dir results_repro` to leave the
committed artifacts untouched.

| Artifact in the paper | Tier | Command |
|---|---|---|
| Table 4, what readouts carry | 1 | `$V -m bench.run_bootstrap_cis --out-dir results_repro` |
| Table 5, minimum detectable effect | 1 | `$V -m bench.run_mde --out-dir results_repro` |
| Table 5, power at observed | 1 | `$V -m bench.run_power_at_observed --mde results_repro/mde.json --out-dir results_repro` |
| Table 6, probe columns | 2 | `$V -m bench.run_probe_cis --model qwen --out-dir results_repro` |
| Table 6, embedding column | 1 | `$V -m bench.run_embedding_reader --out-dir results_repro` |
| Appendix, probe sweep across depth | 1 | `$V -m bench.run_layer_sweep --model qwen --out-dir results_repro` |
| Floor-width robustness check | 1 | `$V -m bench.run_fixed_floor_check` |

Two ordering constraints. `run_probe_cis` is the only GPU step here and it saves every decoder
block's activations to `results/activations/`, which `run_layer_sweep` then reads on CPU — so the
sweep needs the probe run first. And `run_power_at_observed` reads the MDE grid, so it needs
`run_mde` first.

`results/activations/` is gitignored: 582 MB across four files, each over GitHub's 100 MB limit,
and regenerable by the command above.

Once the artifacts exist, the floats are generated rather than hand-written, so the numbers cannot
drift from `results/*.json`:

```bash
$V scripts/gen_tables.py            # Tables 1, 2, 4, 5, 6
$V scripts/gen_influence_box.py     # Table 3, the four influence conditions
$V scripts/gen_appendix_tables.py   # Tables A1 and A2
$V scripts/gen_figures.py           # Figure 1
$V scripts/gen_appendix_figs.py     # Appendix prompt and chat-template figures
$V -m bench.verify_claims           # recompute every derived number in the paper
```

`verify_claims` exists because derived values — "five of six", "89% of the full probe" — are
correct when written and stop being correct when scope shifts. Run it against a draft with
`--check paper.tex` and it flags any `N of M` the artifacts no longer support.

Every script takes `--model {qwen,gemma}`. Pass `--raw` explicitly: without it `--out-dir`
doubles as the input directory and the run fails looking for trials it has not written yet.

## Reproduce from the checkpoints, not from the committed trials

The commands above read the trial sets in `results/trials/`. To regenerate those from the
checkpoints themselves, which takes hours and needs a GPU:

```bash
FULL=1 bash reproduce.sh all
```

This re-runs both models over both corpora, recaptures activations, and regenerates every
readout before any analysis. The sycophancy trials reproduce field-for-field because decoding
is greedy and seeded. The GlobalOpinionQA battery samples its claims, so those trials will
differ in composition while the reported statistics hold.

## Where the outputs live

`results/` is one folder per test, numbered in the order the tests were run. Everything a
script *reads* lives in `results/trials/`; everything it *writes* goes to the matching folder.
[REPRODUCE.md](REPRODUCE.md) has the full map.

The paper's artifacts are the exception: they sit at the root of `results/` rather than in a
per-test folder, because each one spans every test. Those are `bootstrap_cis.json`, `mde.json`,
`power_at_observed.json`, `mde_fixed_floor.json`, `probe_cis_qwen.json`, `layer_sweep_qwen.json`,
`embedding_reader.json` and `axis_alignment.json`. Every number in the paper comes from one of
them, and `bench.verify_claims` recomputes the derived ones.


---

## Upstream

This is a fork. The NLA method, the training code under `nla/`, and the released checkpoints are
the work of the authors cited below; their full training and inference documentation lives in the
upstream repository at <https://github.com/kitft/natural_language_autoencoders>. Everything under
`bench/`, `scripts/` and `results/` is the audit, and is the only part this README documents.

---

## Citation

For attribution in academic contexts, please cite this work as

> Fraser-Taliente, Kantamneni, Ong et al., "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations", Transformer Circuits, 2026.

```bibtex
@article{frasertaliente2026nla,
  author  = {Fraser-Taliente, Kit and Kantamneni, Subhash and Ong, Euan and Mossing, Dan and Lu, Christina and Bogdan, Paul C. and Ameisen, Emmanuel and Chen, James and Kishylau, Dzmitry and Pearce, Adam and Tarng, Julius and Wu, Alex and Wu, Jeff and Zhang, Yang and Ziegler, Daniel M. and Hubinger, Evan and Batson, Joshua and Lindsey, Jack and Zimmerman, Samuel and Marks, Samuel},
  title   = {Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  journal = {Transformer Circuits Thread},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/nla/index.html}
}
```

## License

Apache-2.0 ([LICENSE](LICENSE)). Released checkpoints additionally inherit the
license of their base model (Gemma, Llama-3.3) — see the NOTICE files in each
HF repo.
