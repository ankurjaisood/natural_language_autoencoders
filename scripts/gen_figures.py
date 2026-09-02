"""Figure 1 for the paper: probe accessibility by depth and by dimensionality.

Both panels measure the same thing — how available the counterfactual signal is to a linear
reader — along two different axes, so they share a y-axis and belong in one float.
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = lambda f: json.loads(pathlib.Path(f).read_text())
COND = ["social proof", "sycophancy", "few-shot", "persona"]
COLOR = {"social proof": "#2e6f8e", "sycophancy": "#a8763e",
         "few-shot": "#6b8f5a", "persona": "#8e5b7a"}
lay, prb = R("results/layer_sweep_qwen.json"), R("results/probe_cis_qwen.json")["probes"]
site = lay["read_site"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)

for c in COND:
    p = sorted([x for x in lay["points"] if x["dataset"] == c and x["k"] > 1],
               key=lambda x: x["layer"])
    xs = [x["layer"] for x in p]
    ax1.plot(xs, [x["point"] for x in p], color=COLOR[c], lw=1.8, label=c, zorder=3)
    ax1.fill_between(xs, [x["lo"] for x in p], [x["hi"] for x in p],
                     color=COLOR[c], alpha=0.13, lw=0, zorder=2)
ax1.axvline(site, color="#333", ls="--", lw=1.1, zorder=1)
ax1.annotate("NLA read site", xy=(site, 0.955), xytext=(site - 0.6, 0.955),
             ha="right", va="top", fontsize=8.5, color="#333")
ax1.axhline(0.5, color="#999", lw=0.8, zorder=1)
ax1.set_xlabel("decoder block"); ax1.set_ylabel("probe AUROC")
ax1.set_title("(a) by depth", fontsize=10, loc="left")
ax1.set_ylim(0.42, 0.98); ax1.set_xlim(-0.5, 27.5)

ks = [1, 2, 5, 10, 25, 100, 500, 3584]
for c in COND:
    o = {x["k"]: x for x in prb if x["dataset"] == c}
    ys = [o[k]["point"] for k in ks]
    ax2.plot(range(len(ks)), ys, color=COLOR[c], lw=1.8, marker="o", ms=3.4, zorder=3)
    ax2.fill_between(range(len(ks)), [o[k]["lo"] for k in ks], [o[k]["hi"] for k in ks],
                     color=COLOR[c], alpha=0.13, lw=0, zorder=2)
ax2.axhline(0.5, color="#999", lw=0.8, zorder=1)
ax2.set_xticks(range(len(ks))); ax2.set_xticklabels([str(k) for k in ks], fontsize=8)
ax2.set_xlabel("coordinates selected ($k$)")
ax2.set_title("(b) by dimensionality, at the read site", fontsize=10, loc="left")
ax2.set_ylim(0.42, 0.98)

for a in (ax1, ax2):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)
ax1.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)

out = pathlib.Path("figures/fig1_accessibility.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(f"  wrote {out} and .png")
