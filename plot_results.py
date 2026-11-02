"""Aggregate results/rt_*.csv: per-configuration table (results/rt_aggregate.md) and a 2-panel figure (results/rt_figure.png).
Left: response-time CCDF for the 2 ms / loaded case, every repeat drawn. Right: deadline-miss rate per run."""
import glob, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = []
for p in sorted(glob.glob("results/rt_*.csv")):
    m = re.match(r"rt_(.+)_(other|fifo)_(\d+)ms_(idle|stress)(r\d+)?\.csv", os.path.basename(p))
    if not m: continue
    var, pol, per, load, rep = m.groups()
    d = np.genfromtxt(p, delimiter=",", names=True)
    T = int(per) * 1000.0
    runs.append(dict(pol=pol, per=int(per), load=load, rep=rep or "r0", resp=d["response_us"], T=T,
                     miss=int((d["response_us"] > T).sum()), n=len(d)))

keys = sorted({(r["per"], r["load"], r["pol"]) for r in runs}, key=lambda k: (-k[0], k[1], k[2]))
lines = ["| period | load | policy | runs | periods | misses (per run) | miss rate range | worst response (us) | resp p99 median (us) |", "|---|---|---|---|---|---|---|---|---|"]
for per, load, pol in keys:
    g = [r for r in runs if (r["per"], r["load"], r["pol"]) == (per, load, pol)]
    rates = [100 * r["miss"] / r["n"] for r in g]
    lines.append(f"| {per} ms | {load} | {'SCHED_FIFO' if pol=='fifo' else 'CFS (default)'} | {len(g)} | {sum(r['n'] for r in g)} | "
                 f"{', '.join(str(r['miss']) for r in g)} | {min(rates):.3f}% - {max(rates):.3f}% | {max(r['resp'].max() for r in g):.0f} | "
                 f"{np.median([np.percentile(r['resp'], 99) for r in g]):.0f} |")
open("results/rt_aggregate.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

col = {"other": "#c0392b", "fifo": "#1f6fb2"}
lab = {"other": "CFS (default)", "fifo": "SCHED_FIFO"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
a = ax[0]
for r in [r for r in runs if r["per"] == 2 and r["load"] == "stress"]:
    x = np.sort(r["resp"]); y = 1 - np.arange(len(x)) / len(x)
    a.plot(x, y, color=col[r["pol"]], alpha=.75, lw=1.3)
a.axvline(2000, color="k", ls="--", lw=1); a.text(2050, 0.35, "deadline = 2 ms", fontsize=8)
a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(3e-5, 1.2)
a.set_xlabel("response time (us)"); a.set_ylabel("fraction of periods slower than x")
a.set_title("2 ms period, CPU hog on the same core (each line = one 60 s run)", fontsize=9)
for p in ("other", "fifo"): a.plot([], [], color=col[p], label=lab[p])
a.legend(fontsize=8); a.grid(alpha=.25, which="both")
b = ax[1]
sel = [r for r in runs if r["per"] == 2 and r["load"] == "stress"]
sel.sort(key=lambda r: (r["pol"], r["rep"]))
b.bar(range(len(sel)), [100 * r["miss"] / r["n"] for r in sel], color=[col[r["pol"]] for r in sel])
b.set_xticks(range(len(sel))); b.set_xticklabels([f"{'FIFO' if r['pol']=='fifo' else 'CFS'}\n{r['rep']}" for r in sel], fontsize=8)
b.set_ylabel("deadline misses (% of 30000 periods)"); b.set_title("Deadline-miss rate per run", fontsize=9); b.grid(alpha=.25, axis="y")
fig.suptitle("Quantized LSTM inference on Raspberry Pi 4B (ONNX Runtime INT8, kernel 6.18 PREEMPT)", fontsize=10)
fig.tight_layout(); fig.savefig("results/rt_figure.png", dpi=160); print("saved results/rt_figure.png")
