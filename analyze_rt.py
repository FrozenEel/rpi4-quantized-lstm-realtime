"""Summarise results/rt_*.csv into a table and a plot (run on your laptop: needs matplotlib)."""
import glob, os, numpy as np
rows = []; data = {}
for p in sorted(glob.glob("results/rt_*.csv")):
    d = np.genfromtxt(p, delimiter=",", names=True); n = os.path.basename(p)[3:-4]
    per = float(n.split("_")[-2].replace("ms", "")) * 1000 if n.split("_")[-2].endswith("ms") else None
    data[n] = d
    rows.append((n, len(d), d["response_us"].max(), np.percentile(d["response_us"], 50), np.percentile(d["response_us"], 99),
                 d["wake_late_us"].max(), per))
with open("results/rt_summary.md", "w") as fh:
    fh.write("| run | periods | resp p50 us | resp p99 us | resp max us | wake-late max us | deadline misses |\n|---|---|---|---|---|---|---|\n")
    for n, N, mx, p50, p99, wmx, per in rows:
        miss = int((data[n]["response_us"] > per).sum()) if per else -1
        fh.write(f"| {n} | {N} | {p50:.0f} | {p99:.0f} | {mx:.0f} | {wmx:.0f} | {miss} |\n")
print(open("results/rt_summary.md").read())
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5))
    for n, d in data.items(): plt.hist(d["response_us"], bins=100, histtype="step", label=n, log=True)
    plt.xlabel("response time (us)"); plt.ylabel("count (log)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig("results/rt_response_hist.png", dpi=150); print("saved results/rt_response_hist.png")
except ImportError: pass
