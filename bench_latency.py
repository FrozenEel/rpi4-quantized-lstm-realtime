"""Single-sample (batch=1) latency, model size and process memory per backend.
Each backend runs in its own process so the memory numbers are not mixed."""
import argparse, csv, os, subprocess, sys, time, json
import numpy as np
from src import variants

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="artifacts"); ap.add_argument("--variant")
ap.add_argument("--n", type=int, default=3000); ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--out", default="results/latency.csv")
a = ap.parse_args()

SIZE = {"torch_fp32": "lstm_fp32.pt", "torch_int8": "lstm_fp32.pt", "ort_fp32": "lstm_fp32.onnx", "ort_int8": "lstm_int8.onnx"}

if not a.variant:   # parent: spawn one child per backend
    rows = []
    for v in variants.NAMES:
        r = subprocess.run([sys.executable, __file__, "--variant", v, "--art", a.art, "--n", str(a.n), "--threads", str(a.threads)],
                           capture_output=True, text=True)
        if r.returncode: print(f"skip {v}: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}"); continue
        rows.append(json.loads(r.stdout.strip().splitlines()[-1])); print(rows[-1])
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    sys.exit()

import psutil
proc = psutil.Process()
base = proc.memory_info().rss
f = variants.load(a.variant, a.art, a.threads)
X = np.load(f"{a.art}/samples.npz")["Xte"]
for i in range(300): f(X[i % len(X):i % len(X) + 1])     # warm-up
t = np.empty(a.n)
for i in range(a.n):
    x = X[i % len(X):i % len(X) + 1]
    t0 = time.perf_counter_ns(); f(x); t[i] = (time.perf_counter_ns() - t0) / 1e3   # microseconds
rss = proc.memory_info().rss
print(json.dumps({"variant": a.variant, "threads": a.threads, "mean_us": round(t.mean(), 1), "p50_us": round(np.percentile(t, 50), 1),
                  "p95_us": round(np.percentile(t, 95), 1), "p99_us": round(np.percentile(t, 99), 1), "max_us": round(t.max(), 1),
                  "model_KB": round(os.path.getsize(f"{a.art}/{SIZE[a.variant]}") / 1024, 1),
                  "rss_MB": round(rss / 2**20, 1), "rss_delta_MB": round((rss - base) / 2**20, 1)}))
