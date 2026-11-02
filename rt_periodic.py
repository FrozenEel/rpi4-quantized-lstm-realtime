"""Periodic inference task with absolute-time wakeups.

 sudo python3 rt_periodic.py --variant ort_int8 --period-ms 10 --policy fifo --cpu 3
 python3 rt_periodic.py --variant ort_int8 --period-ms 10 --policy other --cpu 3     (baseline, CFS)

Period k is released at t0 + k*T and has an implicit deadline of t0 + (k+1)*T.
Per period we log: wake-up lateness (actual wake - release) and response time (finish - release).
"""
import argparse, ctypes, ctypes.util, csv, gc, os, sys, time
import numpy as np
from src import variants

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="ort_int8"); ap.add_argument("--art", default="artifacts")
ap.add_argument("--period-ms", type=float, default=10.0); ap.add_argument("--duration-s", type=float, default=60)
ap.add_argument("--policy", choices=["fifo", "other"], default="fifo"); ap.add_argument("--prio", type=int, default=80)
ap.add_argument("--cpu", type=int, default=-1); ap.add_argument("--tag", default="")
ap.add_argument("--out", default="results")
a = ap.parse_args()

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
class TS(ctypes.Structure): _fields_ = [("s", ctypes.c_long), ("ns", ctypes.c_long)]
CLOCK_MONOTONIC, TIMER_ABSTIME = 1, 1

def sleep_until(ns):
    ts = TS(ns // 10**9, ns % 10**9)
    while libc.clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, ctypes.byref(ts), None) == ctypes.get_errno() == 4:
        pass  # EINTR

f = variants.load(a.variant, a.art)
if a.variant == "dummy": X = np.random.randn(100, 30, 24).astype(np.float32)
else: X = np.load(f"{a.art}/samples.npz")["Xte"]
for i in range(300): f(X[i % len(X):i % len(X) + 1])        # warm-up before going real-time

if a.cpu >= 0: os.sched_setaffinity(0, {a.cpu})
if a.policy == "fifo":
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(a.prio))   # needs root / CAP_SYS_NICE
libc.mlockall(3)                                                       # MCL_CURRENT|MCL_FUTURE: no page faults later
gc.collect(); gc.disable()

T = int(a.period_ms * 1e6); N = int(a.duration_s * 1e9 / T)
wake = np.empty(N); resp = np.empty(N)
t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC) + 200_000_000
for k in range(N):
    rel = t0 + k * T
    sleep_until(rel)
    w = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    f(X[k % len(X):k % len(X) + 1])
    e = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    wake[k] = (w - rel) / 1e3; resp[k] = (e - rel) / 1e3
miss = int((resp > T / 1e3).sum())
name = f"{a.variant}_{a.policy}_{int(a.period_ms)}ms{('_' + a.tag) if a.tag else ''}"
os.makedirs(a.out, exist_ok=True)
with open(f"{a.out}/rt_{name}.csv", "w", newline="") as fh:
    w_ = csv.writer(fh); w_.writerow(["k", "wake_late_us", "response_us"]); w_.writerows(zip(range(N), wake.round(1), resp.round(1)))
p = lambda x, q: np.percentile(x, q)
print(f"{name}: periods={N} deadline_misses={miss} ({100*miss/N:.3f}%)  "
      f"wake_late p50/p99/max = {p(wake,50):.0f}/{p(wake,99):.0f}/{wake.max():.0f} us  "
      f"response p50/p99/max = {p(resp,50):.0f}/{p(resp,99):.0f}/{resp.max():.0f} us")
