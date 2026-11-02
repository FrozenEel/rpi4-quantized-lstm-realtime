| period | load | policy | runs | periods | misses (per run) | miss rate range | worst response (us) | resp p99 median (us) |
|---|---|---|---|---|---|---|---|---|
| 10 ms | idle | SCHED_FIFO | 1 | 6000 | 0 | 0.000% - 0.000% | 1561 | 950 |
| 10 ms | idle | CFS (default) | 1 | 6000 | 0 | 0.000% - 0.000% | 1815 | 938 |
| 10 ms | stress | SCHED_FIFO | 1 | 6000 | 0 | 0.000% - 0.000% | 1711 | 1230 |
| 10 ms | stress | CFS (default) | 1 | 6000 | 0 | 0.000% - 0.000% | 5182 | 3343 |
| 2 ms | idle | SCHED_FIFO | 1 | 30000 | 0 | 0.000% - 0.000% | 1756 | 773 |
| 2 ms | idle | CFS (default) | 1 | 30000 | 3 | 0.010% - 0.010% | 3019 | 814 |
| 2 ms | stress | SCHED_FIFO | 4 | 120000 | 0, 0, 0, 32 | 0.000% - 0.107% | 9123 | 918 |
| 2 ms | stress | CFS (default) | 4 | 120000 | 956, 2359, 409, 1130 | 1.363% - 7.863% | 15612 | 4576 |
