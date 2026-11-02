#!/bin/bash
# Run on the Raspberry Pi. Needs: sudo, stress-ng (sudo apt install stress-ng)
# Compares CFS vs SCHED_FIFO, with and without a CPU hog sharing the same core.
V=${1:-ort_int8}; P=${2:-10}; D=${3:-60}; CPU=3
for load in idle stress; do
  for pol in other fifo; do
    if [ $load = stress ]; then taskset -c $CPU stress-ng --cpu 1 --timeout $((D+10))s >/dev/null 2>&1 & SP=$!; sleep 2; fi
    sudo python3 rt_periodic.py --variant $V --period-ms $P --duration-s $D --policy $pol --cpu $CPU --tag $load
    if [ $load = stress ]; then kill $SP 2>/dev/null; wait $SP 2>/dev/null; fi
  done
done
