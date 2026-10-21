#!/usr/bin/env bash
# Mask-width sweep: n160, physics off, 3000 steps, delta in DELTAS x SEEDS.
# Resumable chunks for the Cowork VM (each call advances at most PAR unfinished runs);
# call repeatedly until it prints "all runs finished". Baseline delta=0.05 = runs/grid/n160_physoff_s{0,1}.
set -u
cd "$(dirname "$0")/.."
CHUNK_S=${CHUNK_S:-165}; PAR=${PAR:-2}; THREADS=${THREADS:-2}
DELTAS=${DELTAS:-"0.1 0.2 0.5"}; SEEDS=${SEEDS:-"0 1"}
launched=0
for s in $SEEDS; do for d in $DELTAS; do
  out="runs/delta/d${d}_s${s}"
  [ -f "$out/eval.json" ] && continue
  [ "$launched" -ge "$PAR" ] && continue
  resume=""; [ -f "$out/checkpoint.pt" ] && resume="--resume"
  mkdir -p "$out"
  PYTHONPATH=src timeout "$CHUNK_S" python3 -m pinn_surrogate.train --n-train-geoms 160 --physics off \
    --seed "$s" --steps 3000 --threads "$THREADS" --delta "$d" --ckpt-every 100 --log-every 100 \
    --out "$out" $resume >> "$out.out" 2>&1 &
  launched=$((launched + 1))
done; done
wait
[ "$launched" -eq 0 ] && echo "all runs finished"
for f in runs/delta/*.out; do printf '%s: ' "$f"; grep -E "step +[0-9]+ |median" "$f" | tail -1; done
