#!/usr/bin/env bash
# Wall-shear loss sweep: n160, physics off, 3000 steps, --tau-weight in WEIGHTS x SEEDS.
# Resumable chunks for the Cowork VM; call repeatedly until "all runs finished".
# Baseline (weight 0) = runs/grid/n160_physoff_s{0,1}.
set -u
cd "$(dirname "$0")/.."
CHUNK_S=${CHUNK_S:-165}; PAR=${PAR:-2}; THREADS=${THREADS:-2}
WEIGHTS=${WEIGHTS:-"1 10"}; SEEDS=${SEEDS:-"0 1"}; EXTRA=${EXTRA:-}; TAG=${TAG:-tau}
launched=0
for s in $SEEDS; do for w in $WEIGHTS; do
  out="runs/$TAG/w${w}_s${s}"
  [ -f "$out/eval.json" ] && continue
  [ "$launched" -ge "$PAR" ] && continue
  resume=""; [ -f "$out/checkpoint.pt" ] && resume="--resume"
  mkdir -p "$out"
  PYTHONPATH=src timeout "$CHUNK_S" python3 -m pinn_surrogate.train --n-train-geoms 160 --physics off \
    --seed "$s" --steps 3000 --threads "$THREADS" --tau-weight "$w" --ckpt-every 100 --log-every 100 \
    --out "$out" $EXTRA $resume >> "$out.out" 2>&1 &
  launched=$((launched + 1))
done; done
wait
[ "$launched" -eq 0 ] && echo "all runs finished"
for f in runs/$TAG/*.out; do printf '%s: ' "$f"; grep -E "step +[0-9]+ |median" "$f" | tail -1; done
