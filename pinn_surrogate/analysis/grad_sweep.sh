#!/usr/bin/env bash
# Gradient-loss sweep: n160, physics off, 3000 steps. CONFIGS are "name:tau_weight:grad_weight".
# Resumable chunks for the Cowork VM; call repeatedly until "all runs finished".
set -u
cd "$(dirname "$0")/.."
CHUNK_S=${CHUNK_S:-165}; PAR=${PAR:-2}; THREADS=${THREADS:-2}
CONFIGS=${CONFIGS:-"t1g1:1:1 t0g1:0:1"}; SEEDS=${SEEDS:-"0 1"}
launched=0
for s in $SEEDS; do for c in $CONFIGS; do
  IFS=: read -r name tw gw <<< "$c"
  out="runs/grad/${name}_s${s}"
  [ -f "$out/eval.json" ] && continue
  [ "$launched" -ge "$PAR" ] && continue
  resume=""; [ -f "$out/checkpoint.pt" ] && resume="--resume"
  mkdir -p "$out"
  PYTHONPATH=src timeout "$CHUNK_S" python3 -m pinn_surrogate.train --n-train-geoms 160 --physics off \
    --seed "$s" --steps 3000 --threads "$THREADS" --tau-weight "$tw" --grad-weight "$gw" \
    --ckpt-every 100 --log-every 100 --out "$out" $resume >> "$out.out" 2>&1 &
  launched=$((launched + 1))
done; done
wait
[ "$launched" -eq 0 ] && echo "all runs finished"
for f in runs/grad/*.out; do printf '%s: ' "$f"; grep -E "step +[0-9]+ |median" "$f" | tail -1; done
