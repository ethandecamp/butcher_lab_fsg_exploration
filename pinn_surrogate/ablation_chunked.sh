#!/usr/bin/env bash
# Run the ablation grid in short, resumable chunks.
#
# Why this exists: in the Cowork Linux VM every process started by a shell call is killed
# when that call returns (even with nohup/setsid), and a call lasts at most ~3 minutes. So
# each invocation of this script advances at most two unfinished runs for CHUNK_S seconds
# and exits; train.py checkpoints every 100 steps and --resume continues bit-exactly.
# Call it repeatedly until it prints "all runs finished". On a normal machine, just use
# ablation.sh instead.
set -u
cd "$(dirname "$0")"
CHUNK_S=${CHUNK_S:-160}
STEPS=${STEPS:-3000}
SHARE=${SHARE:-0.1}
PAR=${PAR:-2}
THREADS=${THREADS:-2}
SEED=${SEED:-0}
launched=0
for n in 20 80 160; do
  for phys in on off; do
    out="runs/grid/n${n}_phys${phys}_s${SEED}"
    [ -f "$out/eval.json" ] && continue
    [ "$launched" -ge "$PAR" ] && continue
    resume=""; [ -f "$out/checkpoint.pt" ] && resume="--resume"
    mkdir -p "$out"
    PYTHONPATH=src timeout "$CHUNK_S" python3 -m pinn_surrogate.train --n-train-geoms "$n" \
      --physics "$phys" --physics-share "$SHARE" --seed "$SEED" --steps "$STEPS" --threads "$THREADS" \
      --ckpt-every 100 --log-every 100 --out "$out" $resume >> "$out.out" 2>&1 &
    launched=$((launched + 1))
  done
done
wait
if [ "$launched" -eq 0 ]; then echo "all runs finished"; fi
for f in runs/grid/*_s${SEED}.out; do printf '%s: ' "$f"; grep -E "step +[0-9]+ |median" "$f" | tail -1; done
