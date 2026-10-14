#!/usr/bin/env bash
# Stage 5 ablation: n_train_geoms in {20, 80, 160} x physics in {on, off}, identical
# steps and seed. At most two runs at a time, 2 torch threads each (4-core VM).
# Logs: runs/grid/<name>.stdout.log (appended across restarts), plus each run's own runs/<name>/.
#
# Resume-safe: every run is started with --resume, so after a VM restart or laptop sleep
# just run this script again. Finished runs (checkpoint at the last step + eval.json) are
# skipped by train.py in a second; interrupted runs continue from their last checkpoint;
# a run whose saved config differs (e.g. a different STEPS) stops with an error instead of
# mixing two settings.
#
# Usage (from pinn_surrogate/):  bash ablation.sh            # 3000 steps, seed 0
#                                SEED=1 bash ablation.sh     # the report's second seed
# The report's grid used exactly these defaults (STEPS=3000, SHARE=0.1). In the Cowork VM,
# where processes die with each shell call, use ablation_chunked.sh instead.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
STEPS="${STEPS:-3000}"
SEED="${SEED:-0}"
THREADS="${THREADS:-2}"
SHARE="${SHARE:-0.1}"
MAX_PARALLEL=2
mkdir -p runs/grid

run() {  # $1 = n_train_geoms, $2 = on|off
  local name="n$1_phys$2_s${SEED}"
  echo "$(date '+%F %T') start $name"
  python3 -m pinn_surrogate.train --n-train-geoms "$1" --physics "$2" --steps "$STEPS" \
    --seed "$SEED" --physics-share "$SHARE" --threads "$THREADS" --ckpt-every 100 \
    --log-every 100 --out "runs/grid/$name" --resume >> "runs/grid/$name.stdout.log" 2>&1
  local status=$?
  echo "$(date '+%F %T') done  $name (exit $status)"
  return "$status"
}

# Physics-on runs are the slow ones, so start them first; a simple 2-slot job queue
# (`wait -n`, bash >= 4.3) starts the next run as soon as either slot frees up.
JOBS=("20 on" "80 on" "160 on" "20 off" "80 off" "160 off")
for job in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n || true
  done
  run $job &
done
wait

# A run succeeded iff it wrote eval.json (train.py evaluates at the end) and its config
# has this script's step count (guards against counting a stale run from another STEPS).
failed=0
for job in "${JOBS[@]}"; do
  set -- $job
  dir="runs/grid/n$1_phys$2_s${SEED}"
  if [ -f "$dir/eval.json" ] && grep -q "\"steps\": $STEPS," "$dir/config.json"; then
    :
  else
    echo "FAILED: n$1 physics $2 (see runs/grid/n$1_phys$2_s${SEED}.stdout.log)"
    failed=$((failed + 1))
  fi
done
echo "$(date '+%F %T') all runs finished, $failed failed"
python3 -m pinn_surrogate.summarize --runs runs/grid
python3 -m pinn_surrogate.compare --runs runs/grid
exit "$failed"
