#!/usr/bin/env bash
# The v2 "big" run, natively on the Mac. Safe to re-run at any time: it resumes.
#
#   ./run_big.sh          start, or pick up where it left off (after sleep/reboot/closing Terminal)
#   ./status_big.sh       progress, ETA, latest numbers
#
# What it does, each step skipped if already done:
#   1. packs data/packed_v2.npz from Dan's dataset (~1 min; downloads any iCloud-only files)
#   2. times 150 steps to size the run so training takes TRAIN_HOURS (default 3.5) -> runs/big/STEPS
#   3. trains in the background (nohup + caffeinate), checkpointing every 200 steps
# Lid closed = the Mac sleeps and training pauses; it continues when the Mac wakes.
# Reboot / killed Terminal = run ./run_big.sh again; at most 200 steps are redone.
set -euo pipefail
cd "$(dirname "$0")"
RUN=runs/big
TRAIN_HOURS=${TRAIN_HOURS:-3.5}
DATASET=../steady_state_dataset/dataset_v1_1k
PACKED=data/packed_v2.npz

if [ ! -x .venv/bin/python ]; then
  echo "No .venv here. First run:  python3 -m venv .venv && source .venv/bin/activate && pip install torch numpy matplotlib"
  exit 1
fi
source .venv/bin/activate
export PYTHONPATH=src
THREADS=${THREADS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)}

if [ -f "$RUN/train.pid" ] && kill -0 "$(cat "$RUN/train.pid")" 2>/dev/null; then
  echo "Already running (pid $(cat "$RUN/train.pid")). Check it with ./status_big.sh"; exit 0
fi
if [ -f "$RUN/eval.json" ]; then echo "Finished: $RUN/eval.json exists. ./status_big.sh for the numbers."; exit 0; fi

# The run's settings. Changing any of these after it starts makes --resume refuse (on purpose).
ARGS=(--packed "$PACKED" --n-train-geoms 920 --physics off --tau-weight 3
      --width 288 --depth 6 --enc-width 256 --latent 64 --wall-dist-input
      --batch-solves 16 --nodes-per-solve 512 --lr 1e-3 --lr-min 1e-5 --seed 0
      --log-every 100 --eval-every 1000 --ckpt-every 200 --threads "$THREADS")

if [ ! -f "$PACKED" ]; then
  echo "[1/3] packing $PACKED (~1 min) ..."
  python -m pinn_surrogate.pack --dataset "$DATASET" --out "$PACKED" --version v2
fi

mkdir -p "$RUN"
if [ ! -f "$RUN/STEPS" ]; then
  echo "[2/3] timing 150 steps on $THREADS threads to size the run ..."
  rm -rf "$RUN/_bench"
  python -m pinn_surrogate.train "${ARGS[@]}" --steps 150 --log-every 50 --eval-every 0 \
      --max-eval-solves 2 --out "$RUN/_bench" > "$RUN/_bench.log" 2>&1
  python - "$RUN" "$TRAIN_HOURS" <<'PY'
import json, sys
run, hours = sys.argv[1], float(sys.argv[2])
rows = [json.loads(l) for l in open(f"{run}/_bench/metrics.jsonl")]
sps = (rows[-1]["train_s"] - rows[0]["train_s"]) / (rows[-1]["step"] - rows[0]["step"])  # skip warm-up
steps = max(1000, int(hours * 3600 / sps) // 1000 * 1000)
open(f"{run}/STEPS", "w").write(str(steps))
print(f"      {sps:.3f} s/step -> {steps} steps (~{steps * sps / 3600:.1f} h of training)")
PY
fi
STEPS=$(cat "$RUN/STEPS")

echo "[3/3] training $STEPS steps in the background (log: $RUN/stdout.log)"
nohup caffeinate -i python -m pinn_surrogate.train "${ARGS[@]}" --steps "$STEPS" --out "$RUN" --resume \
    >> "$RUN/stdout.log" 2>&1 &
echo $! > "$RUN/train.pid"
echo "Started (pid $!). You can close this window. Check progress with ./status_big.sh"
