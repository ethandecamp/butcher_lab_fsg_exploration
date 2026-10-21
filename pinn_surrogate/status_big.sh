#!/usr/bin/env bash
# Progress of the v2 big run (see run_big.sh).
cd "$(dirname "$0")"
RUN=runs/big
if [ -f "$RUN/train.pid" ] && kill -0 "$(cat "$RUN/train.pid")" 2>/dev/null; then state="RUNNING (pid $(cat $RUN/train.pid))"
elif [ -f "$RUN/eval.json" ]; then state="FINISHED"
else state="NOT RUNNING - start or resume with ./run_big.sh"; fi
echo "state: $state"
[ -f "$RUN/metrics.jsonl" ] || { echo "no training steps logged yet"; exit 0; }
python3 - "$RUN" <<'PY'
import json, sys
run = sys.argv[1]
rows = [json.loads(l) for l in open(f"{run}/metrics.jsonl") if l.strip()]
total = int(open(f"{run}/STEPS").read()); last = rows[-1]
recent = rows[-10:]
sps = (recent[-1]["train_s"] - recent[0]["train_s"]) / max(1, recent[-1]["step"] - recent[0]["step"]) if len(recent) > 1 else last["sec_per_step"]
left = (total - last["step"]) * sps
print(f"step {last['step']}/{total} ({100*last['step']/total:.0f}%)  {sps:.3f} s/step  ~{left/3600:.1f} h of training left")
print(f"latest losses: data {last.get('loss_data', float('nan')):.3g}  tau {last.get('loss_tau', float('nan')):.3g}")
q = [(r["step"], r["quick_val_tau_rel_l2"]) for r in rows if "quick_val_tau_rel_l2" in r]
if q: print("quick val wall-shear error by step: " + "  ".join(f"{s}:{v:.3f}" for s, v in q))
PY
if [ -f "$RUN/eval.json" ]; then grep -E "median" "$RUN/stdout.log" | tail -2; fi
