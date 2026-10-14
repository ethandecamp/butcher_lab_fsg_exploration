#!/usr/bin/env bash
# One line per grid run of seed $1: last step reached, or the test_B result once evaluated.
for f in runs/grid/*_s${1:-0}.out; do
  r=$(basename "$f" .out)
  if grep -q "test_B median" "$f"; then echo "$r DONE $(grep 'test_B median' "$f" | tail -1 | cut -c50-110)"
  else echo "$r step $(grep -oE 'step +[0-9]+' "$f" | tail -1 | grep -oE '[0-9]+')"; fi
done
