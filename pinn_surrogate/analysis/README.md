# v1 PINN diagnostics (2026-10-20)

Post-mortem of the 12-run grid in `runs/grid/` (branch `feat/pinn-surrogate` @ 63f40c7), done
before designing v2. No training; reads `eval.json`, `metrics.jsonl` and `model.pt`.

## Run

From `pinn_surrogate/` (needs torch for a2–a5; a1 is numpy only). Each takes < 3 min on the VM.

```bash
python3 analysis/a1_eval_breakdown.py                                   # eval.json slices
python3 analysis/a2_near_wall.py runs/grid/n160_physoff_s0 runs/grid/n160_physon_s0
python3 analysis/a3_profile.py runs/grid/n160_physoff_s0 test_B         # near-wall profile
python3 analysis/a4_misc.py                                             # arc region, prior, weights
python3 analysis/a5_abs_error.py runs/grid/n160_physoff_s0              # absolute error vs distance
```

## Findings (n160 physics-off seed 0 unless stated; test_B)

1. **Not a generalization problem.** Median wall-shear error is the same in-distribution (val
   0.38–0.42) as on the OOD FSG shapes (test_B 0.34–0.38). Bulk velocity error is 4–9 %.
2. **More steps on the current loss does not help.** Quick-val shear error bottoms out around
   step 1,000 in most runs and then drifts flat or up while the data loss keeps falling
   (e.g. 0.428 → 0.337 → 0.384 → 0.393 → 0.382 → 0.369). The objective and the metric decouple.
3. **Training makes the peak worse, not better.** The hard-BC prior alone *over*-predicts the
   peak (ratio 1.15); training pulls it to 0.55–0.63, on training shapes too.
4. **The deficit lives inside ~20 µm of the wall.** In the peak-shear region the FEM profile is
   nearly linear out to ~20 µm (u_t / (τ d) = 0.96 → 0.89). The model matches it beyond ~20 µm
   (within ~5 %) but rolls off toward the wall: 0.80 at 15–20 µm, 0.67 at 2–4 µm.
5. **Why the loss allows that: it is absolute MSE.** RMS velocity error is 0.05–0.12 Ū at
   every distance; at 2–5 µm that is a 31 % relative error, in the bulk 6 %. Wall shear is
   never in the loss.
6. **The tanh mask puts a kink at δ in what the trunk must learn.** U = u / tanh(d/δ) must stay
   flat at τδ to ~δ = 20 µm and then rise (1.0 → 1.8 by 60 µm). The trunk has no wall-distance
   input, only (x, y), z, Re, so it has to locate that 20 µm kink from the shape code. It
   smooths it instead, which is exactly the under-shoot in (4).
7. **Trust weights are a minor factor.** Mean weight 0.78 within 5 µm, 0.93 at 5–10 µm, ≈1 beyond.
8. **Error is amplitude, concentrated on the cushion top.** 88 % of squared shear error is at
   arc fraction 0.2–0.7. Best-fit rescaling cuts error per flow bin from 0.34/0.37/0.41/0.51 to
   0.14/0.12/0.13/0.16; the highest-Re bin is worst both ways.
9. **Unused data.** Packed: 240/1,000 geometries, 3,072/~22k nodes per solve (~108 interior
   nodes < 5 µm vs ~420 in the full mesh). Dan's files also carry velocity gradients
   (u_x, u_y, v_x, v_y) at every node and wall τ at full arc resolution; none is in the loss.

## Follow-up: mask-width sweep (tests finding 6) — finding 6 is NOT the cause

`./analysis/delta_sweep.sh` (call repeatedly until "all runs finished"; ~2 min per call on the
VM). n160, physics off, 3,000 steps, seeds 0 / 1; δ = 0.05 is the existing `runs/grid` baseline.

| δ (H) | test_B shear err s0 / s1 | peak ratio (s0) | err after best-fit rescale (s0) | wall p err s0 / s1 |
|---|---|---|---|---|
| 0.05 (baseline) | 0.385 / 0.343 | 0.65 | 0.13 | 0.070 / 0.070 |
| 0.1 | 0.458 / 0.419 | 0.56 | 0.15 | 0.078 / 0.059 |
| 0.2 | 0.401 / 0.409 | 0.68 | 0.22 | 0.063 / 0.063 |
| 0.5 | 0.367 / 0.375 | 0.67 | 0.23 | 0.059 / 0.066 |

(Peak ratio and rescaled error on every other test_B solve.) The amplitude bias does not move
with δ. At δ = 0.5 the target U = u/m has no kink (U/(τδ) falls smoothly 0.96 → 0.75 from 2 to
60 µm), yet the prediction still undershoots at the wall by the same amount as before: u_t/(τ d)
= 0.68 at 2–4 µm vs 0.96 true (`a3_profile.py runs/delta/d0.5_s0 test_B`). Wider δ also makes the
profile *shape* worse (rescaled error 0.13 → 0.22, peak location 0.03 → 0.07). Keep δ = 0.05.
What remains is finding 5: the objective (absolute MSE on values; wall shear unsupervised).

## v2 step 1: direct wall-shear loss (`--tau-weight`)

New loss term `losses.wall_tau_loss` (per-solve relative squared error of τ_t at the 256
outline points, i.e. the square of the evaluation metric), fixed weight `--tau-weight`
(0 = off = bit-identical to v1; tested). n160, physics off, 3,000 steps, 2 seeds;
`./analysis/tau_sweep.sh`, compared with `python3 analysis/a6_compare.py <runs...>`.
~0.085 s/step at 2 threads (3x the data-only cost; one extra autograd pass at 8 x 256 wall points).

test_B medians, seed 0 / seed 1:

| tau weight | wall shear | wall p | u field | v field | peak ratio (s0) | shear by flow k0 → k3 (s0) |
|---|---|---|---|---|---|---|
| 0 (v1) | 0.385 / 0.343 | 0.070 / 0.070 | 0.066 / 0.063 | 0.28 / 0.25 | 0.65 | 0.34 0.37 0.41 0.51 |
| 1 | **0.151 / 0.146** | 0.056 / 0.068 | 0.079 / 0.073 | 0.29 / 0.25 | 0.89 | 0.13 0.14 0.15 0.22 |
| 10 | **0.131 / 0.111** | 0.085 / 0.080 | 0.160 / 0.152 | 0.41 / 0.36 | 0.96 | 0.12 0.12 0.13 0.16 |

- Shear error drops 2.5–3x, and the run beats the v1 run of the same seed on 98–100 % of
  test_B solves (val: 92–98 %). The amplitude bias is mostly gone (peak ratio 0.65 → 0.89 / 0.96).
- Cost: weight 1 raises velocity-field error ~15–35 % (u 0.066 → 0.079 test_B, 0.092 → 0.125 val)
  with wall pressure unchanged; weight 10 doubles u error and worsens wall pressure.
- val is now harder than test_B (shear 0.25 / 0.19–0.21 vs 0.15 / 0.11–0.13). val has higher
  blockage (median 0.53 vs 0.34), and v1 already showed error depends on shape.
- Still under-fit, not over-fit: training-shape shear error (0.19 at w1, 0.14 at w10) is no
  better than held-out. Quick-val shear now keeps falling until ~step 2,000–2,500 instead of
  bottoming out at 1,000, so more steps / capacity now have something to act on.
