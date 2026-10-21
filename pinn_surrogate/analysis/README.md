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

## v2 step 2: near-wall velocity-gradient loss (`--grad-weight`) — does not help

`data/packed_v1_grads.npz` (gitignored, ~7 s to build: `PYTHONPATH=src python3 -m
pinn_surrogate.pack_grads --dataset ../steady_state_dataset/dataset_v1_1k`) holds Dan's
(u_x, u_y, v_x, v_y) at the exact v1 packed nodes (seeded replay; xy checked equal for all 960
solves). Sanity check (`a7_grad_check.py`): in the peak region his du_t/dn → τ_wall near the
wall (0.95 at 2–5 µm, 0.88 at 5–10, 0.78 at 10–20, 0.56 at 20–50), consistent with a3's profile.
`losses.grad_loss` = per-solve weighted relative squared error at 256 interior nodes < 50 µm.
`./analysis/grad_sweep.sh`; n160, physics off, 3,000 steps, 2 seeds.

| tau / grad weight | test_B shear s0 / s1 | wall p | u field |
|---|---|---|---|
| 0 / 0 (v1) | 0.385 / 0.343 | 0.070 / 0.070 | 0.066 / 0.063 |
| 0 / 1 | 0.370 / 0.371 | 0.058 / 0.063 | 0.062 / 0.055 |
| 1 / 0 | **0.151 / 0.146** | 0.056 / 0.068 | 0.079 / 0.073 |
| 1 / 1 | 0.213 / 0.191 | 0.073 / 0.058 | 0.065 / 0.059 |

Why (`a8_grad_fit.py`, model vs Dan du_t/dn ÷ τ_wall in the peak region, test_B s0):

| run | 2–5 µm | 5–10 | 10–20 | 20–50 | at the wall (peak ratio) |
|---|---|---|---|---|---|
| Dan | 0.96 | 0.91 | 0.83 | 0.66 | 1 |
| v1 | 0.77 | 0.86 | 0.87 | 0.65 | 0.65 |
| grad only | 0.76 | 0.83 | 0.84 | 0.63 | ~0.6 |
| tau only | 1.02 | 1.07 | 0.99 | 0.59 | 0.89 |
| tau + grad | 0.93 | 0.97 | 0.91 | 0.59 | — |

- v1 already fits the gradient roughly from 5 µm out. The failure is the last ~2 µm, where the
  pack has ~2 nodes per solve: the wall value is an **extrapolation**, and nothing supervises
  it. Gradient data at ≥ 2 µm cannot fix that; the wall-shear loss supervises it directly.
- Wall-shear-only reaches the wall value with a wrong-shaped profile (too steep at 5–20 µm).
  Adding the gradient term pulls that shape toward Dan's and loses wall accuracy. The true field
  satisfies both losses at once, so the trade-off means **this model cannot represent the
  near-wall layer** — a capacity / representation limit (also: train error ≥ test error).
- Practical: keep tau weight 1, gradient weight 0. Re-try tau + grad once the model is bigger
  or has a wall-distance input; tau + grad matching or beating tau-only is a direct sign the
  model can now represent the layer.

## v2 step 3: the big run (`./run_big.sh`, Mac, 2026-10-21)

920 training shapes (incl. 135 family B), v2 pack (all interior nodes < 20 µm, cap 1,024),
trunk 288 x 6 + wall-distance input, encoder 256 wide, latent 64 (669k params), tau weight 3,
physics off, batch 16, 68,000 steps (sized to 3.5 h; 0.184 s/step on the Mac, 4 threads), seed 0.
All numbers below are on the **v1 pack** (same nodes for every model) via
`python3 analysis/a9_big_eval.py <runs>`; train = every 5th solve of the 160 v1 train shapes.

| | train shear | test_B shear | val shear | test_B peak ratio | test_B wall p | test_B u | test_B v |
|---|---|---|---|---|---|---|---|
| v1 (n160 s0) | 0.425 | 0.385 | 0.375 | 0.62 | 0.070 | 0.066 | 0.28 |
| v2 small (tau w1 s0) | 0.207 | 0.151 | 0.249 | 0.89 | 0.056 | 0.079 | 0.29 |
| **big run** | **0.012** | **0.063** | **0.108** | **0.99** | **0.037** | **0.013** | **0.06** |

- Wall shear by flow index k0 → k3, test_B: 0.063 0.063 0.065 0.067 (v1: 0.34 → 0.51); val:
  0.094 0.104 0.112 0.111. The high-Re penalty is gone.
- Near-wall du_t/dn ÷ τ_wall in the peak region (`a8_grad_fit.py runs/big`): 0.96 / 0.91 / 0.83 /
  0.66 at 2–5 / 5–10 / 10–20 / 20–50 µm, identical to Dan's. The representation limit of
  step 2 is resolved.
- **Now generalization-limited:** training-shape error 0.012 vs 0.063 (FSG) and 0.108 (synthetic).
  Quick-val shear (16 val solves) reached ~0.10 by step ~20k and is flat from ~40k
  (`figures/v3/fig_learning_curve.png`), so more steps alone will not help.
- Ripple (`a10_roughness.py`): median roughness of predicted τ along the arc ÷ truth's = 0.98
  (test_B) / 0.96 (val), but 17–18 % of solves are > 1.5x rougher (e.g. g_00354 in
  `fig_wall_curves_big.png`). Small, real; a smoothness target (Dan saves `wssg`) is an option.
- Caveats: one seed; test_B family is now in training (unseen shapes, seen family), so part of
  the test_B gain vs v2-small is that, not only scale. val (family never in doubt: A/C/D/E)
  improved 0.249 → 0.108 under the same condition as before, which is the cleaner comparison.
- Figures: `figures/v3/` (`analysis/fig_v3_data.py` then `analysis/fig_v3.py`).

### Leakage check for test_B in the big run (2026-10-21)

Family B shapes are cushion outlines harvested from Dan's FSG runs (`meta.json` `params.src` =
run/step mesh), rescaled (sx, sy) and sometimes mirrored: 175 shapes from 156 source meshes and
47 runs (19 meshes give 2 shapes). In the v2 pack 135 B shapes are in train, so of the 40
test_B shapes, 8 share their exact source mesh with a training shape, 28 more share the FSG
run (another step), and 4 come from runs absent from training. Median test_B shear error by group
(big / v2-small / v1): same mesh 0.087 / 0.154 / 0.352; same run 0.056 / 0.150 / 0.404;
unseen run 0.066 / 0.154 / 0.255 (n = 4 shapes, so weak). No sign the big run's 0.063 is inflated
by near-duplicates; a split by FSG run (all steps of a run on one side) would make it airtight.
