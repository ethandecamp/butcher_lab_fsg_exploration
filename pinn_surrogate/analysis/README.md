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
