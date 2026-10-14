# Conditioned PINN surrogate for the AV-cushion steady fluid model — proof of concept

**Date:** 2026-10-14. **Branch:** `feat/pinn-surrogate`. **Data:** Dan's
`steady_state_dataset/dataset_v1_1k` (1,000 cushion shapes × 4 flows, steady 2-D Navier–Stokes,
FEniCS Taylor–Hood P2/P1). Plan: [`PLAN.md`](PLAN.md). How to run: [`INSTRUCTIONS.md`](INSTRUCTIONS.md).
Every number below is in [`LOG.md`](LOG.md) with the command that produced it, and an independent
check of this report is in [`VERIFICATION.md`](VERIFICATION.md). This version incorporates that
check's corrections.

## TL;DR

- **One network, any cushion shape, any flow rate.** `f(x, y | outline, Re) → (u, v, p)` trained
  on up to 160 synthetic shapes and evaluated on **40 held-out FSG-derived cushions (family B,
  never seen in training)**. It predicts wall shear and pressure at 256 wall points in ~3–16 ms
  (median, laptop CPU) vs ~2.5–4.7 s per flow for Dan's FEM solve (his own `cfd_summary.json`
  timings, his hardware).
- **It learns, but it's rough.** Median wall-shear error (relative L2) on the held-out FSG shapes
  falls from **0.54** for the hard-coded boundary prior alone (no learning) to **~0.36** with 80–160
  training shapes. About two-thirds of the remaining error is a single systematic bias: **the
  predicted shear peak is ~0.6× the true peak on 96–100 % of solves**, in all 12 trained models
  and on *training* shapes too. That points to under-fitting near the wall. It is the next thing
  to fix.
- **The physics term, ablated (2 seeds × 3 data sizes × on/off, 12 runs, paired per solve):**
  - **lowers the Navier–Stokes residual** in every cell, in both seeds (lower on 60–100 % of
    solves), by an amount that varies a lot (~5–50 %);
  - **helps wall pressure when data is scarce** (20 shapes: better in both seeds) **and hurts it
    once data is plentiful** (80 and 160 shapes: worse in both seeds);
  - has **no reliable effect on wall shear**: the two seeds disagree in direction at every size. A
    single seed would have suggested a benefit.
- **A textbook data-free PINN, run once on one shape without tuning, failed.** It reduced its
  residual but converged to a wrong, nearly stagnant flow next to the cushion (wall-shear error
  ~100 %). That is why the surrogate trains on data *and* physics.
- **Every layer is checked against Dan's data before it is trusted.** Our residual reproduces
  Dan's stored residuals to float32 precision. Our wall-shear formula reproduces his saved wall
  shear to 0.08–1.2 %. 63 unit tests pass. An independent finite-difference recomputation of wall
  shear agrees with the evaluation pipeline to 0.2–0.3 %.

## What this is, in ML terms

**Task.** Operator learning across problem instances, not a single PINN solve. The network is
conditioned on the cushion outline (256 arc-length-uniform points → MLP encoder → 32-dim latent)
and log Re, and queried at any (x, y). This follows the conditioned-PINN pattern of BioPINN-LM
(Zhu et al. 2026), which Dan shared.

**Why a coordinate network.** The FSG loop needs *wall shear stress*, a derivative of velocity at
the cushion. A coordinate MLP gives it by autograd at the 256 wall points.

**Nondimensionalization.** Lengths ÷ H, velocity ÷ mean inlet speed Ū, pressure and stress ÷ μŪ/H,
so momentum reads `Re (u·∇)u = −∇p + ∇²u`. Targets become O(1)–O(100), and Re enters as one
scalar input instead of a 400× range of velocities.

**Hard boundary conditions (architecture, not loss).**
`u = m·(u_Poiseuille + y(1−y)·g(x)·S_u·N_u)`, `v = m·y(1−y)·g(x)·S_v·N_v`,
`p = 12(10−x) + S_p·N_p`, with `m = tanh(d_cushion/δ)` and `g = (x+10)/20`. No-slip on the cushion
and both walls, and the parabolic inlet, hold exactly for any weights (unit-tested). The outlet's
zero-traction condition is a soft loss. With the network outputs N set to zero this ansatz is
the **prior baseline** used below: Poiseuille flow masked to zero at the cushion.

**Loss.** Trust-weighted MSE on (u, v, p) at sampled FEM nodes, with weights from Dan's own
residual `r_rel`. The physics arm adds the Navier–Stokes + continuity residual by second-order
autograd at interior collocation nodes, and the outlet traction residual. The momentum residual
is normalized per node by the local size of the momentum terms in Dan's solution, so it is exactly
Dan's `r_rel` (FEM median 1.3e-4). Loss weights are set by gradient-norm balancing, with physics
given 10 % of the data term's gradient budget (`--physics-share 0.1`).

**Data used.** 240 of the 1,000 geometries × 4 flows = 960 solves, 3,072 sampled nodes each
(1,024 within 50 µm of the cushion), in one 73 MB file. **Splits:** train = 160 synthetic shapes
(40 each of families A/C/D/E; the first 20 and 80 are family-balanced); val = 40 synthetic shapes
(in-distribution); test_B = 40 FSG-derived shapes (out-of-distribution). No geometry is in two
splits; checked by the tests and independently.

**Model and compute.** 133k parameters (encoder 86k; trunk 5×96 SiLU with 32 Fourier
frequencies). Adam, cosine LR 1e-3 → 1e-5, 3,000 steps, batch 8 solves × 512 nodes (+256 PDE
nodes). CPU only, on the 4-core Linux VM on Ethan's MacBook: ~1.5–2 min per data-only run and
~5–11 min per physics run. The 12-run grid totals ~68 min of training. Thread counts varied
(1–4), so run times are not comparable across runs.

## Verification ladder (before any accuracy number was trusted)

| Check | Result |
|---|---|
| Nondimensionalization round trip, Re formula, upstream dp*/dx* = −12 (Poiseuille) | exact / rtol 1e-12 / −12.000 |
| Our NS residual vs Dan's stored `r_mom_x, r_mom_y, r_div` | ≤ 6.5e-8 of local term size (float32 floor) |
| Wall shear from Dan's velocity gradients vs his saved `tau_t` | rel. L2 0.08–1.2 % (FEM discretization); conventions pinned (normal into fluid, t̂ = (n_y, −n_x)) |
| Torch residual vs analytic Poiseuille and Kovasznay flows, and vs the numpy reference | pass |
| Hard constraints for random weights: u = v = 0 on cushion and walls, inlet = Poiseuille | pass (inlet 1 − m = 0 in float32) |
| Train → crash → `--resume` reproduces an uninterrupted run | pass (bit-exact) |
| Full suite | **63 tests, 0 failures** (python 3.10, torch 2.14.1+cpu) |
| Independent: wall shear by finite differences (no autograd) vs the evaluation pipeline | agrees to 0.2–0.3 % (VERIFICATION.md) |
| Independent: recompute every pooled statistic from per-solve outputs | all match |

## Results

### Stage 3: textbook data-free PINN on one shape (low Re, one untuned run)

The physics loss fell (not monotonically) to 0.012, but wall-shear error ended at **~100 %**
with 29 % sign agreement. Early in training (steps 250–750) the sign agreement was 98 %; it
collapsed as the residual fell. A diagnostic found a near-stagnant layer around the cushion:
tangential velocity 0.01 at 0.1·H from the wall instead of O(1). Field errors were u 71 % and
v 100 %. Low residual, wrong answer: this resembles documented PINN failure modes (e.g.
Krishnapriyan et al. 2021, though their examples are convection/reaction problems). One run on
one shape shows the risk, not a general verdict.

### Stages 4–5: physics on vs off

![scaling](figures/fig_scaling.png)

**Held-out FSG shapes (test_B)**, median over the 160 solves, **seed 0 / seed 1**:

| train shapes | wall shear, on | wall shear, off | wall pressure, on | wall pressure, off | residual, on | residual, off |
|---|---|---|---|---|---|---|
| — (prior, no learning) | 0.544 | | | | | |
| 20 | 0.425 / 0.535 | 0.446 / 0.524 | **0.093 / 0.111** | 0.107 / 0.166 | **0.90 / 0.95** | 1.59 / 1.19 |
| 80 | 0.361 / 0.370 | 0.407 / 0.332 | 0.091 / 0.098 | **0.073 / 0.081** | **0.69 / 1.03** | 1.55 / 1.09 |
| 160 | 0.366 / 0.380 | 0.385 / 0.343 | 0.098 / 0.112 | **0.070 / 0.070** | **0.81 / 0.93** | 1.65 / 1.22 |

Share of solves where physics-on had the lower error (seed 0 / seed 1):

| train shapes | wall shear | wall pressure | residual |
|---|---|---|---|
| 20 | 55 % / 32 % | **59 % / 64 %** | **100 % / 95 %** |
| 80 | 62 % / 44 % | 29 % / 31 % | **100 % / 64 %** |
| 160 | 62 % / 33 % | 15 % / 19 % | **100 % / 91 %** |

The in-distribution val split shows the same directions (`runs/grid/comparison.md`, which also
lists sign-test p-values. Those are conditional on these trained models and say nothing about
retraining variance, so the headline rests on **agreement in direction across seeds**, not on
p-values). Per-run table: `runs/grid/ablation_summary.md`.

**Reading it.**
1. **Learning helps; more data helps, then plateaus.** Wall-shear error goes from 0.54 (prior) to
   ~0.48 at 20 shapes and ~0.36 at 80–160. Going from 80 to 160 shapes buys nothing, so the bottleneck is now the model or its
   supervision, not the amount of data.
2. **Physics reliably improves physical consistency, by a variable amount.** The residual is lower
   in all 12 comparisons (6 cells × 2 seeds), with reductions from ~5 % (seed 1, 80 shapes) to
   ~50 %. Neither arm is pointwise accurate: residuals near 1 mean errors comparable to the local
   term sizes, against the FEM's 1.3e-4.
3. **Physics helps when data is scarce, then gets in the way (wall pressure).** Both seeds agree.
   With 20 shapes the physics prior beats data alone. With 80+ shapes the data is more informative,
   and the physics term, fit to an imperfect velocity field, pulls pressure away from the labels.
4. **Wall shear is not reliably moved by physics.** The seeds disagree in direction at every size.
   Seed 0 alone would have read as a clear win for physics.
5. **Sign and peak location are not evidence of learning here.** The prior alone already gets
   98.8 % sign agreement and puts the peak within 3.9 % of the arc length. The trained models
   reach 96–100 % and ~3 % (and the 20-shape models are worse than the prior on peak location).
   So these metrics mostly reflect the geometry, not learned recirculation.
6. **The highest-Re flow bin is the worst in every run**, though the trend across the lower bins is
   not monotone.

### The dominant error: wall-shear amplitude

![wall curves](figures/fig_wall_curves.png)

On held-out FSG shapes the predicted shear profile has roughly the right shape but too low an
amplitude:

| model (test_B) | median predicted peak / true peak | solves under-predicted |
|---|---|---|
| n160 physics on, seed 0 / 1 | 0.63 / 0.60 | 98 % / 98 % |
| n160 physics off, seed 0 / 1 | 0.62 / 0.63 | 97 % / 100 % |
| other 8 models (independent check) | 0.53–0.65 | 96–100 % |

Rescaling each prediction by its best-fit constant cuts the median error from ~0.38 to ~0.14, so
the amplitude bias is about two-thirds of the error. It is the same with or without physics and
with 20 or 160 shapes. It also appears on the training shapes (peak ratio ~0.55, 100 %
under-predicted), so the model is **under-fitting the near-wall gradient**, not failing to
generalize. Likely causes:

- (a) the data loss supervises velocity *values*, while shear is the *gradient* at the wall, and
  small value errors near the wall barely move the loss;
- (b) the trust weights down-weight 44 % of the nodes within 5 µm of the wall, exactly where shear
  is decided;
- (c) a smooth MLP with low-frequency Fourier features under-resolves the thin near-wall layer.

Dan saves wall shear for every training solve, so supervising it directly is the obvious next
experiment.

## What went wrong during development (and was fixed)

- **Equal-gradient loss balancing let physics override the data.** In the first pilot the data
  loss never fell and wall-shear error was 0.96, worse than the prior. Added `--physics-share`,
  the physics gradient budget as a fraction of the data term's.
- **The adaptive weights eased in too slowly** (~2,000 steps to reach their target). They now jump
  to the target on the first balancing update.
- **A per-solve residual normalization let a few hundred near-wall nodes dominate the physics
  loss.** It is now normalized per node, as Dan's `r_rel` is.
- **Protocol disclosure.** The physics share (0.1) was chosen from a short pilot that looked at
  test_B as well as val (8 solves each). Val alone gives the same choice.
- **Caveat on the final setting.** With share 0.1, the PDE weight sat at its 1e-3 floor for most
  of training in every physics run, so in effect physics had a small fixed weight. A sweep of the
  share and the floor is untried.

## Limitations

- Small model (133k parameters), CPU only, 3,000 steps, 2 seeds, 24 % of the geometries.
  These errors do not represent the method's ceiling.
- Steady, 2-D, rigid cushion. This is the proof of method for the transient surrogate, not a
  drop-in for the FSG loop. Inside the loop, surrogate error would compound over iterations; that
  closed-loop test needs Dan's FEniCS environment.
- The ground truth is itself approximate at the wall: FEM wall shear vs its own gradients differs
  by 0.08–1.2 %.
- Inference times cover wall p and τ at 256 points only, measured during concurrent training at
  1–4 threads (medians 2.9–16.5 ms).

## Next steps (proposed)

1. **Supervise wall shear and wall pressure directly** at the 256 wall points (Dan saves both for
   every solve), and revisit the near-wall trust weights. This targets the amplitude bias.
2. **More capacity and steps on a GPU** (BioHPC or Colab). The plateau at 80–160 shapes and the
   training-set bias both say the model is the limit.
3. **Sweep the physics share and λ floor with ≥ 3 seeds** before drawing further conclusions about
   physics vs data.
4. **Quasi-steady check for the transient model.** A few transient runs from Dan would test
   whether the steady surrogate, swept along a heartbeat waveform, already approximates them
   (Womersley ≈ 0.4).

## Questions for Dan

1. Is wall-shear error of this size meaningful for the growth law, or does it mostly need the peak
   location and sign (which the geometry nearly fixes anyway)?
2. Which wall-shear definition does the FSG loop consume, signed `tau_t` or `tau_mag`?
3. Does the growth law use time-averaged wall loads over a heartbeat? This decides the transient
   design.
4. Could he produce a handful of transient runs to test the quasi-steady idea?

## Reproduce

See `INSTRUCTIONS.md`. In short (from `pinn_surrogate/`):

```bash
PYTHONPATH=src python3 -m unittest discover -s tests          # 63 tests
bash ablation.sh && SEED=1 bash ablation.sh                   # 12 runs -> runs/grid/
PYTHONPATH=src python3 -m pinn_surrogate.summarize --runs runs/grid
PYTHONPATH=src python3 -m pinn_surrogate.compare --runs runs/grid
PYTHONPATH=src python3 -m pinn_surrogate.figures --runs runs/grid --out figures
PYTHONPATH=src python3 -m pinn_surrogate.wall_bias runs/grid/n160_phys{on,off}_s{0,1}
```
