# Independent verification of REPORT.md

**Date:** 2026-10-14. **Verifier:** independent Claude session; it did not build this work.
**Method:** read-only. Scratch scripts live in `$HOME/verify_tmp/` on the VM, outside the repo.
No training, no git, no existing file modified. The test suite was **not** re-run (it writes run
directories), so the "63 tests" figure rests on LOG.md alone.

## Checks

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1a | Pooled test_B medians, "on better" %, sign-test p (tau, p, pde_res, peak_loc; n = 20/80/160) | **PASS** | Recomputed from the per-solve `eval.json` with my own code (exact binomial via `math.comb`). All 24 pooled cells (test_B and val) match `comparison.json` exactly and match the REPORT results table to its rounding. |
| 1b | TL;DR and prose numbers | **PARTIAL** | Most trace to artifacts. Exceptions: (i) "~35–40 % lower [residual] in every cell" and "p < 1e-28 everywhere": val n=80 is 28 % lower with p = 2.4e-28, and val n=160 is 33 % lower. (ii) "under an hour of CPU": the summed `train_s` is 68 min wall-clock at 1–4 threads (physics-on 58.3 min, physics-off 9.8 min). (iii) "52 MB": `packed_v1.npz` is 72.6 MB. (iv) "2 % of the 12 GB" cannot be derived (240/1000 shapes is 24 %; 72.6 MB/12 GB is 0.6 %). (v) Wall-shear medians are "within 0.01 at every size": n=80 differs by 0.011. |
| 1c | Verification table and bias table | **PASS** | The table matches LOG.md (`reference.py` output: r_mom maxerr/scale ≤ 6.49e-8; tau_t relL2 7.7e-4 to 1.19e-2). I re-ran the bias table for n160_physon_s0 and reproduced it exactly (0.63, IQR 0.55–0.71, 98 %). |
| 2 | No leakage | **PASS (one protocol note)** | No geom_id is in two splits (overlaps 0/0/0). test_B is 40/40 family B, and B appears nowhere else. `splits.json` equals the packed splits. The n=20/80/160 train subsets are subsets of train with zero val/test_B overlap, and each `eval.json` evaluates exactly the val/test_B geometries. In-training quick eval uses val only. **Note:** the physics share of 0.1 was chosen in the pilot by looking at test_B as well as val (LOG §Stage 4, 8 solves). Val alone gives the same choice, but this is not disclosed. |
| 2b | Grid configs | **PASS with differences** | All 12 runs use steps 3000, physics_share 0.1, the claimed seed and n, batch 8×512 (+256), and lr 1e-3→1e-5, with 133,443 params. **Threads differ** (1, 2 or 4 per run), so train times and `infer_ms` are not comparable across runs. LOG.md discloses the thread differences only partly. All physics-on runs end with λ_pde at its 1e-3 floor, which REPORT discloses. |
| 3 | Independent wall shear (FD along the normal, no autograd, no `wall_quantities`), n160_physoff_s0 | **PASS** | Test_B solves 812 (k=0, Re 0.60), 870 (k=2, Re 4.3) and 927 (k=3, Re 16.8). u_t(0) = 0 exactly. A second-order one-sided FD with h = 1e-3/2e-3 agrees with `predict_wall` to a relative L2 of 0.22–0.27 %, and h = 2e-4 and 5e-3 agree within 0.5 %. Against Dan's `wall_tau_t`, the FD relative L2 is 0.47 / 0.27 / 0.49 and the pipeline gives 0.47 / 0.27 / 0.49. FD peak ratios are 0.48 / 0.70 / 0.45, consistent with the median of ~0.6 (IQR 0.52–0.76). The evaluation pipeline is sound. |
| 3b | "~0.6× … for every model variant" | **PASS (extended)** | REPORT measured only 5 of the 12 models. I ran the other 7 through the validated pipeline: medians 0.53–0.65, with 96–100 % of solves under-predicted (n20_physon_s0 is 96.3 %, slightly under the stated 97 %). **New:** the bias is also present on *training* solves (n160 off/on: 0.55 / 0.57, 100 % under) and on val (0.52 / 0.51). It is underfitting, not an OOD effect. |
| 4 | Stage 3 data-free PINN | **PARTIAL** | `summary.json` gives tau 1.0019, sign 0.292, loss_pde 0.0117, Re 0.38: matches. "Fell steadily" is roughly right but not monotone (0.030 at step 1000, 0.094 at 1250). "Velocity field error was 71 %" is u only; v relL2 was 1.004. The 0.01 tangential-velocity diagnostic came from `/tmp/diag.py`, which is not in the repo, so it cannot be reproduced. Also notable: at steps 250–750 sign agreement was 98 % (tau 0.87–0.98), and it collapsed as the residual fell. This supports the story. |
| 5 | Claims not checkable from artifacts | see below | |
| 6 | Overclaiming and missing caveats | **FAIL (issues 1–4)** | See below. |

**New baseline (computed by the verifier).** I zeroed the network outputs (N = 0), which leaves only
the hard-BC Poiseuille prior with no learning, and ran it through the same pipeline on test_B. Median
tau relL2 **0.544**, sign agreement **0.988**, peak-location error **0.039**, peak ratio 1.15. The
n=160 models score 0.36 / 1.000 / 0.03; the n=20 models score 0.48 / 0.96–0.99 / 0.06.

## Issues, most severe first

1. **The sign and peak-location claims are mostly reproduced by a model that learned nothing.**
   REPORT: *"it gets the sign of the wall shear right at 99–100 % of wall points (it finds the recirculation zones) and puts the shear peak within ~3 % of the arc length."*
   The N = 0 prior gets 98.8 % sign agreement and 3.9 % peak location, so this metric mostly
   reflects geometry, not learned recirculation. The n=20 models do no better than the prior on
   sign (0.96–0.99) and are worse on peak location (0.06).
   Suggested: *"Wall-shear relative L2 error improves from 0.54 (hard-BC Poiseuille prior, no
   learning) to 0.36. Sign and peak location are already near-correct under the prior (98.8 %,
   3.9 %), so they do not show learned recirculation."* Add the baseline as a row in the results table.

2. **The p-values pseudo-replicate and overstate certainty.** Pooling 2 seeds × 160 solves as 320
   independent pairs counts each solve twice. The tests also condition on the trained models, so
   they say nothing about retraining variance. Averaging each solve's difference over seeds (160
   units) changes the wall-shear picture. n=20 favors off (34 %, p = 1e-4) and n=80 favors on
   (63 %, p = 0.001). Per seed, 4 of 6 wall-shear splits are "significant", in opposite directions.
   REPORT: *"p < 1e-28 everywhere"* and *"the only significant split (20 shapes, p = 0.03)"*. Both
   are artifacts of the pooled test, and val n=160 is also significant (p = 0.038, favoring off).
   Suggested: base the headline on **direction agreement across seeds**: residual and wall pressure
   agree in sign in both seeds at every n, while wall shear does not. State that the p-values are
   conditional on these 2 model pairs.

3. **"Almost all of that is one systematic bias."** A per-solve oracle rescale reduces the median
   relL2 from 0.385 to 0.135 (n160_physoff_s0) and from 0.380 to 0.142 (n160_physon_s1). That is
   about two-thirds of the error, not almost all.
   Suggested: *"about two-thirds of it is an amplitude bias"*. Add that the bias also appears on
   training solves (peak ratio 0.55, 100 % under), which points to underfitting near the wall
   rather than generalization. This supports causes (a)/(c).

4. **"~35–40 % lower in every cell" and "p < 1e-28 everywhere."** The val n=80 cell is 28 % lower,
   val n=160 is 33 % lower, and val n=80 has p = 2.4e-28. In seed 1, val n=80 is only 60 % better
   (p = 0.014).
   Suggested: *"28–40 % lower (median), lower on 80–98 % of solves, in both seeds at every size."*

5. **The "~3–15 ms" inference figure.** `infer_ms` times only wall p/tau at 256 points, not the
   flow field. It was measured during concurrent training at 1–4 threads. Per-run medians are
   2.9–16.5 ms, p95 up to 22 ms, max 40 ms. The ~3 ms runs are the two at 4 threads. Dan's
   "~1–3.5 s" appears only in PLAN.md, with no source and no stated hardware.
   Suggested: *"~3–16 ms (median) to predict wall shear and pressure at 256 points on a laptop CPU, vs ~1–3.5 s per FEM solve (Dan's figure; different hardware)."*

6. **Smaller factual fixes.** "Under an hour of CPU" should read ≈ 68 min wall-clock. "52 MB"
   should read 72.6 MB. Drop or explain "2 % of the 12 GB". "Velocity field error was 71 %"
   should read u 71 %, v 100 %. "Error grows with Re": the highest-Re bin is the worst in every
   run, but the trend is not monotone (n160_physon_s0: 0.37, 0.35, 0.34, 0.41). Undisclosed: test_B
   was consulted when choosing the physics share.

7. **Generalization from one run.** REPORT: *"The textbook data-free PINN fails here."* This rests on
   1 shape, 1 seed, 3,000 steps and no tuning. Suggested: *"A textbook data-free PINN, run once on
   one shape without tuning, failed."*

## Not checkable from the artifacts (wording assessment)

- **Dan's FEM time ~1–3.5 s.** Unsourced (PLAN.md only). Needs a citation or a hedge.
- **Krishnapriyan et al. 2021 as "a known PINN failure mode".** The paper exists, but its failure
  cases are high-coefficient convection and reaction problems, not Re 0.38 Stokes-like flow. "Resembles"
  would be safer than "is".
- **BioPINN-LM's conditioning** ("spherical-harmonic shape vector plus a pressure"). Not verifiable here.
- **Causes (a)–(c) of the bias.** Labelled "likely", so the hedge is acceptable. Cause (b) is
  testable, and the 44 % figure matches LOG.md.
- **"63 tests, 0 failures".** LOG.md only; not re-run by the verifier.
- **"All absolute errors are a lower bound on what the method can do."** Unfalsifiable as written.
  Suggested: "do not represent the method's ceiling".

---

**Follow-up (same day):** REPORT.md was revised after this check to address issues 1–7 (prior baseline added, per-seed direction tables replace pooled p-values, "two-thirds" bias wording, corrected residual ranges, timings, data size, Stage 3 wording, test_B disclosure). The diagnostic behind the Stage 3 near-wall numbers is described in LOG.md; the bias numbers are now reproducible with `python3 -m pinn_surrogate.wall_bias`.
