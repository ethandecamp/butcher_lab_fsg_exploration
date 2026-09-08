# TASK-013 — Re-run the GRN and keep all 24 nodes

**STATUS:** IN PROGRESS — Lane B (Claude Opus 5, overnight session)
**Phase:** 1, Lane B (first). Branch `feat/lane-b-grn`.
**Depends on:** Phase 0 merged. **Blocks:** TASK-014, TASK-018.

## Context

`one_way_fsg_model/networkpoint.py` defines a 24-node Hill-function ODE network across Notch,
Wnt, mechanosensing, TGFb and BMP, solved per spatial point with `solve_ivp`. `simulate()`
returns the **whole 24-vector**, but `networkinput.py` sets
`OUTPUT_NODES = ["Snai1", "Snai2", "EndMT"]` and throws the other 21 away before writing.

That is why demo card H6 had to be refused: the model asked about `SMAD23`, which is computed on
every run and never persisted. Keeping all 24 multiplies the hypothesis space at no new FSG cost.

The GRN layer is **pure numpy/scipy/joblib. No FEniCS, no gmsh.** It runs on Ethan's Mac today.

## Cost — read before starting

Each case's grid is ~8,900 points (8940 / 8923 / 8873), times 3 scenarios, times 3 cases
= **~80,100 ODE solves**. A single solve has been measured at roughly 0.6 s, so this is on the
order of **14 CPU-hours serially, ~1.5-2 h with `joblib` across all cores.** This is the long
pole of Phase 1. Start it first in the lane and let it run while you write TASK-014.

**Do not "optimize" it by loosening `solve_ivp` tolerances or shortening `t_end`.** Changing
either breaks the faithfulness gate below, which is the only thing making the re-run trustworthy.

## What to build

New package `src/llm_insights/grn/`.

- `runner.py`
  - Imports **`networkpoint` only** — never `networkinput`, which has a hardcoded
    `/Users/sophiewang/...` path and does work at import time.
  - Resolves `one_way_fsg_model` the same way `agent/run.py` already does. **Never hardcode a
    sandbox path** (see repo conventions hazard 5).
  - Reads the existing per-case mechanical inputs, reproduces the three scenarios exactly as
    `networkinput.py` does (shear_only, mech_only, combined — read the flags out of that file,
    do not guess them), and persists **all 24 nodes** for each.
  - `joblib.Parallel(n_jobs=-1)`, chunked, with a progress log and a resumable checkpoint per
    case so an interrupted overnight run does not start over.
- Output: `data/grn_full/<case>/activities.csv` — plain text, one row per grid point, columns
  `x_m, y_m, wss_mag_dyn_cm2, von_mises_Pa, wss_norm, mech_norm` plus `<Node>_<scenario>` for all
  24 nodes and 3 scenarios. Roughly 10 MB per case.
- **Add `data/grn_full/` to `.gitignore`.** It is regenerable and the repo already routes binary
  data through LFS; do not add 30 MB of derived output to history. Document the regeneration
  command in `../INSTRUCTIONS.md`.
- A console entry point in `pyproject.toml` and the exact command in `../INSTRUCTIONS.md`.

## Acceptance criteria

1. **Faithfulness gate.** For all three cases, the recomputed `Snai1`, `Snai2` and `EndMT` columns
   must match the shipped `one_way_fsg_model/AHA GRN Plots/3scenarios_results/<case>/results_spatial.csv`
   for every scenario. Report the **max absolute deviation** per node per scenario and paste the
   real numbers. Tolerance 1e-6 absolute. If it does not match, **stop and log the blocker** —
   do not adjust the tolerance to make it pass.
2. **Negative control.** Perturb `Params.n` (2.0 -> 2.5), re-run a 200-point subsample, and show
   the outputs move by far more than the tolerance in (1). Paste the numbers. This proves the
   gate in (1) is not vacuous.
3. **Timing recorded.** Time 20 points, state the extrapolated full cost, then state the actual
   wall time of the real run. Paste both.
4. `tests/test_grn_runner.py`: scenario flags match `networkinput.py`; all 24 nodes present in the
   output; checkpoint/resume produces identical output to an uninterrupted run on a small
   subsample; a synthetic-input case with a known fixed point.
5. Full suite green, count pasted. Ruff clean on touched files.

## Log

_(append below)_
