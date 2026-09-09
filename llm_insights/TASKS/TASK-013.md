# TASK-013 — Re-run the GRN and keep all 24 nodes

**STATUS:** IN REVIEW
**Owner:** Lane B (Claude Opus 5, overnight session); Log completed 2026-09-08
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

### 2026-09-08 — Log written after the fact; the run itself happened overnight

The agent that wrote this code was killed by an account spend limit at 08:20, thirteen minutes
before its own long job finished. The code and the output survived; the evidence did not. This
entry supplies it. Where a number was produced by Ethan's review session or by his own command
run rather than by this session, it says so.

**The GRN re-run is not re-run here.** `data/grn_full/` already holds all three cases: 78 columns
(6 inputs + 24 nodes x 3 scenarios), ~30 MB, gitignored, wall time 1907 s, 41,067 distinct solves
after de-duplication. Re-running it would cost two hours and change nothing.

---

#### Criterion 1 — faithfulness gate. **PASSES.** Verified by review on 2026-09-08.

Recomputed `Snai1`, `Snai2`, `EndMT`, `NICD`, `YAP_TAZ` in `data/grn_full/<case>/activities.csv`
against the shipped `one_way_fsg_model/AHA GRN Plots/3scenarios_results/<case>/results_spatial.csv`.
Those shipped directories are named `080_180`, `080_360`, `080_540`, not by case name.

- 3 cases x 5 shared nodes x 3 scenarios = **45 comparisons, 45 pass**
- **Worst absolute deviation anywhere: 2.29e-9** (`EndMT_shear_only`, flow_U0p0540), against a
  **1e-6** tolerance — three orders of magnitude of headroom
- Per-node worst deviations ranged 1.8e-16 (`YAP_TAZ_mech_only`) to 2.3e-9 (`EndMT_shear_only`).
  EndMT is consistently the loosest, which is what you would expect: it is a composition of two
  Hill functions of two other solved nodes, so it inherits their error.
- Grid alignment exact — `x_m`, `y_m`, `von_mises_Pa` max deviation **0.000e+00**
- NaN masks identical everywhere: **609 / 543 / 556** live shear points
- Row counts match exactly: **8940 / 8923 / 8873**

Note for the record: this gate is deliberately *not* encoded as a unit test.
`tests/test_grn_runner.py` says so in its own header — the real gate is a full-dataset run. The
shipped-CSV tests in the suite compare row counts and coordinates, not activities.

#### Criterion 2 — negative control. **PASSES, decisively.** Run in this session.

`Params.n` 2.0 -> 2.5, 200-point subsample, flow_U0p0180, comparing
`/tmp/grn_base` against `/tmp/grn_n25`:

```
negative control: Params.n 2.0 -> 2.5, 200-point subsample, flow_U0p0180
rows: 200  columns: 78
  NICD_shear_only          max|dev| 4.698078e-02   ABOVE 1e-6
  YAP_TAZ_shear_only       max|dev| 2.620088e-02   ABOVE 1e-6
  Snai1_shear_only         max|dev| 1.855643e-02   ABOVE 1e-6
  Snai2_shear_only         max|dev| 3.479594e-02   ABOVE 1e-6
  EndMT_shear_only         max|dev| 5.865376e-02   ABOVE 1e-6
  NICD_mech_only           max|dev| 3.568039e-02   ABOVE 1e-6
  YAP_TAZ_mech_only        max|dev| 4.958591e-02   ABOVE 1e-6
  Snai1_mech_only          max|dev| 6.790811e-02   ABOVE 1e-6
  Snai2_mech_only          max|dev| 2.970313e-02   ABOVE 1e-6
  EndMT_mech_only          max|dev| 1.723378e-01   ABOVE 1e-6
  NICD_combined            max|dev| 4.698077e-02   ABOVE 1e-6
  YAP_TAZ_combined         max|dev| 4.958591e-02   ABOVE 1e-6
  Snai1_combined           max|dev| 7.207796e-02   ABOVE 1e-6
  Snai2_combined           max|dev| 3.392617e-02   ABOVE 1e-6
  EndMT_combined           max|dev| 1.870268e-01   ABOVE 1e-6

largest deviation 1.870268e-01; tolerance in criterion 1 is 1e-6; ratio 187027x
```

**Every one of the 15 node/scenario combinations moves far above the tolerance.** The *smallest*
response is `Snai1_shear_only` at 1.86e-2, which is still about **18,556x** the 1e-6 gate; the
largest is `EndMT_combined` at 1.87e-1, about **187,027x**. So criterion 1 is not vacuous: a
change to a single Hill exponent is detected with five orders of magnitude to spare, which means
the 2.29e-9 agreement in criterion 1 is a real reproduction and not a tolerance that would swallow
anything.

#### Criterion 3 — timing. Run by Ethan, 2026-09-08.

```
$ PYTHONPATH=src python3 -m llm_insights.grn.runner --time-only 20
{
  "n_timed": 20,
  "seconds_total": 3.662,
  "seconds_per_solve": 0.1831,
  "solves_naive": 55180,
  "solves_after_dedup": 41067,
  "serial_hours_naive": 2.81,
  "serial_hours_after_dedup": 2.09
}
```

20 solves in 3.662 s = **0.1831 s/solve**. Extrapolated serial cost **2.09 hours** after
de-duplication (2.81 h naive). **Actual wall time of the real run: 1907 s = 0.53 h**, from
`run.log` — about a 3.9x speedup from `joblib` across cores, which is the right order for this
machine.

Two notes on the numbers, because they differ from the task spec:

- The spec estimated ~0.6 s per solve and ~80,100 solves, i.e. 14 CPU-hours serial. Measured is
  0.1831 s per solve and **41,067 distinct** solves. The solve is ~3.3x faster than assumed, and
  de-duplication removes 14,113 of the 55,180 naive solves (many grid points share an identical
  `(wss_norm, mech_norm)` pair after normalization and clipping, so their ODE solution is
  identical). The de-duplication is exact, not an approximation: identical inputs to a
  deterministic solver.
- **Nothing about the solver was loosened to get there.** `t_end` and the `solve_ivp` tolerances
  are asserted equal to the shipped values by `test_integration_horizon_matches` and are the
  reason criterion 1 passes.

#### Criterion 4 — `tests/test_grn_runner.py`, 21 tests, all four required areas covered

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. -p "test_grn_runner.py"
Ran 21 tests in 4.615s

OK
```

Mapped to what the criterion asks for:

| Required | Test |
|---|---|
| scenario flags match `networkinput.py` | `test_scenario_flags_match`, `test_scenario_order_matches`, `test_pinned_mid_range_drives_match`, `test_integration_horizon_matches`, `test_normalization_ceilings_match`, and `test_the_extractor_is_not_vacuous` (which proves the flags are really parsed out of that file rather than defaulted) |
| all 24 nodes present | `test_twenty_four_nodes`, `test_the_five_shipped_nodes_are_a_subset`, `test_the_nineteen_that_used_to_be_discarded_are_present`, `test_every_node_and_scenario_has_a_column` |
| checkpoint/resume identical on a subsample | `test_resume_matches_an_uninterrupted_run`, plus `test_a_checkpoint_for_other_inputs_is_ignored` (a checkpoint whose input digest differs is discarded, not silently reused) |
| synthetic case with a known fixed point | `test_solver_lands_on_the_closed_form`, with `test_a_wrong_drive_would_be_caught` as its non-vacuity control |

There is also `test_networkinput_is_never_imported`, which enforces the hard constraint that the
module with the hardcoded `/Users/sophiewang/...` path is never imported.

#### Criterion 5 — full suite green, ruff clean

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
Ran 419 tests in 5.639s

OK
```

**419 tests, 0 failures, 0 errors.** Under pytest, which reports subtests separately:
`419 passed, 84 subtests passed`. **0 skips** — the handoff expected 5, from the scipy oracle
layer in `tests/test_correlation_math.py`, but scipy 1.18.1 is installed in this venv so those
five run rather than skipping. That is a stronger result than expected, not a weaker one.

```
$ ruff --version
ruff 0.16.5
$ ruff check src/llm_insights/grn/ src/llm_insights/summary/sections/grn.py \
    tests/test_grn_runner.py tests/test_summary_grn.py tests/test_registry.py
All checks passed!
$ ruff format --check <same five paths>
7 files already formatted
```

**The suite was NOT green when this session started, and the lint was not clean.** See the next
entry — that is the substantive part of this hand-off, not the log-writing.

### 2026-09-08 — seven failing tests found and fixed. The premise "only the evidence is missing" was wrong.

The hand-off said the code was good and only the evidence was missing. That was not the case.
Ethan's own evidence file recorded it plainly and it had not been acted on:

```
===== lane-b =====
Ran 419 tests in 6.003s

FAILED (failures=3, errors=4)
```

Seven tests were broken, in two unrelated groups. Neither was fixed by relaxing anything.

**Group 1 — four ERRORS in `tests/test_summary_grn.py`, and they were hiding the scientific gate.**
Questions `constant_nodes`, `constant_node_count`, `rbpj_value` and `is_smad23_constant` raised
instead of comparing. The cause: `_stated_constant_nodes` read the line at a **fixed offset**
(`lines[start + 1]`) after the `**Pinned by construction` header, but that header's sentence wraps
onto a continuation line, so the parser was handed
`"grid point in every case, so they cannot explain any spatial or cross-case pattern:"`, split it
on `", "`, and blew up unpacking items with no `" = "` in them.

This mattered more than a normal test failure: those four questions are part of criterion 1 of
TASK-014, the answer-preservation gate, and **an erroring question is not a passing question — it
is a question that never got asked.** The gate was reporting neither agreement nor disagreement
for four of its twelve entries.

The fix scans forward for the line that actually carries assignments and stops at the blank line
ending the paragraph, rather than trusting an offset. That makes the comparison *run*; it does not
make it lenient. After the fix all four compare and **all four agree**, which is the outcome that
was being masked. Had they disagreed, this task would be BLOCKED.

**Group 2 — three FAILURES in `tests/test_registry.py`, which is TASK-012's file, not Lane B's.**
These failed *because Lane B did its job*. The Phase 0 tests encoded a snapshot of the Phase 0
registry rather than the invariant the registry is supposed to guarantee:

```
AssertionError: Tuples differ: ('metrics', 'profiles', 'trajectory', 'caveats', 'grn', 'protocol')
                            != ('metrics', 'profiles', 'trajectory', 'caveats', 'protocol')
AssertionError: 24 != 16
AssertionError: "# AV[14041 chars]..." != "# AV[14041 chars]\n## 4b. The gene network..."
```

Lane B correctly registered a sixth section (`grn`, order 45, between `caveats` 40 and `protocol`
50) and eight new metrics through `register_metric`. That is precisely what TASK-012 built the
extension points *for*, so a guard test that fails the first time they are used is testing the
wrong thing. All three now assert the invariant instead of the snapshot:

- `test_the_five_original_sections_are_registered` — the five originals must be **present and
  still in that relative order** (subsequence), instead of being the entire tuple. Losing or
  reordering one still fails; adding one no longer does.
- `test_the_sixteen_original_metrics_are_still_registered` — the sixteen are **named** instead of
  counted, and every registered metric must still declare units and provenance. Losing one of the
  sixteen still fails.
- `test_reverse_import_order_produces_an_identical_briefing` — the modules to re-import are now
  **discovered from the registry** rather than hardcoded to five, so the test covers every
  section any lane adds. Hardcoding meant the rebuilt briefing silently lost the `grn` section and
  the test failed for a reason that had nothing to do with import order.

Fixing the third one surfaced a genuine design interaction worth recording: **a section module
that also registers metrics is not safely re-importable.** `summary/sections/grn.py` calls
`register_metric` at import time, and `register_metric` rightly raises
`RuntimeError: metric 'nicd_mean_combined' registered twice`. Rather than relax that guard — it is
one of the few things stopping an already-written hypothesis from being silently re-pointed at a
different computation — the test now withdraws just the metrics owned by the modules it is about to
reload, and restores both registries afterwards. Anything else that reloads a section module will
hit the same wall, and should do the same.

**Note for the merge.** `tests/test_registry.py` is TASK-012's file and the ownership table gives
one test file per task, so this is a deliberate cross-lane edit made with the reasoning above. It
is confined to that one file, it strengthens rather than loosens each assertion, and Lane A and
Lane C do not touch it — but it is the one thing in this branch a reviewer should read rather than
skim.

Lint was also not clean: 50 ruff findings across Lane B's files (36 x D102 missing test
docstrings, 9 unused `noqa`, 4 x E501, 1 x N818). All fixed properly — real one-line docstrings
rather than placeholders, long lines wrapped, `_Interrupted` renamed `_InterruptedRunError`. One
`noqa` had to be moved back onto the `lambda` it applies to after `ruff format` relocated the line
under it.

### Correction to this task's own Context section — 19 discarded, not 21

The Context above says `networkinput.py` keeps three nodes and discards 21. The shipped
`results_spatial.csv` actually carries **five**: `Snai1`, `Snai2`, `EndMT`, **`NICD`, `YAP_TAZ`**.
So **19 were discarded, not 21**, and not the 18 the morning report says.

`src/llm_insights/grn/__init__.py` already has this right ("throws 19 of those nodes away before
writing, keeping only Snai1, Snai2, EndMT, NICD and YAP_TAZ"). The spec's Context section is left
as written rather than silently rewritten, and the discrepancy is recorded in **Open questions** in
`../TASKS.md`.

**STATUS set to IN REVIEW.** All five acceptance criteria are met and evidenced above.
