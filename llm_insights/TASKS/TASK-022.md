# TASK-022 — The exploration report shows the test that ran (display half)

**STATUS:** IN REVIEW — 2026-09-09
**Phase:** 2a. Branch `feat/report-shows-test`, from `feat/task-021-explore-fixes` (`6fb8edd`).
**Depends on:** TASK-021. **Blocks:** nothing.

This file covers **only the display half** of what TASK-021's log filed under TASK-022. The
semantic guard (rejecting a claim whose prose disagrees with its test) and the hedge are **still
open under this number** — see "Remaining" below. Nothing here changes what is computed,
classified or ranked.

## Context

`data/explore_*/exploration_report.md` — the one page a reader actually reads — printed each
claim's **English** and never the test that ran. In a live run:

```
claim   : "The expression of Snai1 is directly proportional to mechanical stress..."
params  : field_x=YAP_TAZ_shear_only, field_y=Snai1_shear_only, case=Healthy
```

The prose is about mechanical stress; the test correlated two GRN nodes. Four claims in that run
carried byte-identical params under four different English claims, and the top-level report gave a
reader no way to see it.

The verdicts are correct — the harness ran exactly what `params` asked for. This was a **display**
defect. The project's claim "every verdict was produced by executable code" held throughout; the
claim "the verdict tests the claim" did not, and the report was implying it.

The per-question reports (`q01/report.md`, `cards/render.py`) already showed primitive, params,
decision rule and observed values in full. Only the top-level summary was missing this.

## What was built

`build_exploration_report` in `src/llm_insights/agent/explore.py` gained a final section,
**"Every claim, beside the test that ran"**, after the ranked table. It lists **every** card in
the order the questions produced them — survivors, falsified claims, and ones that could not run —
and for each prints the id, verdict and class on one line, then the claim's prose, then the
mechanical description of the test, then why it was classified as it was.

The mechanical line is read straight off the card (`_mechanical_line`): `card.summary` when the
test ran, `card.error` when it did not. It is never re-derived from `primitive` and `params`,
because a second rendering of the same thing is a second thing to drift.

The section **absorbed** "How each claim was classified", which listed the same ids again with
nothing but their label and reason. A claim is now described in exactly one place.

Verdict wording is imported from `cards.render.verdict_of` rather than restated, so the top-level
report and the per-question reports cannot disagree about what SURVIVED means. `cards/render.py`
itself is untouched.

## Acceptance criteria

1. **The divergence is visible.** `tests/test_report_shows_test.py` builds a card whose prose
   names entities its `params` never touch, renders the report, and asserts **both** the prose and
   the mechanical summary appear — so a reader can see they disagree. ✅
2. **Every card appears**, including `could not run` ones, with the error in place of the summary,
   asserted against a result carrying one survivor, one falsified claim and one that could not
   run. ✅
3. **Regenerated** `data/explore_021/exploration_report.md`; the first three claims are pasted in
   the Log below. ✅
4. **571 tests, 0 failures** (563 before, 8 added). Ruff 0.16.5 clean on both touched files. ✅

## Out of scope

- What is computed, what is classified, how anything is ranked — all unchanged.
- **No semantic guard.** Nothing rejects or flags a claim whose prose disagrees with its test.
  Display only.
- `cards/render.py`, the briefing, `harness/`, `io/`, `cards/card.py` and
  `one_way_fsg_model/` are untouched.

## Remaining under this number

1. **The semantic guard.** Reject, or at minimum flag, a proposed hypothesis whose claim text
   names quantities its `params` do not address. This section makes the failure visible to a
   human; it does not stop it happening.
2. **The hedge** (TASK-021 log, item 1): claims written so loosely that they cannot fail.

## Log

_(append below)_

### 2026-09-09 — display fix

Baseline before touching anything: 563 tests, 0 failures. After: **571 tests, 0 failures**
(`cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..`). Ruff 0.16.5:
`check` and `format --check` both clean on `agent/explore.py` and `tests/test_report_shows_test.py`.

Regenerated `data/explore_021/exploration_report.md` from the saved run's own JSON — no model
call, no re-run of the harness. The first three claims of the new section:

```markdown
### q01_H1 — COULD NOT RUN · COMPUTED

- **claim:** The expression of Snai1 is higher in response to higher mechanical stress across all flow conditions.
- **could not run:** unknown GRN field 'YAP_TAZ'; available fields are ['mech_norm', 'von_mises_pa', 'wss_dyn_cm2', 'wss_norm', 'EndMT_combined', 'EndMT_mech_only', 'EndMT_shear_only', 'NICD_combined', 'NICD_mech_only', 'NICD_shear_only', 'Snai1_combined', 'Snai1_mech_only', 'Snai1_shear_only', 'Snai2_combined', 'Snai2_mech_only', 'Snai2_shear_only', 'YAP_TAZ_combined', 'YAP_TAZ_mech_only', 'YAP_TAZ_shear_only']
- **classified COMPUTED:** the briefing carries no GRN table columns, so a correlation needs the array

### q01_H2 — COULD NOT RUN · COMPUTED

- **claim:** The expression of Snai1 is lower in response to higher mechanical stress in the Underflow case.
- **could not run:** unknown GRN field 'YAP_TAZ'; available fields are ['mech_norm', 'von_mises_pa', 'wss_dyn_cm2', 'wss_norm', 'EndMT_combined', 'EndMT_mech_only', 'EndMT_shear_only', 'NICD_combined', 'NICD_mech_only', 'NICD_shear_only', 'Snai1_combined', 'Snai1_mech_only', 'Snai1_shear_only', 'Snai2_combined', 'Snai2_mech_only', 'Snai2_shear_only', 'YAP_TAZ_combined', 'YAP_TAZ_mech_only', 'YAP_TAZ_shear_only']
- **classified COMPUTED:** the briefing carries no GRN table columns, so a correlation needs the array

### q01_H3 — COULD NOT RUN · COMPUTED

- **claim:** The expression of Snai1 is higher in response to higher mechanical stress in the Overflow case.
- **could not run:** unknown GRN field 'YAP_TAZ'; available fields are ['mech_norm', 'von_mises_pa', 'wss_dyn_cm2', 'wss_norm', 'EndMT_combined', 'EndMT_mech_only', 'EndMT_shear_only', 'NICD_combined', 'NICD_mech_only', 'NICD_shear_only', 'Snai1_combined', 'Snai1_mech_only', 'Snai1_shear_only', 'Snai2_combined', 'Snai2_mech_only', 'Snai2_shear_only', 'YAP_TAZ_combined', 'YAP_TAZ_mech_only', 'YAP_TAZ_shear_only']
- **classified COMPUTED:** the briefing carries no GRN table columns, so a correlation needs the array
```

All five of that run's first question could not run, for the reason now printed in place of the
test: the model asked `correlation` for `YAP_TAZ`, and the GRN table has
`YAP_TAZ_combined`/`_mech_only`/`_shear_only` but no bare `YAP_TAZ`. That was already in the
per-question report; it was not on the page a reader reads.

The clearest instance of the defect this section exists for is in `data/explore_smoke/`, where the
test did run:

```markdown
### H1 — FALSIFIED · COMPUTED

- **claim:** The expression of Snai1 is directly proportional to mechanical stress in the developing heart valve cushion.
- **test that ran:** FAIL: pearson correlation of YAP_TAZ_shear_only with Snai1_shear_only in flow_U0p0360 is r=-0.0451 over 543 rows (8380 dropped for missing values); r >= 0.5 is false
- **classified COMPUTED:** the briefing carries no GRN table columns, so a correlation needs the array
```

The prose says *mechanical stress*; the test correlated two shear-only GRN nodes. The verdict is
arithmetic and correct — it is just not evidence about the sentence above it. A reader can now see
that without opening `q01/report.md`.

**Files changed:** `src/llm_insights/agent/explore.py`, new `tests/test_report_shows_test.py`,
this file (`data/` is gitignored, so the regenerated `data/explore_021/exploration_report.md` is on disk only). Not merged, not pushed.
