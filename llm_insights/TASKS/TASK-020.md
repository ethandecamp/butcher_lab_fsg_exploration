# TASK-020 — Whole-repo verification

**STATUS:** TODO
**Phase:** 3 (last, after everything is merged). Branch `chore/phase2-verification`.

## Context

Four lanes merged into one tree is exactly where a green individual suite hides a broken whole.
This task exists so that "everything works" is a pasted transcript rather than a belief.

## What to do

Run all of the following on the merged tree, in Ethan's venv, and paste every command with its
real output into `VERIFICATION.md` at the repo root of `llm_insights/`.

1. Full suite in the venv, including the five scipy-oracle tests that only run there.
2. Full suite inside the container.
3. `ruff check` and `ruff format --check` at the **pinned 0.16.5**, whole package. Version drift
   reformats unrelated files, so state the version you ran.
4. End-to-end transcript replay, no network, no model. Diff the report against `data/report.md`
   and account for every difference that is not a timestamp.
5. The answer-preservation suites — the original TASK-005 set and TASK-014's GRN set — reported
   with their counts. This is the scientific gate; it comes last and it must be clean.
6. Briefing byte-stability: regenerate and `cmp` against the committed copy.
7. One live run on the local backend, if a model is available, with its cost and wall time.
8. `preflight.py` still refuses to start on a missing runtime import. Test by running outside the
   venv and pasting the refusal.

## Then reconcile the record

- Update `../TASKS.md` **Open questions** with anything Phase 2 surfaced about Dan's model.
  Report, never patch — `one_way_fsg_model/` stays read-only.
- Update the project memory files so the next session starts from the truth: `demo_state.md`,
  `roadmap.md`, `next_phase_feasibility.md`, and `survival_rate_caveat.md` in particular. If
  TASK-015 produced a stratified survival rate, that caveat file needs rewriting rather than
  amending.
- List, in `VERIFICATION.md`, every claim in this repo that is **asserted but not demonstrated.**
  If there are none, say so explicitly.

## Acceptance criteria

1. `VERIFICATION.md` exists with every command and its real output. No summarized results, no
   "passed" without the transcript.
2. Any failure is written up as a numbered defect with a reproduction command. **Do not fix
   defects in this task** unless the fix is a one-line correction of something this phase broke;
   file the rest.
3. A one-paragraph statement of what is now true that was not true before Phase 2, written for
   Dan, in plain language.

## Log

_(append below)_
