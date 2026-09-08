# TASK-015 — Autonomous exploration

**STATUS:** TODO
**Phase:** 1, Lane A. Branch `feat/lane-a-exploration`.
**Depends on:** TASK-011, TASK-012. **Blocks:** TASK-018.

## Context

Today a human supplies the question. The goal is that the system decides what is worth asking,
tests it, and reports back what it found — the human then asks follow-ups.

The trap: "an LLM writes a report about data" is not a result, and every AI-for-science demo does
it. What makes this defensible is that every claim in the report was **pre-registered and
executed**, and that we can say how many survived **split by whether the answer was already
sitting in the briefing.** That split is the honest core of this task and the number worth
reporting.

## What to build

`src/llm_insights/agent/explore.py`, plus `summary/sections/coverage.py` if a coverage summary
needs to reach the prompt. **Do not edit `briefing.py`.**

Pipeline:

1. **Propose questions.** Grounded in the schema, not free association: the metric registry, the
   case list, the GRN nodes and the six primitives are given, and the model proposes K questions
   answerable with them.
2. **Lookup filter — the important part.** Classify each resulting hypothesis mechanically:
   - `LOOKUP` — the observed value appears in the briefing text at the printed precision, or the
     metric is one printed in the briefing's scalar table for that case.
   - `DERIVED` — computable from briefing numbers by one arithmetic step.
   - `COMPUTED` — requires touching arrays the briefing does not contain.
   This is string and numeric matching against the briefing, **not an LLM judgment.**
3. **Investigate** each surviving question through the existing `Study.ask` path. Nothing about
   verification changes.
4. **Coverage bookkeeping.** Track cells of (metric-or-node) x case x primitive, and which have
   been touched.
5. **Stop rule.** Both of:
   - a hard budget: `max_questions`, `max_calls`, `max_cost` — always on, never overridable
     from a model response;
   - coverage plateau: stop when the last K questions opened no new cells.
   **Do not ask a model whether it is finished.** Nothing in the stopping decision may be an LLM
   judgment.
6. **Rank** surviving cards by observed effect size, margin over the pre-registered threshold,
   class from step 2, and whether the result contradicts the naive expectation.
7. **Report** through the existing `agent/synthesis.py`, which already enforces that the prose
   contains no number the cards do not support.

Outputs: `exploration_report.md`, `coverage.json`, and a `stratified_survival.json` carrying the
survival rate split by class.

## Acceptance criteria

1. `tests/test_explore.py` with a **labelled fixture set** of at least 15 questions whose class
   (LOOKUP / DERIVED / COMPUTED) is known by construction. Report the classifier's confusion
   matrix and paste it. State the misclassification rate; do not hide it.
2. **Budget is enforced.** With a stub generator that always proposes more work, assert the run
   halts at `max_questions` and at `max_calls`. A test must show the budget cannot be raised by
   anything a generator returns.
3. Coverage arithmetic tested against a hand-computed small case.
4. Ranking is deterministic: same cards in, same order out, twice. Test it.
5. **One real end-to-end run**, offline where possible, with `stratified_survival.json` pasted
   into the Log. Report survival separately for LOOKUP and COMPUTED. **If nearly all surviving
   claims are LOOKUP, say so plainly in the Log** — that is the known weakness this task exists
   to measure, and hiding it defeats the point.
6. Full suite green, count pasted. Ruff clean on touched files.

## Log

_(append below)_
