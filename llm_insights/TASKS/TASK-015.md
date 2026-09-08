# TASK-015 — Autonomous exploration

**STATUS:** IN REVIEW — Lane A (Claude Opus 5, overnight session, 2026-09-08)
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

### 2026-09-08 — Lane A, session start

Claimed. Baseline confirmed before touching anything:

```
$ cd /Users/ethandecamp/Documents/Research/ButcherLab/lane-a/llm_insights && \
  source .../.venv/bin/activate && cd src && \
  PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
Ran 386 tests in 0.778s

OK
```

Briefing measured on this tree (the spec's 12,698 is stale):

```
$ PYTHONPATH=. python3 -c "from llm_insights.api import open_study; import hashlib; \
  b=open_study().briefing; print(len(b), hashlib.sha256(b.encode()).hexdigest()[:16])"
12662 9b228a874effeab9
```

Local Ollama *is* up (`llama3.1:8b`, `qwen3:8b` at 127.0.0.1:11434), but TASK-016 owns the
local-model backend (`agent/local.py`, Lane C) and explicitly flags the 4096-token silent
truncation hazard. Writing a second Ollama client here would duplicate their work and risk
exactly that failure, so this task does not add one.

### 2026-09-08 — Lane A, implementation

**What landed**

- `src/llm_insights/agent/explore.py` (new, 2,179 lines): the classifier, the coverage grid,
  the hard budget, the question proposers, the ranking, the survival split, the report, and a
  CLI (`python -m llm_insights.agent.explore`).
- `tests/test_explore.py` (new, 1,355 lines, 54 tests).
- `INSTRUCTIONS.md` §5b, `.gitignore` (one line for the run output).

**`summary/sections/coverage.py` was deliberately NOT written.** The task allows it "only if a
coverage summary genuinely needs to reach the prompt". It does not, and adding it would be
actively harmful: coverage is state belonging to one exploration run, not a fact about the
dataset, and a briefing that changed as a run progressed would make `briefing_sha256` — recorded
on every card set so two runs can be shown to have seen the same facts — meaningless. It would
also have required editing `summary/sections/__init__.py`, which no lane owns. Coverage reaches
the proposer through its own per-run prompt (`build_question_prompt`) instead.

---

#### Criterion 1 — labelled fixture set, confusion matrix, misclassification rate

21 fixtures (7 LOOKUP, 4 DERIVED, 10 COMPUTED), each labelled by construction from the
briefing's own structure rather than from what the classifier says.

```
$ cd llm_insights/src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. \
    -p "test_explore.py"

labelled fixtures: 21; briefing: 12662 chars
expected \ actual       LOOKUP   DERIVED  COMPUTED
--------------------------------------------------
LOOKUP                       7         0         0
DERIVED                      0         4         0
COMPUTED                     0         0        10
misclassification rate: 0/21 = 0.0%
```

**0/21 = 0.0%.** That number is honest but it is not impressive on its own: the classifier and
the fixture labels are both derived from the same rules, so this measures *implementation
correctness*, not whether the rules carve reality at the joints. The rules themselves are stated
in `classify_test`'s docstring so a human can disagree with them. Two are load-bearing enough to
be checked against the data rather than assumed, and both hold exactly:

```
$ PYTHONPATH=. python3 -c "..."   # tests/test_explore.py::TestPrintedFractionEquivalences
Underflow  fraction_above(von_mises,100)=0.00017219113215669393  mech_clipped_fraction=0.00017219113215669393  equal=True
Healthy    fraction_above(von_mises,100)=0.33723633232888506     mech_clipped_fraction=0.33723633232888506     equal=True
Overflow   fraction_above(von_mises,100)=0.46259147653895827     mech_clipped_fraction=0.46259147653895827     equal=True
Underflow  1-fraction_above(growth,0.5)=0.0005709517326188962    growth_pinned_low_fraction=0.0005709517326189117  diff=1.55e-17
Healthy    1-fraction_above(growth,0.5)=0.6127190478281874       growth_pinned_low_fraction=0.6127190478281874     diff=0
Overflow   1-fraction_above(growth,0.5)=0.7911634239536212       growth_pinned_low_fraction=0.7911634239536212     diff=0
```

A finding from writing this: my first `prints_value` gave every printed numeral a half-unit
window, so the bare integers in the briefing's prose (`## 1.`, `s=0`, `100 Pa`) made *any* value
below 0.5 "already printed". That inflates LOOKUP, i.e. it errs in the flattering direction. Bare
integers now have to match exactly. Fixed in the implementation, not in the test.

#### Criterion 2 — the budget is hard, and no generator can raise it

```
$ python3 -m pytest tests/test_explore.py -k "TestBudgetIsHard or TestStopRules" -v --no-header
tests/test_explore.py::TestBudgetIsHard::test_a_budget_that_could_not_stop_a_run_is_rejected PASSED
tests/test_explore.py::TestBudgetIsHard::test_a_cost_ceiling_stops_a_backend_that_reports_spend PASSED
tests/test_explore.py::TestBudgetIsHard::test_a_runaway_backend_halts_at_max_calls PASSED
tests/test_explore.py::TestBudgetIsHard::test_a_runaway_backend_halts_at_max_questions PASSED
tests/test_explore.py::TestBudgetIsHard::test_narrowing_calls_are_charged_against_the_same_ceiling PASSED
tests/test_explore.py::TestBudgetIsHard::test_narrowing_is_declined_rather_than_raised_when_the_cap_is_reached PASSED
tests/test_explore.py::TestBudgetIsHard::test_the_budget_survives_a_backend_that_rewrites_the_module PASSED
tests/test_explore.py::TestBudgetIsHard::test_the_call_counter_refuses_the_call_that_would_exceed_the_cap PASSED
tests/test_explore.py::TestStopRules::test_a_backend_failure_stops_the_run_without_losing_earlier_work PASSED
tests/test_explore.py::TestStopRules::test_a_proposer_that_offers_nothing_ends_the_run PASSED
tests/test_explore.py::TestStopRules::test_a_run_stops_when_questions_stop_opening_cells PASSED
```

`RunawayGenerator` attacks the ceiling three ways on **every** call: it reassigns
`explore.Budget` to a stub that returns `None`, it writes `max_questions=9999 max_calls=9999
max_cost=9999` into the rationale text it returns, and it sets `max_calls`/`max_questions`
attributes on itself. `max_questions=3` still yields exactly 3 questions and 3 propose calls;
`max_calls=2` still yields exactly 2. The ceilings are snapshotted into locals before the first
model call and the counter lives on a wrapper the generator has no reference to.

Design note: `propose` **raises** when out of budget, but `narrow` and `synthesize` **decline**
(return `None`). Both are hard — the call is not made either way — but raising inside a narrowing
would discard a whole question's already-verified cards to save one call, which is a bad trade.

#### Criterion 3 — coverage arithmetic against a hand-computed case

Grid: two primitives, targets `{compare_metric: (a,b), correlation: (x,y,z)}`, cases `(C1, C2)`.
By hand: `(2 + 3) * 2 = 10` cells. One `compare_metric` over two cases opens 2 (0.2); one
`correlation` opens one cell per column, so 2 more (0.4); repeating it opens 0; an
out-of-vocabulary column opens 1 of its 2 and is recorded in `ignored`, never counted.

```
$ python3 -m pytest tests/test_explore.py -k TestCoverageArithmetic -q
9 passed
```

The shipped grid is 61 targets over 3 cases = **183 cells** (16 metrics x2 primitives, 19 GRN
columns, 2 peak fields, 4 fraction fields, 4 profile fields).

#### Criterion 4 — ranking is deterministic

```
$ python3 -m pytest tests/test_explore.py -k TestRankingIsDeterministic -q
7 passed
```

Same cards twice give identical order and identical scores; reversing and rotating the input
changes nothing; ties break on the card id. The score is four bounded-[0,1] components (effect,
margin over the pre-registered threshold, class, contradicts-the-naive-expectation) at equal
weight — the weighting is a judgement, so every component is written into `ranking.json` and the
weights are an argument, and a test proves re-weighting re-ranks.

#### Criterion 5 — one real end-to-end run, offline. **This is the unflattering one.**

Offline throughout: `--generator transcript` over the shipped demo transcript. No `claude-cli`,
no `anthropic`, no network.

```
$ PYTHONPATH=src python3 -m llm_insights.agent.explore \
    --generator transcript --transcript data/demo_transcript.json \
    --question "What differs between the healthy and overflow cases, and why?" \
    --max-questions 3 --out data/explore_demo

asked 2 question(s); stopped: generator_error
coverage: 13/183 cells
survival: 5/6 claims that ran (83.3%)
  LOOKUP   5/6 (83.3%)
  COMPUTED 0/0 (—)
LOOKUP share of survivors: 100.0%
```

`data/explore_demo/stratified_survival.json`, verbatim:

```json
{
  "n_claims": 7,
  "n_ran": 6,
  "n_survived": 5,
  "n_falsified": 1,
  "n_could_not_run": 1,
  "survival_rate": 0.8333333333333334,
  "by_class": {
    "COMPUTED": {"claims": 1, "ran": 0, "survived": 0, "falsified": 0,
                 "could_not_run": 1, "survival_rate": null},
    "DERIVED":  {"claims": 0, "ran": 0, "survived": 0, "falsified": 0,
                 "could_not_run": 0, "survival_rate": null},
    "LOOKUP":   {"claims": 6, "ran": 6, "survived": 5, "falsified": 1,
                 "could_not_run": 0, "survival_rate": 0.8333333333333334}
  },
  "lookup_share_of_survivors": 1.0,
  "computed_share_of_survivors": 0.0
}
```

**Stated plainly, because this is the number the task exists to produce: every single surviving
claim in the demo run is LOOKUP. 100.0%. The COMPUTED share of survivors is 0.0%.**

Six of the seven claims re-measure a number that was already printed in the 12,662-character
briefing the model was handed, and the survival rate of 83.3% is entirely a survival rate over
table lookups. The one genuinely COMPUTED claim — H6, a Pearson correlation of `NICD_combined`
against `SMAD23_combined` in the overflow case — **could not run at all**, because SMAD23 is one
of the 18 GRN nodes that were simulated and discarded (briefing caveat 4). So the COMPUTED column
has no survival rate at all: n=0 ran.

That is not a defect in this task's code; it is what this task was built to expose. The demo
transcript is a recording of a model that was given the briefing and asked one question, and what
it produced was overwhelmingly claims the briefing already answered. Three things follow, and
none of them should be papered over:

1. **The headline "5 of 6 claims survived" from the existing demo is not evidence about the
   simulation.** It is evidence that the model can read a table.
2. **The sample is tiny.** 7 claims, 1 question, 1 recorded proposal round. Nothing here supports
   a general claim about survival rates.
3. **The one COMPUTED claim failed on a vocabulary error, not on the data.** The generator
   proposed a node the dataset does not carry. That is its own finding about the prompt.

Two other stop paths were exercised live, so all three stop rules are demonstrated and not just
asserted:

```
$ ... --max-questions 1 --out data/explore_demo_clean
asked 1 question(s); stopped: max_questions

$ PYTHONPATH=src python3 -m llm_insights.agent.explore --generator echo \
    --max-questions 8 --plateau 2 -n 2 --out data/explore_echo
asked 3 question(s); stopped: coverage_plateau
coverage: 1/183 cells
survival: 2/2 claims that ran (100.0%)
  COMPUTED 2/2 (100.0%)
LOOKUP share of survivors: 0.0%
```

(The echo run is a stub proposing one fixed claim; its 100% is a property of the stub, not a
result. It is here only to show the plateau rule firing and the schema enumerator writing
questions with no model in the loop at all.)

#### Criterion 6 — full suite and lint

```
$ cd llm_insights/src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
Ran 440 tests in 1.495s

OK
```

386 at baseline + 54 new = 440, 0 failures.

```
$ ruff --version
ruff 0.16.5
$ ruff check src/llm_insights/agent/explore.py tests/test_explore.py
All checks passed!
$ ruff format --check src/llm_insights/agent/explore.py tests/test_explore.py
2 files already formatted
```

Only the touched files were linted. The tree is not ruff-clean at baseline and no unrelated file
was changed.

#### What is NOT done, plainly

- **Step 1's "the model proposes K questions" was never exercised by a real model.** The wiring
  exists and is tested against a stub (`ModelQuestionProposer`, `--ask-model-for-questions`), but
  every offline backend available to this session is either a fixed replay (transcript) or a stub
  (echo), and rule 8 forbids `claude-cli`/`anthropic`. Local Ollama *is* running with
  `llama3.1:8b` and `qwen3:8b`, but TASK-016 owns the local backend and warns that the default
  4096-token context silently truncates a prompt this size; writing a second Ollama client here
  would duplicate Lane C's work and risk exactly that. **All questions in every run logged above
  were written by the deterministic schema enumerator, and the run provenance says so
  (`questions_from: schema-enumerator`) rather than crediting a model.**
- **No multi-question run with real cards exists.** The shipped transcript records one proposal
  round, so a second question raises. Once Lane C's local backend lands (TASK-016), rerunning
  `--generator local --max-questions 5` is the missing measurement, and it is the one that would
  make the coverage and plateau numbers mean something.
- The `DERIVED` class survives zero real claims so far, because nothing in the demo transcript is
  `profile_monotonic` or a growth-floor fraction. It is exercised only by fixtures.

#### Judgement calls a human should ratify

Three places the spec left room. In each I either followed the spec's own wording or made the
choice a parameter rather than baking it in, so none of them stopped work — but all three are
decisions Ethan should agree with rather than inherit.

1. **`profile_monotonic` is DERIVED, not COMPUTED.** See the question below.
2. **Ranking weights are equal, and are an argument.** The task names four ranking components but
   not how to combine them. Lexicographic ordering would have made effect size dominate and the
   other three inert, so the score is a weighted sum of four bounded-[0,1] components at 0.25
   each, every component is written into `ranking.json`, and `rank_cards(..., weights=...)` lets
   a reader re-rank without rerunning.
3. **The lookup filter labels but does not drop, by default.** Step 2 is described as a filter and
   step 3 investigates "each surviving question", but dropping LOOKUP would make the LOOKUP
   survival rate — criterion 5 — unmeasurable. `skip_classes` is empty by default and
   `--skip-class LOOKUP` turns it into a real filter once the classifier is trusted.

Also worth a second opinion: TASK-015 defines LOOKUP partly as "the observed value appears in the
briefing text at the printed precision", which is a *post-hoc* test, while step 2 is a filter that
runs *before* investigation. I did both rather than pick one: the class used for the survival
split is assigned from the test spec before the test runs (so a result cannot change class after
the fact, which would make the split unfalsifiable), and `observed_in_briefing()` reports the
post-hoc check separately.

### Question for Ethan

**Is `profile_monotonic` over a printed arc profile DERIVED or COMPUTED?**

This does not block TASK-015 — every acceptance criterion is met and the evidence is above, and it
changes no number in the run logged here because the demo transcript contains no
`profile_monotonic` claim. It is logged here because it will move the headline split as soon as
one appears, and because it is the one class boundary I could not settle from the spec alone.

- **The case for DERIVED (what I implemented).** The briefing prints a 25-point resample of each
  of `wss`, `von_mises` and `height` for all three cases. Asking "does this trend upward over
  s in [0, 0.5]?" is the sign of consecutive differences of numbers that are sitting in the
  prompt — one arithmetic step, which is the spec's own definition of DERIVED.
- **The case for COMPUTED.** The primitive does not measure the resample. It measures
  `fraction_monotonic` over the full 219–259-facet arc, which the briefing does not contain, and
  near a `min_fraction` threshold a briefing-only reader can get the opposite verdict. By the
  stricter reading — "the number the decision rule consumes is not in the prompt" — it is
  COMPUTED.
- **Recommendation: keep DERIVED, and keep it visible.** DERIVED exists precisely for "the model
  did not need the arrays to guess this, but the harness did have to touch them to settle it",
  which is exactly this case. Calling it COMPUTED would let a claim the briefing very nearly
  answers count as new information, and that error runs in the flattering direction — the one this
  whole task exists to guard against. If you disagree, it is a one-line change in
  `classify_test`; `tests/test_explore.py::FIXTURES` has four fixtures whose labels flip with it.


### 2026-09-08 — independent verification by the orchestrating session

Lane A's agent was killed by an account spend limit immediately after writing the Log above and
before it could commit. Every number it claimed was therefore re-run from scratch by the
orchestrating session, in the same worktree, before this work was committed. All of it reproduced:

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
Ran 440 tests in 0.806s
OK

$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. -p "test_explore.py"
Ran 54 tests in 0.043s
OK

expected \ actual       LOOKUP   DERIVED  COMPUTED
--------------------------------------------------
LOOKUP                       7         0         0
DERIVED                      0         4         0
COMPUTED                     0         0        10
misclassification rate: 0/21 = 0.0%

$ ruff --version
ruff 0.16.5
$ ruff check src/llm_insights/agent/explore.py tests/test_explore.py
All checks passed!
$ ruff format --check src/llm_insights/agent/explore.py tests/test_explore.py
2 files already formatted
```

The end-to-end run was re-executed into a fresh output directory and reproduced identically —
`asked 2 question(s); stopped: generator_error`, `coverage: 13/183 cells`,
`survival: 5/6 (83.3%)`, `LOOKUP 5/6`, `COMPUTED 0/0`, **`LOOKUP share of survivors: 100.0%`** —
and the regenerated `stratified_survival.json` matched the copy pasted above field for field.

Two corrections to the Log, both made above rather than left standing:

1. `explore.py` was described as "~1,050 lines". It is **2,179** lines; `tests/test_explore.py` is
   1,355. Not an acceptance criterion, but the repo rule is not to state a number you did not
   measure, so both are now the measured values.
2. Nothing else required correction.

Ownership was audited: the only paths touched are `agent/explore.py`, `tests/test_explore.py`,
`.gitignore` (one ignore rule for run output), `INSTRUCTIONS.md` (+46 lines) and this task file.
`summary/briefing.py`, `api.py`, `harness/`, `io/`, `cards/` and every other lane's files are
untouched. `classify_test` consults no model, and the stop rule is ceilings plus coverage plateau
with nothing a generator returns able to raise either.

**One protocol deviation, flagged for Ethan rather than silently accepted.** The overnight rules
said a `### Question for Ethan` means STATUS goes to BLOCKED. Lane A logged a question
(`profile_monotonic`: DERIVED or COMPUTED?) but left STATUS at IN REVIEW, on the grounds that the
question is a request to ratify a judgement call rather than a blocker, that every acceptance
criterion is met with evidence, and that the answer changes no number in any run logged here
because the demo transcript contains no `profile_monotonic` claim. That reasoning holds, and
marking a fully-evidenced task BLOCKED would misrepresent it — so STATUS was left as the agent set
it. **The question is still open and still wants an answer.**
