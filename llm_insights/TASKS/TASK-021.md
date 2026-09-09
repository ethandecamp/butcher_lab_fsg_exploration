# TASK-021 — Make autonomous exploration actually run

**STATUS:** IN REVIEW
**Phase:** 2a. Branch `feat/task-021-explore-fixes`, from `feat/phase2-merged`.
**Depends on:** the three-lane merge (`a409531`). **Blocks:** TASK-018, TASK-020.

## Context

The first two live runs of `agent.explore --generator local --ask-model-for-questions` against
`llama3.1:8b` on the merged tree, 2026-09-09, proposed **20 hypotheses and produced 4 verdicts.**
Sixteen were lost to two mechanical defects. Neither is about the model's scientific judgement, so
the quality of what it proposes is still, after all of Phase 1, **unmeasured**.

Both defects were caught loudly by the existing guards — nothing produced a wrong verdict, and
that is not in question here. What is in question is that the loop cannot currently get far enough
to measure anything.

## Defect 1 — run-scoped ids make every question after the first produce zero cards

`api.Study._seen_ids` is one set for the whole study. `loop._validate` rejects any repeated id at
the **propose** stage. Each proposal call is independent, so the model restarts at `H1` for every
question, and the whole batch is rejected:

```
question 2 -> rejected proposal 0..4 at stage propose: id 'H1' was already used in this run
question 3 -> rejected proposal 0..4 at stage propose: id 'H1' was already used in this run
asked 3 question(s); stopped: coverage_plateau   coverage: 1/231   survival: 0/0
```

The plateau stop then fires correctly for the wrong reason — two barren questions in a row.

**The repair already exists in this file and is not applied here.** `loop._unique_id` (line ~138)
suffixes a colliding id, and the **narrow** path calls it (line ~439). The propose path does not.

This is also why Lane A never had a working multi-question run. TASK-015's Log recorded the
limitation as a property of the transcript backend — "a second question raises `generator_error`"
— which put a plausible explanation in front of a real defect. Worth stating in the Log.

## Defect 2 — three vocabularies, overlapping names, none of it in the prompt

A primitive's `field`/`metric` parameter draws from one of three disjoint namespaces, and the same
physical quantity is spelled differently in each:

| quantity | metric (`compare_metric`, `metric_ordering`) | GRN field (`correlation`) | profile field (`profile_monotonic`, `peak_location`) |
|---|---|---|---|
| wall shear stress | `wss_mean_dyn_cm2` | `wss_dyn_cm2` | `wss` |
| von Mises | `von_mises_mean_pa` | `von_mises_pa` | `von_mises` |
| a GRN node | only 8 `*_mean_combined` are registered | `Snai1_shear_only`, all 24 x 3 | not available |

All four confusions were observed in one question:

```
unknown metric 'Snai1_shear_only'                       <- GRN field used as a metric
unknown GRN field 'wss_mean_dyn_cm2'                    <- metric used as a GRN field
parameter 'field' must be one of ['wss', 'von_mises'], got 'Snai1_shear_only'
```

`build_system_prompt` already introspects `PRIMITIVES` and lists the registered **metric** names.
It lists neither the GRN correlation fields nor the profile/peak fields, so the briefing names 24
nodes the model cannot address and gives it no way to know which spelling belongs to which
primitive.

Related, and in scope because it is the same code path: Lane B reported that
`build_system_prompt()` offers a **stale** metric vocabulary — the 8 GRN metrics register as an
import side effect of `summary/sections/grn.py`, and nothing in `agent/generator.py` imports
`summary.sections`, so a bare call lists 16 metrics instead of 24. Both real CLI paths happen to
build a briefing first, so it is latent rather than live. Fix it here.

## What to build

- **The id fix.** Multi-question runs must produce cards. Prefer prefixing per question
  (`q02_H1`) over suffixing (`H1_r2`), because it keeps provenance readable and preserves the
  model's own within-question references — each proposal call is independent, so ids are only ever
  self-consistent inside one question. Applying `_unique_id` at propose is acceptable if you
  justify it. Whatever you choose, an id must still appear exactly once in a run and a card must
  still be traceable to the claim that produced it.
- **The vocabulary fix.** `build_system_prompt` must tell the model, per primitive, which names
  that primitive accepts — metrics, GRN fields, and profile/peak fields as three named
  vocabularies, with the overlap called out explicitly so the model is not left to guess that
  `wss` and `wss_dyn_cm2` and `wss_mean_dyn_cm2` are the same quantity in three registers.
  **Derive every list from the same source the validator uses.** A hand-maintained copy in the
  prompt is the exact failure this project exists to avoid: it would drift, and the drift would be
  invisible.
- **Optional, your judgement, and say which you chose and why:** one repair re-prompt when a
  hypothesis fails on a name error only. The harness already returns an excellent message naming
  the valid set. A repair loop does not put a model in the verification path — it still judges no
  result — but it costs a call per repair and it must be bounded by the existing budget, not
  outside it. If you add it, it is off by default behind a flag.

`harness/primitives.py` is verification core: prefer reading what it already exposes over editing
it. If exposing a vocabulary genuinely requires a change there, that is allowed under the
architecture rule's defect exception, but it must come with a test layer that does not depend on
the thing being changed, and you must say in the Log why the change was unavoidable.

## Acceptance criteria

1. **A multi-question regression test.** A stub generator that always returns ids `H1`..`H5`, run
   over at least 3 questions, must produce cards for **every** question. This test must fail
   against the current `feat/phase2-merged`. Paste both runs — failing before, passing after.
   `tests/test_explore_ids.py`.
2. **Vocabulary/validator agreement, tested both directions.** A test asserting that every name
   `build_system_prompt` offers for a primitive is accepted by that primitive, **and** that every
   name the primitive accepts is offered. A name in one and not the other is a failure. This is
   [[verification-philosophy]] applied to the prompt: the vocabulary is a compression of the
   validator and must be checked against it. Paste the run.
3. **The stale-vocabulary bug is fixed and tested.** `build_system_prompt()` called with nothing
   else imported must list all 24 metrics, not 16. Paste a test that fails without the fix.
4. **A real live run, `--max-questions 5`, on `--generator local` with `llama3.1:8b`.** Paste the
   full console output and the resulting `stratified_survival.json`. Report:
   - how many hypotheses were proposed, and how many reached a verdict;
   - the count of `could not run` **split by cause** (name error, id collision, anything else);
   - the survival rate split by LOOKUP / DERIVED / COMPUTED, and the LOOKUP share of survivors.
   **State the numbers plainly whatever they are.** A low survival rate honestly reported is the
   result; a flattering one that hides a mechanical failure is not. If most hypotheses still fail
   for a reason this task did not fix, name it — that is the finding, and it is what the next task
   is scoped from.
5. **Wall time recorded** for that run, and the per-call latency, so the demo can be paced. The
   bake-off measured 37.4 s per proposal call at 6,248 prompt tokens; the merged briefing is
   larger, and the first live runs showed 41-65 s. Paste real numbers.
6. Full suite green, count pasted. Expect **550** plus whatever this task adds. Ruff clean at the
   pinned 0.16.5 on touched files.

## Out of scope

- **The hedge.** In the smoke run the model proposed a claim (`r >= 0.5`) and its own negation
  (`r <= 0.1`) over one identical test, so one had to pass, and the "surviving claim" was the null
  result. Nothing currently rejects a batch containing mutually exclusive claims on one test.
  Real, and TASK-022.
- **Prose/test correspondence.** Four hypotheses carried byte-identical params under four
  different English claims; two of them described mechanical stress while testing a correlation
  between two GRN nodes. "Every verdict was produced by executable code" still holds. "The verdict
  tests the claim" is a different statement and is not established. Also TASK-022, and it is the
  more important of the two.
- The briefing's character budget (19,538 of 20,000). Do not add anything to the briefing here.
- Merging, pushing, TASK-018, TASK-019, TASK-020.

## Log

_(append below)_

---

### 2026-09-08 — Ethan / Claude Opus 5. Both defects fixed; two live runs measured.

**STATUS: IN REVIEW.** Branch `feat/task-021-explore-fixes`, from `feat/phase2-merged` (a409531).

Worked the criteria in order. Each is logged with the command and its real output. Nothing
below is asserted without a pasted run.

---

#### Criterion 1 — multi-question regression test

`tests/test_explore_ids.py`. A stub generator (`RestartingGenerator`) always returns ids
`H1`..`H5`, whatever it is asked, because that is what a real backend does: every proposal
call is independent and has no memory of the ids the last one spent. Three questions, five
claims each. The dataset is a stand-in on which every primitive fails — the defect is at
the *propose* stage, and a claim that could not run is still a card the loop must account
for, so the assertion holds without the 2.1 GB results tree.

**Failing against `feat/phase2-merged`** (production code stashed back to that state, tests
kept):

```
$ git stash push llm_insights/src/llm_insights/agent/loop.py \
    llm_insights/src/llm_insights/api.py llm_insights/tests/test_api.py
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. -p test_explore_ids.py -v

rejected proposal 0 at stage propose: hypothesis: id 'H1' was already used in this run; ids must be unique so outcomes can be matched to claims
rejected proposal 1 at stage propose: hypothesis: id 'H2' was already used in this run; ids must be unique so outcomes can be matched to claims
rejected proposal 2 at stage propose: hypothesis: id 'H3' was already used in this run; ids must be unique so outcomes can be matched to claims
rejected proposal 3 at stage propose: hypothesis: id 'H4' was already used in this run; ids must be unique so outcomes can be matched to claims
rejected proposal 4 at stage propose: hypothesis: id 'H5' was already used in this run; ids must be unique so outcomes can be matched to claims
   ... (the same five, again, for question 3)

AssertionError: 0 != 5 : question 2 produced 0 card(s); 5 were proposed and none should be lost to an id
AssertionError: 0 != 5 : question 3 produced 0 card(s); 5 were proposed and none should be lost to an id
AssertionError: 5 != 15

Ran 3 tests in 0.002s
FAILED (failures=4)
```

That is the live-run symptom of 2026-09-09 reproduced in 2 ms.

**Passing after:**

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. -p test_explore_ids.py

hypothesis q01_H1 could not run: ... q01_H5 ... q02_H1 ... q03_H5
Ran 4 tests in 0.002s
OK
```

**What the fix is.** `loop._apply_id_prefix` stamps a per-question prefix onto every id at
the propose stage and at the narrow stage; `api.Study` supplies it from a counter advanced
*before* the run, so a question that fails part way through never lends its prefix to the
next one. Prefixing was chosen over `_unique_id`'s suffixing, as the spec preferred: `q02_H1`
says where a card came from, `H1_r2` does not, and prefixing rewrites the whole batch by one
rule so the model's own within-question numbering survives. A duplicate *inside* one question
is still a genuine collision and is still rejected.

**A trade worth naming.** The one-question CLI (`agent/run.py`) also goes through `Study`, so
the demo's ids are now `q01_H1` rather than `H1`. Uniform is better than asymmetric here — a
report reading `H1..H5, q02_H1..q02_H5` would be worse than one reading `q01_*, q02_*` — and
nothing depended on the old spelling (full suite green). Recorded transcripts are unchanged;
only the ids assigned on replay differ.

**Two tests in `tests/test_api.py` were updated, not deleted.**
`test_ids_are_unique_across_two_questions_in_one_study` asserted that a re-proposed `H1` is
*rejected* (`n_rejected == 2`, `n_admissible == 0`). That assertion encoded Defect 1 as
intended behaviour, which is exactly what this task exists to change. It now asserts the real
contract: both claims survive, ids stay unique, and a card still names its question.

**On TASK-015's Log.** It recorded the multi-question limitation as a property of the
transcript backend — "a second question raises `generator_error`". That put a plausible
explanation in front of a real defect and is why Lane A never had a working multi-question
run. The defect was in `loop`/`api`, not in any backend.

**A defect in the fix, found by the first live run and fixed.** `build_narrow_prompt`
serializes the refuted claim with its id already prefixed, so a model asked to narrow
`q05_H5` writes `q05_H6` back — and the prefix was stamped on a second time:

```
INFO llm_insights.agent.loop: narrowed q05_H5 into q05_q05_H6
```

Unique and traceable, and still wrong. `_apply_id_prefix` is now idempotent, with
`TestNarrowingDoesNotRepeatThePrefix` reproducing the exact id before the fix:

```
AssertionError: 2 != 1 : successor id 'q02_q02_H9' repeats its question prefix
```

---

#### Criteria 2 and 3 — vocabulary/validator agreement, and the stale registry

`tests/test_prompt_vocabulary.py`, 9 tests.

`build_system_prompt` now emits a `VOCABULARIES` block in which every fixed name list is
attributed to the exact `primitive.parameter` that accepts it, parameters sharing a list are
grouped so the model can see that they do, and a derived table spells out the three registers
of each physical quantity. Every list is resolved at render time from the object the
harness's own validator consults (`METRIC_NAMES` via the section registry, `GRN_NODES ×
GRN_SCENARIOS`, `CONDITION_BY_CASE`/`CASE_BY_CONDITION`, `_PEAK_FIELDS`/`_FRACTION_FIELDS`/
`_PROFILE_FIELDS`). There is no second copy to drift.

`primitive_vocabularies()` is the single accessor the prompt renders from and the tests read.

**How the reverse direction avoids being tautological.** "Every name the primitive accepts is
offered" cannot be checked against the constant the prompt already reads. So the test passes
each parameter a bogus name and parses the accepted set out of **the validator's own
rejection message**, which enumerates it (`unknown metric 'x'; known metrics are [...]`,
`unknown GRN field 'x'; available fields are [...]`, `parameter 'field' must be one of [...]`,
`unknown case 'x'; expected one of [...] or [...]`). Which parameters are constrained at all
is discovered the same way rather than declared, so a primitive that gains one fails the test
without anyone remembering to add it to a list.

**Failing with each defect reintroduced** (the GRN-field attribution removed, and the section
import removed from `_metric_names`) — this is the `feat/phase2-merged` state of both:

```
FAIL: test_a_bare_process_offers_every_metric
AssertionError: Lists differ: ['arc_length_mm', 'cushion_height_mm', ...] != ['arc_length_mm', 'beta_catenin_mean_combined', ...]
  (a bare process offers 16 metric(s); 24 are registered once every provider module is imported)

FAIL: test_every_constrained_parameter_has_a_vocabulary_in_the_prompt (primitive='correlation', param='field_x')
AssertionError: correlation.field_x tells the code exactly what it accepts (unknown GRN field
'definitely_not_a_field_name'; available fields are ['mech_norm', 'von_mises_pa', 'wss_dyn_cm2',
'wss_norm', 'EndMT_combined', ... 'YAP_TAZ_shear_only']) but the prompt offers the model nothing

FAIL: test_every_constrained_parameter_has_a_vocabulary_in_the_prompt (primitive='correlation', param='field_y')
   (the same)

Ran 6 tests in 0.248s
FAILED (failures=3)
```

**Passing after:**

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. -p test_prompt_vocabulary.py
Ran 9 tests in 0.316s
OK
```

**The stale registry (criterion 3), measured.** `METRIC_NAMES` is *rebound* by
`register_metric`, and the eight GRN metrics register as an import side effect of
`summary/sections/grn.py`:

```
$ PYTHONPATH=. python3 -c "
import llm_insights.summary.metrics as m
print('before:', len(m.METRIC_NAMES))
import llm_insights.summary.sections.grn
print('after :', len(m.METRIC_NAMES))"
before: 16
after : 24
```

`_metric_names()` now imports `llm_insights.summary.sections` first — the same import the
briefing itself does to populate the registry, so it reads the same source rather than a
second list. The test runs in a **subprocess**, because a test process that has already
imported the briefing machinery cannot observe the bug.

**`harness/primitives.py`, `io/` and `cards/card.py` were not touched.** Every vocabulary was
already reachable by reading what those modules expose; the architecture rule's defect
exception was not needed.

---

#### Criterion 6 — suite and lint

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
Ran 563 tests in 13.727s
OK

$ ruff --version
ruff 0.16.5
$ ruff check <the six touched files>
All checks passed!
$ ruff format --check <the six touched files>
6 files already formatted
```

563 = the expected 550 baseline + 4 (`test_explore_ids.py`) + 9 (`test_prompt_vocabulary.py`).
Touched files: `agent/generator.py`, `agent/loop.py`, `api.py`, `tests/test_explore_ids.py`,
`tests/test_prompt_vocabulary.py`, `tests/test_api.py`. Ruff was run on touched files only;
the tree was already not ruff-clean at baseline (42 pre-existing ARG00x in `test_explore.py`
and `test_agent.py` alone), which is TASK-020's problem, not this one's.

---

#### Criterion 4 and 5 — live run 1, `data/explore_021/`

Preflight, as specified:

```
$ curl -s .../api/generate -d '{"model":"llama3.1:8b","prompt":"ok","stream":false,"keep_alive":"30m"}' >/dev/null
$ curl -s http://127.0.0.1:11434/api/ps | python3 -m json.tool | grep -i context
            "context_length": 32768
```

```
$ export LLM_INSIGHTS_LOCAL_URL="http://127.0.0.1:11434"
$ export LLM_INSIGHTS_LOCAL_MODEL="llama3.1:8b"
$ time PYTHONPATH=src python3 -m llm_insights.agent.explore \
    --generator local --ask-model-for-questions --max-questions 5 --out data/explore_021

asked 5 question(s); stopped: max_questions
coverage: 21/231 cells
survival: 13/23 claims that ran (56.5%)
  LOOKUP   12/17 (70.6%)
  DERIVED  0/4 (0.0%)
  COMPUTED 1/2 (50.0%)
LOOKUP share of survivors: 92.3%

INFO llm_insights.agent.local: local generator finished: 15 local call(s), 353.0s, 134981 in / 5959 out tokens, $0.00
PYTHONPATH=src python3 -m llm_insights.agent.explore --generator local  5:53.56 total
```

`data/explore_021/stratified_survival.json`:

```json
{
  "n_claims": 33,
  "n_ran": 23,
  "n_survived": 13,
  "n_falsified": 10,
  "n_could_not_run": 10,
  "survival_rate": 0.5652173913043478,
  "by_class": {
    "COMPUTED": {"claims": 11, "ran": 2, "survived": 1, "falsified": 1, "could_not_run": 9, "survival_rate": 0.5},
    "DERIVED":  {"claims": 4,  "ran": 4, "survived": 0, "falsified": 4, "could_not_run": 0, "survival_rate": 0.0},
    "LOOKUP":   {"claims": 18, "ran": 17, "survived": 12, "falsified": 5, "could_not_run": 1, "survival_rate": 0.7058823529411765}
  }
}
```

**Proposed vs verdicts.** 25 raw proposals over 5 questions, plus 8 narrowing successors =
**33 admissible claims, 0 rejected at validation**, of which **23 reached a verdict**. Against
the two runs this task was scoped from — 20 proposed, 4 verdicts — that is 33 proposed and 23
verdicts.

**`could not run`, split by cause (10 total):**

| n | cause |
|---|---|
| 8 | `unknown GRN field` — a name from the wrong register handed to `correlation` |
| 1 | `test 'fraction_above' is missing required setting(s) ['value']` |
| 1 | `parameter must be one of [...]` |
| **0** | **id collision** |

The defect this task was written for is gone. Every remaining failure is a name.

Splitting the 8 further: **5 were the bare node name `YAP_TAZ`** (question 1, all five claims)
and **3 were the metric `yap_taz_mean_combined`** (question 5) — while the prompt listed all
nineteen valid GRN fields, including `YAP_TAZ_combined`, under that exact parameter two lines
away.

**Survival split.** 13/23 = 56.5% overall. LOOKUP 12/17 (70.6%), DERIVED 0/4 (0.0%), COMPUTED
1/2 (50.0%). **LOOKUP share of survivors: 12/13 = 92.3%.**

That last number is the honest headline and it is not flattering: essentially everything that
survived was a claim whose answer was already printed in the briefing. One survivor out of
thirteen required the harness to compute something the model could not read off. The COMPUTED
column is where the mechanical losses land — 9 of its 11 claims could not run — so the
COMPUTED rate of 50% rests on two claims and means very little yet.

**Wall time and per-call latency (criterion 5).** 5:53.56 wall; 353.1 s of that inside the
model, so pacing is model-bound and nothing else matters for the demo.

| call kind | n | min | mean | max |
|---|---|---|---|---|
| proposal (5 hypotheses, ~16.3k prompt tokens) | 5 | 36.2 s | **45.7 s** | 64.5 s |
| narrow (one successor) | 8 | 8.7 s | 10.4 s | 12.3 s |
| question list / narrative summary | 2 | 6.9 s | 20.6 s | 34.4 s |

Against the bake-off's 37.4 s at 6,248 prompt tokens: the proposal prompt is now ~16.3k
tokens (merged briefing plus the vocabulary block) and costs 45.7 s on average — consistent
with the 41–65 s the first live runs showed. **Budget ~6 minutes for a five-question demo.**
The system prompt grew from 6,250 to 9,157 characters; the briefing was not touched and
remains 19,538 of its 20,000 ceiling.

---

#### Live run 2, `data/explore_021b/` — and the finding

Run 1's ten `could not run` were eight name errors, five of them the bare node name
`YAP_TAZ` and three the metric `yap_taz_mean_combined`. So the prompt was given the
near miss by name: `invalid_name_examples()` derives the bare node stems (by splitting the
activity names on the loader's own scenario suffixes) and the GRN-flavoured metrics, and the
prompt now carries

```
REJECTED BY correlation.field_x and correlation.field_y, however reasonable they look:
["EndMT", "NICD", "Snai1", "Snai2", "YAP_TAZ", "beta_catenin_mean_combined", ...].
A node name on its own is not a column. Every activity carries a scenario, so write the
full name from the GRN FIELDS list above and never the node alone.
```

A named trap is a claim like any other, so `TestNamedTrapsAreRealTraps` checks that every
name the prompt calls invalid really is rejected by that primitive, and that no name is both
offered and warned against.

Same command, same model, `--out data/explore_021b`:

```
asked 5 question(s); stopped: max_questions
coverage: 19/231 cells
survival: 11/18 claims that ran (61.1%)
  LOOKUP   11/18 (61.1%)
  COMPUTED 0/0 (—)
LOOKUP share of survivors: 100.0%

INFO llm_insights.agent.local: local generator finished: 13 local call(s), 358.0s, 116785 in / 5867 out tokens, $0.00
PYTHONPATH=src python3 -m llm_insights.agent.explore --generator local  5:58.74 total
```

```json
{
  "n_claims": 29, "n_ran": 18, "n_survived": 11, "n_falsified": 7, "n_could_not_run": 11,
  "survival_rate": 0.6111111111111112,
  "by_class": {
    "COMPUTED": {"claims": 10, "ran": 0, "survived": 0, "falsified": 0, "could_not_run": 10, "survival_rate": null},
    "DERIVED":  {"claims": 0,  "ran": 0, "survived": 0, "falsified": 0, "could_not_run": 0,  "survival_rate": null},
    "LOOKUP":   {"claims": 19, "ran": 18, "survived": 11, "falsified": 7, "could_not_run": 1, "survival_rate": 0.6111111111111112}
  },
  "lookup_share_of_survivors": 1.0,
  "computed_share_of_survivors": 0.0
}
```

Per-call latency: proposal n=5, min 41.6 s, **mean 50.2 s**, max 73.9 s at ~16.6k prompt
tokens; narrow n=6, mean 10.6 s; 358.1 s in-model of 5:58.74 wall.

**It did not help. The errors moved; the total did not.**

| cause | run 1 | run 2 |
|---|---|---|
| `unknown GRN field` | 8 | 4 |
| `unknown metric` | 0 | 3 |
| wrong value for a fixed-vocabulary parameter | 1 | 4 |
| missing required parameter | 1 | 0 |
| **id collision** | **0** | **0** |
| **total `could not run`** | **10 / 33** | **11 / 29** |

The exact names, run 2: `snai1_mean_combined` used as a metric (×3 — not registered; the
model extrapolated the naming convention of the ten that are), `op: 'eq'` (×2),
`field: 'grn_saturated_fraction'` (×2 — a metric used as a `fraction_above` field),
`yap_taz_mean_combined`, `nicd_mean_combined`, `von_mises_mean_pa` and `wss_mean_dyn_cm2`
used as GRN fields (×1 each).

The last two are the metric↔GRN-field inversion that the prompt tabulates *by those exact
names*, three lines above the list it should have copied from.

**The finding, stated plainly: naming is not an information problem, and this task's
hypothesis about it was wrong.** The prompt now lists every valid name, attributes every list
to the exact `primitive.parameter` that accepts it, tabulates the three registers of each
physical quantity, and names the plausible near misses as rejected. `llama3.1:8b` commits the
same class of error at the same rate anyway — roughly **a third of all claims** in both runs
(10/33, 11/29). What changed between the runs is only which register it guessed wrong in.
Two runs is a small sample and the per-cause split is noise-dominated; the *total* is not.

Also worth separating out: `op: 'eq'` is not a naming error at all. There is no equality
operator — `_OPS` is `gt/ge/lt/le` — and the model wanted one. No amount of vocabulary fixes
that; it is a capability gap.

**On the optional repair re-prompt — I did not add it, and here is why.** The spec left it to
my judgement. Before these runs, the case for it was a guess; after them it is evidence-backed,
and it is now the obvious next move — precisely *because* two runs show that more naming
information in the prompt does not reduce the error rate. But it is a fourth model-interaction
path, with its own budget accounting and its own failure surface, and it deserves its own
before/after measurement rather than being bolted on unmeasured at the end of this task. It
also would not have fixed `op: 'eq'` or the missing required parameter. Scoped below.

**A second thing the runs showed.** Two proposals were rejected at validation in run 2:

```
rejected proposal N at stage propose: hypothesis: id 'q05_H2' was already used in this run
rejected proposal N at stage propose: hypothesis: id 'q05_H5' was already used in this run
```

That is the model returning the same id twice *inside one question*, which is a genuine
collision and correctly rejected. The prefix deliberately does not paper over it.

**And the one that matters most for what the demo can claim.** The LOOKUP share of survivors
was 92.3% in run 1 and **100% in run 2**. COMPUTED claims ran 2/11 and then 0/10 — the
mechanical losses land almost entirely on the class whose answers are *not* in the briefing,
so what survives is overwhelmingly the model re-reading its own briefing. Every verdict was
still produced by executable code, and the stratified table reports this honestly rather than
hiding it in an overall rate. But on the strength of these two runs, the system has not yet
demonstrated that it can falsify a claim whose answer was not already printed in front of it.

---

### Where this leaves things

**Fixed and measured.** Defect 1 is gone: 0 id collisions across two five-question runs, 33
and 29 claims where the baseline produced 20 proposals and 4 verdicts. Defect 2 (the stale
16-of-24 metric vocabulary, and the three unattributed namespaces) is fixed and both
directions are tested against the validator's own error text.

**Not fixed, and now the dominant failure mode.** Name errors, ~1 claim in 3, invariant to
prompt content. **This is what TASK-022 should be scoped from**, ahead of the hedge:

1. **A bounded repair re-prompt on a name error**, off by default behind a flag, counted
   against the existing budget. The harness already returns a message naming the valid set;
   feeding that back is the model fixing its own syntax, not grading an outcome. Measure it
   the same way: before/after on the same command.
2. **The 24-node briefing versus the 5-node `correlation`.** `data/grn_full/*/activities.csv`
   has 78 columns — 24 nodes × 3 scenarios — and the briefing discusses them, while
   `io/dataset.py` still loads `GRN_NODES = (Snai1, Snai2, EndMT, NICD, YAP_TAZ)`, so
   `correlation` can address 5 of the 24 nodes the model has just been told about. Several of
   both runs' name errors are the model reaching for a node it read about and cannot test.
   This is a Lane B merge left half-finished, in our own code, not Dan's. Not patched here:
   `io/` is verification core and this is outside TASK-021's two named defects.
3. **No equality operator.** `_OPS` has no `eq`; the model asked for one twice in run 2.
   Either add it (with a tolerance, since float equality is meaningless here) or say in the
   prompt that equality claims must be written as a two-sided bound.

**Still out of scope and still real:** the hedge, and prose/test correspondence. Both TASK-022
per the spec; on this evidence they are less urgent than the name-error rate, because a claim
that never runs cannot be hedged or mismatched.

**Open question for Ethan.** Nothing in `one_way_fsg_model/` was touched and no new suspected
defect in it was found. Item 2 above is ours, not Dan's, so it is not going under **Open
questions** in TASKS.md.

**Files changed:** `agent/generator.py`, `agent/loop.py`, `api.py`, `tests/test_api.py` (two
assertions updated, none deleted), new `tests/test_explore_ids.py` and
`tests/test_prompt_vocabulary.py`. `harness/`, `io/`, `cards/card.py`, `summary/briefing.py`
and the briefing's contents untouched. Not merged, not pushed.
