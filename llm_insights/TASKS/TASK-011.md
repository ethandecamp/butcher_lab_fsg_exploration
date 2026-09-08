# TASK-011 — Callable API and an investigation session

**STATUS:** IN REVIEW
**Owner:** Claude Code (overnight Phase 2 session, 2026-09-07)
**Phase:** 0 (serial). Branch `feat/phase2-prologue`.
**Depends on:** nothing. **Blocks:** everything.

## Context

`agent/loop.py:investigate()` is already the real entry point; `agent/run.py` is CLI plumbing
wrapped around it. Two problems block Phase 1:

1. There is no way to ask a question from Python without going through `argparse`.
2. `investigate()` is **stateless**. Every run starts blank. A follow-up question cannot see what
   was already asked, so it will happily re-propose a hypothesis that was already falsified, and
   it rebuilds the Dataset and briefing from scratch every time (the expensive part).

TASK-015 (autonomous exploration) and TASK-018 both need memory across questions.

## What to build

New module `src/llm_insights/api.py`. This is the internal package API — nothing is published
anywhere, it is simply the stable surface the rest of the repo calls.

```python
study = open_study(root=None, *, generator=None, briefing_file=None) -> Study
answer = study.ask("How much does blood flow matter?", n=6) -> Answer
answer2 = study.follow_up("Why did H4 fail?", n=3) -> Answer
```

- `Study` holds: resolved `Dataset`, the briefing string, the generator, and the accumulated
  `list[Card]` plus per-question metadata.
- **The Dataset and briefing are built exactly once per Study**, lazily on first use.
- `Answer` is a frozen dataclass: `question`, `cards`, `meta`, `blurb`, `headline`, and the
  written artifact paths if an output directory was given.
- `Study.cards` returns every card so far; `Study.history` returns the ordered `Answer` list.
- `follow_up` differs from `ask` in exactly one way: prior hypothesis ids, claims and verdicts are
  passed into the proposal prompt so the model does not re-propose settled ground. Add a
  `build_followup_prompt` in `agent/generator.py` (that file is editable by this task).
- Id collisions across questions must not happen. The existing `_unique_id` logic in `loop.py`
  now needs to be seeded from `Study`-level ids, not per-run ids.
- `close()` / context-manager support that forwards to the generator's `close()`.

`agent/run.py` becomes a thin wrapper: parse args, build a `Study`, call `ask`, write output.
**Its observable CLI behavior must not change** — same flags, same stdout, same files.

## Do not

- Touch `harness/`, `io/`, `summary/`, or `cards/card.py`.
- Change any number the existing code produces.
- Introduce a new generator backend (that is TASK-016).

## Acceptance criteria

1. `tests/test_api.py` exists and covers, at minimum:
   - Dataset and briefing are constructed **once** across three `ask` calls (assert with an
     injected counter or monkeypatched constructor, not by timing).
   - `follow_up` receives the prior cards: with a stub generator that records its prompt, assert
     prior hypothesis ids appear in the follow-up prompt and do not appear in a fresh `ask`.
   - Hypothesis ids are unique across two questions in one Study.
   - `close()` propagates to the generator.
2. **CLI parity.** With `--generator transcript --transcript data/demo_transcript.json`, the
   cards produced through `api.open_study(...).ask(...)` and through `run.main([...])` are equal
   field-for-field. Test this, do not assert it.
3. Full suite green. Paste the real output of:
   `cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..`
   The count must be **>= 356 tests, 0 failures**, and the new test count stated.
4. `data/report.md` regenerated from the transcript replay is unchanged except for timestamps.
   Show the diff.
5. Ruff clean on files this task touched, at the pinned version. Lint only those files.

## Log

### 2026-09-07 — Claude Code, overnight Phase 2 session

**What was built.**

- `src/llm_insights/api.py` — `Study`, `Answer`, `open_study`, plus `DEFAULT_ROOT` and
  `_default_root()`, moved here from `agent/run.py` so `api.py` does not have to import the
  CLI. `run.py` re-imports `DEFAULT_ROOT` from `api`, so anything that referenced
  `run.DEFAULT_ROOT` still works.
- `agent/run.py` is now argument parsing, error messages and stdout. `main()` builds a
  `Study`, asks it one question and prints the result. The three exit-2 paths (bad `--root`,
  backend construction failure, backend failure mid-run) and the stale-replay warning are in
  the same order they were, so the observable CLI is unchanged.
- `agent/loop.py` — `investigate()` gained one optional parameter, `seen_ids`. Default `None`
  means "a fresh set for this run", which is exactly the old behaviour. A `Study` passes its
  own set, which the run mutates.
- `agent/generator.py` — `build_followup_prompt(question, prior)`.

**Design decision: how prior work reaches the model.** Every backend builds its own user turn
from `build_user_prompt(briefing, question, n)` (confirmed: `AnthropicGenerator.propose`,
`ClaudeCLIGenerator.propose:685`, `PasteGenerator.propose:927`). So the *question* is the only
place session context can be injected without widening the `Generator` protocol or editing
five backends. `Study.follow_up` therefore wraps the backend in a private
`_FollowUpGenerator` whose `propose` rewrites the question through `build_followup_prompt`
and forwards everything else by `__getattr__`. The loop still receives the human's question,
so `meta["question"]` and the report header are unchanged — tested by
`test_the_question_recorded_in_meta_is_the_one_the_human_asked`.

The prior-work block carries **ids, claims and verdicts only, never observed numbers**. Handing
the model the numbers would let it write a "prediction" whose answer it had already been told,
which is the one thing this project must not do. Tested by
`test_observed_numbers_are_not_leaked_into_the_prompt`.

**Consequence Ethan should know about — id collisions across questions are rejected, not
renamed.** `_validate` in `loop.py` rejects a duplicate id and counts it in `n_rejected`. Now
that `seen_ids` spans the whole study, a follow-up whose backend restarts its numbering at
`H1` gets every claim rejected. That is what the spec asked for ("Id collisions across
questions must not happen", seeded from Study-level ids) and changing `_validate` to rename
instead would change `n_rejected` for existing single-question runs, which this task is
forbidden to do. It is mitigated in the prompt: `build_followup_prompt` tells the model that a
spent id is rejected outright and the claim is lost with it. **If a live follow-up turns out to
lose claims this way, the fix is a decision for you, not for an agent** — the honest options are
(a) leave it and rely on the prompt, (b) rename proposals the way `_unique_id` already renames
successors, accepting a changed `n_rejected`. Recorded here rather than guessed at.

---

**Acceptance criterion 1 — `tests/test_api.py` exists and covers the four required cases.**

16 new tests. The four required ones are
`TestBuiltOnce.test_dataset_and_briefing_are_built_once_across_three_asks` (monkeypatched
`api.Dataset` counter plus a monkeypatched `build_briefing`, not timing),
`TestFollowUpMemory.test_prior_ids_appear_in_a_follow_up_and_not_in_a_fresh_ask`,
`TestFollowUpMemory.test_ids_are_unique_across_two_questions_in_one_study`, and
`TestFollowUpMemory.test_close_propagates_to_the_generator`.

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. -p "test_api.py"
...
Ran 16 tests in 0.124s

OK
```

**Acceptance criterion 2 — CLI parity, tested rather than asserted.**
`TestCLIParity.test_api_cards_equal_cli_cards` runs the same question through
`api.open_study(...).ask(...)` and through `run.main([...])` with
`--generator transcript --transcript data/demo_transcript.json`, then compares
`dataclasses.asdict()` of every card pairwise. It also asserts the card list is non-empty, so
the comparison cannot pass vacuously. It runs against the real dataset and is skipped only if
the dataset or the demo transcript is absent; both were present, and it passed (see the run
above, and the `ok` lines below).

```
test_api_cards_equal_cli_cards (tests.test_api.TestCLIParity.test_api_cards_equal_cli_cards) ... ok
test_the_cli_still_writes_all_four_artifacts (tests.test_api.TestCLIParity....) ... ok
test_a_bad_root_still_exits_two_before_any_backend_is_built (tests.test_api.TestCLIParity....) ... ok
```

**Acceptance criterion 3 — full suite green.**

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
....................................................................................................................................................
----------------------------------------------------------------------
Ran 372 tests in 0.722s

OK
```

372 = the verified 356 baseline + **16 new tests** from this task. 0 failures, 0 errors.
(The `Environment check: ERROR broken / Refusing to start` lines in the middle of that output
come from `tests/test_preflight.py` exercising the refusal path; they are expected output from
a passing test.)

**Acceptance criterion 4 — `data/report.md` regenerated from the transcript replay is
unchanged except for timestamps.**

The committed `data/report.md` was generated in a different environment, so it carries a
`/mnt/user-data/uploads/...` dataset root. To isolate *this task's* effect from that, the
report was regenerated twice on this machine: once from `b5724e8` (the pre-task tree, in a
throwaway worktree) and once from this branch, with identical arguments.

```
$ cd src && PYTHONPATH=. python3 -m llm_insights.agent.run --generator transcript \
    --transcript ../data/demo_transcript.json \
    --question "What differs between the healthy and overflow cases, and why?" \
    --out $SCRATCH/after --log-level ERROR
$ diff $SCRATCH/before/report.md $SCRATCH/after/report.md
17c17
< - **Generated:** 2026-09-08T02:01:21Z
---
> - **Generated:** 2026-09-08T02:00:50Z

$ diff $SCRATCH/before/cards.json $SCRATCH/after/cards.json
(exit 0 — byte-identical)

$ diff $SCRATCH/before/transcript.json $SCRATCH/after/transcript.json
6c6
<     "generated_at": "2026-09-08T02:01:21Z",
---
>     "generated_at": "2026-09-08T02:00:50Z",

$ diff $SCRATCH/before/report.html $SCRATCH/after/report.html
(one line, differing only in the same `Generated` timestamp)
```

Against the committed copy, only the timestamp and the environmental dataset root differ:

```
$ diff data/report.md $SCRATCH/after/report.md
16,17c16,17
< - **Dataset root:** /mnt/user-data/uploads/butcher_lab_fsg_exploration/one_way_fsg_model
< - **Generated:** 2026-09-01T23:33:29Z
---
> - **Dataset root:** /Users/ethandecamp/.../one_way_fsg_model
> - **Generated:** 2026-09-08T02:00:50Z
```

`cards.json` being byte-identical across the refactor is the strongest form of criterion 4:
every number, every verdict and every field order survived unchanged.

**Acceptance criterion 5 — ruff clean on the touched files, at the pinned version.**

```
$ ruff --version
ruff 0.16.5
$ ruff check tests/test_api.py src/llm_insights/api.py src/llm_insights/agent/run.py \
    src/llm_insights/agent/loop.py src/llm_insights/agent/generator.py
All checks passed!
$ ruff format --check <same five files>
5 files already formatted
```

One unrelated auto-fix was required to get there: `loop.py` used
`datetime.now(timezone.utc)`, which ruff 0.16.5 flags as UP017, so it is now
`datetime.now(UTC)`. Same object, same output; noted because it is the one hunk in this commit
that is not part of the task.

### Finding for the morning report — the tree is not ruff-clean at baseline

Not a blocker for this task, but TASK-020 criterion 3 asks for `ruff check` and
`ruff format --check` over the **whole package** and that will fail on pre-existing code.
Measured on `b5724e8`, before any Phase 2 edit:

```
$ git stash && ruff check src tests
Found 151 errors.
[*] 16 fixable with the `--fix` option (4 hidden fixes ...)
$ ruff format --check src tests
11 files would be reformatted, 21 files already formatted
$ ruff check src
Found 17 errors.
$ ruff format --check src
6 files would be reformatted, 17 files already formatted
```

So 17 of the 151 are in `src/` and the rest are in `tests/`. This is a pre-existing condition,
not something Phase 2 introduced, and cleaning it is its own commit per the style guide's
"do not reformat unrelated files in a feature PR". Flagging it so TASK-020 does not read as a
Phase 2 regression.

**STATUS set to IN REVIEW.** All five acceptance criteria met.
