# TASK-011 — Callable API and an investigation session

**STATUS:** TODO
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

_(append below)_
