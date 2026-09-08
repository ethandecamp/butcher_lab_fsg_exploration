# Phase 2 tasking — TASK-011 through TASK-020

Closed tasks TASK-000 through TASK-010 stay in `../TASKS.md`. **New tasks live one-per-file in
this directory.** That is deliberate: `TASKS.md` is the most-touched file in the repo, and
parallel agents appending to it is a guaranteed merge conflict.

## Protocol (unchanged from TASKS.md)

- Claim a task by setting **STATUS: IN PROGRESS** at the top of its file, with your agent name.
- Append to that file's **Log** section. Never delete log history.
- Move to **IN REVIEW** when done. A human closes tasks.
- **Blocked beats guessing at intent.** Write the blocker in the Log and stop.
- Any acceptance criterion that mentions a number requires **the command and its real pasted
  output** in the Log. No asserted numbers.

## Phases

Phase 0 is serial. Nothing else may start until both its tasks are IN REVIEW, because Phase 0
creates the extension points that keep Phase 1 from colliding.

| Phase | Tasks | Parallel? |
|---|---|---|
| 0 — prologue | TASK-011, TASK-012 | No. In order. |
| 1 — body | Lane A: TASK-015 · Lane B: TASK-013 then TASK-014 · Lane C: TASK-016 then TASK-017 · Lane D: TASK-019 | Yes, four lanes at once |
| 2 — join | TASK-018 | No. Needs Lane A and Lane B merged. |
| 3 — epilogue | TASK-020 | No. Last. |

TASK-019 is **BLOCKED** and must not be started. See its file.

## Branches

Phase 0 runs on `feat/phase2-prologue`, branched from tag `v0.1-demo-working`.
Each Phase 1 lane branches from the merged prologue:

- Lane A → `feat/lane-a-exploration`
- Lane B → `feat/lane-b-grn`
- Lane C → `feat/lane-c-infra`
- Lane D → `feat/lane-d-fenics` (not started)

## File ownership — the anti-collision map

An agent may create or edit **only** the paths its lane owns. If a task needs a change in a file
another lane owns, that is a blocker: write it in the Log and stop. Do not edit it "just a
little."

| Path | Owner |
|---|---|
| `src/llm_insights/api.py` | TASK-011 (then read-only) |
| `src/llm_insights/agent/run.py` | TASK-011 |
| `src/llm_insights/summary/registry.py`, `summary/sections/` | TASK-012 creates; lanes A and B each add their **own new file** under `sections/` |
| `src/llm_insights/summary/briefing.py` | TASK-012 only. **Nobody edits it after Phase 0.** |
| `src/llm_insights/grn/` | Lane B |
| `src/llm_insights/summary/sections/grn.py` | Lane B |
| `src/llm_insights/agent/explore.py` | Lane A |
| `src/llm_insights/summary/sections/coverage.py` | Lane A |
| `src/llm_insights/agent/local.py`, `agent/bakeoff.py` | Lane C |
| `Dockerfile`, `.dockerignore`, `INSTRUCTIONS_DOCKER.md` | Lane C |
| `src/llm_insights/harness/experiments.py` | TASK-018 |
| `tests/test_<task>.py` | one new file per task, never append to an existing test file |
| `pyproject.toml` | anyone, **one line at a time**, expect trivial conflicts |

## Rules that outrank convenience

1. **`one_way_fsg_model/` is read-only.** It is the oracle every verdict is checked against.
   Import from it, never edit it. Suspected defects go to **Open questions** in `../TASKS.md`.
2. **No language model judges any result.** Verdicts are arithmetic. Nothing added in Phase 2
   may put an LLM in the verification path.
3. **The verification core stays stable.** `harness/`, `io/`, and `cards/card.py` are not edited
   by feature work. New capability arrives through the existing registries
   (`@primitive`, `summary/metrics._register`) or a new module, never by rewriting the core.
   TASK-018 is the one place this is stretched, and it carries a guard-test requirement.
4. **Every compression is checked against what it compresses.** Any new summary must come with a
   test answering the same questions from the full arrays and from the summary, and agreeing.
5. **Registries iterate in sorted order.** Output must not depend on import order, or the
   answer-preservation tests go flaky depending on which lane's module loaded first.
