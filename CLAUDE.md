# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

**The project is live** (as of 2026-09-02). `llm_insights/` contains a working system that
proposes qualitative claims about Dan's FSG/GRN model and then falsifies them by running
executable tests against the model's output — no language model grades any result. The
one-question demo runs end to end, live, with no Anthropic API key (it authenticates through the
Claude Code CLI and a Claude subscription).

Start at [`llm_insights/README.md`](llm_insights/README.md) for what it is and
[`llm_insights/INSTRUCTIONS.md`](llm_insights/INSTRUCTIONS.md) for how to run it.

Repo layout:

- **`llm_insights/` — Ethan's project code.** Five packages: `io/` (the only module that knows
  the FSG file layout), `summary/` (metrics, arc-length profiles, the briefing the model reads),
  `harness/` (the falsifier — six executable primitives), `agent/` (generator backends and the
  CLI), `cards/` (the report). `llm_insights/CLAUDE.md` governs all work in there and
  `llm_insights/TASKS.md` is the work queue and inter-agent handoff channel — read both before
  writing code.
- `one_way_fsg_model/` — a fluid-solid-growth (FSG) + gene-regulatory-network (GRN) model of a
  developing heart AV cushion, handed off by Daniel (dpp48@cornell.edu). This is the simulator
  the project reasons about and the ground truth every hypothesis is verified against.
  **Read-only** — see the hard rule below. Its own `README.txt` is canonical for field/column
  definitions, how to load results, and how to rerun it.
- `resources_for_ethan/` — background PDFs/slides (EndMT biology, FSG modeling papers, related
  computational-modeling projects). Read-only.
- [`RESOURCES.md`](RESOURCES.md) — indexes both of the above. Sections 1–5 summarize every file in
  `resources_for_ethan/`; **section 6 is a source-code-level walkthrough of `one_way_fsg_model/`**
  (pipeline architecture, module-by-module algorithms/parameters, data formats, gotchas, and a
  module-boundary map for refactoring) written from a full read of every `.py` file there. Check
  RESOURCES.md before re-reading source files or the PDFs/slides from scratch — it exists so that
  work doesn't have to be redone.

There is still no build system or test suite inside `one_way_fsg_model/`, and none should be
added there.

## Hard rule: `one_way_fsg_model/` is read-only

**No code changes may be made to Dan's original FSG model.** Everything under
`one_way_fsg_model/` is Daniel Pearce's fluid-solid-growth + GRN model, and this project treats
it as ground truth — it is the oracle that every generated hypothesis is verified against.
Editing it silently invalidates every result built on top of it, and any comparison against
previously generated output becomes meaningless.

That means: do not edit, refactor, reformat, lint, "fix", or reorganize any file in that
folder. Import from it, read its outputs, and wrap it in new code instead. The same applies to
`resources_for_ethan/` — those are Dan's handoff files, and nothing in that folder gets renamed,
edited, or deleted.

**`RESOURCES.md` is the one exception, and only in one direction.** It is this project's own
index, not handoff material, and its whole purpose is to spare the next reader a cold read. When
a new file lands in `resources_for_ethan/`, **append an entry for it** in the existing house
style (title line with citation, `*Summary:*`, `*Takeaways:*` aimed at this project). What stays
prohibited is rewriting, reorganizing, condensing, or reformatting entries that are already
there — that destroys work someone did by reading a source end to end.

If something in there looks like a genuine bug (for example the `MECH_MAX` normalization
clipping ~44% of the overflow case), **do not patch it.** Write it up under **Open questions**
in `llm_insights/TASKS.md` and raise it with Dan. Deciding whether that behavior is intentional
is his call, not an agent's.

Ethan's own work goes in `llm_insights/` — see `llm_insights/CLAUDE.md` for the conventions
that apply there.

## Commands

Everything runs from `llm_insights/`. `INSTRUCTIONS.md` is the full reference — every flag, the
backends, and a failure-mode table in §9. Only the essentials are repeated here, so there is one
place to keep current rather than two.

```bash
cd llm_insights
source .venv/bin/activate      # not optional — see INSTRUCTIONS.md §1

# run it live, no API key. any question.
PYTHONPATH=src python3 -m llm_insights.agent.run --question "..." --out data/live

# see the prompt and its size without spending anything
PYTHONPATH=src python3 -m llm_insights.agent.run --dry-run --question "..."

# tests
python3 -m pytest tests -q
```

**356 tests, 0 failures** (measured 2026-09-07). Five of them are a scipy-based oracle layer in
`tests/test_correlation_math.py` that skips wherever scipy is absent and runs in Ethan's venv.

Lint is ruff and only ruff, configured in `llm_insights/pyproject.toml`; see
`llm_insights/STYLE_GUIDE.md` for the conventions it enforces.
