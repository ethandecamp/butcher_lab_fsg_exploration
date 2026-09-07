# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

This project hasn't started yet — Ethan is beginning it shortly. The repo currently just holds
handoff/reference material, not an established codebase:

- `one_way_fsg_model/` — a fluid-solid-growth (FSG) + gene-regulatory-network (GRN) model of a
  developing heart AV cushion, handed off by Daniel (dpp48@cornell.edu) as a starting reference
  for LLM-based hypothesis generation work. Its own `README.txt` documents it in full (what the
  model does, folder layout, field/column definitions, how to load results, how to rerun it).
  Read that file before touching anything in this folder — don't assume this codebase's eventual
  shape or conventions; treat it as input material until the actual project direction is set.
- `resources_for_ethan/` — background PDFs/slides (EndMT biology, FSG modeling papers, related
  computational-modeling projects).
- [`RESOURCES.md`](RESOURCES.md) — indexes both of the above. Sections 1–5 summarize every file in
  `resources_for_ethan/`; **section 6 is a source-code-level walkthrough of `one_way_fsg_model/`**
  (pipeline architecture, module-by-module algorithms/parameters, data formats, gotchas, and a
  module-boundary map for refactoring) written from a full read of every `.py` file there. Check
  RESOURCES.md before re-reading source files or the PDFs/slides from scratch — it exists so that
  work doesn't have to be redone.

Ethan's own project code lives in `llm_insights/` (scaffolded 2026-08-30): ruff + pytest
config, the team style guide, and `TASKS.md`, which is the work queue and the handoff channel
between planning agents and implementing agents. There is still no build system or test suite
inside `one_way_fsg_model/`, and none should be added there.

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

## When real project work begins

Update this file to reflect the actual project once it takes shape — commands to build/lint/
test, and the architecture of whatever gets built. Don't extrapolate that structure from
`one_way_fsg_model/` in the meantime; it's reference material, not a template to extend.
