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

There is no build system, test suite, or lint config anywhere in the repo yet, and no project
code of Ethan's own so far.

## When real project work begins

Update this file to reflect the actual project once it takes shape — commands to build/lint/
test, and the architecture of whatever gets built. Don't extrapolate that structure from
`one_way_fsg_model/` in the meantime; it's reference material, not a template to extend.
