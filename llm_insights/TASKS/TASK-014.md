# TASK-014 — Put the gene network into the briefing

**STATUS:** TODO
**Phase:** 1, Lane B (second). Branch `feat/lane-b-grn`.
**Depends on:** TASK-012, TASK-013. **Blocks:** TASK-018.

## Context

The briefing today carries GRN **outputs** only — three aggregate EndMT scalars and a saturation
fraction. It contains no network structure: no nodes, no pathways, no parameters. A model reading
it can only make correlational claims. Given the topology it can make mechanistic ones, which is
the qualitative insight this project exists to produce.

## What to build

A new file `src/llm_insights/summary/sections/grn.py`, registered through TASK-012's
`@briefing_section`. **Do not edit `briefing.py`.**

The section covers:

- The 24 nodes grouped by pathway, and the input wiring in words: which nodes read shear, which
  read mechanical stress, and which are the three outputs.
- The `Params` values actually used, and the two normalization constants `WSS_MAX = 30.0` and
  `MECH_MAX = 100.0`, cross-referenced to the existing saturation caveat.
- Per-node summary statistics for the newly available nodes, from TASK-013's output.

New metrics registered through TASK-012's `register_metric` for the nodes worth testing —
**not all 21 blindly.** Each metric carries units and a provenance string like every existing one.

## The budget constraint

The briefing is currently 12,698 characters (~3,200 tokens). Twenty-one nodes times three cases
times two statistics is 126 new numbers, and dumping them would roughly double the prompt and
bury the existing content.

**Hard ceiling: the briefing must stay under 20,000 characters.** Summarize rather than dump —
for example, report the k most spatially variable nodes and the k most flow-sensitive ones by
name with numbers, and state plainly that full arrays for any node are available on request
through the harness. State the k you chose and why in the Log.

## Acceptance criteria

1. **Answer preservation, the scientific gate.** Extend the TASK-005 pattern in a new file
   `tests/test_summary_grn.py`: at least **12 GRN questions**, each answered twice — once from
   the full arrays, once from the briefing section alone — all agreeing. Plus at least one
   negative control proving the suite is not vacuous. Paste the run.
2. Briefing length under 20,000 characters. Paste `wc -c data/briefing.md` before and after.
3. `--dry-run` still works and reports the new size. Paste it.
4. Every new metric has units and provenance, and appears in the generator's allowed-value list
   automatically (it is derived from the registry — confirm, do not assume).
5. Full suite green, count pasted. Ruff clean on touched files.

## Log

_(append below)_
