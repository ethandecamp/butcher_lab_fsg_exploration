# TASK-012 — Extension points, so Phase 1 does not collide

**STATUS:** TODO
**Phase:** 0 (serial, after TASK-011). Branch `feat/phase2-prologue`.
**Depends on:** TASK-011. **Blocks:** Lane A, Lane B.

## Context

Two Phase 1 lanes need to add content to the briefing. If both edit `summary/briefing.py`, the
merge is painful and the failure mode is silent (a dropped section). This task converts the two
shared files into registries so lanes add **new files** instead of editing shared ones.

This is a **pure refactor**. It must not change a single number or a single byte of briefing
output.

## What to build

1. `src/llm_insights/summary/registry.py` — a `@briefing_section(order: int, name: str)`
   decorator writing into a module-level `_SECTIONS` dict, mirroring the `@primitive` pattern in
   `harness/primitives.py`. Provide `registered_sections()` returning them sorted by
   `(order, name)` — **sorted, never import order.**
2. Move the five existing section builders out of `briefing.py` into
   `summary/sections/{metrics,profiles,trajectory,caveats,protocol}.py`, each decorated with its
   current order. `summary/sections/__init__.py` imports all five so registration happens on
   package import.
3. `briefing.py` becomes a composer: build the header, iterate `registered_sections()`, join.
   Nothing else. **After this task, no other task may edit `briefing.py`.**
4. `summary/metrics.py` already has `_REGISTRY` and `_register`. Expose them as public
   `METRIC_REGISTRY` and `register_metric` so other modules can add metrics without editing
   `metrics.py`, and make every consumer iterate `sorted(METRIC_REGISTRY)`. Keep the existing
   16 metric definitions exactly where they are.
5. Document both extension points in `../STYLE_GUIDE.md` (short — a paragraph each with a usage
   snippet).

## Do not

- Change section content, ordering, wording, number formatting, or rounding.
- Touch `harness/`, `io/`, `cards/`.
- Add any new section or metric. That is Lane A's and Lane B's job.

## Acceptance criteria

1. **Byte-identical briefing.** Save the pre-refactor briefing, refactor, regenerate, and show
   `cmp` reporting no difference. Paste the commands and their real output. This is the whole
   gate for this task — a refactor that changes one character has failed.
2. `tests/test_registry.py`: registration works; iteration order is sorted and is **independent
   of import order** (import the section modules in reverse and assert identical output);
   duplicate `(order, name)` raises rather than silently overwriting.
3. Full suite green, count pasted, still >= 356 + TASK-011's additions.
4. Ruff clean on touched files.

## Log

_(append below)_
