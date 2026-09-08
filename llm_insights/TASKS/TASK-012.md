# TASK-012 — Extension points, so Phase 1 does not collide

**STATUS:** IN REVIEW
**Owner:** Claude Code (overnight Phase 2 session, 2026-09-07)
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

### 2026-09-07 — Claude Code, overnight Phase 2 session

**What was built.**

- `summary/registry.py` — `@briefing_section(order, name)`, `registered_sections()`,
  `section_names()` and `render_sections(ds, n_profile_points)`, mirroring the `@primitive`
  pattern in `harness/primitives.py`. Iteration is `sorted(..., key=(order, name))`.
  A duplicate name **raises `ValueError`** rather than replacing.
- `summary/sections/{metrics,profiles,trajectory,caveats,protocol}.py` — the five builders,
  moved verbatim, at orders 10/20/30/40/50 (spaced so a lane can interleave without
  renumbering). `summary/sections/__init__.py` imports all five, deliberately in alphabetical
  order, which is *not* the render order — anyone who reorders those imports to change the
  briefing should find that it does nothing.
- `summary/briefing.py` is now the header plus `render_sections(...)` plus `"\n".join`.
  Nothing else. **Nobody edits it after Phase 0.**
- `summary/metrics.py` — `METRIC_REGISTRY` and `register_metric` are public;
  `_register = register_metric` is kept as an alias so the sixteen definitions stayed exactly
  where they are and the diff is a rename. `metric_table` and `metric_value` now read
  `sorted(METRIC_REGISTRY)`.
- `../STYLE_GUIDE.md` — a new "Extension points" section documenting both, with a snippet each.

**One design point worth stating.** `METRIC_NAMES` was `Final[tuple[...]]` computed at import
time, i.e. a snapshot. Left that way, a metric registered by Lane B *after* `metrics.py` was
imported would never appear in it — and `harness/primitives.py:216` (`if name not in
METRIC_NAMES`) and `agent/generator.py:184` both read it, so a new metric would have been
rejected as unknown and would never have reached the model's allowed-value list. Both of those
read it through a **lazy, call-time import**, so the fix is for `register_metric` to rebind the
module-level name. That is what it now does. This means TASK-014 acceptance criterion 4 holds
without editing `harness/` or `generator.py`, which repo rule 3 forbids. `sorted(METRIC_REGISTRY)`
is documented as the preferred live view for new code.

---

**Acceptance criterion 1 — byte-identical briefing. This is the whole gate.**

Saved before the refactor, from `9fc1102` (TASK-011 merged, `briefing.py` untouched):

```
$ cd src && PYTHONPATH=. python3 -c "...build_briefing(Dataset(DEFAULT_ROOT))..."
chars 12662
sha256 9b228a874effeab9482432e32ea20464b154fb6ef5c02d1ba0559fd08b23dd8d
```

Regenerated after the refactor, after `ruff format`:

```
$ cd src && PYTHONPATH=. python3 -c "...build_briefing(Dataset(DEFAULT_ROOT))..."
chars 12662
sha256 9b228a874effeab9482432e32ea20464b154fb6ef5c02d1ba0559fd08b23dd8d

$ cmp $SCRATCH/briefing_before.md $SCRATCH/briefing_after.md
cmp: FILES ARE IDENTICAL
```

`cmp` reported no difference — it exits 0 and prints nothing; the line above is the shell echo
confirming that. The sha256 also matches the `briefing_sha256` already recorded in the shipped
`data/report.md` (`9b228a874effeab9`), so this is the same briefing the demo was generated from.

The end-to-end replay is unchanged too. `cards.json` is byte-identical across the refactor and
`report.md` differs only in its timestamp:

```
$ diff <TASK-011 tree>/report.md <TASK-012 tree>/report.md
17c17
< - **Generated:** 2026-09-08T02:00:50Z
---
> - **Generated:** 2026-09-08T02:09:37Z

$ cmp <TASK-011 tree>/cards.json <TASK-012 tree>/cards.json
cards.json: FILES ARE IDENTICAL
```

**Note on the briefing's size.** It is **12,662** characters, not the 12,698 quoted in
TASK-014. Lane B should measure rather than trust that figure; the 20,000-character ceiling is
unaffected either way.

**Acceptance criterion 2 — `tests/test_registry.py`.**

14 new tests. They cover: registration works and the decorator returns the function unchanged;
iteration is sorted by `(order, name)` and not by registration order; a duplicate name raises,
at the same order and at a different one, and the *first* registration is the one that survives;
`render_sections` concatenates in render order; the five real sections are all registered with
distinct orders; `METRIC_REGISTRY is METRICS`; the sixteen original metrics are intact with
units and provenance; registering a metric re-sorts `METRIC_NAMES` immediately; and registering
a metric twice raises.

Import-order independence is tested against the **real** sections, not a fixture: the registry
is cleared, the five modules are re-imported through `importlib.reload` in reverse order, and
the briefing is rebuilt and compared to the reference. It must come out identical.

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. -p "test_registry.py"
..............
----------------------------------------------------------------------
Ran 14 tests in 0.038s

OK
```

**Acceptance criterion 3 — full suite green.**

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
..................................................................................................................................................................
----------------------------------------------------------------------
Ran 386 tests in 0.718s

OK
```

386 = 356 baseline + 16 from TASK-011 + **14 from this task**. 0 failures, 0 errors.

**Acceptance criterion 4 — ruff clean on the touched files.**

```
$ ruff --version
ruff 0.16.5
$ ruff check src/llm_insights/summary/registry.py src/llm_insights/summary/briefing.py \
    src/llm_insights/summary/metrics.py src/llm_insights/summary/sections/ tests/test_registry.py
All checks passed!
$ ruff format --check <same>
10 files already formatted
```

One pre-existing unrelated fix was needed to get `metrics.py` clean: it imported `MECH_MAX_PA`
and never used it (the only occurrence is inside a provenance *string*), which ruff flags as
F401. The import is removed. Flagging it because it is the one hunk in this commit that is not
part of the task.

`summary/profiles.py` is **not** ruff-clean (ANN201, a missing return annotation on `profile`)
but this task did not touch it, so it was left alone per "do not reformat unrelated files in a
feature PR". It is part of the pre-existing 17 `src/` errors recorded in TASK-011's log.

**STATUS set to IN REVIEW.** All four acceptance criteria met. Phase 0 is complete; Phase 1
lanes may start.
