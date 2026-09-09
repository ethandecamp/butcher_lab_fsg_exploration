# TASK-014 — Put the gene network into the briefing

**STATUS:** IN REVIEW
**Owner:** Lane B (Claude Opus 5); claimed and evidenced 2026-09-08
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

### 2026-09-08 — claimed late; the code existed before the task did

`summary/sections/grn.py` and `tests/test_summary_grn.py` were written overnight by the Lane B
agent that was killed at 08:20, which never claimed this task — STATUS was still TODO with the work
already on disk. Claimed now, verified against all five criteria, gaps filled.

What ships:

- `src/llm_insights/summary/sections/grn.py` — registered through TASK-012's `@briefing_section`
  at **order 45**, so it renders between `caveats` (40) and `protocol` (50) as section "4b".
  `briefing.py` was **not** edited, which is the whole point of the registry.
- Eight new metrics registered through `register_metric`.
- `tests/test_summary_grn.py` — 12 answer-preservation questions plus structural and metric tests.

#### The budget decision: k = 5, and why

`TOP_K = 5` in `summary/sections/grn.py:53`. The section states the pathway grouping and the input
wiring as *facts* rather than as numbers, then names only the **5 most spatially variable** and the
**5 most flow-sensitive** nodes with numbers, and says plainly that full arrays for any node are one
harness call away.

The alternative was dumping 21 nodes x 3 cases x 2 statistics = 126 new numbers, which the task
spec itself warns would roughly double the prompt and bury the existing content. k=5 is the
smallest k that still covers both Notch and TGFb/BMP arms of the network in the spatial ranking, so
a reader is not shown a single pathway and left to assume it is the whole story — the section says
that in as many words ("these two rankings are a way in, not the whole story").

#### Criterion 1 — answer preservation. **The scientific gate. PASSES — but it was silently broken.**

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. -p "test_summary_grn.py"
............
----------------------------------------------------------------------
Ran 12 tests in 0.744s

OK
```

12 questions, each answered twice — once from the full arrays in `data/grn_full/`, once from the
briefing section text alone — all agreeing. Non-vacuity is covered by
`test_the_suite_is_not_vacuous`, which damages the section text four ways and asserts the checks
then fail.

**This gate was not actually running when this session started.** Four of the twelve questions —
`constant_nodes`, `constant_node_count`, `rbpj_value`, `is_smad23_constant` — were **erroring**, not
passing and not failing: the helper `_stated_constant_nodes` read a fixed line offset after the
`**Pinned by construction` header, that header's sentence wraps onto a continuation line, and the
parser crashed unpacking the wrong line. An erroring question is a question that never got asked,
so a third of this gate was reporting nothing at all.

The parser now locates the line that actually carries `name = value` assignments. That makes the
comparison run; it does not make it lenient. All four then **agree**. Had any disagreed, this task
would be BLOCKED rather than IN REVIEW. Full detail is in TASK-013's Log under the seven-failing-
tests entry.

#### Criterion 2 — briefing under 20,000 characters. **PASSES, with 462 characters to spare.**

```
$ git show c5cb465:llm_insights/data/briefing.md | wc -c
   12698                       # BEFORE, bytes
$ git show c5cb465:llm_insights/data/briefing.md | python3 -c "import sys; print(len(sys.stdin.read()))"
12662                          # BEFORE, characters

$ wc -c data/briefing.md
   19590 data/briefing.md      # AFTER, bytes
                               # AFTER, characters: 19538
```

**12,662 -> 19,538 characters** (12,698 -> 19,590 bytes). The ceiling is 20,000 characters, so it
passes on characters and, as it happens, on bytes too. `wc -c` counts bytes and the briefing
contains multi-byte characters (em dashes, the `->` arrows), which is the whole reason the two
figures in circulation — 12,662 and 12,698 — were never in conflict.

**Flagging the headroom, because it is thin: 462 characters, 2.3% of the ceiling.** The GRN section
consumed 6,876 of the 7,338 characters that were available. The next section anyone adds will
breach 20,000 unless something is cut, and TASK-018 is the next task expected to want prompt space.
This is not a criterion failure — it is a warning that the budget is now effectively spent.

`data/briefing.md` was regenerated as part of this, so the committed artifact matches what the code
now produces. That is deliberate: TASK-020 criterion 6 regenerates the briefing and `cmp`s it
against the committed copy, and leaving a stale 12,698-byte artifact next to code that emits 19,590
would fail that check for a bookkeeping reason.

#### Criterion 3 — `--dry-run` still works and reports the new size

```
$ PYTHONPATH=src python3 -m llm_insights.agent.run --dry-run
...
dry run: 6250 + 19808 = 26058 characters, roughly 6514 tokens for the first call. No model was called.
```

Works, and reports the new size: **26,058 characters total, ~6,514 tokens** for the first call
(6,250 system + 19,808 user; the user turn is the briefing plus the question and instruction
wrapper). Worth noting for Lane C: this is comfortably inside the 32,768-token context TASK-016
confirmed, but it is **~1.6x** the 4,096-token Ollama default that would have truncated it
silently.

#### Criterion 4 — new metrics carry units and provenance, and reach the generator's vocabulary. **Confirmed, not assumed.**

```
registry total: 24   listed in system prompt: 24
new GRN metrics: 8
every new metric present in the prompt vocabulary: True
  beta_catenin_mean_combined     units='dimensionless' source=data/grn_full/<case>/activities.csv column beta_catenin_
  bmp2456_mean_combined          units='dimensionless' source=data/grn_full/<case>/activities.csv column BMP_2456_comb
  hey_mean_combined              units='dimensionless' source=data/grn_full/<case>/activities.csv column HEY_combined,
  nicd_mean_combined             units='dimensionless' source=data/grn_full/<case>/activities.csv column NICD_combined
  smad23_mean_combined           units='dimensionless' source=data/grn_full/<case>/activities.csv column SMAD23_combin
  smad23_sd_combined             units='dimensionless' source=data/grn_full/<case>/activities.csv column SMAD23_combin
  smad67_mean_combined           units='dimensionless' source=data/grn_full/<case>/activities.csv column SMAD67_combin
  yap_taz_mean_combined          units='dimensionless' source=data/grn_full/<case>/activities.csv column YAP_TAZ_combi
```

Read out of `build_system_prompt()`'s own "metric names currently registered" line, so this is the
list the model is actually handed, not the registry re-printed. 16 -> 24 metrics, all eight new ones
present, all eight carrying units and a provenance string naming the column they come from.

**One real caveat found while confirming this, and it is an ordering trap.** The eight metrics are
registered as an import side effect of `summary/sections/grn.py`. Nothing in
`agent/generator.py` imports `summary.sections`, so calling `build_system_prompt()` *before* a
briefing has been built yields the old 16:

```
# generator imported, sections never imported:
registry total: 16   listed in system prompt: 16   new GRN metrics: 0
```

Both real CLI paths are safe — `run.main` and `run._dry_run` each call `_load_briefing` before any
prompt is built, and `Study.briefing` imports the sections — which is why criterion 4 genuinely
holds in use. But a future caller that builds a system prompt without touching a briefing will
silently offer the model a smaller vocabulary than the registry holds. Recorded in **Open
questions**; not patched, because the fix touches files this lane does not own.

`smad23_mean_combined` is worth calling out specifically: SMAD23 is the node whose absence made
Lane A's single genuinely COMPUTED claim impossible to run. It exists now.

#### Criterion 5 — full suite green, ruff clean on touched files

```
$ cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
Ran 419 tests in 5.639s

OK
```

419 tests, 0 failures, 0 errors, 0 skips (pytest: `419 passed, 84 subtests passed`).

```
$ ruff --version
ruff 0.16.5
$ ruff check src/llm_insights/grn/ src/llm_insights/summary/sections/grn.py \
    tests/test_grn_runner.py tests/test_summary_grn.py tests/test_registry.py
All checks passed!
$ ruff format --check <same five paths>
7 files already formatted
```

Reaching that required fixing 50 findings, described in TASK-013's Log. Only the files these two
tasks touched were linted; the tree is not ruff-clean at baseline.

**STATUS set to IN REVIEW.** All five acceptance criteria are met and evidenced above. The two
things a reviewer should look at rather than skim are the 462-character briefing headroom and the
cross-lane edit to `tests/test_registry.py`.
