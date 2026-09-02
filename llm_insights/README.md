# llm_insights

LLM-driven hypothesis generation and verification over the AV cushion fluid-solid-growth (FSG)
+ gene-regulatory-network (GRN) model in `../one_way_fsg_model/`.

**To run anything, see [`INSTRUCTIONS.md`](INSTRUCTIONS.md).** No Anthropic API key is needed:
the live backend reaches the model through the Claude Code CLI, which authenticates with a
Claude subscription.

## What this is

Dan's FSG model produces raw fields — velocity, wall shear stress, von Mises stress, growth
factor, and GRN node activities across 11,615 solid nodes, 15 timesteps, 3 flow cases, totalling
roughly 27.4 million field values. The raw output is faithful but not *legible*: it is hard for
a researcher to look at it and come away with a claim they can defend.

This package builds an agent that proposes qualitative claims about that model and then
**falsifies them by running the model**, rather than by asking another language model whether
they sound right. The design rule is that a hypothesis is admissible only if it compiles into an
executable test with a decision rule stated *before* the test runs. The agent never grades its
own work.

The FSG model is treated as ground truth. We are not validating biology; we are extracting
defensible statements about a simulator.

## How it works

```
FSG output  ->  io  ->  summary  ->  briefing (~3,200 tokens)  ->  generator (LLM)
                                                                        |
                                                                   hypotheses
                                                                        |
                                          harness (executable code, no LLM)  ->  outcomes
                                                                        |
                                                              cards  ->  report.html
```

Three ideas do the work:

**Compression that is tested.** The summary layer collapses 27.4 M values onto normalized arc
length — the cushion is a shallow cap and ~94% of the mesh never touches the fluid, so the
honest low-dimensional view is one-dimensional. That compression is not assumed correct: 22
questions are answered twice, once from the full arrays and once from the summary alone, and the
answers must agree. `tests/test_summary.py` includes a negative control that corrupts the summary
and asserts the same checks then fail.

**Pre-registration.** A claim carries its decision rule before the outcome is known. Prediction
first, result second.

**Executable verification.** The comparison happens in code, through six primitives. If a claim
cannot be compiled into one, it is not admissible yet — it goes back to be sharpened. No language
model, including a critic, decides any outcome.

## Layout

```
llm_insights/
├── INSTRUCTIONS.md        <- how to run it. Start here.
├── NOTES_data_formats.md  <- what is on disk, what is unreadable, and the traps.
├── CLAUDE.md              <- agent-facing rules.
├── STYLE_GUIDE.md         <- code conventions.
├── TASKS.md               <- the work queue and inter-agent comms channel.
├── data/                  <- briefing, transcript, cards, and rendered reports.
└── src/llm_insights/
    ├── io/                <- dataset access. The only module that knows the file layout.
    ├── summary/           <- metrics, arc-length profiles, the briefing.
    ├── harness/           <- the falsifier: primitives, spec, runner.
    ├── agent/             <- generator backends, the investigate loop, the CLI.
    │                         generator.py: prompts, parsing, API-key backends.
    │                         subscription.py: the keyless backends (Claude Code CLI, paste).
    └── cards/             <- card model and the markdown/HTML report.
```

## Status

The one-question demo works end to end. On *"What differs between the healthy and overflow cases,
and why?"* it returns 7 cards: 5 survived, 1 falsified-then-narrowed, 1 refused as unanswerable.

The falsified card is the point. The agent predicted that the overflow case would show the most
spatially heterogeneous EndMT response, because it has the highest and most widely spread
mechanical stress. The simulator said no — variability is roughly twice as high in the healthy
case. The narrowed claim that replaced it identifies why: 46% of the overflow domain is clipped
at the GRN's 100 Pa normalization ceiling, and clipped nodes all return the same pinned activity,
which compresses spatial variance. That claim survived its own stricter test.

Any question can be asked, live, with no API key — see `INSTRUCTIONS.md` §2. A recorded replay
of the demo run is always available offline as a fallback (§3).

**228 tests.** See `INSTRUCTIONS.md` §6.

## Relationship to `one_way_fsg_model/`

Read-only. See `CLAUDE.md` — no code in that folder may be modified. Findings about it go to
**Open questions** in `TASKS.md` and then to Dan; they are never patched here.
