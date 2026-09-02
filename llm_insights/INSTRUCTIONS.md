# INSTRUCTIONS.md — how to run the hypothesis tester

Everything you need to run the demo live, replay it, run the tests, and understand what each
backend costs.

**No Anthropic API key is required.** The live backend goes through the Claude Code CLI, which
authenticates with your Claude subscription. If you do have an API key, `--generator anthropic`
still works exactly as before.

---

## 1. One-time setup

```bash
cd ~/Documents/Research/ButcherLab/butcher_lab_fsg_exploration/llm_insights
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Check it worked:

```bash
python3 -c "import numpy, scipy, pandas; print('deps ok')"
python3 -c "import llm_insights; print('package ok')"
claude --version          # the live backend needs this on PATH
```

If the second line says `No module named 'llm_insights'`, the editable install is not active in
this shell. Re-run `pip install -e ".[dev]"`. Every command below also sets `PYTHONPATH=src`,
which imports the package whether or not it is installed — one less thing that can fail in front
of an audience.

**`PYTHONPATH=src` does not substitute for activating the environment.** It makes *this package*
importable; it does nothing about numpy, pandas or anything else the harness needs. On
2026-09-02 a live run outside the virtualenv reached the correlation primitive and produced
`COULD NOT RUN: No module named 'scipy'` — a working project, a broken interpreter choice. Every
run command below therefore activates the venv first, and the tool now refuses to start if a
required module is missing.

If `claude` is missing: `npm install -g @anthropic-ai/claude-code`, then run `claude` once and
log in. Nothing in this package ever reads or stores your credentials.

---

## 2. Run it live — the demo (no API key)

This is what to show Dan. **Any question can be asked.** The model writes the hypotheses; the
verdicts are computed in code.

```bash
cd llm_insights
source .venv/bin/activate
PYTHONPATH=src python3 -m llm_insights.agent.run \
  --question "What differs between the healthy and overflow cases, and why?" \
  --out data/live
open data/live/report.html
```

The first line matters. Without it the run may use a different interpreter, and a dependency
that interpreter lacks will surface as a failed card rather than as a setup error. The tool
prints an `Environment check:` warning when it notices, and refuses to start outright if
something it needs is missing.

`--generator claude-cli` is the default, so it does not need to be typed. Roughly 6–10 model
calls, about 30–90 seconds.

To use a different model — Haiku is roughly a tenth the cost per call and is enough for a
smoke test, though it has not produced the more interesting behaviours (see §10):

```bash
PYTHONPATH=src python3 -m llm_insights.agent.run --model haiku --question "..." --out data/live
```

### What it costs

The default model is **Sonnet**. On a subscription these calls draw down your rolling usage
allowance rather than a bill; the CLI still reports a dollar-equivalent, and this tool prints the
total when the run finishes:

```
Usage:       8 call(s), $0.1400, 41203 in / 15877 out tokens
```

A measured single call on **Haiku** cost **$0.0161**, most of it the CLI's own ~7.5k-token
system prompt, which is charged whatever the model. A full Haiku run is therefore around
**$0.15–0.25 equivalent**. Sonnet has not been measured here; expect a few times that, so
roughly **$0.50–1.00 equivalent** for a full run — still a rounding error against a Pro
allowance. You can run the demo repeatedly without noticing.

To see the exact prompt and its size before spending anything:

```bash
PYTHONPATH=src python3 -m llm_insights.agent.run --dry-run --question "..."
```

### The guardrails, and why each one is there

| Guardrail | Effect |
|---|---|
| `--model sonnet` (default) | Opus costs several times more per turn than Sonnet, Sonnet more than Haiku. Sonnet is the default because the behaviours worth showing — refusing an unanswerable question, correctly blaming a normalization artifact when narrowing — have only been observed from the stronger tiers. `--model haiku` is the cheap smoke test; it falls back to Haiku automatically if your account cannot reach Sonnet. |
| `--max-calls 12` (default) | Hard ceiling on model calls. A full run makes 6–10, so this catches a runaway without firing normally. |
| `--max-budget-usd 0.50` (default) | Cumulative ceiling. The run stops rather than continuing past it, and the same value is passed to the CLI as a per-call cap. |
| Tools disabled, `--max-turns 1` | Each call is one round trip, not an agent session. A tool call would cost a second full-context request. |
| Empty scratch directory | The CLI runs in an empty temp dir, so `CLAUDE.md` auto-discovery finds nothing. This is a **correctness** property as much as a cost one: a claim must be written from the briefing alone. |
| `ANTHROPIC_API_KEY` stripped from the child environment | If a key happened to be exported, the call would silently route through metered API billing. It is removed so a keyless run stays keyless. |

Raise a ceiling only deliberately:

```bash
PYTHONPATH=src python3 -m llm_insights.agent.run \
  --max-calls 20 --max-budget-usd 1.00 --question "..."
```

> **Do not add `--bare`.** It looks like the right way to suppress `CLAUDE.md` discovery, but it
> also restricts authentication to `ANTHROPIC_API_KEY` — OAuth and keychain are never read — so
> with no key every call would fail to authenticate. The empty scratch directory achieves the
> same thing without touching auth. There is a test pinning this.

---

## 3. Replay the recorded demo

Deterministic, free, and offline. Use it if the CLI is unavailable or your allowance is spent.

```bash
PYTHONPATH=src python3 -m llm_insights.agent.run \
  --generator transcript \
  --transcript data/demo_transcript.json \
  --question "What differs between the healthy and overflow cases, and why?" \
  --out data
```

Expected output:

```
  proposed        7
  admissible      7
  rejected        0
  survived        5
  falsified       0
  narrowed        1
  could not run   1
```

A replay re-runs every verification in code, so the numbers on the cards are computed live even
though the claims are replayed. The provenance block always says it was a replay; it cannot be
passed off as a live call.

**A replay ignores the question it is given.** If you ask a transcript a question it did not
record, the tool prints a loud warning before it runs — because otherwise the report would carry
your new question in the header and the old run's answers underneath it. That is the one failure
mode that would actually mislead an audience, so it is impossible to hit silently.

---

## 4. Manual fallback (`--generator paste`)

Last resort: no CLI, no key, but a browser. The tool copies each prompt to your clipboard, you
paste it into a Claude chat, copy the reply, and press Enter. Two exchanges for a typical run.

```bash
PYTHONPATH=src python3 -m llm_insights.agent.run --generator paste --question "..." --out data/live
```

Prompts and replies are also written to `data/paste/` so nothing depends on the clipboard
working. This backend is never selected automatically — you have to ask for it by name.

It has one genuine advantage worth knowing: the prompt that goes in is *exactly* the one this
package built, with no CLI agent-wrapper around it. If you ever want the run to be a clean
instrument rather than a demo, this is the cleanest one available without an API key.

---

## 5. Run it live with an API key (unchanged)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
PYTHONPATH=src python3 -m llm_insights.agent.run --generator anthropic --model claude-sonnet-4-5 \
  --question "..." --out data/live
unset ANTHROPIC_API_KEY
```

The key is read from the environment on every request and is never stored on the object, logged,
or written into the report — there are tests asserting all four.

---

## 6. Run the tests

```bash
cd llm_insights
python3 -m pytest tests -q
```

Or without pytest installed:

```bash
cd llm_insights/src
python3 -m unittest discover -s ../tests -t .. -v
```

**228 tests.** The `-t .` form fails because `tests/` is a sibling of `src/`, not a child.

Neither test command needs `PYTHONPATH`: pytest picks up `pythonpath = ["src"]` from
`pyproject.toml`, and the unittest form gets it from running inside `src/`.

The suite that matters most is `tests/test_summary.py`. It asks 22 questions of the data, answers
each one twice — once from the full arrays and once from the summary alone — and asserts the two
agree. It also includes a negative control that corrupts the summary and asserts the same checks
then fail, so the suite is provably not vacuous.

`tests/test_subscription.py` covers the two keyless backends. No test there runs the real CLI or
touches the network: the subprocess boundary is crossed only through an injected transport or a
stub executable written into a temp directory, and the response fixtures mirror a real
`claude -p --output-format json` envelope captured from the CLI.

---

## 7. What the pieces are

| Module | What it does |
|---|---|
| `io/dataset.py` | Reads the baked FSG output. The only module that knows where files live. |
| `summary/metrics.py` | 16 named scalar metrics, each with units and provenance. |
| `summary/profiles.py` | Fields collapsed onto normalized arc length, 25 points. |
| `summary/briefing.py` | Assembles the ~3,200-token briefing the model reads. |
| `harness/` | The falsifier: six executable primitives, strict parsing, a runner that never raises. |
| `agent/generator.py` | The prompts, the response parser, and the API-key backends. |
| `agent/subscription.py` | The keyless backends: Claude Code CLI and the paste fallback. |
| `agent/loop.py`, `agent/run.py` | The investigate loop and the CLI. |
| `cards/` | Card model and the markdown + HTML report renderer. |

Data flow:

```
FSG output  ->  io  ->  summary  ->  briefing  ->  generator (LLM)
                                                        |
                                                   hypotheses
                                                        |
                                     harness (executable, no LLM)  ->  outcomes
                                                        |
                                              cards  ->  report.html
```

Nothing downstream of the generator can tell which backend wrote a claim. That is what makes
swapping them safe, and it is why adding a keyless backend changed no verification code.

---

## 8. Every flag

```
--generator {claude-cli,anthropic,transcript,paste,echo}
                                          backend (default: claude-cli)
--question TEXT                           the question to investigate
--n INT                                   how many hypotheses to ask for (default 5)
--model TEXT                              model id (claude-cli and anthropic)
--out DIR                                 where cards and report go (default: data/)
--dry-run                                 print the prompt and its size; call nothing

--max-calls INT                           call ceiling         (claude-cli, default 12)
--max-budget-usd FLOAT                    cost ceiling in USD  (claude-cli, default 0.50)
--claude-binary PATH                      claude executable, if not on PATH
--paste-dir PATH                          where --generator paste writes its files

--skip-preflight                          skip the startup environment check
--synthesize                              after verification, ask for a written summary
                                          of the run and render it, fenced, at the top
                                          of the report (default: off)
--min-rel-diff FLOAT                      floor on the relative difference a
                                          compare_metric claim must clear (default 0.0)

--transcript PATH                         recorded run to replay
--root PATH                               dataset root (defaults to ../one_way_fsg_model)
--briefing-file PATH                      use a saved briefing instead of rebuilding it
--no-narrow                               don't ask for a narrowed claim after a failure
--log-level {DEBUG,INFO,WARNING,ERROR}
```

Regenerate the briefing on its own, to read what the model actually sees:

```bash
PYTHONPATH=src python3 -c "
from llm_insights.io.dataset import Dataset
from llm_insights.summary.briefing import build_briefing
print(build_briefing(Dataset('../one_way_fsg_model')))
" > data/briefing.md
```

---

## 8b. Two flags worth knowing

### `--synthesize` — a written summary of the run

```bash
PYTHONPATH=src python3 -m llm_insights.agent.run --question "..." --synthesize --out data/live
```

Off by default. When on, one extra model call is made **after every claim has already been
verified**, asking for three to six sentences of plain English about what the run found. It is
rendered at the top of the report, fenced, and labelled as narration rather than as a result.

It cannot change a verdict. The harness is never re-entered, and the summary is written from the
finished cards.

Every number in the summary is checked against the cards before it renders. A numeral is allowed
only if it matches a value in some card's `observed` (rounding is fine — `0.128` is accepted for
an observed `0.12808567238007387`), appears verbatim in a claim or decision rule, or is a small
integer no bigger than the number of cards. **Anything else and the summary is not rendered at
all.** One retry is spent naming the offending numerals back to the model; if that also fails the
summary is suppressed permanently and the provenance block says so:

```
- **Narrative summary:** suppressed: unsupported numerals ['0.003', '12']
```

A suppressed summary is always visible in the report. It is never dropped silently.

Why the check exists: this is the one piece of model-written prose in a report whose whole claim
is that no language model judged any result. Fencing it in the layout is not enough on its own,
so the containment check is what actually keeps the claim true.

### `--min-rel-diff` — an operator floor on the noise guard

```bash
PYTHONPATH=src python3 -m llm_insights.agent.run --question "..." --min-rel-diff 0.10 --out data/live
```

`compare_metric` takes a `min_rel_diff`: the relative difference two cases must show before an
ordering counts. It defaults to `0.0`, so a proposal that omits it passes on any difference at
all, including floating-point noise. This flag sets a floor beneath whatever the model asked for
— the effective margin is `max(model_supplied, floor)`, so a model may be **stricter** than you
but never looser. The floor is recorded in the provenance block and shown on every card it
raised.

`0.10` is a reasonable starting value: above float noise, and roughly at the 12% spread that the
GRN grid choice alone produces (`mech_clipped_fraction` is 0.337 on the full mesh and 0.296 on
the 80% grid). Two limits to know:

- It applies only to `compare_metric`. **`metric_ordering` has no margin parameter at all**, so a
  monotonicity claim still passes on an arbitrarily small step. Adding one is open work.
- A single relative number does not fit every metric. For arc positions relative difference is the
  wrong measure — 10% of `s=0.51` is 0.05, about the 25-point grid spacing — and for fractions
  already in [0,1] an absolute floor near 0.01 is the meaningful one. Per-metric floors are open
  work.

---

## 9. If the live backend fails

The error message is the CLI's own, which is where the real cause is reported. Common ones:

| Message contains | What to do |
|---|---|
| `No module named 'llm_insights'` | The editable install is not active in this shell. Every command here already sets `PYTHONPATH=src`, so this should not appear; if it does, you dropped that prefix. `pip install -e ".[dev]"` fixes it permanently. |
| `is not on PATH` | `npm install -g @anthropic-ai/claude-code`, then `claude` to log in. |
| `/login`, `Invalid API key` | Run `claude` once interactively and log in. |
| `unknown model`, `not available on your plan` | The tool retries once on Haiku automatically. If it still fails, pass `--model haiku` explicitly. |
| `unknown option` | The tool retries once with a minimal command line, keeping fewer guardrails. Nothing to do. |
| `Environment check: ERROR ... cannot import ...` | The interpreter is missing something the harness needs. `source .venv/bin/activate && pip install -e ".[dev]"`. No model calls were made, so nothing was spent. |
| `Environment check: WARNING ... running outside the project virtualenv` | The run will proceed, but `source .venv/bin/activate` first and re-run. This is the 2026-09-02 failure. |
| A card reading `ENVIRONMENT FAULT` | That test could not run because of this machine's setup. It is not a result: fix the setup and re-run before reading anything into it. |
| limit or allowance exhausted | Fall back to §3, the recorded replay. |
| `stopping: this run has reached its $0.50 ceiling` | Working as intended. Raise it with `--max-budget-usd` if you meant to. |

Whatever happens, §3 always works offline, so a demo is never dead.

---

## 10. Known limits, stated plainly

- **The survival rate is inflated, and you should say so before anyone works it out.** The
  briefing prints every registered metric for all three cases, and `compare_metric` and
  `metric_ordering` operate only on registered metrics. So a claim using either of those is
  reading an answer that is already in its own prompt — four of the six primitives are lookups
  rather than predictions. In the 2026-09-02 live run, all four lookup claims survived and the
  single claim needing something the briefing did not state (a WSS/EndMT correlation) was
  falsified. Quote the rate as a reading-comprehension score until a blind-briefing mode exists,
  or classify each card as lookup vs. extrapolation and report the two rates separately.
- **`--synthesize` has never been run against the real `claude` binary.** Every test covering it
  runs against fakes and stub executables, including deliberately misbehaving ones. One live run
  with the flag is the missing evidence; do it before showing the feature to anyone.
- **The CLI backend is not a clean instrument.** Claude Code wraps the prompt in its own agent
  instructions. This package replaces them with `--system-prompt` when the installed CLI
  supports that flag, and concatenates otherwise; either way the wrapper is an uncontrolled
  variable. Fine for a demo. For the roadmap's "run N questions, report the survival rate"
  measurement, prefer `--generator anthropic` or `--generator paste`, where the prompt going in
  is exactly the one this code built.
- **Per-step solid stress exists for the healthy case only.** `extracted_fields/` and
  `dynamic_inputs/` were only generated for `flow_U0p0360`. Full-mesh von Mises for all three
  cases is available at step 14 only, via `grn_inputs/`. Installing `h5py` and reading
  `solid_fields.h5` would unlock the full 15-step × 3-case stress evolution — the single
  highest-value next step.
- **Only 5 of 23 GRN nodes were written out.** Ask about `SMAD23` and the harness refuses rather
  than inventing an answer. That is deliberate; card H6 in the demo shows it.
- **Growth `g` is per-cell in the reference frame**, while stress is per-node in the deformed
  frame. They are not co-registered, so no claim currently correlates them directly.
- **`step_k/arc_data.npz` is the geometry at the end of step k-1.** Step 0 is the pristine
  undeformed cap and is not a simulation result.
- **The harness needs no scipy.** Pearson and Spearman are computed in numpy
  (`harness/primitives.py`), verified against hand-computable golden values and, where scipy is
  installed, against scipy itself in `tests/test_correlation_math.py`. The oracle layer skips
  where scipy is absent, so run the suite in the venv at least once after touching that math.
- **Logging uses stdlib `logging`, not loguru**, and tests are `unittest.TestCase` rather than
  pytest-native. Both were forced by a sandbox with no package installs; both run fine under your
  venv now. See TASK-006.
