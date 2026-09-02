# INSTRUCTIONS.md — how to run the hypothesis tester

Everything you need to run the demo, run the tests, and run it live against the API.

---

## 1. One-time setup

```bash
cd ~/Documents/Research/ButcherLab/butcher_lab_fsg_exploration/llm_insights
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Check it worked:

```bash
python3 -c "import numpy, scipy, pandas; print('ok')"
```

`h5py` is in the dependency list but **nothing in this package needs it yet** — every field
is read from the `.npz`, `.npy`, and `.csv` artifacts. See §7.

---

## 2. Run the demo (no API key needed)

This is what to show Dan. It replays the recorded generations and re-runs every verification
in code, so the numbers on the cards are computed live even though the claims are replayed.

```bash
cd llm_insights
python3 -m llm_insights.agent.run \
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

Then open the report:

```bash
open data/report.html
```

`data/report.md` is the same content as markdown if you'd rather paste it into slides.

---

## 3. Run it live against the Anthropic API

Do this in the terminal you're presenting from, right before the demo. **Do not put the key in
`.zshrc`, and do not commit it.**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python3 -m llm_insights.agent.run \
  --generator anthropic \
  --model claude-sonnet-4-5 \
  --question "What differs between the healthy and overflow cases, and why?" \
  --n 5 \
  --out data/live
```

Notes:

- The key is read from the environment on every request and is **never stored on the object,
  logged, or written into the report** — there are tests asserting all four.
- Each run writes `data/live/transcript.json`. You can replay that exact run later with
  `--generator transcript --transcript data/live/transcript.json`, which is how you turn a good
  live run into a reproducible one.
- A replay always reports itself as a replay in the provenance block. It cannot be passed off as
  a live call.
- Cost is small: the briefing is ~3,200 tokens and a run makes roughly 6-10 calls.

Unset it when you're done:

```bash
unset ANTHROPIC_API_KEY
```

---

## 4. Run the tests

```bash
cd llm_insights
python3 -m pytest tests -q
```

Or without pytest installed:

```bash
cd llm_insights/src
python3 -m unittest discover -s ../tests -t .. -v
```

**177 tests.** The command above is the one that works — `-t .` fails because `tests/` is a
sibling of `src/`, not a child.

The suite that matters most is `tests/test_summary.py`. It asks 22 questions of the data, answers
each one twice — once from the full arrays and once from the summary alone — and asserts the two
agree. It also includes a negative control that corrupts the summary and asserts the same checks
then fail, so the suite is provably not vacuous.

---

## 5. What the pieces are

| Module | What it does |
|---|---|
| `io/dataset.py` | Reads the baked FSG output. The only module that knows where files live. |
| `summary/metrics.py` | 16 named scalar metrics, each with units and provenance. |
| `summary/profiles.py` | Fields collapsed onto normalized arc length, 25 points. |
| `summary/briefing.py` | Assembles the ~3,200-token briefing the model reads. |
| `harness/` | The falsifier: six executable primitives, strict parsing, a runner that never raises. |
| `agent/` | Generator backends (anthropic / transcript / echo), the investigate loop, the CLI. |
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

---

## 6. Useful flags

```
--generator {anthropic,transcript,echo}   backend (default: transcript)
--transcript PATH                         recorded run to replay
--question TEXT                           the question to investigate
--n INT                                   how many hypotheses to ask for (default 5)
--model TEXT                              model id, anthropic backend only
--out DIR                                 where cards and report go (default data/)
--root PATH                               dataset root (defaults to ../one_way_fsg_model)
--no-narrow                               don't ask for a narrowed claim after a failure
--briefing-file PATH                      use a saved briefing instead of rebuilding it
--log-level {DEBUG,INFO,WARNING,ERROR}
```

Regenerate the briefing on its own, to read what the model actually sees:

```bash
python3 -c "
from llm_insights.io.dataset import Dataset
from llm_insights.summary.briefing import build_briefing
print(build_briefing(Dataset('../one_way_fsg_model')))
" > data/briefing.md
```

---

## 7. Known limits, stated plainly

- **Per-step solid stress exists for the healthy case only.** `extracted_fields/` and
  `dynamic_inputs/` were only generated for `flow_U0p0360`. Full-mesh von Mises for all three
  cases is available at step 14 only, via `grn_inputs/`. Installing `h5py` and reading
  `solid_fields.h5` would unlock the full 15-step × 3-case stress evolution — that is the single
  highest-value next step.
- **Only 5 of 23 GRN nodes were written out.** Ask about `SMAD23` and the harness refuses rather
  than inventing an answer. That is deliberate; card H6 in the demo shows it.
- **Growth `g` is per-cell in the reference frame**, while stress is per-node in the deformed
  frame. They are not co-registered, so no claim currently correlates them directly.
- **`step_k/arc_data.npz` is the geometry at the end of step k-1.** Step 0 is the pristine
  undeformed cap and is not a simulation result.
- **Logging uses stdlib `logging`, not loguru**, and tests are `unittest.TestCase` rather than
  pytest-native. Both were forced by a sandbox with no package installs; both run fine under your
  venv now. See TASKS.md for the reconciliation notes.
