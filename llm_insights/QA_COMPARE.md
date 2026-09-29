# QA_COMPARE — Dan's questions, four ways

Runs Dan's 10 questions (`qa/dan_qa.json`) through four arms and writes one comparison table:

| arm | what happens |
|---|---|
| **Llama (harness)** | llama3.1:8b proposes 5 claims, the harness tests each against the FSG output, failures get one narrowing round, then llama writes a summary that is withheld if it cites any number not on a card. The normal pipeline. |
| **Llama (no harness)** | llama3.1:8b reads the *same* briefing and just answers. Nothing tested, nothing checked. |
| **Sonnet (harness)** | Same as above, through the Claude Code CLI on your subscription. No API key. |
| **Sonnet (no harness)** | Same as above. |

Both arms are asked for the same shape of answer (a 1–2 sentence headline, then a 3–6 sentence
summary), so the table compares content rather than length. Nothing grades anything — Dan's
answer sits in its own column for a human to compare against.

Code: `src/llm_insights/agent/qa_compare.py`. Tests: `tests/test_qa_compare.py`.

---

## 1. Run it

Two terminals, both starting in the repo:

```bash
cd ~/Documents/Research/ButcherLab/butcher_lab_fsg_exploration/llm_insights
```

**Terminal 1 — Ollama with a context window big enough for the briefing** (about 16.6k tokens).
Skip this if Ollama is already running the way it was for the Sep 9 run:

```bash
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

**Terminal 2 — the whole comparison:**

```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m llm_insights.agent.qa_compare run
```

That runs all four arms in order (llama harness → llama baseline → Sonnet harness → Sonnet
baseline) and then writes the table. It prints one line per question as it goes.

**The output is `data/qa_compare/qa_comparison.csv`** (open in Numbers/Excel and copy cells into
PowerPoint), with the same table in `data/qa_compare/qa_comparison.md`.

```bash
open data/qa_compare/qa_comparison.csv
```

## 2. If it stops part way — just run the same command again

Every question is saved the moment it finishes, and a question that already has a result is
skipped. So a usage limit, a closed laptop, or Ollama dying just means:

```bash
PYTHONPATH=src python3 -m llm_insights.agent.qa_compare run
```

and it picks up where it stopped. An arm stops by itself after **2 failures in a row** (that
nearly always means the backend is down or out of usage) and the command exits 1 with
`run the same command again to resume`.

A question where *any* model call failed part way (e.g. the usage limit hit during the summary
call) is **not** saved, so it is redone in full on the next run — no half-finished answers end up
in the table.

## 3. Other commands

```bash
# only one model, or one arm
PYTHONPATH=src python3 -m llm_insights.agent.qa_compare run --models llama
PYTHONPATH=src python3 -m llm_insights.agent.qa_compare run --models sonnet --arms baseline

# only some questions
PYTHONPATH=src python3 -m llm_insights.agent.qa_compare run --questions 1 6

# redo questions that already have results
PYTHONPATH=src python3 -m llm_insights.agent.qa_compare run --force --questions 3

# rebuild the table from whatever has finished, calling no model
PYTHONPATH=src python3 -m llm_insights.agent.qa_compare report

# start completely fresh (keeps the old run, just moves it aside)
mv data/qa_compare data/qa_compare_old_$(date +%Y%m%d_%H%M)

# the tests (no model is called)
python3 -m pytest tests/test_qa_compare.py -q
```

Every flag: `PYTHONPATH=src python3 -m llm_insights.agent.qa_compare run --help`.

## 4. What gets written

All under `data/qa_compare/`:

| file | what |
|---|---|
| `qa_comparison.csv` | **The table.** One row per question: #, Question, Dan, then the four arms. |
| `qa_comparison.md` | Same table, then run statistics and notes on how to read it. |
| `qa_run_stats.csv` | Per arm: questions answered, calls, time, claims survived/falsified/could-not-run, how many survivors were LOOKUPs, numbers the no-harness answers used that aren't in the briefing. |
| `qa_cards.csv` | Every harness claim beside the test that actually ran, its verdict and LOOKUP/COMPUTED class. Backup slides. |
| `briefing.md` | The exact briefing every arm read. |
| `<model>/<arm>/qNN/answer.json` | One question's full result. Harness arms also get `cards.json`, `report.md/.html`, `transcript.json` there. |

How to read a harness cell: the answer, then `[Tests: 3 survived, 1 falsified, 1 could not run]`.
If the summary was withheld by the numeric check, the cell says so and lists the claims that
survived instead of being blank.

## 5. Time and cost

- **Llama:** free. Harness ≈ 1.5–2.5 min per question (proposal call ~50 s, narrowings ~10 s
  each, summary), baseline ≈ 30–50 s. **≈ 25–35 min for both llama arms.** Measured from earlier
  llama runs on this briefing, not from this script.
- **Sonnet:** up to ~9 calls per harness question plus 1 per baseline question, so **~40–90 CLI
  calls** in total. Per-call latency through the CLI has **not** been measured for Sonnet on this
  prompt size — budget 20–40 min. This counts against your Pro usage; if the limit hits, wait for
  it to reset and rerun (section 2).
- The CLI reports an equivalent-spend estimate even on a subscription; the run stops if that
  passes `--max-budget-usd` (default 15) or `--max-calls` (default 120). These are runaway guards,
  not bills.

## 6. Things to say when presenting

- **Sonnet has no fallback.** The CLI backend normally retries on Haiku if the account can't use
  the requested model. That's turned off here, so the Sonnet column is Sonnet or it fails loudly.
  The stats table's "Model actually used" column confirms it (`claude-cli:sonnet`).
- **Sonnet goes through Claude Code**, not the raw API. Our system prompt is passed with
  `--system-prompt`, but it's still Claude Code's harness around the call. Llama gets the prompt
  directly.
- **All four arms read the same briefing**; the notes under the table confirm the SHA-256 matched,
  and say loudly if it didn't.
- **Dan's answers are expectations**, several hedged, and Q10 is blank. They're the reference
  column, not an answer key.
- **LOOKUP share**: a surviving claim whose answer was already printed in the briefing. Earlier
  runs had 92–100% of survivors as LOOKUPs; if that holds here, "survived" mostly means "read the
  table correctly."
- The **no-harness** answers aren't checked, so the stats count numbers they cite that don't appear
  in the briefing. The harness summaries can't contain any (that's what withholding enforces).

## 7. When something goes wrong

| you see | cause | fix |
|---|---|---|
| `could not reach the local model server ... Connection refused` | Ollama isn't running | start it (section 1, terminal 1), rerun |
| `ContextTooSmallError: ... reports a context of 4096 tokens` | Ollama started without a big context | restart it with the `OLLAMA_CONTEXT_LENGTH=32768 ...` command, rerun |
| `[sonnet] cannot start this backend -- ... 'claude' is not on PATH` | Claude Code CLI not installed / not on PATH | `npm install -g @anthropic-ai/claude-code`, run `claude` once to log in, rerun. Llama arms still run. |
| `... usage limit reached` then `stopping this arm` | Pro usage limit | wait for reset, rerun the same command |
| `unknown model 'sonnet'` / `not available on your plan` | account can't use Sonnet via the CLI | no silent fallback by design; try `--cli-model claude-sonnet-4-5` or tell me |
| `call or cost ceiling was reached` | hit `--max-calls` / `--max-budget-usd` | rerun with e.g. `--max-calls 200 --max-budget-usd 30` |
| `WARNING: the briefing has changed since results in this folder were produced` | code or data changed between runs | table will flag it; move `data/qa_compare` aside and rerun everything for a fair comparison |
| preflight refuses to start | venv not active | `source .venv/bin/activate` |
