# TASK-016 — A local model backend, and a bake-off to choose it

**STATUS:** IN PROGRESS — Lane C (Claude Opus 5), started 2026-09-07
**Phase:** 1, Lane C (first). Branch `feat/lane-c-infra`.
**Depends on:** Phase 0 merged. **Blocks:** TASK-017.

## Context

The demo reaches a model by shelling out to the `claude` CLI on Ethan's Claude Pro subscription.
That works, but it is wrong for two things we now want:

- **It is not portable.** A container that needs Ethan's personal login is useless to Dan.
- **It is not a clean instrument.** Claude Code wraps the prompt in its own agent instructions, so
  any survival rate measured through it has an uncontrolled variable in it.

A local open-weights model fixes both, costs nothing per call, and keeps the door open to
reward-based fine-tuning later.

## Hardware reality

Ethan's machine is a 14-inch M5 MacBook Pro with **16 GB** unified memory and 153 GB/s of memory
bandwidth. Local decode is bandwidth-bound, not capacity-bound: an 8B model at 4-bit is about
5 GB and roughly 30 tok/s; a 14B is about 8.5 GB and roughly 18 tok/s and leaves little room
beside a loaded dataset. **Target the 8B class. Do not pull anything above 14B.**

## Context-window hazard — check this before anything else

Ollama's default context on this machine is **4096 tokens** (it derives it from VRAM: the M5
reports 11.8 GiB usable). The briefing is already ~3,200 tokens, and TASK-014 may take it to
~5,000. **The prompt will not fit in the default context**, and the failure mode is silent
truncation, not an error — the model answers about a briefing whose end it never saw.

Before running any bake-off, restart the server with a real context and confirm it:

```
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

Then **verify the model actually received the whole prompt** — put a canary sentence at the very
end of the briefing and assert the model can quote it back. A test must cover this. Add a startup
check to `LocalGenerator` that reads the server's reported context length and **refuses to run**
if the assembled prompt exceeds it, rather than sending a request that will be truncated.

## What to build

- `src/llm_insights/agent/local.py` — `LocalGenerator`, satisfying the existing `Generator`
  protocol (`propose`, `narrow`, `synthesize`, `close`), speaking the OpenAI-compatible
  `/v1/chat/completions` shape that Ollama and llama.cpp both serve.
- Configuration by **environment variable only** — `LLM_INSIGHTS_LOCAL_URL`,
  `LLM_INSIGHTS_LOCAL_MODEL`. No credentials in code, no credentials in the repo, no prompting the
  user for a key. `../INSTRUCTIONS.md` must give the exact `export` commands to copy.
- Register `--generator local` in the CLI.
- `src/llm_insights/agent/bakeoff.py` — runs N proposal calls against each candidate model and
  reports, per model: strict-validator rejection rate, mean latency, mean tokens, and the
  distribution of rejection reasons.

## Acceptance criteria

1. **Fault injection, not just unit tests.** Stand up stub servers that misbehave on purpose —
   malformed JSON, truncated mid-object, valid JSON of the wrong schema, HTTP 500, connection
   refused, a hang past the timeout — and run the real code end to end against each. This is the
   technique that found a live bug in TASK-007 that all 49 unit tests had passed. Paste the
   results. `tests/test_local.py`.
2. **No network in the test suite.** Every test uses an injected transport or a local stub. The
   suite must pass with no model running.
3. **Bake-off result pasted** for at least two candidate models, with the rejection rate per
   model and the chosen default stated with a reason. If the best rejection rate is bad, say the
   number plainly rather than picking a winner among poor options.
4. `ANTHROPIC_API_KEY` is stripped from any child environment, as `subscription.py` already does.
   Test it.
5. Existing `claude-cli`, `transcript`, `echo` and `anthropic` backends are unchanged and their
   tests still pass. Paste the full suite count.
6. **Canary test passes**: with the full briefing assembled, the model quotes back a sentence
   placed at its very end, proving nothing was truncated. Paste it.
7. Ruff clean on touched files.

## Log

### 2026-09-08 — Lane C (Claude Opus 5)

Session note: an earlier attempt at this task was killed mid-flight by an account spend
limit, after `agent/local.py` had been written but before anything was verified. That
file was re-read and re-verified from scratch before being trusted; nothing below is
carried over on faith.

#### Baseline, before any change

```
$ cd llm_insights/src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
Ran 386 tests in 0.694s

OK
```

#### What was built

- `src/llm_insights/agent/local.py` — `LocalGenerator`, speaking
  `POST /v1/chat/completions` (the shape Ollama and `llama.cpp --server` share).
  Implements `propose`, `narrow`, `synthesize`, `close`. Configuration is by
  environment variable only: `LLM_INSIGHTS_LOCAL_URL`, `LLM_INSIGHTS_LOCAL_MODEL`, and
  the optional `LLM_INSIGHTS_LOCAL_CONTEXT`. No credential is built, read or sent.
- `src/llm_insights/agent/bakeoff.py` — the bake-off (`run_model`, `format_table`) and
  the canary (`run_canary`), both behind one CLI.
- `tests/test_local.py` — 54 tests, including six fault-injection cases run against real
  loopback `http.server` stubs.
- `src/llm_insights/agent/generator.py` — **exactly two edits, as instructed**: `"local"`
  added to `GENERATOR_KINDS`, and one `if kind == "local":` branch in `make_generator`
  with a lazy import of `llm_insights.agent.local`. Nothing else in that file was
  touched. `--generator local` reaches the CLI automatically because `run.py` builds its
  `choices` from `GENERATOR_KINDS`.

#### The context-window hazard — dealt with first, as the spec requires

The server on this machine already reports a real context; it did not need restarting.
Measured:

```
$ curl -s http://127.0.0.1:11434/api/ps
{"models":[{"name":"llama3.1:8b",...,"context_length":32768}]}
```

`LocalGenerator.check_context` reads that number before every call and raises
`ContextTooSmallError` rather than sending a prompt that would not fit. It reads it
from, in order: `LLM_INSIGHTS_LOCAL_CONTEXT`, Ollama's `/api/ps`, llama.cpp's `/props`,
and then those two again after a one-token warm-up — because Ollama reports nothing
about a model it has not loaded, so a check made against a cold server would learn
nothing and pass vacuously.

**Finding — the first token estimator erred in the wrong direction.** The guard
estimates prompt tokens from character count, since no tokenizer is available over the
OpenAI-compatible surface. The first draft used 3.0 characters per token, reasoned from
"English is about 4". Measured against the real assembled proposal prompt (18,980
characters: 6,057 system + 12,923 user):

```
$ PYTHONPATH=src python3 -c "...post the real prompt with max_tokens=1..."
llama3.1:8b prompt_tokens = 6248
qwen3:8b    prompt_tokens = 7277
```

That is 3.04 and **2.61** characters per token — the briefing is mostly numbers, units
and identifiers, which tokenize far worse than prose. At 3.0 the estimate was 6,327,
i.e. **950 tokens under qwen3's real count**. An under-estimate is exactly the failure
the guard exists to prevent: it waves through a prompt the server then truncates in
silence. `CHARS_PER_TOKEN` is now **2.0**, which clears both measured tokenizers by
about 30%. `tests/test_local.py::ContextWindowTests::test_estimate_errs_high` pins the
estimate above both measured counts, with the numbers in the test.

**Briefing size, measured — the two figures in circulation are both right.** It is
**12,662 characters** and **12,698 bytes** (`wc -c` counts bytes; the briefing has
multi-byte characters). sha256 of the file: `9b228a874effeab9482432e32ea20464b154fb6ef5c02d1ba0559fd08b23dd8d`.

#### Criterion 6 — the canary. PASS on both models.

**Finding — the canary was invalid at first, and the failure was mine, not the
model's.** With a 32-token completion budget, `qwen3:8b` returned
`finish_reason: length` with empty `content`: it is a reasoning model, Ollama routes its
`<think>` block to a separate `reasoning` field, and the whole budget went there. That
is indistinguishable from a truncated prompt, so the instrument was measuring the wrong
thing. The budget is now `CANARY_MAX_TOKENS = 512`. The assertion is unchanged — the
passphrase must appear in the reply.

```
$ PYTHONPATH=src python3 -m llm_insights.agent.bakeoff --canary \
    --models llama3.1:8b qwen3:8b --briefing-file data/lane_c/briefing.md \
    --log-level WARNING
briefing: 12662 characters

model:            llama3.1:8b
briefing + canary: 12956 characters sent
server context:   32768
prompt tokens:    4657
passphrase:       ELDERBERRY-6840
model replied:    'ELDERBERRY-6840'
CANARY:           PASS

model:            qwen3:8b
briefing + canary: 12956 characters sent
server context:   32768
prompt tokens:    5672
passphrase:       ELDERBERRY-2194
model replied:    'ELDERBERRY-2194'
CANARY:           PASS

EXIT=0
```

The passphrase is randomised per run, so neither model can have answered from memory of
an earlier one.

#### Criterion 5 — the full suite

```
$ cd llm_insights/src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
Ran 440 tests in 8.179s

OK
```

386 baseline + 54 new. The `claude-cli`, `transcript`, `echo`, `paste` and `anthropic`
backends are untouched and their tests still pass.

#### Criterion 2 — no network in the suite

Every test uses either an injected `transport` callable or a loopback `http.server`
stub on an ephemeral port. The suite passes with nothing listening:

```
$ LLM_INSIGHTS_LOCAL_URL=http://127.0.0.1:1 PYTHONPATH=. \
    python3 -m unittest discover -s ../tests -t ..
Ran 440 tests in 7.469s

OK
```

#### Criterion 7 — ruff, on touched files only

```
$ ruff check src/llm_insights/agent/local.py src/llm_insights/agent/bakeoff.py \
    src/llm_insights/agent/generator.py tests/test_local.py
All checks passed!
$ ruff format --check <same files>
4 files already formatted
```

(The tree is not ruff-clean at baseline — 191 errors in `tests/`, 14 in `src/` — and
none of those files were touched.)

