# TASK-016 — A local model backend, and a bake-off to choose it

**STATUS:** TODO
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

_(append below)_
