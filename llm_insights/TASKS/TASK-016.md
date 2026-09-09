# TASK-016 — A local model backend, and a bake-off to choose it

**STATUS:** IN REVIEW — Lane C (Claude Opus 5), started 2026-09-07, all 7 criteria evidenced 2026-09-08
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


### 2026-09-08 (evening) — Lane C (Claude Opus 5), continuation session

The 08:20 session was killed by an account spend limit with criteria 1, 3 and 4
unevidenced. This entry finishes those three and resolves the ruff-debt conflict. It
appends; nothing above is edited or removed.

#### The context check, re-confirmed against a warm server before anything else

`/api/ps` reports only on models already *loaded*, so on a cold server it returns
`{"models":[]}` and any check made against it passes vacuously. Both candidates were
warmed first, then read — the same order `LocalGenerator.warm_up` /
`LocalGenerator.server_context_length` implement in code:

```
$ curl -s http://127.0.0.1:11434/api/ps
{"models":[]}                       <-- cold: this answer means nothing

$ curl -s http://127.0.0.1:11434/api/generate \
    -d '{"model":"llama3.1:8b","prompt":"ok","stream":false,"keep_alive":"30m"}' >/dev/null
$ curl -s http://127.0.0.1:11434/api/ps | python3 -m json.tool
{
    "models": [
        {
            "name": "llama3.1:8b",
            "model": "llama3.1:8b",
            ...
            "parameter_size": "8.0B",
            "quantization_level": "Q4_K_M"
            ...
            "size_vram": 7122106449,
            "context_length": 32768
        }
    ]
}

$ curl -s http://127.0.0.1:11434/api/generate \
    -d '{"model":"qwen3:8b","prompt":"ok","stream":false,"keep_alive":"30m"}' >/dev/null
$ curl -s http://127.0.0.1:11434/api/ps | python3 -c \
    "import json,sys; [print(m['model'], m['context_length']) for m in json.load(sys.stdin)['models']]"
qwen3:8b 32768
```

**32768 on both, measured warm.** Not the 4096 default, so the bake-off below was not
run against a server that would truncate the prompt in silence.

One incidental fact worth recording: after warming `qwen3:8b`, `/api/ps` lists only
`qwen3:8b` — `llama3.1:8b` was evicted. **The two 8B models do not co-reside in 16 GB**,
so a multi-model bake-off pays a model-load cost at each switch. That is inside the
first call's latency for each model, and is called out again in the latency numbers.

#### Criterion 1 — fault injection. PASS, with the results pasted this time.

The earlier entry asserted "54 tests, including six fault-injection cases" without
pasting the run. Here it is.

```
$ cd llm_insights/src
$ PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. -p "test_local.py" -v
[...]
test_a_dead_server_does_not_defeat_the_context_probe (tests.test_local.FaultInjectionTests...) ... ok
test_body_truncated_mid_object (tests.test_local.FaultInjectionTests...) ... ok
test_connection_refused (tests.test_local.FaultInjectionTests...) ... ok
test_hang_past_the_timeout (tests.test_local.FaultInjectionTests...) ... ok
test_http_500 (tests.test_local.FaultInjectionTests...) ... ok
test_malformed_json_body (tests.test_local.FaultInjectionTests...) ... ok
test_valid_json_of_the_wrong_schema (tests.test_local.FaultInjectionTests...) ... ok
[...]
----------------------------------------------------------------------
Ran 54 tests in 7.076s

OK
```

A green test tells you an assertion held; it does not tell you *what the code did*. The
criterion asks for the latter, so the same six stubs were re-run through the real
`LocalGenerator.propose` with the resulting exception printed verbatim. Nothing below
is mocked under the socket: each is a real `http.server` on a real loopback port.

```
1. malformed JSON body ('not json at all, just an apology from the proxy')
  raised LocalBackendError 0.01s into the call
  the local model server at http://127.0.0.1:49749/v1/chat/completions returned a body that is not JSON (Expecting value: line 1 column 1 (char 0)); it began: 'not json at all, just an apology from the proxy'

2. body truncated mid-object
  raised LocalBackendError 0.00s into the call
  the local model server at http://127.0.0.1:49751/v1/chat/completions returned a body that is not JSON (Unterminated string starting at: line 1 column 38 (char 37)); it began: '{"choices": [{"message": {"content": "[{\"id\": \"H1\"'

3. valid JSON of the wrong schema ({"result":"hello","done":true})
  raised LocalBackendError 0.00s into the call
  the local model server's reply has no 'choices' list, so it is not an OpenAI-shaped chat completion; keys were ['done', 'result']

4. HTTP 500
  raised LocalBackendError 0.00s into the call
  the local model server returned HTTP 500 for http://127.0.0.1:49755/v1/chat/completions: {"error":"internal server error, model failed to load"}

5. connection refused (port bound then closed)
  raised LocalBackendError 0.00s into the call
  could not reach the local model server at http://127.0.0.1:49757/v1/chat/completions ([Errno 61] Connection refused); start it, or point LLM_INSIGHTS_LOCAL_URL at one that is running

6. hang past the timeout (server sleeps 5s, client timeout 0.5s)
  raised LocalBackendError 0.50s into the call
  the local model server at http://127.0.0.1:49759/v1/chat/completions did not respond within 0.5s
```

Per case, what the real code did:

| fault | what happened | is that right? |
|---|---|---|
| malformed JSON | `http_json` caught `JSONDecodeError` and raised `LocalBackendError` naming the URL, the decoder's complaint and the first 200 characters of the body | yes — the proxy's apology is quoted, so the cause is diagnosable without a packet capture |
| truncated mid-object | same path, `Unterminated string starting at: line 1 column 38` | yes — and note it is caught at the *envelope* level, before any hypothesis parsing, so a half-written reply cannot become a half-parsed claim |
| valid JSON, wrong schema | passed the JSON decode, then `extract` refused it: `no 'choices' list ... keys were ['done', 'result']` | yes — this is the fault that most looks like success, and the observed keys are printed so a server-shape mismatch is obvious |
| HTTP 500 | `urllib`'s `HTTPError` converted to `LocalBackendError` carrying the status **and the server's own body** (`model failed to load`) | yes — the reason the model failed survives to the operator |
| connection refused | `URLError` converted, and the message names the fix and the env var (`LLM_INSIGHTS_LOCAL_URL`) | yes — 0.00s, no retry storm |
| hang past the timeout | the socket timeout fired at **0.50s** against a server sleeping 5s, i.e. it did not wait for the hang | yes |

**Instrument error found and fixed while doing this — the third of this task.** Two
problems, both mine, both in the timeout case:

1. *The message was wrong.* It formatted the timeout with `{timeout:.0f}`, so a 0.5s
   timeout printed as **`did not respond within 0s`** — a number that reads as a
   misconfiguration rather than the timeout doing its job. `local.py:180` now uses
   `{timeout:g}` and prints `within 0.5s`. `test_hang_past_the_timeout` asserts on the
   exception type and the elapsed time, not the string, so it was blind to this. It is
   the only change to `local.py` in this session.
2. *My first measurement of it was wrong.* The first transcript reported the timeout
   case as firing after **5.00s**, which would have meant the timeout never fired. It
   did fire at 0.5s; my harness was timing the whole `with _FaultServer(...)` block,
   whose `__exit__` joins a handler thread that is still inside its 5-second `sleep`.
   The transcript above times the call itself, inside the block, which is what
   `test_hang_past_the_timeout` already did correctly. Recording this because a
   measurement that indicts the code when the harness is at fault is exactly the error
   this task keeps finding.

#### Criterion 4 — `ANTHROPIC_API_KEY` is stripped from any child environment. PASS.

Three tests pin it, and they run in the suite above:

```
$ PYTHONPATH=. python3 -m unittest discover -s ../tests -t .. \
    -p "test_local.py" -k ApiKeyIsolation -v
test_no_request_contains_the_api_key (tests.test_local.ApiKeyIsolationTests.test_no_request_contains_the_api_key)
No request contains the api key. ... ok
test_shared_child_environment_still_strips_the_key (tests.test_local.ApiKeyIsolationTests.test_shared_child_environment_still_strips_the_key)
Shared child environment still strips the key. ... ok
test_the_key_is_not_needed_to_construct_a_local_generator (tests.test_local.ApiKeyIsolationTests.test_the_key_is_not_needed_to_construct_a_local_generator)
The key is not needed to construct a local generator. ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.002s

OK
```

Those use an injected transport. Because the criterion is about a *child environment*,
it was also demonstrated against a real `subprocess` and a real socket, with a
deliberately recognisable fake key exported in the parent:

```
parent environment: ANTHROPIC_API_KEY='sk-ant-do-not-leak-0123456789'

1. child_environment() contains ANTHROPIC_API_KEY: False
   it kept 64 of the parent's 65 variables

2. real subprocess, env=child_environment():
     ANTHROPIC_API_KEY visible to child: False; value=None
   control -- same probe with the unscrubbed parent environment:
     ANTHROPIC_API_KEY visible to child: True; value='sk-ant-do-not-leak-0123456789'

3. captured 1 real request(s), 6870 bytes on the wire
   headers sent      : {'content-type': 'application/json', 'accept': 'application/json'}
   secret value on wire : False
   the string 'ANTHROPIC_API_KEY' on wire : False
   any 'authorization'/'x-api-key' header: False
```

The control line in (2) is the part that makes this non-vacuous: the identical probe
*does* see the key when handed the unscrubbed environment, so (1) and (2) are measuring
the scrubber and not a key that was never set. `child_environment()` removes exactly one
variable of 65.

Worth stating plainly: `LocalGenerator` **starts no child process at all**, so it has no
child environment of its own to leak through. What is pinned here is (a) the shared
`subscription.child_environment()` scrubber the subprocess-backed backends use, and (b)
that a local call carries no credential on the wire in any form — no `Authorization`
header, no `x-api-key`, and the key's value nowhere in the 6,870 bytes actually sent.
`request_headers()` returns a fixed two-entry mapping and reads nothing from the
environment, so there is nowhere for a credential to hide.

#### Criterion 3 — the bake-off. Run, and the numbers are below.

Command, exactly as specified, at the committed defaults (`DEFAULT_ROUNDS = 5`,
`DEFAULT_N = 5`, `DEFAULT_MAX_TOKENS = 2048`, `DEFAULT_TEMPERATURE = 0.4`), against the
saved briefing so the two models see byte-identical input:

```
$ cd ~/Documents/Research/ButcherLab/lane-c/llm_insights
$ PYTHONPATH=src python3 -m llm_insights.agent.bakeoff \
    --models llama3.1:8b qwen3:8b \
    --briefing-file data/lane_c/briefing.md \
    --out data/lane_c/bakeoff_evening.json

briefing: 12662 characters

| model | json | calls | failed | proposed | rejected | rejection rate | usable yield | mean latency | mean tok in | mean tok out |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.1:8b | no | 5 | 0 | 25 | 0 | 0.0% | 100.0% | 37.4 s | 6248 | 764 |
| qwen3:8b | no | 5 | 1 | 16 | 0 | 0.0% | 64.0% | 86.5 s | 7277 | 1694 |

### llama3.1:8b — why claims were rejected

(nothing was rejected)

### qwen3:8b — why claims were rejected

(nothing was rejected)
- CALL FAILED — round 4: LocalBackendError: the local model server returned a choice with no message content; finish_reason was 'length'
EXIT=0
```

Per call, from `data/lane_c/bakeoff_evening.json`:

```
llama3.1:8b:
  latencies (s)     : [49.5, 36.0, 33.6, 35.6, 32.2]
  prompt tokens     : [6248, 6248, 6248, 6248, 6248]
  completion tokens : [735, 807, 754, 799, 725]
  asked=25 proposed=25 accepted=25 rejected=0 call_failures=0

qwen3:8b:
  latencies (s)     : [104.4, 70.0, 57.5, 100.2, 100.2]
  prompt tokens     : [7277, 7277, 7277, 7277, 7277]
  completion tokens : [1732, 1449, 1193, 2048, 2048]
  asked=25 proposed=16 accepted=16 rejected=0 call_failures=1
```

**Distribution of rejection reasons: empty, for both models.** Nothing that reached
`hypothesis_from_dict` was thrown out — 25 of 25 for llama3.1:8b, 16 of 16 for qwen3:8b.
The first call in each column also carries that model's load into VRAM (49.5 s and
104.4 s against later calls of ~33 s and ~60 s); the two 8B models do not co-reside in
16 GB, so each switch pays it once.

##### An empty rejection histogram is the answer a vacuous gate would also give, so the gate was checked

A 0% rejection rate is worth nothing unless the validator can reject. Nine objects an
8B model plausibly emits — built against the real schema, so each fails for the reason
it is named for — were put through the same `hypothesis_from_dict` and `score_batch`
the bake-off uses:

```
ACCEPTED  the control -- a well-formed hypothesis, which MUST be accepted

REJECTED  a bare string, not an object
          bucket : not an object
          message: hypothesis: must be an object/dict, got str
REJECTED  missing the 'claim' field
          bucket : missing field 'claim'
          message: hypothesis: missing required field 'claim'
REJECTED  an empty claim
          bucket : empty field 'claim'
          message: hypothesis: field 'claim' must not be empty
REJECTED  'id' is a number, not a string
          bucket : field 'id' wrong type
          message: hypothesis: field 'id' must be str, got int (7)
REJECTED  an invented extra field
          bucket : unexpected field(s) ['confidence']
          message: hypothesis: unexpected field(s) ['confidence']; allowed fields are ['claim', 'id', 'parent_id', 'rationale', 'test']
REJECTED  a primitive that does not exist
          bucket : unknown primitive
          message: hypothesis 'H1' test: field 'primitive' names 'vibes_check', which is not a known test; known tests are ['compare_metric', 'correlation', 'fraction_above', 'metric_ordering', 'peak_location', 'profile_monotonic']
REJECTED  params as a string instead of an object
          bucket : field 'params' wrong type
          message: hypothesis 'H1' test: field 'params' must be dict, got str ('field=stress')
REJECTED  the whole test block missing
          bucket : not an object
          message: hypothesis 'H1' test: must be an object/dict, got NoneType

score_batch: proposed=9 accepted=1 rejected=8 rejection_rate=89%
distinct reason buckets: 7
```

One well-formed object accepted, eight malformed ones rejected, sorted into seven
distinct buckets. The gate bites, and `classify_rejection` can tell the failures apart,
so the empty histograms above are a real result and not an instrument that never fires.

##### The qwen3 reasoning confound — how it was checked, and what it does and does not touch

`qwen3:8b` is a reasoning model, and a small completion budget already produced one
false failure earlier in this task. Three separate things could have been confounded,
and they have different answers. Measured on the **real** assembled proposal prompt
(6,057 system + 12,923 user characters), `data/lane_c/reasoning_split.txt`:

```
=== llama3.1:8b, max_tokens=2048 ===
  finish_reason               : 'stop'
  usage.completion_tokens     : 821
  reasoning field present     : False
  reasoning chars             : 0
  content chars               : 3283
  '<think>' inside content    : False
  parse_response              : 5 object(s), 5 valid

=== qwen3:8b, max_tokens=2048 ===
  finish_reason               : 'stop'
  usage.completion_tokens     : 1193
  reasoning field present     : True
  reasoning chars             : 1968
  content chars               : 3092
  '<think>' inside content    : False
  parse_response              : 5 object(s), 5 valid

=== qwen3:8b, max_tokens=4096 ===
  finish_reason               : 'stop'
  usage.completion_tokens     : 3756
  reasoning field present     : True
  reasoning chars             : 11554
  content chars               : 2548
  '<think>' inside content    : False
  parse_response              : 5 object(s), 5 valid
```

1. **Rejection classification is NOT confounded.** Ollama puts the `<think>` block in
   `choices[0].message.reasoning`, a *sibling* of `content`. `LocalGenerator.extract`
   reads only `content`, so no reasoning text ever reaches `parse_response` or
   `hypothesis_from_dict`. `'<think>' inside content` is `False` on every call, and the
   content parsed to 5 valid hypotheses in all three. qwen3's 0% rejection rate is a
   real measurement of its JSON, not an artefact of thinking text being parsed as data.

2. **Token accounting IS confounded, and the table's `mean tok out` column must be read
   with that in mind.** `usage.completion_tokens` counts reasoning. At the 4,096 budget
   qwen3 reported 3,756 completion tokens for 2,548 characters of JSON (~640 tokens)
   and 11,554 characters of thinking (~2,900 tokens) — the two sum to the reported
   figure, so roughly **three quarters of qwen3's billed output never reaches the
   parser.** llama3.1:8b has no `reasoning` field at all, so its 764 is all content.
   The columns are therefore not like-for-like: qwen3's true JSON output is around
   600-700 tokens, comparable with llama's, and the rest is deliberation.

3. **The budget is where it actually bites, and that is a real defect in the
   measurement at the shipped default.** Rounds 4 and 5 both consumed exactly 2,048
   tokens — the cap. Round 4 came back with empty `content` and `finish_reason:
   'length'`, which `extract` correctly refuses, counting as a call failure. Round 5
   came back with a *partial* reply that still parsed, yielding **1** hypothesis instead
   of 5. So 9 of 25 requested hypotheses were lost to the completion budget, not to the
   schema — and only one of those nine is visible as a "failure". The rejection-rate
   column cannot see any of it, which is exactly why `usable_yield` exists.

Re-run with the budget doubled, to separate "cannot hold the schema" from "spent the
budget thinking":

```
$ PYTHONPATH=src python3 -m llm_insights.agent.bakeoff --models qwen3:8b \
    --max-tokens 4096 --briefing-file data/lane_c/briefing.md \
    --out data/lane_c/bakeoff_qwen4096.json

briefing: 12662 characters

| model | json | calls | failed | proposed | rejected | rejection rate | usable yield | mean latency | mean tok in | mean tok out |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3:8b | no | 5 | 0 | 21 | 0 | 0.0% | 84.0% | 119.3 s | 7277 | 2406 |

### qwen3:8b — why claims were rejected

(nothing was rejected)
EXIT=0

round 1 returned 5 object(s)
round 2 returned 1 object(s)
round 3 returned 5 object(s)
round 4 returned 5 object(s)
round 5 returned 5 object(s)
completion tokens: [2037, 4096, 1310, 2827, 1758]
```

Doubling the budget removed the hard failure but **did not remove the loss**: round 2
hit the 4,096 cap exactly and still returned 1 hypothesis instead of 5. Yield went
64% → 84%, and mean latency went 86.5 s → 119.3 s. qwen3's deliberation length is not
bounded by anything this package controls, so there is no budget that makes this go
away — only one that makes it rarer and slower.

##### The choice: **`llama3.1:8b`**, which is already `DEFAULT_LOCAL_MODEL`. No code change needed.

Said plainly first, because the spec asks for the number rather than a coronation:
**the rejection rate does not discriminate between these two models. Both are 0%.**
Nothing either model returned was rejected by the strict validator, so if rejection
rate were the only criterion this bake-off would be a tie and the honest answer would
be "either".

It is not a tie on the other three columns:

| | llama3.1:8b | qwen3:8b (2048) | qwen3:8b (4096) |
|---|---|---|---|
| strict-validator rejection rate | 0.0% (0/25) | 0.0% (0/16) | 0.0% (0/21) |
| **usable yield** (valid hypotheses / asked) | **100% (25/25)** | 64% (16/25) | 84% (21/25) |
| mean latency | **37.4 s** | 86.5 s | 119.3 s |
| completion tokens, spread across 5 calls | **725-807** | 1193-2048 (capped twice) | 1310-4096 (capped once) |
| calls that returned nothing usable | **0** | 1 | 0 |

`llama3.1:8b` is the default because it delivered every hypothesis asked for, in a
third of the time, with a completion length that varied by 11% across five calls
against qwen3's 3x — and a predictable completion length is what lets
`check_context`'s reserved budget be a real guard rather than a guess. qwen3:8b's
answer quality is not being judged here and may well be better; what is measured is
that it cannot reliably deliver five schema-valid objects inside a bounded budget, and
that is the property this package needs from a proposal backend.

**Two honest limits on the number 0%.** First, n is small: 25 objects per model. Zero
failures in 25 draws bounds the true rejection rate at roughly **12%** at 95%
confidence (rule of three), not at zero — the right reading is "no schema failure was
observed in 25", not "this model never fails". Second, 0% is a measurement of *schema
survival only*. It says nothing about whether a claim is true, interesting or
falsifiable; that is the harness's job, and no model output here reached a card.

**One thing this bake-off does not cover, flagged for TASK-014.** It was run against
the 12,662-character pre-GRN briefing, at 6,248 prompt tokens for llama and 7,277 for
qwen3. TASK-014's ceiling is 20,000 characters. At that size the prompt is roughly
10,000-11,500 tokens, still far inside the 32,768 context, but qwen3's already-marginal
completion budget gets no easier. If the briefing grows, this bake-off should be re-run
rather than assumed to carry over.

#### The ruff-debt conflict, resolved by re-measuring — both circulating figures were misread

Two numbers were in circulation for the pre-existing lint debt:

- this Log's earlier entry: **191 errors in `tests/`, 14 in `src/`**
- `MORNING_REPORT.md`: **151, and 17 in `src/`**

Neither pair describes the current tree. Re-measured at the pinned **ruff 0.16.5**, from
`llm_insights/` so the package's own `pyproject.toml` is the config in force:

```
$ ruff --version
ruff 0.16.5

$ cd ~/Documents/Research/ButcherLab/lane-c/llm_insights   # feat/lane-c-infra @ 589bd26
$ ruff check src/
Found 14 errors.
[*] 1 fixable with the `--fix` option.
$ ruff check tests/
Found 132 errors.
[*] 10 fixable with the `--fix` option (4 hidden fixes can be enabled with the `--unsafe-fixes` option).
$ ruff check src tests
Found 146 errors.
[*] 11 fixable with the `--fix` option (4 hidden fixes can be enabled with the `--unsafe-fixes` option).
```

The same measurement on every other live tree gives the identical figures, so this is
the package's debt and not something a lane introduced:

```
feat/phase2-prologue    @ c5cb465   src/ 14   tests/ 132
feat/lane-a-exploration @ 75e028d   src/ 14   tests/ 132
feat/lane-b-grn         @ 137eb3c   src/ 14   tests/ 132
feat/lane-c-infra       @ 589bd26   src/ 14   tests/ 132
```

**Where 151 came from — it is right, but it is a different commit and a different
scope.** Measured against `b5724e8`, the last commit *before* any Phase 2 edit
(extracted with `git archive` so no branch was touched):

```
$ git archive b5724e8 llm_insights | tar -x -C $TMP && cd $TMP/llm_insights
$ ruff check src/
Found 19 errors.
$ ruff check tests/
Found 132 errors.
$ ruff check src tests
Found 151 errors.
$ ruff format --check src tests
11 files would be reformatted, 20 files already formatted
```

So:

| claim | verdict |
|---|---|
| this Log's "191 errors in `tests/`" | **wrong.** `tests/` is 132, and has been 132 at every commit checked. No invocation I can construct at 0.16.5 produces 191. |
| this Log's "14 in `src/`" | **right** for the current tree. |
| `MORNING_REPORT.md`'s "151" | **right, but stale and mis-scoped.** 151 is `src` *and* `tests` together at `b5724e8`, not `tests/` alone and not current. Phase 0 fixed 5 of the `src/` errors incidentally, taking the combined figure to 146. |
| `MORNING_REPORT.md`'s "17 in `src/`" | **close but wrong.** `src/` at `b5724e8` measures **19**, not 17; it is **14** now. |

**The number TASK-020 should expect is 146** (`ruff check src tests`), plus 9 files that
`ruff format --check` would rewrite — 4 under `src/`, 5 under `tests/`. None of it is a
Phase 2 regression; the debt predates the phase, and per-task criteria only ever asked
for ruff-clean on the files a task touched. Per the no-deleting rule the earlier entry
above is left exactly as written; this is the correction, not an edit.

#### Suite and ruff, after this session's one code change

The only edit to shipped code in this session is the timeout-message format described
under criterion 1 (`local.py:180`).

```
$ cd llm_insights/src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
Refusing to start: fix the errors above first. No model calls were made.
..................................................................................................................................................................
----------------------------------------------------------------------
Ran 459 tests in 7.512s

OK
```

**459 tests, 0 failures.** (`Refusing to start` is `tests/test_preflight.py` asserting
its own failure path; it is a passing test.) This is the same 459 Ethan measured
independently for this lane before the session started, so the edit changed no count.

```
$ ruff --version
ruff 0.16.5
$ ruff check src/llm_insights/agent/local.py src/llm_insights/agent/bakeoff.py \
    src/llm_insights/agent/generator.py tests/test_local.py
All checks passed!
$ ruff format --check <same four files>
4 files already formatted
```

#### A pre-existing bake-off artifact, not used

`data/lane_c/bakeoff.txt` and `bakeoff.json` (mtime 08:37) hold a three-model run —
`llama3.1:8b`, `qwen3:8b`, `qwen3:4b` — made after the previous session was killed. It
is not referenced by any Log entry, and its completion tokens (a qwen3:8b call at 3,506,
qwen3:4b capped at 4,096) show it ran at a larger `max_tokens` than the committed
`DEFAULT_MAX_TOKENS = 2048`, so its provenance cannot be vouched for. It was left in
place, untouched, and nothing above is taken from it. The runs above are written to
`bakeoff_evening.{txt,json}`, `bakeoff_qwen4096.{txt,json}` and `reasoning_split.txt`.

Worth keeping from it as corroboration only, not as evidence: `qwen3:4b` failed all five
rounds with `finish_reason: 'length'` and an empty `content` — the same budget-versus-
thinking failure measured properly above, but total. That is the reason to stay off the
smaller reasoning models here, and it is consistent with the hardware note in this
task's spec.

#### Criterion status at the end of this session

| # | criterion | status |
|---|---|---|
| 1 | fault injection, results pasted | **met** — 54 tests green plus a per-case transcript of what the real code did |
| 2 | no network in the suite | met (previous session) |
| 3 | bake-off result pasted, default chosen with a reason | **met** — `llama3.1:8b`; both models 0% rejection, chosen on yield, latency and bounded output |
| 4 | `ANTHROPIC_API_KEY` stripped from any child environment | **met** — 3 tests, plus a real subprocess and a real socket, with a control arm |
| 5 | existing backends unchanged, full suite count pasted | met (previous session); re-confirmed at 459 above |
| 6 | canary passes on the full briefing | met (previous session), reviewed by Ethan |
| 7 | ruff clean on touched files | met; re-run above after this session's edit |

Seven of seven. Setting **STATUS: IN REVIEW**.

Not done here, deliberately, and each is somebody else's step:

- **No merge.** `INSTRUCTIONS.md` has the known three-way collision — Lane A and Lane C
  both insert at line 185 and Lane C is the only lane that modifies existing lines.
  Ethan resolves that by hand.
- **No push.** Committed to `feat/lane-c-infra` only.
- **TASK-017 untouched.** `Dockerfile`, `.dockerignore`, `INSTRUCTIONS_DOCKER.md` and
  `tests/test_docker.py` are exactly as the previous session left them.

### Question for Ethan

`DEFAULT_MAX_TOKENS = 2048` is right for `llama3.1:8b` (measured 725-807 tokens per
proposal call, ~2.5x headroom) and is a real constraint on any reasoning model — qwen3:8b
loses 36% of requested hypotheses at 2048 and still 16% at 4096, because its thinking is
counted against the same budget. Three options, and it is your call which:

1. **Leave it at 2048.** The default backend is llama3.1:8b, which never comes close to
   it. A reasoning model is then effectively unsupported at the default, which is
   honest but undocumented.
2. **Raise it to 4096.** Costs nothing for llama (it stops at `stop`, not `length`), but
   `check_context` reserves `max_tokens` against the server's context, so the guard's
   reserve doubles — irrelevant at 32,768, relevant if anyone runs a 8,192-context
   server.
3. **Make the reserve per-model**, e.g. a larger budget when the server reports a
   `reasoning` field. That is new behaviour in `local.py`, so it is a code change I did
   not make without your say-so.

My recommendation is (1) plus a sentence in `INSTRUCTIONS.md` §5b saying reasoning
models need `--max-tokens 4096` and will still occasionally truncate. That keeps the
default honest for the default model and does not add a code path for a backend nobody
has chosen.
