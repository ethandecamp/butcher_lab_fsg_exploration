# TASK-017 — Container, and instructions a non-programmer can follow

**STATUS:** IN REVIEW — Lane C (Claude Opus 5), all 6 criteria evidenced 2026-09-08
**Phase:** 1, Lane C (second). Branch `feat/lane-c-infra`.
**Depends on:** TASK-016.

## Context

Right now this runs on exactly one machine. Dan should be able to run it, and so should anyone
he hands it to. This task covers the Python side only. The FEniCS solver image is TASK-019 and is
a separate, harder problem.

## What to build

- `llm_insights/Dockerfile`, multi-arch (`linux/amd64` and `linux/arm64` — Ethan's Mac is arm64,
  most other people are amd64).
- `.dockerignore` excluding `.venv/`, `data/`, caches, and the LFS payload.
- **Data is mounted, never baked.** The FSG output is ~2.1 GB behind Git LFS. The image stays
  small; `one_way_fsg_model/` comes in as a read-only bind mount.
- The **test suite runs during the image build**, so a broken image cannot be produced.
- A `smoke` entrypoint running the transcript replay end to end — no network, no model, no API
  key — proving the container works before anyone configures an LLM.
- `INSTRUCTIONS_DOCKER.md`, written for someone who is not a programmer: numbered steps, the exact
  command to copy at each one, what correct output looks like, and a failure table mapping
  symptom to fix. Assume they have never used Docker. Cover installing Docker, cloning, running
  `git lfs pull`, and the mount syntax.

## Acceptance criteria

1. `docker build` succeeds on arm64 and the build-time test run is visible in the log. Paste the
   tail of the build output showing the test count.
2. `docker run ... smoke` produces a report from the transcript replay with **no network access**
   (`--network none`) and no API key present. Paste the command and its output.
3. The generated report matches the host-generated one except for timestamps. Show the diff.
4. Image size stated. If it exceeds 2 GB, explain why in the Log.
5. `INSTRUCTIONS_DOCKER.md` is verified by following it literally from a fresh clone in a
   scratch directory. Paste the transcript of doing so. **Do not mark this criterion met from
   reading the document.**
6. Ruff clean on touched files; full suite green on the host too.

## Log

_(append below)_

### 2026-09-08 — Lane C (Claude Opus 5)

Session note: the `Dockerfile`, `.dockerignore`, `INSTRUCTIONS_DOCKER.md` and
`tests/test_docker.py` on this branch were written by an earlier session that was killed
by a spend limit four minutes in, before claiming the task and before building anything.
**Nothing here had ever been built or run.** The first `docker build` failed. Every
number below is from a command run in this session.

Docker: client 29.1.3, server 29.1.3 (Docker Desktop 4.57.0), buildx v0.30.1.
Host: 14-inch M5 MacBook Pro, darwin/arm64.

---

#### The inherited image could not be built at all

`docker build` failed at the build-time test gate with **30 failures**, from two separate
omissions. Both are the same class of mistake: a packaging decision whose consequence
surfaces several layers later as something that looks like a code bug.

```
#12 7.773 30 failed, 400 passed, 29 skipped, 1 warning in 7.50s
#12 ERROR: process "/bin/sh -c python3 -m pytest tests -q" did not complete successfully: exit code: 1
------
Dockerfile:54
--------------------
  54 | >>> RUN python3 -m pytest tests -q
--------------------
```

**1. Nineteen failures — the packaging files were never copied into the image.**
`tests/test_docker.py` asserts against the *text* of `Dockerfile`, `.dockerignore` and
`INSTRUCTIONS_DOCKER.md`, resolving them as `Path(__file__).parents[1]`. The Dockerfile
copied `pyproject.toml`, `src`, `tests` and `data`, and none of those three.

```
#12 7.770 E   AssertionError: /app/Dockerfile is missing; the container packaging is incomplete
#12 7.771 E   AssertionError: /app/.dockerignore is missing; the container packaging is incomplete
```

**2. Eleven failures — `.dockerignore` excluded a fixture the suite loads.**
`data/` was excluded and only `demo_transcript.json` re-included, but
`tests/test_cards.py:38` loads `data/example_cards.json`.

```
#12 7.768 E   FileNotFoundError: [Errno 2] No such file or directory: '/app/data/example_cards.json'
```

Fixed in `1cb2058`: `COPY Dockerfile .dockerignore INSTRUCTIONS_DOCKER.md ./` placed
after `COPY tests`/`COPY data` so the pip layer stays cached, and
`!data/example_cards.json` added to `.dockerignore`. Copying the packaging files in is
also the stronger arrangement — the image carries the assertions made about it, so the
guards run against exactly the Dockerfile that built it. It costs 33 kB.

`COPY .dockerignore` is legal under BuildKit; that was checked with a two-line throwaway
build before relying on it, not assumed:

```
#6 [2/3] COPY Dockerfile .dockerignore ./
#7 0.085 -rw-r--r--    1 root     root             3 Sep  9 01:12 /.dockerignore
#7 0.085 -rw-r--r--    1 root     root            84 Sep  9 01:12 /Dockerfile
```

---

#### Criterion 1 — build succeeds and the test run is visible in the log. **MET, both architectures.**

**arm64** (native):

```
$ cd ~/Documents/Research/ButcherLab/lane-c/llm_insights
$ docker build --platform linux/arm64 --progress=plain -t llm-insights:arm64 .
[...]
#13 7.541 =============================== warnings summary ===============================
#13 7.541 src/llm_insights/harness/spec.py:35
#13 7.541   /app/src/llm_insights/harness/spec.py:35: PytestCollectionWarning: cannot collect
#13 7.541   test class 'TestSpec' because it has a __init__ constructor (from: tests/test_preflight.py)
#13 7.541
#13 7.541 434 passed, 29 skipped, 1 warning in 7.89s
#13 DONE 7.6s
[...]
#15 naming to docker.io/library/llm-insights:arm64 done
#15 unpacking to docker.io/library/llm-insights:arm64 1.1s done
#15 DONE 6.8s
EXIT=0
```

**434 + 29 = 463**, which is the host count exactly (see criterion 6). The 29 skips are
the data-dependent tests: the dataset is deliberately absent at build time.

Note the expected figure in the task prompt was 459. It was 459 when this session
started and is 463 now: four guard tests were added for the defects criterion 5 turned
up. The pre-existing 459 is pinned in the first failing build above (30 + 400 + 29).

**amd64** (cross-built on an arm64 Mac):

```
$ docker build --platform linux/amd64 --progress=plain -t llm-insights:amd64 .
#13 8.894 434 passed, 29 skipped, 1 warning in 8.46s
#15 naming to docker.io/library/llm-insights:amd64 done
EXIT=0
```

The spec warned amd64 might be impractical under emulation. It was not — Docker Desktop
uses Rosetta 2 rather than QEMU on Apple Silicon, and the suite ran in 8.46 s emulated
against 7.89 s native. **It is a real x86-64 image, not a mislabelled arm64 one**, and it
runs:

```
$ docker run --rm --platform linux/amd64 --entrypoint uname llm-insights:amd64 -m
x86_64
$ docker run --rm --platform linux/arm64 --entrypoint uname llm-insights:arm64 -m
aarch64
```

---

#### Criterion 2 — `smoke`, `--network none`, no API key. **MET.**

```
$ echo "ANTHROPIC_API_KEY set: $([ -n "${ANTHROPIC_API_KEY:-}" ] && echo YES || echo no)"
ANTHROPIC_API_KEY set: no

$ docker run --rm \
    --network none \
    -v "/Users/ethandecamp/Documents/Research/ButcherLab/butcher_lab_fsg_exploration/one_way_fsg_model:/data/one_way_fsg_model:ro" \
    -v "$OUT:/out" \
    llm-insights:arm64 smoke

INFO llm_insights.api: opening dataset at /data/one_way_fsg_model
INFO llm_insights.summary.briefing: briefing built: 12662 characters, 25 profile points
INFO llm_insights.agent.loop: proposing 5 hypotheses via transcript:/app/data/demo_transcript.json
INFO llm_insights.harness.runner: hypothesis H1 PASS
INFO llm_insights.harness.runner: hypothesis H2 PASS
INFO llm_insights.harness.runner: hypothesis H3 FAIL
INFO llm_insights.harness.runner: hypothesis H4 PASS
INFO llm_insights.harness.runner: hypothesis H5 PASS
WARNING llm_insights.harness.runner: hypothesis H6 could not run: unknown GRN field 'SMAD23_combined'; ...
INFO llm_insights.harness.runner: ran 6 hypotheses: 4 passed, 1 refuted, 1 could not run
INFO llm_insights.agent.loop: narrowing round 1: 1 refuted claim(s)
INFO llm_insights.agent.loop: narrowed H3 into H3b
INFO llm_insights.harness.runner: hypothesis H3b PASS
INFO llm_insights.agent.loop: run complete: 7 admissible, 5 survived, 0 falsified, 1 narrowed, 1 could not run
INFO llm_insights.cards.card: wrote 7 cards to /out/cards.json
INFO llm_insights.cards.render: wrote report for 7 cards to /out/report.md and /out/report.html

Generator:   transcript:/app/data/demo_transcript.json
Model:       claude-opus-5
Dataset:     /data/one_way_fsg_model

  proposed        7
  admissible      7
  rejected        0
  survived        5
  falsified       0
  narrowed        1
  could not run   1

cards:       /out/cards.json
markdown:    /out/report.md
html:        /out/report.html
transcript:  /out/transcript.json
EXIT=0
```

The same run on the amd64 image, also `--network none`, also exit 0. Both are recorded
under criterion 3.

The 1 refuted / 1 narrowed / 1 could-not-run are the recorded demo's own results, not
container faults. `H6` names a GRN field that does not exist in the dataset; the harness
reports that rather than inventing a verdict, which is the intended behaviour.

---

#### Criterion 3 — the container's report matches the host's except for timestamps. **MET.**

Host reference, same transcript, same question:

```
$ cd ~/Documents/Research/ButcherLab/lane-c/llm_insights
$ PYTHONPATH=src python3 -m llm_insights.agent.run \
    --generator transcript --transcript data/demo_transcript.json \
    --question "What differs between the healthy and overflow cases, and why?" \
    --root .../butcher_lab_fsg_exploration/one_way_fsg_model --out $HOST_OUT
```

**`cards.json` is byte-identical, with no normalisation of any kind** — and it is
`cards.json` that carries every observed number and every verdict:

```
$ shasum -a 256 host_out/cards.json ctr_out/cards.json ctr_out_amd64/cards.json
55462244174cd76b0f4cc36dd00a949ce775bb6a254df7aae5f92c266d1e7e71  host_out/cards.json
55462244174cd76b0f4cc36dd00a949ce775bb6a254df7aae5f92c266d1e7e71  ctr_out/cards.json
55462244174cd76b0f4cc36dd00a949ce775bb6a254df7aae5f92c266d1e7e71  ctr_out_amd64/cards.json
```

Three platforms — macOS/arm64 host, linux/arm64 container, linux/amd64 container — one
hash. **There is no last-ulp float difference to report: there is no float difference at
all**, in any significant figure, in any of the observed values. numpy on manylinux and
numpy on macOS agreed bit-for-bit on this workload.

The rendered files do differ, on exactly three fields. `diff -u`:

```
$ diff -u host_out/report.md ctr_out/report.md
@@ -13,9 +13,9 @@
 ## Provenance

-- **Dataset root:** /Users/ethandecamp/.../butcher_lab_fsg_exploration/one_way_fsg_model
-- **Generated:** 2026-09-09T01:12:02Z
-- **Generator:** transcript:data/demo_transcript.json
+- **Dataset root:** /data/one_way_fsg_model
+- **Generated:** 2026-09-09T01:13:53Z
+- **Generator:** transcript:/app/data/demo_transcript.json
 - **Model:** claude-opus-5
 - **Question:** What differs between the healthy and overflow cases, and why?
 - **Briefing sha256:** 9b228a874effeab9

$ diff -u host_out/transcript.json ctr_out/transcript.json
@@ -1,9 +1,9 @@
-    "generator": "transcript:data/demo_transcript.json",
+    "generator": "transcript:/app/data/demo_transcript.json",
-    "generated_at": "2026-09-09T01:12:02Z",
+    "generated_at": "2026-09-09T01:13:53Z",

$ for f in report.md report.html transcript.json cards.json; do
    echo -n "$f: "; diff host_out/$f ctr_out/$f | grep -c '^[<>]'; done
report.md: 6          (3 line pairs)
report.html: 2        (1 line pair -- the whole <header> is one line)
transcript.json: 4    (2 line pairs)
cards.json: 0
```

Accounting for every difference that is not a timestamp — there are two, and both are
the same directory or file named differently on the two sides:

| Field | Host | Container | Why it differs |
|---|---|---|---|
| `Generated` | `...T01:12:02Z` | `...T01:13:53Z` | Timestamp. The two runs are 111 s apart. |
| `Dataset root` | the host path | `/data/one_way_fsg_model` | **The same directory.** It is a bind mount; the mount point is what the container can see. A container reporting the host's path would be reporting a path that does not exist inside it. |
| `Generator` | `transcript:data/...` | `transcript:/app/data/...` | **The same file**, spelled relative on the host command line and absolute in the entrypoint. Proven, not assumed: <br>`f3a8865f96b4fec8f9d1997777f1e785a33a8077e5a2543659eb9f0455fe6d6e  llm_insights/data/demo_transcript.json`<br>`f3a8865f96b4fec8f9d1997777f1e785a33a8077e5a2543659eb9f0455fe6d6e  /app/data/demo_transcript.json` |

Substituting those two and the timestamp, all four artefacts are identical:

```
cards.json       identical after normalising: True
report.md        identical after normalising: True
report.html      identical after normalising: True
transcript.json  identical after normalising: True
```

---

#### Criterion 4 — image size. **MET. 813 MB, under the 2 GB threshold.**

```
$ docker image ls llm-insights:arm64 --format '{{.Size}}'
813MB
$ docker history llm-insights:arm64 --format '{{.Size}}\t{{.CreatedBy}}'
460MB   RUN /bin/sh -c pip install --no-cache-dir -e…
52MB    RUN /bin/sh -c set -eux;  savedAptMark="$(a…      (python:3.11-slim's own layers)
13.1MB  RUN /bin/sh -c set -eux;  apt-get update;  a…
881kB   COPY tests ./tests
1.07MB  COPY src ./src
32.8kB  COPY Dockerfile .dockerignore INSTRUCTIONS_D…
28.7kB  COPY data ./data
20.5kB  RUN /bin/sh -c printf '%s\n' … (the entrypoint)
```

Essentially all of it is the dependency layer: numpy, scipy, pandas, matplotlib and h5py
wheels. The project's own code is 1.07 MB and the packaging 33 kB. No explanation is owed
under the criterion, but for the record: `docker image ls` reports 179MB for the amd64
tag against 813MB for arm64. That is not a smaller image — it is the compressed manifest
size for a platform the local store has not unpacked, versus the unpacked size for the
native one. Both were built from the same Dockerfile and both run.

**The dataset is mounted, never baked**, and the build context confirms it. Measured with
a throwaway `FROM busybox / COPY . /ctx` build under the real `.dockerignore`:

```
--- build context, as Docker actually received it ---
2.0M    /ctx
1.0M    /ctx/src
860.0K  /ctx/tests
20.0K   /ctx/data
 [markdown files]
--- data/ ---
-rw-r--r--    1 root     root          7914 demo_transcript.json
-rw-r--r--    1 root     root          6933 example_cards.json
```

**2.0 MB total.** `llm_insights/data/` is 216 KB on disk and only those two checked-in
fixtures reached the context; `lane_c/`, the generated reports and the briefing did not.
The 2.1 GB dataset is `one_way_fsg_model/`, a *sibling* of the build context, so it is
not reachable from here even by accident. Caches and `*.egg-info` are absent.

One honest gap: **the `.venv/` rule was not exercised**, because this worktree has no
`.venv` — it borrows the one in the main checkout, which is 346 MB. The rule is correct
and `tests/test_docker.py::DockerignoreTests::test_the_virtualenv_is_excluded` pins it,
but no build in this session had a `.venv` in the context to exclude.

---

#### Criterion 5 — `INSTRUCTIONS_DOCKER.md` followed literally from a fresh clone. **MET, and it found four defects.**

Done in a scratch directory, from a clone, executing each command as written rather than
reading it. Two substitutions were unavoidable and are called out where they occur.

```
### STEP 1 — docker info --format 'Server: {{.ServerVersion}}'
Server: 29.1.3

### STEP 2 — git lfs install
Git LFS initialized.

### STEP 3 — git clone <REPOSITORY-URL> butcher_lab_fsg_exploration
Cloning into 'butcher_lab_fsg_exploration'...
Filtering content: 100% (1299/1299), 1.57 GiB | 883.99 MiB/s, done.

### STEP 4 — git lfs pull
(no output)
### STEP 4 check — du -sh one_way_fsg_model
2.1G    one_way_fsg_model
### is it real data or 130-byte pointers?
  19998 bytes    one_way_fsg_model/FSG Results/flow_U0p0540/step_014/arc_data.npz
  1754184 bytes  one_way_fsg_model/FSG Results/flow_U0p0540/step_014/fluid_velocity.h5
  4286796 bytes  one_way_fsg_model/FSG Results/flow_U0p0540/step_014/solid_fields.h5

### STEP 5 — docker build -t llm-insights:latest .
EXIT=0
#13 7.649 430 passed, 29 skipped, 1 warning in 7.37s
#15 naming to docker.io/library/llm-insights:latest done
#15 unpacking to docker.io/library/llm-insights:latest 1.2s done

### environment: is an API key present?
ANTHROPIC_API_KEY set: no

### STEP 6 — copied verbatim from the document
mkdir -p out
docker run --rm \
  --network none \
  -v "$(cd .. && pwd)/one_way_fsg_model:/data/one_way_fsg_model:ro" \
  -v "$(pwd)/out:/out" \
  llm-insights:latest smoke

  proposed        7
  admissible      7
  rejected        0
  survived        5
  falsified       0
  narrowed        1
  could not run   1

cards:       /out/cards.json
markdown:    /out/report.md
html:        /out/report.html
transcript:  /out/transcript.json
EXIT=0

### the out/ folder the doc says to open
-rw-r--r--  1 ethandecamp  staff  10607 cards.json
-rw-r--r--  1 ethandecamp  staff  30009 report.html
-rw-r--r--  1 ethandecamp  staff  10433 report.md
-rw-r--r--  1 ethandecamp  staff   8003 transcript.json
```

Steps 5 and 6 ran verbatim, and step 6's output matches the document's "correct output"
block line for line.

**The two substitutions, stated plainly.** (a) Step 3's `<REPOSITORY-URL>` is a
placeholder, not a URL — see defect 1. (b) The clone had to come from the local
repository rather than GitHub, because the packaging branch `feat/lane-c-infra` is not
pushed and this task must not push it. Everything from step 4 onward is unaffected by
that; the clone is a real clone with a real LFS checkout.

##### Defect 1 — step 3 tells the reader to type a placeholder

`git clone <REPOSITORY-URL> butcher_lab_fsg_exploration`. Someone who has never used
Docker will not know what belongs there, and the document tells them to copy each command
exactly. **Fixed:** the real URL is in the document now.

##### Defect 2 — the step-4 data check does not discriminate, and would pass a broken clone

This is the important one, and it is exactly the hazard the task spec warned about: a
clone with pointer files instead of data, where the real-data tests skip silently and a
green suite verifies nothing.

The document said the check `du -sh one_way_fsg_model` "must print a number in the
hundreds of megabytes or more", and the failure table said a few megabytes means LFS did
not fetch. **Both are wrong.** `one_way_fsg_model/` also contains several hundred MB of
ordinary PDFs and PNGs (`FSG Modeling Papers`, `AHA GRN Plots`) that arrive with any
clone. Measured on a clone that fetched no LFS content at all:

```
$ GIT_LFS_SKIP_SMUDGE=1 git clone -b feat/lane-c-infra <repo> nolfs
$ du -sh one_way_fsg_model
981M    one_way_fsg_model                <-- passes the document's stated check

$ f="one_way_fsg_model/FSG Results/flow_U0p0540/step_014/solid_fields.h5"
$ echo "$(stat -f%z "$f") bytes"; head -c 200 "$f"
132 bytes
version https://git-lfs.github.com/spec/v1
oid sha256:5e8d38ae3a38278bec0020847db57b8e4b51242ef80ed2c3b1ee3a0a79fe1552
size 4286796
```

981M is not "a few megabytes". A reader following the document would have concluded the
download worked. **Fixed:** the check is now a single file's size, which does
discriminate (4,286,796 bytes against 132), with both folder sizes given for reference.

##### Defect 3 — the container's failure on pointer data was incomprehensible

The entrypoint's guard tests for the *directory* `$ROOT/FSG Results`, which exists in a
pointer clone — all the directories and all the filenames are there; only the contents
are stubs. So the guard passed and the run died deep inside numpy:

```
  File "/app/src/llm_insights/io/dataset.py", line 351, in arc
    with np.load(path) as d:
  File "/usr/local/lib/python3.11/site-packages/numpy/lib/_npyio_impl.py", line 489, in load
ValueError: This file contains pickled (object) data. If you trust the file you can load
it unsafely using the `allow_pickle=` keyword argument or `pickle.load()`.
```

That message names neither Git LFS nor the fix, and invites a non-programmer to reach for
`allow_pickle`. **Fixed in the entrypoint**, not compensated for in prose: it now probes
the first `.h5` under `FSG Results` for the LFS pointer magic and refuses to start.
Verified against a real pointer-only clone, with the rebuilt image:

```
$ docker run --rm --network none -v "$(pwd)/one_way_fsg_model:/data/one_way_fsg_model:ro" \
    -v "$(pwd)/out:/out" llm-insights:arm64 smoke
error: the dataset at /data/one_way_fsg_model is Git LFS pointer files, not the real data
  /data/one_way_fsg_model/FSG Results/flow_U0p0180/step_002/fluid_velocity.h5 is 132 bytes; it should be megabytes.
  Run these in the repository folder you mounted, then try again:
    git lfs install
    git lfs pull
  see INSTRUCTIONS_DOCKER.md step 4.
EXIT=3
```

And the document's own fix recovers that same clone, end to end:

```
$ git lfs pull
$ du -sh one_way_fsg_model
2.1G    one_way_fsg_model                                  (was 981M)
  4286796 bytes  .../solid_fields.h5                       (was 132)
$ docker run --rm --network none ... smoke
cards:       /out/cards.json
$ shasum -a 256 out/cards.json
55462244174cd76b0f4cc36dd00a949ce775bb6a254df7aae5f92c266d1e7e71
```

The same hash as the host and both containers. **Criterion 5's git-lfs requirement is
therefore exercised, not just documented**: the pointer state was produced, shown to
defeat the old check, shown to be caught by the new one, and shown to be repaired by the
documented command. Two failure-table rows were added — one for the new message, one for
the old `ValueError` in case someone is running an older image.

##### Defect 4 — the build's test count was a stale asserted number

The document said "You will see a line like `440 passed, 27 skipped` go by". The real
line is `430 passed, 29 skipped, 1 warning in 7.37s` — the figure predated the packaging
fixes and the pytest warning was undocumented, so a reader would have seen an unexplained
warning and a count that did not match. **Fixed**, with the warning explained.

Not changed: the document shows the build ending in ` => => naming to ...` lines, which is
the TTY progress format. A non-TTY run prints `#15 naming to ...`. A reader in a terminal
gets the form the document shows, so this is not a defect.

---

#### Criterion 6 — ruff clean on touched files, full suite green on the host. **MET.**

Ruff is pinned at 0.16.5 in `pyproject.toml`. Only `tests/test_docker.py` is Python among
this task's touched files; `Dockerfile`, `.dockerignore` and `INSTRUCTIONS_DOCKER.md` are
not.

```
$ ruff --version
ruff 0.16.5
$ ruff check tests/test_docker.py
All checks passed!
$ ruff format --check tests/test_docker.py
1 file already formatted
```

Full suite on the host:

```
$ cd llm_insights/src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
----------------------------------------------------------------------
Ran 463 tests in 7.929s

OK
```

Lint was scoped to touched files deliberately: the tree is not ruff-clean at baseline
(recorded in TASK-016's Log — 191 errors in `tests/`, 14 in `src/`), and none of those
files were touched here. Whole-package lint is TASK-020's problem.

---

#### Summary

| # | Criterion | Verdict |
|---|---|---|
| 1 | build succeeds, test run visible | **MET** — arm64 and amd64, `434 passed, 29 skipped` on both |
| 2 | `smoke`, `--network none`, no key | **MET** — exit 0, report written |
| 3 | matches the host report but for timestamps | **MET** — `cards.json` byte-identical on 3 platforms |
| 4 | image size stated | **MET** — 813 MB, under 2 GB |
| 5 | instructions followed literally from a fresh clone | **MET** — 4 defects found and fixed |
| 6 | ruff clean on touched files, suite green | **MET** — 463 tests, OK |

Files changed this session: `Dockerfile`, `.dockerignore`, `INSTRUCTIONS_DOCKER.md`,
`tests/test_docker.py` (four guard tests added, 459 → 463), `TASKS/TASK-017.md`.
Nothing outside Lane C's ownership was touched; `src/llm_insights/summary/briefing.py`
was not opened. Not pushed, not merged — `INSTRUCTIONS.md` has the known three-way
collision at line 185 that Ethan resolves by hand.

### Question for Ethan

Nothing blocking. One thing worth a decision: **the image is 813 MB, and 460 MB of it is
scipy + matplotlib.** Neither is a runtime dependency of the analysis — `pyproject.toml`
lists matplotlib under `dependencies` and the Dockerfile installs scipy only because
`tests/test_correlation_math.py` uses it as an oracle. A build that installed them for
the test layer and dropped them from the final stage would roughly halve the image. That
is a multi-stage-build change and I did not make it, because it is outside what TASK-017
asked for and it changes what `docker run ... shell` has available. Say the word if you
want it.
