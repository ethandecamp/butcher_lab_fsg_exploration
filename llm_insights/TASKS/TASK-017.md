# TASK-017 — Container, and instructions a non-programmer can follow

**STATUS:** TODO
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
