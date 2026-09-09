# The hypothesis tester, packaged so that someone who is not Ethan can run it.
#
# Three properties this image is built around, each of which is a decision:
#
# 1. **The dataset is mounted, never baked.** The FSG output is ~2.1 GB behind Git LFS.
#    Copying it in would produce an image nobody could pull and would freeze one
#    snapshot of Dan's results inside a container image, where nothing could tell you
#    it had gone stale. It arrives at runtime as a read-only bind mount at
#    /data/one_way_fsg_model. The build context is `llm_insights/`, so the dataset --
#    its sibling -- is not even reachable from here.
#
# 2. **The test suite runs during the build.** A broken image cannot be produced: the
#    RUN below is a build step like any other, so a failing test fails `docker build`.
#    The data-dependent tests skip, because the dataset is deliberately absent at build
#    time; everything else runs.
#
# 3. **`smoke` needs no network, no model and no API key.** It replays the recorded
#    transcript, which re-executes every verification in code. That is the entrypoint
#    to run first: it proves the container works before anyone configures an LLM.
#
# Multi-arch. `python:3.11-slim` publishes linux/amd64 and linux/arm64, and every
# dependency here ships manylinux and aarch64 wheels for cp311, so no compiler is
# needed on either. Build both with:
#
#   docker buildx build --platform linux/amd64,linux/arm64 -t llm-insights:latest .
#
# See INSTRUCTIONS_DOCKER.md for the copy-paste version of all of this.

FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE keeps .pyc files out of the layers; PYTHONUNBUFFERED makes
# `docker run` show progress as it happens rather than in one burst at the end.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_INSIGHTS_ROOT=/data/one_way_fsg_model

WORKDIR /app

# Dependencies first, in their own layer, so editing source does not reinstall numpy.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e . \
 && pip install --no-cache-dir pytest scipy

# Everything the suite and the smoke run need. `data/` is excluded by .dockerignore
# except for the one recorded transcript, so this copies a few kilobytes, not the
# generated reports.
COPY tests ./tests
COPY data ./data

# The packaging files themselves. tests/test_docker.py asserts against their text --
# that the dataset is mounted rather than baked, that this gate exists at all, that the
# instructions cover `git lfs pull` -- so without them here those nineteen guard tests
# do not skip, they fail, and the build cannot complete. Copying them in is also the
# stronger arrangement: the image carries the assertions made about it, so the guards
# run against exactly the Dockerfile that built it. A few kilobytes.
COPY Dockerfile .dockerignore INSTRUCTIONS_DOCKER.md ./

# The build-time gate. If any test fails, this layer fails and no image is produced.
RUN python3 -m pytest tests -q

# The entrypoint is written here rather than shipped as a file in the repository,
# because the packaging task owns the Dockerfile and nothing else in the tree.
RUN printf '%s\n' \
    '#!/bin/sh' \
    '# Entrypoint for the llm-insights image. See INSTRUCTIONS_DOCKER.md.' \
    'set -e' \
    '' \
    '# A local model would be reached over the host network, which `smoke` does not' \
    '# have and does not want. Any Anthropic key that leaked in through `docker run -e`' \
    '# is removed here as well, so a run that claims to need no key provably uses none.' \
    'unset ANTHROPIC_API_KEY' \
    '' \
    'OUT="${LLM_INSIGHTS_OUT:-/out}"' \
    'ROOT="${LLM_INSIGHTS_ROOT:-/data/one_way_fsg_model}"' \
    '' \
    'case "${1:-smoke}" in' \
    '  smoke)' \
    '    shift 2>/dev/null || true' \
    '    if [ ! -d "$ROOT/FSG Results" ]; then' \
    '      echo "error: no FSG dataset at $ROOT" >&2' \
    '      echo "  mount it read-only, e.g.:" >&2' \
    '      echo "    -v /path/to/one_way_fsg_model:/data/one_way_fsg_model:ro" >&2' \
    '      echo "  see INSTRUCTIONS_DOCKER.md step 5." >&2' \
    '      exit 2' \
    '    fi' \
    '    mkdir -p "$OUT"' \
    '    exec python3 -m llm_insights.agent.run \' \
    '      --generator transcript \' \
    '      --transcript /app/data/demo_transcript.json \' \
    '      --question "What differs between the healthy and overflow cases, and why?" \' \
    '      --root "$ROOT" \' \
    '      --out "$OUT" "$@"' \
    '    ;;' \
    '  test)' \
    '    shift' \
    '    exec python3 -m pytest /app/tests -q "$@"' \
    '    ;;' \
    '  run)' \
    '    shift' \
    '    exec python3 -m llm_insights.agent.run --root "$ROOT" --out "$OUT" "$@"' \
    '    ;;' \
    '  shell)' \
    '    exec /bin/sh' \
    '    ;;' \
    '  *)' \
    '    exec "$@"' \
    '    ;;' \
    'esac' \
    > /usr/local/bin/llm-insights \
 && chmod +x /usr/local/bin/llm-insights

# Where the dataset is mounted and where reports are written. Declared so that
# `docker run` without a mount fails with this image's own message rather than a
# traceback from three modules down.
VOLUME ["/data/one_way_fsg_model", "/out"]

ENTRYPOINT ["/usr/local/bin/llm-insights"]
CMD ["smoke"]
