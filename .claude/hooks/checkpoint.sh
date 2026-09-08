#!/usr/bin/env bash
# PreCompact hook: snapshot any dirty work into a WIP commit before context
# compaction summarizes it away.
#
# Covers EVERY linked worktree, not just the main checkout. Phase 1 lanes run in
# sibling worktrees (../lane-a, ../lane-b, ../lane-c), so a hook that only looked
# at CLAUDE_PROJECT_DIR would silently protect nothing where the work actually is.
#
# Never blocks compaction (always exits 0).
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

REASON="${1:-auto}"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Parenthesised body = runs in a subshell, so the cd never leaks to the caller.
checkpoint_one() (
  dir="$1"
  cd "$dir" 2>/dev/null || return 0
  [ -n "$(git status --porcelain 2>/dev/null)" ] || return 0

  git add -A -- ':!one_way_fsg_model' ':!resources_for_ethan' ':!llm_insights/data' 2>/dev/null

  # Nothing staged after the exclusions? Then there is nothing worth committing.
  git diff --cached --quiet 2>/dev/null && return 0

  git commit -m "chore: pre-compaction checkpoint (${REASON}, ${TS})

Automated WIP snapshot from the PreCompact hook. Safe to squash/amend
away once the task lands a real commit." --no-verify >/dev/null 2>&1

  branch="$(git symbolic-ref --short HEAD 2>/dev/null || echo detached)"
  echo "[checkpoint] committed in ${dir} (${branch})" >&2
)

while IFS= read -r line; do
  case "$line" in
    "worktree "*) checkpoint_one "${line#worktree }" ;;
  esac
done < <(git worktree list --porcelain 2>/dev/null)

echo "[checkpoint] ${REASON} compaction at ${TS}" >&2
exit 0
