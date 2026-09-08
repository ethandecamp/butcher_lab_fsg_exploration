#!/usr/bin/env python3
"""SessionStart hook (resume/compact): re-inject the state a fresh context window
would otherwise have to rediscover.

Phase 2 task status lives one-per-file in ``llm_insights/TASKS/TASK-0NN.md``, not in
``llm_insights/TASKS.md`` (which holds only the closed TASK-000..010). A brief that
pointed at TASKS.md would hand a compacted session a snippet of finished work and tell
it that was the handoff channel, which is worse than saying nothing.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess

TASKS_DIR = pathlib.Path("llm_insights/TASKS")
STATUS_RE = re.compile(r"^\*\*STATUS:?\*\*[: ]*(.+?)\s*$|^\*\*STATUS:\s*(.+?)\*\*\s*$", re.M)


def sh(cmd: str, cwd: str | None = None) -> str:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10, cwd=cwd
        )
        return r.stdout.strip()
    except Exception:
        return ""


def worktrees() -> list[str]:
    out = sh("git worktree list --porcelain")
    return [ln[len("worktree ") :] for ln in out.splitlines() if ln.startswith("worktree ")]


def task_status() -> list[str]:
    rows: list[str] = []
    if not TASKS_DIR.is_dir():
        return rows
    for path in sorted(TASKS_DIR.glob("TASK-*.md")):
        text = path.read_text(errors="ignore")
        m = STATUS_RE.search(text)
        status = "UNKNOWN"
        if m:
            status = (m.group(1) or m.group(2) or "").strip().strip("*").strip()
        title = text.split("\n", 1)[0].lstrip("# ").strip()
        rows.append(f"- {path.stem}: {status} — {title}")
        for q in re.findall(r"^### Question for Ethan\s*\n(.+)$", text, re.M):
            rows.append(f"    QUESTION LOGGED: {q.strip()[:160]}")
    return rows


def main() -> None:
    parts = ["## Checkpoint brief (auto-injected on resume/compact)"]

    for wt in worktrees() or ["."]:
        branch = sh("git symbolic-ref --short HEAD", cwd=wt) or "detached"
        log = sh("git log --oneline -5", cwd=wt)
        dirty = sh("git status --porcelain", cwd=wt)
        block = f"### Worktree {wt}\nBranch: {branch}\nRecent commits:\n{log}"
        if dirty:
            block += f"\nUncommitted changes:\n{dirty}"
        parts.append(block)

    rows = task_status()
    if rows:
        parts.append(
            "### Phase 2 task status (from llm_insights/TASKS/)\n"
            + "\n".join(rows)
            + "\n\nRead the relevant TASK file in full before acting on this summary."
        )

    parts.append(
        "Reminder: Phase 2 tasks live one-per-file in `llm_insights/TASKS/`, governed by "
        "`llm_insights/TASKS/README.md` (phase order, lane map, file-ownership table). "
        "`llm_insights/TASKS.md` holds only the closed TASK-000..010 and the Open questions "
        "list. `one_way_fsg_model/` and `resources_for_ethan/` are read-only. "
        "Never ask the user a question during an unattended run: log it under "
        "`### Question for Ethan` in the task's Log, set that task to BLOCKED, and move on to "
        "the next task you can do."
    )

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": "\n\n".join(parts),
                }
            }
        )
    )


if __name__ == "__main__":
    main()
