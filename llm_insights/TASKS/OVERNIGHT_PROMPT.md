You are implementing Phase 2 of this project overnight. Work autonomously. Do not wait for me.

## Read first, before doing anything

1. `llm_insights/TASKS/README.md` — the protocol, the phase order, the lane map, and the file
   ownership table. This governs everything below.
2. The project memory: `MEMORY.md` and, at minimum, `demo_state.md`, `repo_conventions.md`,
   `next_phase_feasibility.md`, `verification_philosophy.md`.
3. `llm_insights/TASKS/TASK-011.md` through `TASK-020.md`.

## Ground rules, in priority order

1. **Never weaken an acceptance criterion to make it pass.** If a faithfulness gate, tolerance,
   byte-comparison or test fails, that is a finding. Write it in the task's Log with the real
   output and mark the task BLOCKED. Do not loosen a tolerance, skip a test, delete an assertion,
   or narrow a comparison so it goes green. This is the single most important rule here.
2. **`one_way_fsg_model/` is read-only.** Import from it, never edit it. Suspected defects go to
   the Open questions section of `llm_insights/TASKS.md`.
3. **No language model judges any result.** Verdicts are arithmetic. Nothing you add may put an
   LLM in the verification path or in a stopping decision.
4. **Stay inside your lane's owned files** per the ownership table. If a task needs a change to a
   file another lane owns, that is a blocker. Log it and stop.
5. **Acceptance criteria that mention a number require the command and its real pasted output**
   in the Log. Never assert a number you did not run.
6. **Do not push.** Commit to local branches only. Ethan pushes.
7. **Do not start TASK-019.** It is blocked pending an answer from Dan. Do not decide the blocker
   is resolved.
8. **Stay inside this repository and its sibling lane worktrees.** Specifically:
   - Do not read, write, move or delete anything elsewhere on this machine. The only files you
     create outside the repo are the three lane worktrees at `../lane-a`, `../lane-b`, `../lane-c`.
   - Do not run `rm -rf`. If something must go, move it into a `_to_delete/` folder inside the
     repo and say so in the report.
   - Do not install anything globally. Python packages go into the existing venv and nowhere else.
     Do not run `brew install`, `pip install --user`, `sudo`, or `conda install` outside the venv.
     If a dependency is genuinely missing, that is a blocker: log it and stop.
   - Do not `git push`, `git push --force`, or change any remote, in this repo or any other.
   - Do not change git global config, shell rc files, or anything in `~/.ssh`, `~/.aws`,
     `~/.config`, or any other credential location.
   - Do not make network requests beyond the local model server on localhost.
9. **Never ask me a question. Never guess a design decision either.** I am asleep. If you reach a
   point where you would want to ask me something — an ambiguity in a spec, a design choice the
   task does not settle, a tradeoff with no obvious right answer — do all three of these:
   - Write the question in that task's Log under a heading `### Question for Ethan`, with the
     options you considered and what you would recommend and why.
   - Set the task's **STATUS to BLOCKED** and stop working on it. Do not pick an answer and
     proceed. A blocked task with a clear question is a good outcome; a finished task built on a
     guess is a bad one, because I cannot see the guess.
   - **Then move on to the next task you can legitimately work on.** A blocker blocks that task,
     not the night. Only stop the session entirely when nothing remains that you can do without
     an answer from me.
   Do not use AskUserQuestion or any other mechanism that waits for a human. Nobody is there.

## Order of work

**Step 0 — baseline.** Confirm the tag `v0.1-demo-working` exists and that there are no uncommitted changes to **tracked** files. Untracked scratch output under `llm_insights/data/` (`live/`, `_to_delete/`) is expected and is not a reason to stop. Confirm
the suite is green before you change anything, and record the count. **The verified baseline is
356 tests, 0 failures.** If you see fewer than 356, something is wrong — stop and report. If the
baseline is not green, stop and report; do not start work on a broken tree.

Note: `tests/test_preflight.py` deliberately prints `Environment check: ERROR broken` and
`Refusing to start` while exercising the refusal path. That is expected output from a passing
test, not a failure.

Environment facts already verified on this machine: Ollama 0.33.3 is serving on
`127.0.0.1:11434` with `llama3.1:8b` and `qwen3:8b` pulled. **Docker is installed but the daemon
was not running** — if it is still down when Lane C reaches TASK-017, that is a blocker to log,
not something to work around.

Run tests with the venv active:
```
cd llm_insights && source .venv/bin/activate
cd src && PYTHONPATH=. python3 -m unittest discover -s ../tests -t ..
```

**Step 1 — Phase 0, serial, do it yourself, no subagents.** Branch `feat/phase2-prologue` from
**current `HEAD`** (which already carries the task specs and hooks). Do not branch from the tag
`v0.1-demo-working` — that predates the specs and branching from it would remove them from the
working tree. The tag remains the rollback point, not the branch point. TASK-011, then TASK-012. Commit each separately with a conventional-commit
message. Do not proceed until both are IN REVIEW with their acceptance criteria met and pasted.
TASK-012's gate is a byte-identical briefing — if `cmp` reports any difference, stop.

**Step 2 — Phase 1, parallel, three subagents, each in its own git worktree.** Three branches
cannot be checked out in one working tree, so create a worktree per lane before you fan out:

```
git worktree add ../lane-a -b feat/lane-a-exploration feat/phase2-prologue
git worktree add ../lane-b -b feat/lane-b-grn         feat/phase2-prologue
git worktree add ../lane-c -b feat/lane-c-infra       feat/phase2-prologue
```

Each worktree checks out the LFS payload again, so to avoid three extra copies of the ~2.1 GB
dataset, set `GIT_LFS_SKIP_SMUDGE=1` for these adds and symlink each worktree's
`one_way_fsg_model` at the real one in the main checkout. The dataset resolver already looks for
`../one_way_fsg_model`, so a symlink is enough. Verify a lane can read the dataset before handing
it to a subagent.

The venv lives in the main checkout and is shared. Activate it by absolute path from each
worktree; do not create three venvs.

- Lane A → worktree `../lane-a`, branch `feat/lane-a-exploration` — TASK-015
- Lane B → worktree `../lane-b`, branch `feat/lane-b-grn` — TASK-013 then TASK-014
- Lane C → worktree `../lane-c`, branch `feat/lane-c-infra` — TASK-016 then TASK-017

Do not remove the worktrees when you finish. Ethan merges from them in the morning.

Give each subagent: the ground rules above, its own task files, `TASKS/README.md`, and the
ownership table. Tell each one explicitly that it may not edit `summary/briefing.py` and may not
touch another lane's files.

Lane B is the long pole — TASK-013 is roughly 80,000 ODE solves, about 1.5 to 2 hours with
`joblib` across all cores. Start Lane B first and let it run.

**Step 3 — stop there.** TASK-018 and TASK-020 need the lanes merged, and the merge is Ethan's in
the morning. Do not merge the lanes yourself.

## When you finish, or when you run out of things you can do

Write `llm_insights/MORNING_REPORT.md` containing:

- What each task's status actually is, and the real test counts.
- Every blocker, with the command that reproduces it.
- Every acceptance criterion that is **not** met, listed plainly. Do not round up to "done."
- Anything you found about Dan's model, for the Open questions list.
- What Ethan has to do first thing: the merge order, and any decision you deliberately left to him.
- If TASK-015 produced a stratified survival rate, state it, including the LOOKUP share, even if
  it is unflattering.

Be accurate rather than encouraging. I would much rather read "three of five criteria met, here
is the failure" than find out at the meeting.
