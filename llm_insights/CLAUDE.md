# CLAUDE.md — llm_insights

Rules for any agent doing work in this folder. Read this before writing code.

## Hard rules

1. **Never modify anything in `../one_way_fsg_model/`.** That is Dan Pearce's FSG + GRN model
   and it is the ground truth this whole project is measured against. Changing it silently
   invalidates every result. Import from it, read its outputs, wrap it — never edit it, never
   "fix" it, never reformat it, never lint it. If something in there genuinely appears to be a
   bug, do not patch it: write it up under **Open questions** in `TASKS.md` and leave it alone.
   The same goes for `../resources_for_ethan/`. **`../RESOURCES.md` is different:** it is our
   own index, and appending an entry when a new resource arrives is what it is for. Never
   rewrite or reorganize entries that already exist there.

2. **Follow `STYLE_GUIDE.md`.** Full type annotations, Google-style docstrings with
   Args/Returns/Raises, loguru with the lazy brace form (never f-strings in log calls), ruff as
   the only linter, conventional commit messages.

3. **`ruff check` and `ruff format` must pass before any commit.** Suppress with a specific
   code and a reason (`# noqa: ARG002 - signature fixed by the solver callback`), never bare.

4. **A human reviews every PR.** Per the style guide, AI must not be the only reviewer. Ethan
   has to be able to explain every line — if you write something he would not be able to defend
   in a meeting, either simplify it or explain it in the handoff log.

5. **Do not invent numbers.** If you report a timing, an error bound, or a fraction, it must
   come from code that actually ran. Paste the command and the output into the task's log.

## Working protocol

- `TASKS.md` is the communication channel between the planning agent (Cowork), Ethan, and
  implementing agents (Claude Code). Read it at the start of a session.
- Claim a task by setting its status to `IN PROGRESS` and putting your name in Owner.
- Append to the task's **Log** as you go. Never delete history, never rewrite someone else's
  entries.
- If you are blocked or the spec is wrong, set status `BLOCKED`, write why under the task, and
  stop. Do not guess at intent and build the wrong thing.
- Questions that are not about one specific task go under **Open questions** at the bottom.

## Context you should not have to rediscover

- `../RESOURCES.md` section 6 is a source-level walkthrough of the entire FSG model. Read it
  before opening the model's `.py` files.
- `../RESOURCES.md` §1 covers the closest published prior art for this package, including Zhu et
  al. 2026 (BioPINN-LM), whose 42-dim "MechToken" solves the same compress-a-simulation-for-an-LLM
  problem as `summary/briefing.py`. Read that entry before designing any experiment on the
  briefing's contents — their encoding ablation already tested several variants.
- `../one_way_fsg_model/README.txt` is the canonical reference for field/column layouts.
- Only `run_fsg.py` needs FEniCS/gmsh. Everything downstream (export, GRN, plotting) runs on
  plain numpy/scipy/h5py/matplotlib/pandas.
- There are exactly **three** simulation runs (`flow_U0p0180`, `flow_U0p0360`, `flow_U0p0540`).
  Any claim phrased "as flow increases" is n=3. Say so.
