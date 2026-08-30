# TASKS.md

The communication channel between the planning agent (Cowork), Ethan, and the implementing
agents (Claude Code). Read `CLAUDE.md` first.

## Protocol

- **Statuses:** `TODO` -> `IN PROGRESS` -> `IN REVIEW` -> `DONE`, or `BLOCKED`.
- **Claiming:** set status to `IN PROGRESS` and fill in Owner before writing code.
- **Logging:** append dated entries to the task's Log. Never delete or rewrite history —
  including someone else's entries. The log is the handoff.
- **Blocking:** if the spec is wrong or ambiguous, set `BLOCKED`, say why, and stop. Building
  the wrong thing confidently is worse than stopping.
- **Finishing:** move to `IN REVIEW`, not `DONE`. A human closes tasks.
- **Evidence:** acceptance criteria that mention a number require the command and its real
  output pasted into the Log. No asserted numbers.

---

## TASK-000 — Pin exact tool versions

**Status:** IN REVIEW
**Owner:** Claude Code (session "Task 0"), 2026-08-30
**Size:** ~10 minutes
**Blocks:** nothing, but do it first so lint is deterministic.

### Context
The style guide requires an exact ruff pin in both `pyproject.toml` and
`.pre-commit-config.yaml`. The machine that scaffolded this repo had no PyPI access, so rather
than guess a version that may not exist, both spots carry a TODO.

### Acceptance criteria
1. Resolve the current ruff release (`pip index versions ruff`, or check
   https://github.com/astral-sh/ruff/releases).
2. Pin that exact version in `pyproject.toml` under `[project.optional-dependencies] dev`.
3. Set the matching `rev:` tag in `.pre-commit-config.yaml` (replacing `v0.0.0-REPLACE-ME`).
4. Remove both TODO comments.
5. `pre-commit run --all-files` passes.

### Log
- 2026-08-30 (Claude Code) — Done, ready for review. Ruff pinned to **0.16.5** (the current
  release) in `pyproject.toml` and `.pre-commit-config.yaml`; both TODO comments removed.

  Version resolved with `pip3 index versions ruff`:

  ```
  ruff (0.16.5)
  Available versions: 0.16.5, 0.16.4, 0.16.3, ...
    INSTALLED: 0.12.0
    LATEST:    0.16.5
  ```

  Matching `astral-sh/ruff-pre-commit` tag confirmed to exist before pinning
  (`git ls-remote --tags https://github.com/astral-sh/ruff-pre-commit | grep -E 'v0\.16\.[0-9]+$'`
  -> `1f1e8bf348ff38fc88619a38d3ca4d9c56abea49  refs/tags/v0.16.5`), and the hook env that
  `rev: v0.16.5` actually builds reports `ruff 0.16.5`, so the two pins really do agree:

  ```
  $ ~/.cache/pre-commit/reposjp7ywz1/py_env-python3.14/bin/ruff --version
  ruff 0.16.5
  ```

  Criterion 5, `pre-commit run --all-files` from `llm_insights/`, exit code 0:

  ```
  ruff check...........................................(no files to check)Skipped
  ruff format..............................................................Passed
  fix end of files.........................................................Passed
  trim trailing whitespace.................................................Passed
  check yaml...........................................(no files to check)Skipped
  check toml...........................................(no files to check)Skipped
  check for added large files..............................................Passed
  ### exit=0
  ```

- 2026-08-30 (Claude Code) — **Incident: the first `pre-commit run --all-files` modified 530
  files under `one_way_fsg_model/`. Fully reverted; no damage remains. Read this before
  running pre-commit here.**

  Cause: the git repo root is the *parent* of `llm_insights/`, and `--all-files` means every
  file `git ls-files` returns for the whole repo. `trailing-whitespace` and `end-of-file-fixer`
  are not Python-only, so they rewrote Dan's `.msh` and `grn_input.json` files — e.g.
  `flow_U0p0180/solid.msh` went from 1,100,698 to 1,077,407 bytes. That is a hard-rule-1
  violation and it happened because I ran the command in criterion 5 literally without checking
  its scope first.

  Recovery: `git checkout -- one_way_fsg_model .gitignore README.md`. (A stale
  `.git/index.lock` dated 15:45, left by the scaffolding session with no git process running,
  had to be removed first.) Verified restored:

  ```
  $ ls -l "one_way_fsg_model/FSG Results/flow_U0p0180/solid.msh"
  -rw-r--r--  1 ethandecamp  staff  1100698 ...      # matches the LFS pointer's size field
  $ git lfs fsck --pointers
  Git LFS fsck OK
  $ git status --porcelain          # one_way_fsg_model/ no longer appears
   M .gitignore
   M CLAUDE.md
   M README.md
  ?? llm_insights/
  ```

  This machine *does* have working Git LFS (`git-lfs/3.7.1`), so the checkout restored real
  content rather than pointers. On a machine without it, that same recovery would have replaced
  1,327 real binaries with ~130-byte pointer stubs. See the LFS open question below.

  Permanent fix (small scope addition beyond the stated criteria, flagged here deliberately): a
  top-level `exclude` in `.pre-commit-config.yaml` covering `one_way_fsg_model/`,
  `resources_for_ethan/`, and `RESOURCES.md`, so `--all-files` can never reach the read-only
  material again. Re-verified after the change: `git status` shows nothing under
  `one_way_fsg_model/`.

- 2026-08-30 (Claude Code) — Three other things changed; all are visible in the diff and none
  were in the task text, so calling them out explicitly:

  1. **Hook id `ruff` -> `ruff-check`.** At v0.16.5 the run printed `ruff (legacy alias)`.
     `ruff-check` is the current id (confirmed in the tag's `.pre-commit-hooks.yaml`); this
     avoids depending on a deprecated alias at the moment we pin the version.
  2. **`tests/__init__.py` got a docstring.** It was empty and `ruff check` failed it with
     `D104 Missing docstring in public package`, so criterion 5 could not pass without it.
     Fixed by writing the docstring rather than by adding `D104` to `per-file-ignores` —
     style guide rule 3 says don't widen `ignore` to silence a line.
  3. **Whitespace fixes to the repo-root `.gitignore` and `README.md`** (trailing spaces on two
     comment lines; missing final newline). These are Ethan's files, not Dan's, but they are
     unrelated to TASK-000, which the style guide's "don't reformat unrelated files in a feature
     PR" rule discourages. I let them stand because criterion 5 cannot pass on a clean checkout
     otherwise — the hooks just re-apply them. **Reviewer's call:** either accept them here or
     split them into their own commit. `llm_insights/STYLE_GUIDE.md` also had trailing
     whitespace trimmed.

- 2026-08-30 (Claude Code) — Caveat on the criterion-5 evidence: `ruff check`, `check-yaml`, and
  `check-toml` report *"no files to check"* under `--all-files` because `llm_insights/` is still
  untracked, so `git ls-files` returns nothing from it. That run proves the hooks install and
  the config parses, but it does not prove ruff passes on our code. This does:

  ```
  $ pre-commit run --files $(find . -type f -not -name ".DS_Store" \
      -not -path "./.ruff_cache/*" -not -path "*/__pycache__/*" | tr '\n' ' ')
  ruff check...............................................................Passed
  ruff format..............................................................Passed
  fix end of files.........................................................Passed
  trim trailing whitespace.................................................Passed
  check yaml...............................................................Passed
  check toml...............................................................Passed
  check for added large files..............................................Passed
  ```

  Once `llm_insights/` is committed, plain `--all-files` will cover these on its own.
  `python3 -m pytest` also runs clean (`collected 0 items` — there are no tests yet).

---

## TASK-001 — Precompute the GRN response surface

**Status:** TODO
**Owner:** unassigned
**Size:** ~half a day of work + ~40 min of compute
**Blocks:** the hypothesis-verification loop. Do this early.

### Context

`../one_way_fsg_model/networkpoint.py` is a 23-node ODE gene-regulatory network. Its entire
input is **two scalars** — shear stress and tissue (von Mises) stress:

```python
simulate_physical(shear_dyn, mech_pa) -> (t, y_trajectory, steady_state)
```

So the model's complete steady-state behavior is a 2-D surface, and it can simply be computed
once instead of re-solved per query.

Measured on a 2026-08-30 run: **one solve takes ~0.64 s**. The verification loop the project is
built around will need to ask "what would happen if shear were higher here?" thousands of
times. At 0.64 s each that is a bottleneck; from a lookup table it is free.

Two important details, both confirmed by reading the source:

- `normalize_shear` / `normalize_mech` (lines 116-119) **clip to [0, 1]** against
  `SHEAR_MAX = 30.0` dyn/cm² and `MECH_MAX = 100.0` Pa. So `simulate_physical(s, 859)` is
  literally the same call as `simulate_physical(s, 100)`. Grid the **normalized** `[0,1]²`
  space via `simulate()` directly — gridding physical values above the maxima computes the same
  point repeatedly.
- That clipping is exactly why 44% of the overflow-case domain is saturated. The surface will
  make the plateau visible, which is a deliverable in its own right — do not smooth it away.

### Acceptance criteria

1. New package `src/llm_insights/grn_surface/` with `__init__.py` (style guide: every service is
   a package).
2. `precompute.py` — CLI entry point (`precompute-grn-surface`), with flags for grid resolution
   (default 64), output path (default `data/grn_surface_{N}x{N}.npz`), and `--n-jobs`.
   Parallelize with `joblib`.
3. It **imports** `networkpoint` from `../one_way_fsg_model/`. It does not copy it, vendor it,
   or modify it. See rule 1 in `CLAUDE.md`.
4. The `.npz` stores: `shear_norm_grid`, `mech_norm_grid`, `node_names` (from
   `networkpoint.NODES`), and `steady_state` with shape `(n_shear, n_mech, 23)`. Plus metadata:
   grid resolution, generation timestamp, `SHEAR_MAX` / `MECH_MAX`, and every field of the
   `Params` dataclass used. A surface computed under different `Params` is a different surface
   and must not be silently reused.
5. `surface.py` — a `GRNSurface` class that loads the `.npz` and exposes:
   - `query(shear_norm: float, mech_norm: float) -> dict[str, float]`
   - `query_physical(shear_dyn: float, mech_pa: float) -> dict[str, float]`
   - both vectorized over array input, using bilinear interpolation.
   It warns (loguru) if the loaded `Params` metadata does not match the current
   `networkpoint.Params()` defaults.
6. **Accuracy gate — this is the real acceptance test.** `tests/test_grn_surface.py` samples
   >= 50 random *off-grid* points, compares `GRNSurface.query` against a live
   `networkpoint.simulate()` call at the same point, and asserts max absolute error across all
   23 nodes is below a stated threshold. Put the actual measured max error in the Log. If it is
   larger than ~0.01, raise the grid resolution rather than loosening the threshold — and say
   so.
7. Logging per the style guide: `INFO` for start/finish and grid size, `DEBUG` for progress and
   timing, `TRACE` for entry/exit. Lazy brace form, never f-strings.
8. `ruff check` and `ruff format` clean; `pytest` green.

### Optional, only if criterion 6 still passes afterward
`simulate()` accepts a `y0` initial condition. Warm-starting each solve from the previous grid
point's steady state may cut the 0.64 s substantially. If you try it, prove equivalence against
cold-start results before keeping it — a faster wrong surface is worse than a slow right one.

### Notes
- 64 x 64 x 23 float64 is ~750 KB, so the artifact is small enough to commit if the team
  prefers reproducible-by-checkout. `data/*.npz` is gitignored by default; change it in its own
  PR if you decide otherwise.
- A contour plot of `EndMT_combined` over the surface would be a good sanity check and is
  probably worth showing Dan. Not required for acceptance.

### Log
- 2026-08-30 (Cowork planning agent) — Task written. Timing figure (0.64 s/solve, avg over 100
  solves spanning the full input range) measured directly against `networkpoint.py`. The clip
  behavior in criterion 3's note was read from source, not inferred.

---

## Open questions

Questions for Ethan or Dan that are not scoped to a single task. Add, don't delete.

- **Is `MECH_MAX = 100.0` Pa deliberate?** Peak von Mises is 597 Pa (healthy) and 859 Pa
  (overflow), so 29.7% / 44.1% of nodes clip to `mech_norm = 1.0` in those two cases,
  pinning YAP_TAZ at its ceiling for ~30% / ~45% of the domain. This may be intentional
  physiological normalization, or the dynamic range may simply be exhausted. It materially
  affects how any gene-expression pattern in the high-flow case should be read. **Ask Dan
  before building analysis on top of the GRN outputs.** Do not "fix" it — see `CLAUDE.md`
  rule 1.
- **Can `run_fsg.py` actually be re-run, and where?** FEniCS + gmsh are not installed on
  Ethan's machine as of 2026-08-30. BioHPC (Weill) is rent-by-the-hour, so the run budget is a
  real dollar figure. This gates every hypothesis that needs new mechanics rather than new GRN
  evaluations.
- **Git LFS is not initialized in at least one environment touching this repo.**
  `.gitattributes` routes `*.png *.npz *.npy *.h5 *.msh *.xdmf *.pdf *.pptx` through LFS, but
  `git lfs` was not installed on the machine that scaffolded this folder — so 1,327 binary
  files under `one_way_fsg_model/` show as modified (index holds a ~130-byte pointer, working
  tree holds the real ~98 KB file). **No source file is affected — 0 `.py` files differ.**
  Risk: a `git add -A` commit from an environment without LFS would write ~2.1 GB of binaries
  straight into git history. Verify `git lfs version` and run `git lfs install` before the next
  commit, and stage explicitly rather than with `-A` until that is confirmed.

  - _Update 2026-08-30 (Claude Code, TASK-000):_ on **this** machine LFS is present and working
    — `git lfs version` -> `git-lfs/3.7.1 (GitHub; darwin arm64; go 1.25.3)`, `git lfs fsck
    --pointers` -> `Git LFS fsck OK`, and `git status` is clean under `one_way_fsg_model/`. So
    the problem is environment-specific, not repo-wide, and the check stays necessary. It is
    also now load-bearing for *recovery*, not just for commits: reverting an accidental edit to
    the model with `git checkout` restores real file content on an LFS-enabled machine and
    130-byte pointer stubs on one without. See the TASK-000 incident log.
