# TASK-019 — Reproduce one FSG run in a rebuilt solver environment

**STATUS: BLOCKED — DO NOT START**

**Phase:** 1, Lane D. Branch `feat/lane-d-fenics`.
**Unblock condition:** Dan has sent his `conda env export -n fenics-LEGACY` output **and**
answered the growth-floor question below. Ethan will move this to TODO himself. An agent must
not start this task, and must not decide on its own that the blocker is resolved.

## Why it is blocked

`run_fsg.py` needs legacy FEniCS (`dolfin`) and `gmsh`. We can build **a** working FEniCS
ourselves. What we cannot do blind is rebuild **the solver that produced our ground truth**, and
the three existing runs are what every verdict in this project is checked against.

Sources of divergence, in order of size:

1. **The source differs, and Docker cannot fix that.** The shipped runs have a growth floor of
   0.5; `one_way_fsg_model/solid_solver.py:51` says 0.3. That is different code, not a different
   library version. Only Dan can say which is correct.
2. **The mesh.** A different gmsh version can produce a different mesh from the same geometry
   spec, which moves everything downstream more than solver settings do.
3. **The nonlinear solve.** PETSc/SLEPc versions and their solver and preconditioner defaults
   land at different points inside the same tolerance ball, and 15 growth steps compound it.

The risk is not wrong output. It is **silent incomparability**: if new runs differ from old ones
by more than the effect size a hypothesis tests, claims spanning both families measure the solver
rather than the biology, and nothing errors to tell you.

## Scope when unblocked — the reproduction test only

Do **not** build LLM-proposed FSG runs in this task. That is TASK-021, and it should not be
written until this task produces a number.

1. Build a Docker image for legacy FEniCS from Dan's env export. Pin gmsh explicitly.
2. Re-run `flow_U0p0180` — measured at 6010 s for 15 steps on Dan's machine, so budget several
   hours, more if the image is amd64 under emulation on arm64.
3. Diff against the shipped `arc_data.npz` and `g_field.npy` for every step.
4. **Report the max relative deviation** in the fields hypotheses are actually built on, per
   step. That number is the deliverable.

## Acceptance criteria

1. The deviation number, pasted, per field and per step.
2. A written verdict with the number as evidence: either new runs are comparable to old ones and
   may be mixed freely, or they are a separate provenance family and cross-family comparisons must
   be refused. If the latter, open a follow-up task for the harness guard — do not implement it
   here.
3. The image is reproducible from a committed Dockerfile plus the env file, and the exact build
   and run commands are in `../INSTRUCTIONS.md`.
4. `one_way_fsg_model/` is unmodified. Confirm with `git status` on that directory (on Ethan's
   Mac, where `git status` is trustworthy).

## Log

_(append below)_
