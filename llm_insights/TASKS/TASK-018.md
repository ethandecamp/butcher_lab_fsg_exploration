# TASK-018 — Counterfactual GRN experiments the agent designs itself

**STATUS:** TODO
**Phase:** 2 (serial, after Lane A and Lane B are merged). Branch `feat/phase2-experiments`.
**Depends on:** TASK-013, TASK-014, TASK-015.

## Context

Everything so far queries data that already exists. This is the first task where **the agent
designs an experiment that has never been run**, predicts the outcome, and then runs it. That is
out-of-sample prediction, and it is the strongest epistemic claim in the project.

The GRN layer makes it affordable: a perturbed re-solve is seconds to minutes, with no FEniCS and
no new FSG run.

## The two modes, which must not be mixed

- **Confirmatory** — the prediction is written and locked **before** the run executes.
- **Exploratory** — the run happens, and its output is new data fed back into TASK-015's loop for
  fresh insight.

Both are wanted. The failure to avoid is one run serving both roles: if the model predicts and
then mines the same run, the "prediction" is contaminated. Every proposed run is tagged
`confirmatory` or `exploratory` **before** it executes, and the tag is recorded on the card.

## What to build

- `src/llm_insights/harness/experiments.py` — new primitives registered through the existing
  `@primitive` decorator. Perturbations available to the agent:
  - node knockout (clamp a node's activity to 0 through the ODE),
  - a `Params` field change,
  - a normalization-constant change (`MECH_MAX`, `WSS_MAX`),
  each applied to a chosen case and scenario, re-solved on a stated subsample, and compared
  against the unperturbed baseline.
- Prediction locking: the hypothesis, its threshold, and a digest of both are written with a
  timestamp before the perturbed run starts; the card records both digest and timestamp.
- Perturbations run through TASK-013's runner. Subsample sizes are a parameter with a documented
  default so a single experiment stays under a minute.

## The core-stability guard

Repo rule 3 says feature work does not modify `harness/`. This task adds a file inside it, so it
carries an extra obligation: **the original six primitives must be proved unchanged.** Capture
golden outputs for all six against the real dataset **before** any edit, and assert them
afterwards in `tests/test_primitives_golden.py`. That test must not import anything this task
adds.

## Acceptance criteria

1. Golden-output test for the original six primitives passes, written before the new file exists.
   Paste both runs.
2. **A card cannot be marked confirmatory if its prediction digest postdates the run.** Enforced
   by test, not by convention. `tests/test_experiments.py`.
3. Perturbation correctness: a knockout of `EndMT` yields zero `EndMT` activity everywhere; a
   knockout of an upstream node changes downstream nodes and leaves unconnected pathways alone.
   Paste the numbers.
4. **At least one real confirmatory experiment**, end to end: the prediction, the timestamp, the
   perturbed run, the verdict. Paste it in the Log **whether it survived or was falsified.** A
   falsification here is a better result than a survival, not a worse one.
5. At least one exploratory run feeding TASK-015's loop, with the resulting report.
6. Full suite green, count pasted. Ruff clean on touched files.

## Log

_(append below)_
