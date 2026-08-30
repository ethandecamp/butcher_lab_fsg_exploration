# llm_insights

LLM-driven hypothesis generation and verification over the AV cushion fluid-solid-growth (FSG)
+ gene-regulatory-network (GRN) model in `../one_way_fsg_model/`.

## What this is

Dan's FSG model produces raw fields — velocity, wall shear stress, von Mises stress, growth
factor, and 23 GRN node activities across ~9k spatial nodes, 15 timesteps, 3 flow cases. The
raw output is faithful but not *legible*: it is hard for a researcher to look at it and come
away with a claim they can defend.

This package builds an agent that proposes qualitative claims about that model and then
**falsifies them by running the model**, rather than by asking another language model whether
they sound right. The design rule is that a hypothesis is admissible only if it can be compiled
into an executable test with a decision rule stated *before* the test runs. The agent never
grades its own work.

The FSG model is treated as ground truth. We are not validating biology; we are extracting
defensible statements about a simulator.

## Layout

```
llm_insights/
├── CLAUDE.md          <- agent-facing rules. Read first.
├── STYLE_GUIDE.md     <- code conventions. Non-negotiable.
├── TASKS.md           <- the work queue and the inter-agent comms channel.
├── pyproject.toml     <- ruff + pytest config, dependencies.
├── data/              <- generated artifacts (gitignored).
├── src/llm_insights/
│   └── grn_surface/   <- precomputed GRN response surface + lookup.
└── tests/
```

## Setup

```bash
cd llm_insights
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Verify your environment

The GRN model needs only `numpy` + `scipy`. The mechanics solver (`run_fsg.py`) needs FEniCS
and gmsh, which are **not** required for anything in this package:

```bash
python3 -c "import numpy, scipy, h5py, pandas; print('ok')"
```

## Relationship to `one_way_fsg_model/`

Read-only. See `CLAUDE.md` — no code in that folder may be modified.
