# PINN surrogate for the cushion fluid model — plan (v1, steady state)

**Written:** 2026-10-14. **Target:** a trained first version plus its ablation by ~2026-10-21.
**Branch:** `feat/pinn-surrogate-plan` (from `feat/qa-compare` @ `649945b`).
**Data:** Dan's `steady_state_dataset/dataset_v1_1k/` (its `README.md` is canonical for fields).

## Why this exists

The long-run goal is for LLM agents to *run their own experiments* on the FSG model rather than
only query three precomputed runs. A full FSG run costs ~100 min today. The eventual FSG model
will use **transient** (heartbeat-driven) fluid, where the fluid solve becomes the dominant cost.
This steady-state surrogate is the **proof of method** before the transient version: can a
physics-informed network predict the wall loads on cushion shapes it has never seen?

Measured context (2026-10-14):
- FSG log (`flow_U0p0180`): ~340 s per FSG iteration = 1 steady NS solve + ~11 solid/growth
  solves. Dan's steady solves in this dataset take ~1–3.5 s each, so today the steady fluid is
  roughly 1% of an FSG run (inferred — the log does not time the modules separately). A steady
  surrogate alone does **not** speed up today's FSG runs; it proves the method for transient.
- The only thing the FSG loop takes from the fluid is **wall pressure and wall shear along the
  cushion surface** (`traction_mapper.py`: traction = −p·n + τ). The GRN also only uses WSS.

## Decisions already made

| Decision | Choice | Why |
|---|---|---|
| Location | `pinn_surrogate/`, next to `llm_insights/`, own venv + `pyproject.toml` | Keeps torch out of the harness env; Dan's model stays read-only |
| PINN style | Conditioned PINN, BioPINN-LM pattern (`RESOURCES.md` §1) | The paper Dan sent; Dan's dataset ships the derivatives/residuals this needs |
| Network | f(x, y, shape code, Re) → (u, v, p) | One network across all shapes and flows |
| No-slip | **Hard-coded**: velocity = (distance-to-wall factor) × network output | Walls exact by construction; one less loss term to balance |
| Shape encoding (v1) | 256-point cushion outline (`arc_n256.npy`) → small MLP encoder | Closest to their spherical-harmonics vector; SDF/CNN is a v2 option |
| Scoring target | Wall p and WSS along the arc, vs `arc_k.npz` `p_256`, `tau_t_256` | That is the FSG interface; full-field error is secondary |
| Framework | PyTorch | Needs 2nd derivatives through autograd |
| Held-out test | **Family B** (175 real FSG-derived shapes, never trained on) + ~10% random held-out geometries from A/C/D/E | B is the realistic distribution; the random split measures in-distribution error |

### Nondimensionalization (fixed now, because it is the #1 cause of PINNs not training)

Lengths ÷ H (0.4 mm), velocity ÷ mean inlet speed Ū = (2/3)·U_max, pressure and shear ÷ μŪ/H
(viscous scale). Momentum then reads `Re (u·∇)u = −∇p + ∇²u`, which stays well-behaved as
Re → 0 (the dataset goes down to Re = 0.16). Wall shear in these units is O(1–50) across the whole
dataset instead of spanning ~100× in Pa.

## Data facts (measured 2026-10-14)

- 1,000 geometries × 4 flows = 4,000 solves; **all 4,000 converged** and have `derived_k.npz`.
- 20.8k–23.5k mesh nodes per geometry (median 21.6k); 2-D domain x ∈ [−4, 4] mm, y ∈ [0, 0.4] mm.
- Boundary tags in `mesh.npz` `line_tags` match Dan's FSG convention: 10 inlet, 11 outlet,
  12 top wall, 13 cushion arc, 14 bottom wall.
- FEM trust (`r_rel`, one geometry at Re 44): median 6e-5 in the bulk, **4e-3 median / 5e-2 p95
  within 5 µm of the cushion.** The "ground truth" is least accurate exactly where WSS lives —
  weight the data loss by trust, and remember this sets a floor on achievable WSS error.
- `sdf.npy` covers only x ∈ [−2, 2] mm — fine for v2, irrelevant for the outline encoder.
- On disk: 12 GB unzipped + 12 GB zip. Training uses a subsample, packed once (Stage 1).

## Stages

Each stage ends with a check whose command and real output get pasted into `LOG.md` — no
asserted numbers (same rule as `llm_insights/TASKS/`).

### Stage 0 — Setup (Ethan + agent, ~½ day)
- Add `steady_state_dataset/` to the root `.gitignore` (**done in this commit**).
- `pinn_surrogate/` skeleton: `pyproject.toml`, `src/pinn_surrogate/`, `tests/`,
  `INSTRUCTIONS.md` with every run command, `LOG.md`.
- **Ethan installs torch** (blocking; an agent cannot):
  ```bash
  cd pinn_surrogate
  python3 -m venv .venv && source .venv/bin/activate
  pip install torch numpy scipy matplotlib pytest
  python3 -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
  ```
- **Check:** that last line prints a version and `True` (Apple GPU usable).

### Stage 1 — Data pipeline (~1 day)
- `data.py`: per (geometry, flow) load nodes, fields, trust weights, outline, Re; nondimensionalize.
- Pack a compact training set once: per solve keep ~4k nodes, **oversampled near the cushion**
  (by `dist_arc`), plus all boundary nodes by tag, plus the 256-point wall targets.
- Write `splits.json` (family B + random 10% held out), fixed seed.
- **Checks:** (a) round trip — nondim → dim reproduces Dan's values exactly; (b) Dan's own data is
  self-consistent — WSS recomputed from his saved velocity gradients at the wall matches his
  `arc_k.npz` `tau_t` (if it doesn't, we learn which WSS definition to score against before
  training anything).

### Stage 2 — Physics residual, validated before any network sees it (~½ day)
- `physics.py`: the nondimensional NS + continuity residual in torch.
- **Check:** feed it Dan's FEM fields and derivatives (`u_x`, `lap_u`, `p_x`, …) instead of a
  network; it must reproduce Dan's `r_mom_x`, `r_mom_y`, `r_div` to tolerance. This is the
  project rule — test each layer against the thing it replaces — applied to the loss itself.

### Stage 3 — Textbook PINN on one case (~1 day)
- One geometry, one flow, physics + inlet/outlet terms only, **no data**. The classic PINN.
- **Check:** wall WSS along the arc vs `tau_t_256`, relative L2 error reported. Also measure
  seconds/step on the Mac — this number decides whether Stage 4 needs a GPU.
- Purpose: proves the loss machinery works before adding the hard part (generalization).

### Stage 4 — Conditioned PINN across shapes (~2 days)
- Add shape encoder + Re input. Loss = data (trust-weighted) + PDE residual + inlet profile +
  outlet, with adaptive loss reweighting (as in BioPINN-LM / Wang et al.).
- Start on ~200 geometries × 4 flows; scale up only if Stage 3 timings allow.
- **Check:** wall p and WSS error on the held-out sets.

### Stage 5 — The ablation (the headline result) (~1 day)
- Same network, same budget, **physics term off** (pure data fit). This arm is also the
  "baseline the PINN must beat".
- Sweep training-set size (e.g. 50 / 200 / 800 geometries): physics priors usually help most
  when data is scarce, and that curve is the result worth showing.
- **Check:** table of PINN vs data-only, per held-out set, per training-set size.

### Stage 6 — Evaluation report for Dan (~½ day)
Metrics, all on held-out shapes, broken out by family and by Re:
- relative L2 error of wall WSS and pressure along the arc (the FSG interface);
- **location of peak WSS** (arc-length error) — what the biology responds to;
- recirculation detection: does the sign of WSS flip in the same places as Dan's;
- NS residual of the prediction (physical plausibility, as in the point-cloud paper in §4);
- inference time per (shape, flow) vs Dan's ~1–3.5 s solve.

## Out of scope this week (next phases)

1. **Quasi-steady check.** Rough Womersley number here is ≈0.4 (H/2 = 0.2 mm, ~2 Hz heart rate),
   below the ~1 where flow lags the heartbeat — so a heartbeat might be approximable by
   evaluating the steady surrogate along the inflow waveform. Needs a few real transient runs
   from Dan to test.
2. **Transient surrogate** — needs transient training data from Dan. If the growth law uses
   time-averaged wall loads, map (shape, waveform) → time-averaged WSS/p directly and skip time
   stepping.
3. **Closed-loop test** — swap the surrogate into the FSG loop and compare the final shape after
   15 iterations (errors compound). Needs Dan's FEniCS environment.
4. **Harness integration** — a "fluid-only" experiment tier in `llm_insights/` so agents can test
   claims on any shape/flow in milliseconds.
5. **Solid/growth surrogate** — the solid side caps how fast a full FSG run can get.

## Risks

- **Loss balancing** (data vs physics vs boundary) is where PINNs usually stall → adaptive
  weights, and Stage 3 shakes this out on one case first.
- **Near-wall accuracy:** WSS is a gradient at the wall, where Dan's FEM is least trustworthy.
- **High-Re cases** (recirculation, Re up to 65) will likely be worst; report error vs Re.
- **Compute:** second derivatives across thousands of shapes may be slow on a MacBook. If
  Stage 3 timings say so, Stage 4–5 move to a GPU (BioHPC / Colab) — undecided.

## Questions for Dan

1. Will the eventual FSG model run transient fluid *inside* the growth loop?
2. Does the growth law use time-averaged wall loads, or the full in-cycle history?
3. Is a quasi-steady approximation known to fail here (e.g. at high flow / recirculation)?
4. Which WSS definition should we score against — `tau_t` (signed tangential) or `tau_mag`?
5. Can he produce a handful of transient runs to validate the quasi-steady idea?

## Open decisions for Ethan

- GPU access this week? (Decides Stage 4–5 scale.)
- Shape encoding for v2: SDF image + CNN, or stay with the outline?
