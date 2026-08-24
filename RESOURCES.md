# Resources Index

A guide to `resources_for_ethan/` — background PDFs and slide decks handed off as reference
material for the LLM-based hypothesis-generation project on the AV cushion fluid-solid-growth
(FSG) + gene-regulatory-network (GRN) model. Read this instead of opening every file cold.

Entries are grouped by topic, not alphabetically. Each entry gives the file, what it is, a
summary, and takeaways aimed specifically at this project. Sections 1–5 cover the background
PDFs/slides in `resources_for_ethan/`. **Section 6 covers the `one_way_fsg_model/` code itself** —
a from-the-source-code walkthrough (pipeline architecture, module-by-module algorithms/params,
data formats, and gotchas) written after reading every `.py` file in that folder, so a future
reader doesn't have to redo that pass. `one_way_fsg_model/README.txt` is still the canonical
reference for exact field/column layouts and is excellent on its own — section 6 complements it
with implementation-level detail (algorithms, parameters, control flow, caveats) the README
doesn't go into.

**Note on the three `.pptx` decks:** they're huge (24–170MB, mostly embedded images/video) and
were summarized from programmatically extracted on-slide text only — titles, labels, and
bullets, no charts/images/video. Treat those three summaries as best-effort starting points, not
substitutes for opening the deck.

---

## 1. Core methodology: LLMs for hypothesis generation

The two most directly relevant documents in the folder — read these first.

### [`LLMs for Hypothesis Generation.pdf`](resources_for_ethan/LLMs%20for%20Hypothesis%20Generation.pdf)
**Large Language Models for Causal Hypothesis Generation in Science** — perspective paper,
*Machine Learning: Science and Technology* 6 (2025), Cohrs et al., 20pp.

*Summary:* Frames LLMs as "imperfect experts" to be queried, characterized, and combined with
data and domain experts along a spectrum — not oracles to be trusted outright, and not dismissed
just because they can't "truly" reason causally. Reviews existing LLM-causality approaches
(pairwise relations, full-graph elicitation, LLM-as-prior, LLM-as-post-hoc-corrector, causal
agents), proposes four axes for characterizing an LLM expert (reliability, consistency,
uncertainty, content-vs-reasoning), and demonstrates a hybrid LLM+data causal-discovery method
("chatPC") on a real causal-graph benchmark.

*Takeaways:*
- Treat the LLM as one input alongside domain experts and data-driven discovery, not a
  replacement for either — maps directly onto combining LLM hypotheses with the GRN/FSG model's
  own mechanistic output.
- "chatPC" (github.com/IPL-UV/CHG_LLM): LLM-answered conditional-independence queries plugged
  into the PC algorithm's edge-pruning steps — a concrete recipe for building a candidate GRN
  from literature knowledge plus whatever data exists.
- Practical mitigations for LLM noise: query multiple times and aggregate by vote (don't trust a
  single answer), and don't trust self-reported confidence — it doesn't track actual reliability.
- Evaluation template: benchmark generated graphs against a known ground-truth DAG (e.g. the
  Sachs et al. consensus network) using structural/interventional distance metrics (via the
  `gadjid` package) — directly reusable for validating LLM-proposed GRN edges.
- Named pitfalls to design around: hallucination (worse for vague prompts), training-cutoff blind
  spots, and RAG as a mitigation — worth grounding hypotheses in retrieved EndMT/valve literature.

### [`Language Model and Morphogenesis.pdf`](resources_for_ethan/Language%20Model%20and%20Morphogenesis.pdf)
**A Transformer-Based Language Model Reveals Developmental Constraint and Network Complexity
During Zebrafish Embryogenesis** — *PNAS Nexus*, June 2026, Poyatos, 9pp.

*Summary:* Trains "Zebraformer," a small BERT-style transformer, on single-cell RNA-seq from a
zebrafish developmental atlas — each cell is a "sentence," each gene a "token," expression rank
plays the role of word order. With no developmental labels, the model's embeddings recapitulate
known biology, and its attention matrices, reinterpreted as data-driven GRNs, are used to test
the "developmental hourglass" hypothesis: mid-embryogenesis shows a transient, stage-specific
drop in GRN cross-module connectivity and hub dominance, without a corresponding rise in
embedding-level fragility.

*Takeaways:*
- A direct methodological template for turning single-cell/expression data into a masked-LM
  transformer with no labels needed — transferable if EndMT/cushion single-cell data exists.
- Concrete recipe for extracting a GRN-like structure from attention weights post hoc (top-5
  outgoing edges per gene, normalized, sparsified) — comparable/complementary to this project's
  GRN component.
- Demonstrates testing a developmental-stage hypothesis quantitatively via graph metrics computed
  on the LLM-derived network over time — same pattern could ask whether EndMT onset shows an
  analogous transient GRN reorganization.
- Small-scale and tractable: ~10M params, 2 transformer layers, trainable on a single GPU — this
  approach doesn't require large compute to replicate for a heart-valve dataset.
- Code/data: github.com/juanfpoyatos/Zebraformer, zenodo.org/records/18559841.

---

## 2. EndMT / valve mechanobiology (core biology background)

### [`Computational Model of EMT.pdf`](resources_for_ethan/Computational%20Model%20of%20EMT.pdf)
**A Computational Model of the Endothelial to Mesenchymal Transition** — *Frontiers in Genetics*,
2020, Weinstein, Mendoza & Álvarez-Buylla, 26pp.

*Summary:* A literature-curated Boolean network model (29 molecules, 77 interactions —
GATA2/FLI1/ETS1/SNAI1/SNAI2/TWIST1/ZEB1/ZEB2/LEF1 plus VEGF/HIF1/NOTCH/TGF/WNT/PDGF signaling)
of the GRN controlling endothelial cell identity (Phalanx/Tip/Stalk) versus EndMT/mesenchymal
identity. Finds 444 attractors, validates against 58 known mutant phenotypes (63.8% recovered),
and generates novel predictions, including that Tip cells cannot transition directly to
mesenchymal cells — they must pass through a Stalk/non-Tip state first.

*Takeaways:*
- A fully specified, ready-to-use GRN module for EC identity/EndMT (code at
  github.com/NathanWeinstein/EndMT) — a natural component to couple to or compare against the
  FSG model's cell-fate logic.
- Concrete trigger conditions for EndMT: loss of FLI1 and GATA2, absent VEGFA, sufficient oxygen
  (no HIF1α), with SNAI1/SNAI2/TWIST1/ZEB1/ZEB2 active — a strong prior for AV-cushion EndMT
  hypothesis generation.
- Confirms EndMT in valve formation is driven by TGF/WNT/NOTCH and inhibited by VEGF, and that
  shear stress gates EndMT via KLF2/KLF4/ERK5 — a plausible bridge point between the FSG model's
  mechanical side and the GRN side.
- Explicit limitation the authors flag: synchronous Boolean update, no continuous dynamics, no
  cell geometry/mechanics — exactly the gap an FSG+GRN hybrid model is positioned to fill.
- Table 1 is a ready-made literature citation index keyed to each regulatory interaction — useful
  as a citation seed set.

### [`Shear From the Butcher Lab - Stress Regulates Growth Through Biological Signaling.pdf`](resources_for_ethan/Shear%20From%20the%20Butcher%20Lab%20-%20Stress%20Regulates%20Growth%20Through%20Biological%20Signaling.pdf)
**Shear and Hydrostatic Stress Regulate Fetal Heart Valve Remodeling through YAP-Mediated
Mechanotransduction** — *eLife* 2023, Wang et al. (Butcher Lab), 15pp.

*Summary:* Shows YAP acts as the central mechanosensor converting hemodynamic force into
cell-fate/growth decisions in developing valves. In endothelial cells (VEC), oscillatory shear
stress activates YAP while unidirectional shear stress suppresses it; in interstitial cells
(VIC), compressive stress activates YAP (→ proliferation/growth) while tensile stress deactivates
it (→ compaction). In vivo flow/pressure disruption (left atrial ligation) mis-regulates YAP and
produces valve malformation resembling hypoplastic left heart syndrome.

*Takeaways:*
- YAP/TAZ is a strong candidate hub node for the GRN model's mechanotransduction layer — largely
  independent of canonical Hippo (LATS1/2) regulation in this context.
- Explicit force-to-signal mapping to encode: VEC reads shear direction/pattern; VIC reads
  hydrostatic stress mode (compressive vs. tensile). Different cell types, different mechanical
  inputs.
- Validated transcriptional readout genes for YAP activity: THBS1, ANKRD1, PTX3 (~10-fold up at
  peak activation) — usable as measurable proxies in a GRN model.
- Proposed feedback loop (Fig. 6): local stress state → local YAP state → local growth/shape
  change → altered geometry → altered stress — a candidate rule-set for spatially-resolved
  FSG-GRN coupling.
- Directly supports the hypothesis-generation framing: altered flow (via FSG output) → YAP
  mis-regulation → predictable malformation phenotype, validated in vivo.

---

## 3. The AV cushion FSG/GRN model itself (handoff material)

### [`Daniel Pearce - AV Cushion FSG.pptx`](resources_for_ethan/Daniel%20Pearce%20-%20AV%20Cushion%20FSG.pptx)
**A Mechanogenetic Growth Simulation Framework for Spatiotemporal Predictions of Cardiovascular
Morphogenesis** — Daniel Pearce's presentation (with Sophie Wang & Jonathan T. Butcher), 12
slides. *(Extracted-text summary — open the original for figures.)*

*Summary:* This is almost certainly Daniel's own walkthrough of the exact model handed off in
`one_way_fsg_model/`. Motivated by Hypoplastic Left Heart Syndrome (HLHS — ~40% of CHD-related
neonatal deaths, >90% involving severe valve malformation, largely unexplained phenotypic
variability). Central thesis: morphometry emerges from tight coupling between geometry,
mechanics, and genetics, tracked across chick HH stages (~HH14–HH31). Model architecture is a
three-step loop — **fluid solve → solid solve → grow tissue** — iterated over ~48-hour windows.
The GRN layer maps mechanical stimuli (wall shear stress, compressive stress) through
receptor/ligand/signaling-protein/transcription-factor logic gates to remodeling behavior.
Validated against ex vivo AV cushion perfusion culture (24hr, 70 mL/min steady flow).

*Takeaways:*
- **Core model loop — fluid solve → solid solve → grow tissue — is very likely the operative
  structure of the code in `one_way_fsg_model/`.** Read this before the code.
- Central hypothesis worth interrogating: malformations like HLHS can emerge from otherwise
  *normal* growth rules combined with *abnormal* mechanical loading — mechanics, not genetics
  alone, drives divergent phenotypes.
- Key spatial mechanical variables: wall shear stress (up to ~3.6 cm/s) and compressive stress,
  mapped across chick HH stages 14–31.
- Explicitly framed as producing testable, spatially-resolved predictions meant to generate
  hypotheses — i.e. this deck is the direct predecessor of the new project's goal.
- Foundational references to chase for methods/assumptions: Rodriguez 1994, Buskohl 2012,
  Rosenbusch 2025, LaBelle 2025 (FSG methods); Saucerman 2019, Moonen 2015 (mechanosensitive
  GRN/signaling).

---

## 4. Related Butcher Lab computational-modeling projects

Surrogate modeling, ML, and growth-prediction methods from adjacent lab work — useful for
technique, not for AV-cushion biology directly.

### [`Computational Projects June 2026.pptx`](resources_for_ethan/Computational%20Projects%20June%202026.pptx)
**Potential Computational Projects – Butcher Lab (June 2026)** — 19 slides, appears to be a
recruiting/overview deck. *(Extracted-text summary.)*

*Summary:* A "map" of the lab's computational project space, not a methods document. Lists: (1)
a prior multi-scale FSI model combining morphometrics, local gene expression, and FSG modeling
for cardiac growth trajectories — closely mirrors the AV cushion handoff; (2) a Calcific Aortic
Valve Disease (CAVD) project hypothesizing that heterogeneous leaflet mechanics and stress
concentrations at calcified/healthy interfaces drive disease, via an echo-imaging-to-mechanics
pipeline; (3) RV/LVOT geometry segmentation; (4) ML/PINN surrogate models for growth and CFD.

*Takeaways:*
- Best single "map" of adjacent lab efforts in this folder — use it to see how the new project
  fits into the lab's broader computational portfolio.
- The CAVD project's core hypothesis (local mechanics → stress concentration → calcification
  signaling) is a strong structural analog for GRN-linked mechanobiology hypothesis generation,
  even though it targets a different disease/tissue.
- Confirms PINN/surrogate modeling is an active lab interest for speeding up growth/CFD
  simulations — relevant if the FSG model becomes a bottleneck for hypothesis generation at scale.
- No explicit mention of the AV cushion EndMT/GRN model — treat as adjacent context, not an
  extension of it.

### [`Real Time Surrogate Modeling for Personalized Predictions.pdf`](resources_for_ethan/Real%20Time%20Surrogate%20Modeling%20for%20Personalized%20Predictions.pdf)
**Real-Time Surrogate Modeling for Personalized Blood Flow Prediction and Hemodynamic Analysis**
— arXiv preprint, April 2026, Anagnostopoulos et al. (EPFL/Stanford/Athens), 19pp.

*Summary:* Trains a DNN surrogate (forward: parameters→pressure; inverse: parameters→cardiac
output) for a validated 1-D arterial pulse-wave network model, using a 2000-patient in silico
cohort sampled from real clinical correlations. Enables near-real-time global sensitivity
analysis and rejection of non-physiological parameter combinations, replacing ~50 CPU-days of
solver time. Key finding: cardiac output estimation from noninvasive parameters alone is
ill-posed, but becomes identifiable with one additional pressure measurement.

*Takeaways:*
- Directly transferable pattern: train a DNN surrogate for the FSG solver to get cheap forward
  predictions and fast sensitivity/parameter screening.
- Their "a priori rejection of non-physiological parameter combinations" is a concrete way to
  handle FSG/GRN parameter spaces where random sampling would waste effort on implausible cases.
- Rigorous inverse-problem identifiability analysis (Table 2) is a template for figuring out
  which GRN/FSG parameters are recoverable from limited data.
- Caution: their surrogate, trained on idealized simulated data, generalized only moderately to
  real clinical noise (CO correlation dropped from ~0.94–0.998 in silico to ~0.68 clinically) —
  a warning for any FSG/GRN surrogate trained mostly on simulation output.

### [`Surrogate CFD Model in Abdominal Aorta.pdf`](resources_for_ethan/Surrogate%20CFD%20Model%20in%20Abdominal%20Aorta.pdf)
**A Predictive Surrogate Model for Hemodynamics and Structural Prediction in Abdominal Aorta for
Different Physiological Conditions** — *Computer Methods and Programs in Biomedicine*, 2024,
12pp.

*Summary:* Combines Proper Orthogonal Decomposition (POD) with LSTM networks to accelerate
fluid-structure-interaction (FSI) simulation of an idealized abdominal aorta. A full-order FSI
model (STAR-CCM+, Mooney-Rivlin hyperelastic wall, non-Newtonian viscosity) generates
snapshot data under rest/exercise conditions; POD reduces fields to dominant spatial modes, LSTM
predicts their temporal evolution. Reconstruction/prediction errors stay below ~10% for most
variables, with pressure and solid-side quantities most accurate.

*Takeaways:*
- Concrete POD+LSTM reduced-order pipeline for cyclic FSI problems — structurally analogous to
  building a surrogate for the FSG+GRN heart-valve model's own outputs.
- Different field types (pressure vs. velocity vs. wall shear stress) need different numbers of
  POD modes — a reminder that FSG/GRN fields may need field-specific dimensionality reduction.
- Notable methodological overlap with `one_way_fsg_model/` (STAR-CCM+ conventions, Mooney-Rivlin
  wall, POD-based reduction) — worth comparing solver/BC conventions directly against that
  model's README.
- Key limitation: accuracy degrades sharply extrapolating beyond the training window — a caution
  for any FSG/GRN surrogate applied outside its training regime.

### [`Fluid Flow Predictions on Irregular Geometries.pdf`](resources_for_ethan/Fluid%20Flow%20Predictions%20on%20Irregular%20Geometries.pdf)
**A Point-Cloud Deep Learning Framework for Prediction of Fluid Flow Fields on Irregular
Geometries** — *Physics of Fluids* 33, 2021, Kashefi, Rempe & Guibas (Stanford), 28pp.

*Summary:* A PointNet-based network predicts steady 2D velocity/pressure fields directly on
unstructured mesh point clouds — no voxelization, so object boundaries are represented exactly.
Trained on flow past seven parametrized cylinder cross-sections (2595 geometries), it achieves
~1800x speedup over CFD, generalizes to unseen multi-object and airfoil configurations despite
training only on single objects, and shows approximate conservation of mass/momentum.

*Takeaways:*
- Point-cloud/PointNet-style surrogates handle irregular, boundary-exact geometry — directly
  relevant to AV cushion shapes, which are irregular and change during EndMT/remodeling, unlike
  grid-based CNN surrogates.
- Demonstrated generalization from single-object training to unseen multi-object/shape
  configurations suggests such a surrogate could plausibly generalize across evolving cushion
  morphology within an FSG loop without per-shape retraining.
- Validates surrogate accuracy via conservation-of-mass/momentum residuals, not just pointwise
  error — a good template for validating any learned replacement of the FSG fluid solver.
- Explicitly flags what it does *not* do — PINN loss terms, unsteady/moving-boundary flow — both
  gaps map directly onto the AV cushion problem (pulsatile flow, growing/deforming boundary).

### [`Shape Growth Predictions in Plans.pdf`](resources_for_ethan/Shape%20Growth%20Predictions%20in%20Plans.pdf)
**A Novel Shape-Based Plant Growth Prediction Algorithm Using Deep Learning and Spatial
Transformation** — *IEEE Access*, April 2022, Kim, Lee & Kim, 12pp.

*Summary:* Predicts a future plant image from a short past sequence by working in a binary
"shape" domain rather than raw pixels: a spatial transformer network estimates affine transform
parameters between successive silhouettes and extrapolates a future shape, then a hierarchical
patch-based autoencoder reconstructs local per-leaf detail (since different leaves grow at
different rates) onto the predicted shape.

*Takeaways:*
- Predicting growth as a small set of geometric/transform parameters rather than raw pixels/mesh
  is a useful pattern — analogous to summarizing AV cushion shape evolution as low-dimensional
  deformation parameters for an LLM to reason over.
- The paper's two-tier "global shape + local reconstruction" split mirrors the AV cushion problem,
  where different tissue regions (leaflet tip vs. base, EndMT front vs. bulk) plausibly grow at
  different rates.
- Explicitly finds that stripping background/irrelevant content before prediction outperforms
  predicting on raw imagery — a reminder to isolate the biologically meaningful field/mesh region
  before hypothesis generation.
- Pure 2D computer-vision paper with no cardiac/GRN content — relevance is purely methodological.

---

## 5. Valve leaflet imaging/tracking (adjacent lab projects)

Both from imaging pipelines tracking real (adult, diseased) valve motion — a different project
from the developmental AV cushion model, related by biomechanics methodology rather than biology.

### [`Aortic Leaflet Tracking - 6-22-2026.pptx`](resources_for_ethan/Aortic%20Leaflet%20Tracking%20-%206-22-2026.pptx)
37 slides, almost entirely image/video content (echo frames, segmentation overlays, strain maps).
*(Extracted-text summary only — open the original for the actual figures.)*

*Summary:* A U-Net-based pipeline segments aortic valve leaflets from echocardiography video,
computes strain and coaptation metrics per patient (pre-valve-replacement echo, 4 patients), and
a separate workflow feeds segmented LVOT/aorta geometry into CFD with a planned PINN surrogate
for millisecond-scale fluid predictions.

*Takeaways:*
- Could supply patient-derived mechanical loading data (strain, wall shear stress) as boundary
  conditions or validation targets for the FSG model, or as inputs for mechanotransduction
  hypothesis generation.
- A distinct, adult/diseased-valve project — not a developmental extension of the AV cushion
  model, but shares imaging-to-mechanics-to-ML-surrogate methodology worth borrowing.
- Open the original .pptx directly for the actual strain plots/segmentation results — text
  extraction captured only slide titles/labels.

### [`Mitral Leaflet Tracking - 6-22-2026.pptx`](resources_for_ethan/Mitral%20Leaflet%20Tracking%20-%206-22-2026.pptx)
45 slides, almost entirely image/video content. *(Extracted-text summary only.)*

*Summary:* Same U-Net tracking/strain pipeline applied to mitral valve leaflets, comparing
healthy subjects against mitral valve prolapse (MVP) cases. Computes Green-Lagrange strain and
finds that parietal (bottom) leaflet strain, via PCA, best separates healthy from MVP.

*Takeaways:*
- Produces quantitative healthy-vs-diseased strain/geometry data that could serve as validation
  targets or comparison data for the AV cushion FSG model.
- The healthy-vs-MVP strain/PCA findings could motivate specific phenotype axes or hypotheses for
  the GRN/FSG hypothesis-generation work.
- Same caveat as above: open the original .pptx for the actual figures — ~169MB of this file is
  image/video not captured by text extraction.

---

## 6. Code walkthrough: `one_way_fsg_model/`

This section is a source-code-level summary of every `.py` file in `one_way_fsg_model/`, not
just the README. Read `README.txt` first for the canonical field/column/folder-layout reference
(it's already excellent) — this section adds the algorithms, parameters, control flow, and
gotchas the README doesn't cover, plus a module-boundary map for anyone modularizing this into a
reusable package. Nothing in `one_way_fsg_model/` needs FEniCS/gmsh except `run_fsg.py` itself —
every downstream stage (export, GRN, plotting) only needs `numpy, scipy, h5py, matplotlib,
pandas` (+`joblib` for the GRN spatial driver).

### 6.1 Pipeline architecture

Two independent stages chained by intermediate CSV/npz files:

```
Mechanics (FSG loop)  ->  post-processing/export  ->  GRN model (ODE network)  ->  hypothesis figures
run_fsg.py + 4 modules     process_fsg_results.py      networkpoint.py +           plot_grn_hypothesis_trends.py
                            extract_fsg_fields.py       networkinput.py
                            build_grn_inputs.py
```

Each FSG iteration (`run_fsg.py`): (1) mesh the fluid channel around the current cushion shape,
(2) solve steady Navier-Stokes for wall shear stress (WSS) + pressure on the surface, (3) map
those tractions onto the solid mesh, (4) solve hyperelastic solid mechanics under that load, (5)
grow the solid via a compression-homeostasis law over several sub-steps, (6) extract the new
deformed outline and repeat. Three canonical runs ship in `FSG Results/`: `flow_U0p0180`
(1.8 cm/s, "Underflow"), `flow_U0p0360` (3.6 cm/s, "Healthy"), `flow_U0p0540` (5.4 cm/s,
"Overflow"), each `step_000`-`step_014` (production runs used `N_FSG=15`, `N_HOLD=10` — the
script's own top-of-file default is `N_FSG=48`, and the module docstring cites yet other stale
numbers; always trust the live constants/CLI args over the docstring).

### 6.2 Mechanics core modules

**`mesh_builder.py`** — all gmsh geometry/meshing plus the meshio -> dolfin `Mesh` converter.
- `build_solid_mesh` / `build_solid_mesh_from_arc`: half-ellipse or arbitrary-arc solid mesh, with
  localized corner refinement near the base to avoid slivers.
- `build_fluid_mesh`: builds the channel domain via a gmsh OCC boolean cut (channel minus cap),
  then re-identifies inlet/outlet/wall/arc boundaries by bounding-box geometry (since OCC assigns
  new curve IDs after the cut); refines near the arc and base corners.
- `msh_to_fenics`: reads the `.msh` via `meshio`, builds a dolfin `Mesh` + `cell_tags` +
  `facet_tags` by matching line-element vertex pairs to dolfin facets.
- Physical tag convention: solid `1`=base/clamped, `2`=arc/loaded, `10`=solid surface; fluid
  `10`=inlet, `11`=outlet, `12`=top wall, `13`=solid arc (no-slip + traction extraction),
  `14`=bottom wall, `20`=fluid surface.
- **In:** cap geometry params or an arc coordinate array (for remeshing), channel bounding box,
  mesh sizing (`H_FLUID, H_ARC, H_SOLID, H_CORNER`). **Out:** `.msh` file -> tagged dolfin `Mesh`.
- Gotcha: a subprocess-watchdog helper (`_worker_main`) exists for killing gmsh if it hangs on
  degenerate geometry, but `run_fsg.py` calls the mesh builders in-process — that protection isn't
  actually engaged; only a post-hoc "0 triangles" `RuntimeError` check catches self-intersection.

**`fluid_solver.py`** — steady incompressible Navier-Stokes + traction extraction.
- `solve_navier_stokes`: Taylor-Hood P2(velocity)/P1(pressure), no stabilization (comment notes
  SUPG/PSPG would be needed for Re >> 100). Weak form `rho(u.grad(u)).v dx + sigma_f(u,p):eps(v) dx
  + div(u) q dx = 0`, `sigma_f = 2*mu*eps(u) - p*I`. Newton solve via `mumps`, 50 iters, rtol 1e-8.
  BCs: parabolic inlet profile (tag 10), no-slip top/arc/bottom (12/13/14), do-nothing outlet (11).
  A `symmetry_bottom` flag exists but is never passed `True` from `run_fsg.py`.
- `_extract_arc_traction`: projects Cauchy stress, evaluates `t = sigma.n` at each arc facet
  midpoint, decomposes into pressure (`t.n`) and WSS (`|t - (t.n)n|`) components.
- **In:** tagged fluid mesh, `rho`, `mu`, inlet velocity, channel `y0/y1`. **Out:** `u_f`, `p_f`,
  and `arc_data` dict (`x,y,p,tau_x,tau_y,tau_mag,nx,ny`) sampled at the surface only.
- Also contains `solve_transient_navier_stokes` (BDF1 + Picard, periodic inlet, TAWSS/OSI) — a
  pulsatile-flow variant that is **not called by `run_fsg.py`**, kept for reuse/experimentation.

**`traction_mapper.py`** — thin bridge from CFD arrays to FEM boundary loads.
- `build_traction_expressions(arc_data)`: 1-D linear interpolation (`scipy.interpolate.interp1d`
  along arc `x`) wrapped as FEniCS `UserExpression`s `p_expr`, `wss_expr`. Used as
  `traction = -p_expr*n + wss_expr` in the solid solver's residual.
- Also has `get_deformed_arc_coords` — a simpler, less robust duplicate of the arc-extraction logic
  that actually lives in `run_fsg.get_deformed_arc`; this duplicate is **not called** anywhere.
- **In:** `arc_data` from the fluid solver. **Out:** two `UserExpression`s. Stateless.

**`solid_solver.py`** (largest/most complex module, ~1000 lines) — hyperelastic mechanics +
growth.
- Constitutive model: multiplicative growth split `F = Fe.Fg` (`Fg = sqrt(g)*I` isotropic by
  default; an anisotropic tensor variant exists but is off unless `anisotropic=True` is passed).
  Exponential (Demiray/Fung-type) strain energy
  `psi = (C/2)*(exp(alpha*(I1e-2))-1) + (1/D)*(Je-1)^2`, defaults `C_exp=200.0`, `alpha_exp=0.30`,
  `D_exp=6.0e-3`. Computes von Mises, closed-form principal stresses, tension/compression
  magnitude, Green-Lagrange strain, Jacobians. Newton solve via `mumps`, 200 iters, rtol 1e-9.
- Growth law (`_growth_dg`, compression-homeostasis, Buskohl-style): `sigma_comp = max(-sigma_min,
  0)`; setpoint `sigma_comp_home_Pa=38.0`, dead band `3.0` Pa; `dg = dt_g * k_g * stimulus /
  setpoint` with `k_g=0.007`, `dt_g=0.5`; clipped to `dg_max_step=0.05` per hold step; masked near
  the clamped base by a smoothstep height mask; result clipped to `g in [0.3, 1.3]`.
- Adaptive growth sub-stepping (`run_hold`): if applying the full growth increment fails Newton
  convergence, halves the fraction and retries down to `1/32` before raising — **this exception is
  not caught anywhere in `run_fsg.py`**, so it will crash an unattended run (unlike a fluid-solve
  failure, which is caught and just skips the iteration).
- Large surface area of alternate config (`base_bc` variants: roller/asym_roller/atrial_pin/
  yroller_spring; atrial-compaction setpoint shift; the anisotropic growth path) is implemented but
  **inert under `run_fsg.py`'s default `SolidSolver(mesh, facet_tags)` call** — only active if a
  caller explicitly passes a custom `config` dict, which the shipped runs never do.
- **In:** tagged solid mesh, material constants, traction expressions, growth-law constants, and
  *previous-iteration state* (current `u`, `g` — this solver is a stateful object, not a pure
  function, unlike the other three mechanics modules). **Out:** updated `u`, `g`, plus the full
  derived-field bundle (von Mises, principal stresses, Cauchy stress, `F`, Green-Lagrange `E`,
  `J_e`, `J_g`).

**`run_fsg.py`** — orchestrator, `python3 run_fsg.py [--u_inlet] [--n_fsg] [--n_hold]`.
- Loops the four modules above per iteration; `get_deformed_arc` (lines ~193-256) extracts the new
  boundary from the converged solid state — sorts by *reference* angle (robust to large asymmetric
  deformation, unlike sorting by deformed x), clamps `y >= CHANNEL_Y0`, refits with an exact cubic
  spline and resamples at `N_ARC_PTS=300` uniform arc-length points. This is the closure of the
  loop: its output feeds the next iteration's `build_fluid_mesh` input.
- Every `N_REMESH` iterations, `remesh_solid` rebuilds the solid mesh on the current deformed/grown
  shape as a new stress-free reference (updated-Lagrangian), resets `g` to 1 and `u` to 0, and
  rotates to a new `final_genN` XDMF; `_save_cumulative_g` recomposes growth across generations
  post-hoc via `LinearNDInterpolator`/nearest-neighbor fallback interpolation.
- Key parameters (all in the top-of-file constants, metres unless noted): `LS=1e-3`,
  `A_CAP/B_CAP=0.40/0.14*LS`, channel `CHANNEL_X0/X1=+-4*LS`, `CHANNEL_Y0/Y1=0/0.40*LS`,
  `H_FLUID=0.020*LS`, `H_ARC=0.004*LS`, `H_SOLID=0.004*LS`, `N_ARC_PTS=300`, `RHO=1060.0` kg/m^3,
  `MU=3.5e-3` Pa.s, `U_INLET=0.040` m/s default, `N_FSG=48` default, `N_RAMP_FIRST=10`,
  `N_RAMP=1`, `N_HOLD=10` default, `N_REMESH=5`, `N_RAMP_REMESH=5`, `RESULTS_ROOT="FSG Results"`
  (relative — run from `one_way_fsg_model/`).
- Output tree per case: `fsg_log.txt`, `solid.msh`, `cell_centroids.npy`,
  `solid_remesh_XXX.msh`/`cell_centroids_remesh_XXX.npy` per remesh generation, `cumulative_g.npy`,
  `final_genN/solid_fields.xdmf` (+`.h5`, time series per generation), and `step_000...step_NNN/`
  each holding `fluid.msh`, `fluid_velocity.xdmf`, `fluid_pressure.xdmf`, `arc_data.npz`,
  `solid_fields.xdmf`(+`.h5`), `g_field.npy`.
- Gotcha: a Navier-Stokes solve failure is caught and just `continue`s to the next iteration
  (prior solid/arc state carried forward, but the `step_XXX` folder still gets created without
  fluid/solid outputs for that step) — downstream consumers should tolerate gaps in `step_XXX/`.

### 6.3 Post-processing / export scripts (mechanics -> portable tables)

Three scripts with overlapping logic (surface-band KDTree detection, grid-decimation
downsampling) and no CLI args — all configuration is module-level constants edited by hand:

| Script | Scope | Coords used | Output |
|---|---|---|---|
| `extract_fsg_fields.py` | one case, one step (hardcoded `flow_U0p0360/step_014`) | **reference** (undeformed) — inconsistent with the other two | `extracted_fields/{wss_arc_surface,cauchy_stress_cushion}_final.{csv,json,npz,txt}` |
| `process_fsg_results.py` | one case (`FLOW_CASE` const), every step (auto-discovered), 5 downsample fractions (100/80/60/40/20%) + a custom fixed-count grid, 4 file formats | deformed (ref + displacement) | `extracted_fields/` (final step) + `dynamic_inputs/step_XXX/` (all steps), incl. `mechanical_inputs_trimmed.*` — the schema the GRN side consumes |
| `build_grn_inputs.py` | all 3 canonical cases at once, final step only (`STEP_NAME="step_014"` hardcoded, not auto-discovered) | deformed | `FSG Results/<case>/grn_inputs/{downsampled_XXXpct,custom_140node}/grn_input.*` — 5-column schema `x_m, y_m, von_mises_Pa, wss_mag_dyn_cm2, is_surface`; also a 3-case comparison PNG with a shared color scale |

Common conventions across all three: WSS converted Pa -> dyn/cm^2 via `PA_TO_DYN(_CM2) = 10.0`;
all stress/pressure fields left in Pa; surface-band detection = KDTree median nearest-neighbor
spacing x a threshold multiplier (`BAND_MULT=2.0`); downsampling = nearest-point grid decimation,
**not** interpolation/averaging. `VisualisationVector/N` in every `solid_fields.h5` means the same
thing everywhere: `0`=displacement, `1`=von Mises, `2/3`=max/min principal stress, `4`=tension,
`5`=compression magnitude, `6/7`=`J_e`/`J_g`, `8`=growth factor `g`, `9/10/11`=deformation
gradient `F` / Green-Lagrange `E` / Cauchy stress (each flattened 3x3, row-major — 2D uses
`xx=[:,0], xy=[:,1], yy=[:,4]`).

**This 3-way schema overlap is the main thing worth consolidating if refactoring** — the true
interface boundary between the FEniCS-dependent mechanics half and the pure-NumPy GRN half is
just `x_m, y_m, wss_mag_dyn_cm2, von_mises_Pa[, is_surface]`; right now three scripts each produce
a slightly different variant of it.

### 6.4 GRN model

**`networkpoint.py`** — the GRN itself: a 23-node ODE network relaxing toward Hill-function
targets, `dy/dt = (target - y) / tau`, three timescales by node role (`tau_signal=0.1h` for
ligand/receptor nodes, `tau_tf=1h` for TF/SMAD nodes, `tau_output=10h` for Snai1/Snai2/EndMT).
- `NODES` (23, in solver order): `JAG, DLL, NOTCH_receptors, NICD, MAML, RBPJ, HEY` (Notch),
  `LRP, beta_catenin, TCF_LEF` (Wnt), `YAP_TAZ` (mechanosensing), `TGFb_TypeI, TGFb_TypeII,
  TGFb_123, SMAD23, SMAD4` (TGFb), `BMP_TypeI, BMP_TypeII, BMP_2456, SMAD67, SMAD158` (BMP),
  `Snai1, Snai2, EndMT` (outputs).
- Mechanical inputs enter at exactly three/four points: `JAG`/`DLL` (shear, opposite sign), `NICD`
  (direct shear term), `klf2` (shear, gates Wnt/BMP), `YAP_TAZ` (tissue/mechanical stress — its
  sole input). `EndMT = mean(hill(Snai1), hill(Snai2))`.
- Hill function: `hill(x,n,ec50) = x^n / (ec50^n + x^n)`, clipped `x` to `[0,1]`; `AND` = product,
  `OR` = iterated probabilistic-or, `NOT(x)=1-x`.
- `Params` dataclass: `n=2.0` (Hill coefficient), `ec50=0.2` (internal), `ec50_input=0.5` (for
  shear/mech inputs specifically), fixed ligand levels `Noggin_Chordin=0.05, VEGF=0.30, DKK=0.1,
  WNT=0.2, Frizzled=0.5`, timescales as above. Normalization ranges `SHEAR_MAX=30.0` dyn/cm^2,
  `MECH_MAX=100.0` Pa.
- Entry points: `simulate(shear01, mech01, p=Params(), t_end=300.0, y0=None)` and
  `simulate_physical(shear_dyn, mech_pa, ...)` (converts via `SHEAR_MAX`/`MECH_MAX` then calls
  `simulate`) — both integrate with `scipy.integrate.solve_ivp(method="LSODA", rtol=1e-8,
  atol=1e-10)` and return `(t, y(t), steady_state)`.
- `python3 networkpoint.py` run directly sweeps shear alone, mech alone, one time course, and a
  30x30 EndMT(shear,mech) heatmap, saving 4 PNGs into the script's own directory.

**`networkinput.py`** — drives the point model across a full spatial field.
- Reads a mechanical-input CSV (`x_m, y_m, wss_mag_dyn_cm2, von_mises_Pa` — the schema from 6.3),
  normalizes by `WSS_MAX=30.0`/`MECH_MAX=100.0`, runs three scenarios per point (`shear_only`,
  `mech_only`, `combined`) in parallel via `joblib.Parallel(n_jobs=-1, backend="loky")`, writes
  `results_spatial.csv` + per-node/scenario contour plots.
- **Two things to fix before rerunning:** `CSV_PATH` (line 14) is hardcoded to a nonexistent path
  on a different collaborator's machine — repoint it at e.g.
  `FSG Results/flow_U0p0360/extracted_fields/mechanical_inputs_trimmed.csv` or any
  `dynamic_inputs/step_XXX/.../mechanical_inputs_trimmed.csv`; and `OUTPUT_NODES` (line 18) is
  currently `["Snai1", "Snai2", "EndMT"]` (3 nodes), narrower than the **5-node** output
  (`+ NICD, YAP_TAZ`) already sitting in `AHA GRN Plots/3scenarios_results/*/results_spatial.csv`
  — confirmed directly against those CSVs. This is a known, documented gap (an earlier pipeline
  version produced the 5-node files); reproducing it is a one-line fix since both extra keys
  already exist in `networkpoint.NODES`/`IDX`. Also requires `joblib` (not needed elsewhere in the
  folder).

### 6.5 Plotting/visualization scripts (leaf modules — consumers only, nothing feeds back downstream)

- **`load_and_plot_fields_example.py`** — the recommended starting point, fully self-contained
  (h5py+numpy+matplotlib only). `--case flow_U0pXXXX --step N` CLI flags. Its docstring/code has
  the authoritative `VisualisationVector` index table (reproduced in 6.3 above).
- **`plot_fsg_fields.py`** — no CLI; hardcoded to `flow_U0p0360/step_014` via `STEP_DIR`/
  `EXTRACTED` constants; depends on `extracted_fields/` already existing (i.e. run after
  `extract_fsg_fields.py`/`process_fsg_results.py`).
- **`visualize_growth.py <run_dir> [--stride] [--equal] [--no-anim] [--anchor]`** — animates
  (`growth_anim.gif`) and overlays (`growth_overlay.png`) cushion growth across all steps of a run
  (profile evolution, stress/growth glyphs, height/area trends, WSS/pressure vs. arc length, OSI).
  Default `run_dir` points at a case not present in this handoff — always pass one of the three
  shipped case paths explicitly, e.g. `"FSG Results/flow_U0p0180"`.
- **`plot_grn_hypothesis_trends.py`** — no CLI; builds the 7 figures in
  `AHA GRN Plots/hypothesis_figures/` from the 3 canonical GRN CSVs (`mechanosensing_axes`,
  `spatial_switch_profile`, `lr_asymmetry_vs_flow`, `yap_taz_saturation`/`fig_saturation_fraction`
  — note the filename/function-name mismatch, `input_field_profiles`,
  `surface_expression_profiles`, `grn_dashboard`). Hardcoded to case folders `080_180/360/540`.

### 6.6 Caveats checklist

- **Hardcoded absolute paths from other collaborators' machines** — cosmetic in most scripts
  (error-message text, docstring usage examples: `/Users/danielpearce/...`), but
  `networkinput.py`'s `CSV_PATH` (line 14, `/Users/sophiewang/...`) must be changed before that
  script will run at all.
- **Units are consistent but mixed**: WSS in dyn/cm^2 (= Pa x 10), everything else (pressure, von
  Mises, principal stresses, Cauchy components) in Pa; coordinates in meters at the mechanics
  layer, sometimes re-expressed in mm/um for plotting.
- **The GRN 3-node vs. 5-node output discrepancy** (6.4) is the one substantive documented gap
  between current code and shipped results — trivial to fix (extend `OUTPUT_NODES`) but a
  future rerun will not reproduce the existing `results_spatial.csv` files as-is otherwise.
- **No CLI args** except on `run_fsg.py`, `load_and_plot_fields_example.py`, and
  `visualize_growth.py` — every other script requires hand-editing module-level constants.
- **Exception handling is inconsistent** in `run_fsg.py`'s main loop: a fluid-solve failure is
  caught and skips the iteration; a growth-substep failure in `solid_solver.run_hold` is not
  caught and will crash an unattended run.
- **FEniCS/gmsh only needed to rerun `run_fsg.py` itself** — every other script (export, GRN,
  plotting) works on the already-baked `.h5`/`.npz`/`.csv` output with plain
  numpy/scipy/h5py/matplotlib/pandas.

### 6.7 Suggested module boundaries (if refactoring into a package)

The code is already close to this shape; the main gaps are (a) hardcoded constants that should
become function parameters, and (b) implicit object state (`SolidSolver`'s `u`/`g` fields,
`run_fsg.py`'s `arc_coords`) that should become explicit return values passed between steps.

| Module | Inputs | Outputs |
|---|---|---|
| Geometry/mesh (`mesh_builder.py`) | cap params or arc coords, channel box, mesh sizing | tagged dolfin `Mesh` |
| Fluid solver (`fluid_solver.py`) | tagged fluid mesh, `rho`/`mu`, inlet velocity | `u_f`, `p_f`, `arc_data` (surface WSS/pressure) |
| Traction mapping (`traction_mapper.py`) | `arc_data` | FEniCS `UserExpression`s for the solid load |
| Solid solver + growth (`solid_solver.py`) | tagged solid mesh, material + growth-law constants, traction exprs, **previous `(u, g)` state** | updated `(u, g)`, full stress/strain field bundle |
| Arc extraction (currently `run_fsg.get_deformed_arc`, not its own module) | deformed solid mesh + `u` | resampled arc coords -> feeds the *next* iteration's mesh step |
| Orchestrator (`run_fsg.py`) | run config (`u_inlet, n_fsg, n_hold, n_remesh`, geometry/mesh sizing) | run directory tree; ideally a pure-ish `fsg_step(arc_in, solid_state_in, config) -> (arc_out, solid_state_out, step_record)` looped and persisted by the orchestrator |
| Export (`process_fsg_results.py` + siblings, ideally merged) | raw run dir, case/step selection, downsample resolutions, surface-band threshold | portable tables at each resolution, in one canonical schema (`x_m, y_m, wss_mag_dyn_cm2, von_mises_Pa[, is_surface]`) |
| GRN point kinetics (`networkpoint.py`) | shear, mech (normalized or physical), `Params`, integration horizon | trajectory `(t, y(t))` + steady state — already a clean pure function |
| GRN spatial driver (`networkinput.py`) | mechanical-inputs table (export schema), scenario flags, `OUTPUT_NODES` | `results_spatial.csv` (per-node, per-scenario) |
| Plotting scripts | exported tables/HDF5 | PNGs/GIFs only — leaf nodes, no downstream consumers, safe to leave as-is |
