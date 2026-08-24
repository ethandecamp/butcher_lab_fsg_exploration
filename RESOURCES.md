# Resources Index

A guide to `resources_for_ethan/` — background PDFs and slide decks handed off as reference
material for the LLM-based hypothesis-generation project on the AV cushion fluid-solid-growth
(FSG) + gene-regulatory-network (GRN) model. Read this instead of opening every file cold.

Entries are grouped by topic, not alphabetically. Each entry gives the file, what it is, a
summary, and takeaways aimed specifically at this project. For the FSG/GRN model code itself
(not covered here), see [`one_way_fsg_model/README.txt`](one_way_fsg_model/README.txt).

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
