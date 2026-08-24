ONE-WAY FLUID-SOLID-GROWTH (FSG) MODEL — AV CUSHION
=====================================================
Handed off to: Ethan DeCamp
Purpose: mechanics + gene-regulatory-network (GRN) outputs to explore for
LLM-based hypothesis generation.


1. WHAT THIS MODEL IS
----------------------
A one-way (fluid -> solid, not fully coupled) fluid-solid-growth model of a
single atrioventricular (AV) cushion in the developing heart. Each run
iterates:

  1. Build a fluid mesh from the current (possibly already-grown) cushion
     shape.
  2. Solve steady laminar Navier-Stokes in that channel at a prescribed
     inlet velocity -> wall shear stress (WSS) and pressure on the cushion
     surface.
  3. Map those fluid tractions onto the solid mesh.
  4. Solve solid mechanics (hyperelastic) under that load.
  5. Grow the solid using a compression-homeostasis growth law (elements
     under compressive stress grow; see sigma_comp_mag below), holding for
     several sub-steps to approximate ~2 days of biological growth.
  6. Extract the new deformed cushion outline and repeat.

Each repetition of steps 1-6 is one "FSG step" / "FSG iteration". The runs
in this folder used N_FSG=15 iterations, N_HOLD=10 growth sub-steps per
iteration (see run_fsg.py header for the full parameter list and algorithm
docstring).

Three inlet flow speeds were run, spanning underflow / healthy / overflow
hemodynamic conditions:

  flow_U0p0180  ->  U_inlet = 0.018 m/s = 1.8 cm/s   ("Underflow", Re ~ 2.7)
  flow_U0p0360  ->  U_inlet = 0.036 m/s = 3.6 cm/s   ("Healthy",   Re ~ 5.5)
  flow_U0p0540  ->  U_inlet = 0.054 m/s = 5.4 cm/s   ("Overflow",  Re ~ 8.2)

These three cases are the ones referenced throughout (mechanics results,
comparison figures, and the GRN scenario outputs).


2. FOLDER LAYOUT
------------------
One Way FSG Model/
|
|-- README.txt                          <- this file
|-- load_and_plot_fields_example.py     <- START HERE: quick-load + plot demo
|
|-- run_fsg.py                          <- main FSG driver (see section 4)
|-- mesh_builder.py                     <- \
|-- fluid_solver.py                     <-  } modules run_fsg.py imports
|-- solid_solver.py                     <-  }
|-- traction_mapper.py                  <- /
|-- process_fsg_results.py              <- post-processing: extracts every
|                                           step into GRN-ready tables
|-- extract_fsg_fields.py               <- post-processing: final-step only
|-- build_grn_inputs.py                 <- builds the mechanical input
|                                           tables the GRN model consumes
|-- plot_fsg_fields.py                  <- example final-step field plots
|-- visualize_growth.py                 <- animates cushion growth over
|                                           FSG iterations
|-- plot_grn_hypothesis_trends.py       <- builds the hypothesis figures
|                                           in AHA GRN Plots/hypothesis_figures
|-- networkpoint.py                     <- the GRN model itself (ODE network,
|                                           single spatial point) — see 5a
|-- networkinput.py                     <- runs networkpoint.py's model over
|                                           a full spatial field — see 5a
|
|-- FSG Results/                        <- MECHANICS output (this repo's
|   |                                      normal results root; scripts
|   |                                      above expect this exact name)
|   |-- flow_U0p0180/                   <- 1.8 cm/s run, steps 000-014
|   |-- flow_U0p0360/                   <- 3.6 cm/s run, steps 000-014
|   |-- flow_U0p0540/                   <- 5.4 cm/s run, steps 000-014
|   `-- comparison_figures/             <- cross-case summary figures +
|                                          metrics_summary.csv
|
`-- AHA GRN Plots/                      <- GRN output (gene expression)
    |-- 3scenarios_results/
    |   |-- 080_180/                    <- GRN run driven by flow_U0p0180
    |   |-- 080_360/                    <- GRN run driven by flow_U0p0360
    |   `-- 080_540/                    <- GRN run driven by flow_U0p0540
    `-- hypothesis_figures/             <- derived hypothesis-generation
                                            figures (mechanosensing axes,
                                            L-R asymmetry, saturation, etc.)


3. INSIDE ONE MECHANICS RUN: FSG Results/flow_U0pXXXX/
----------------------------------------------------------
  fsg_log.txt          Full run log: solver settings, per-iteration timing,
                        Newton convergence, etc. Good for a run summary.
  solid.msh            Initial (undeformed) solid mesh, gmsh format.
  cell_centroids.npy   (22769, 2) reference-config centroid of every solid
                        mesh triangle, in meters. Indexes align with
                        g_field.npy inside each step folder.
  grn_inputs/           Mechanical fields resampled onto uniform grids at
                        several resolutions (100/80/60/40/20% + a fixed
                        140-node set), for feeding into the GRN model. See
                        build_grn_inputs.py docstring for column layout.
  final/                Growth-only visualization frames (growth_low.*.png)
                        plus a copy of the final solid_fields.
  step_000/ ... step_014/    One folder per FSG iteration:
      arc_data.npz         Fluid-side data on the cushion SURFACE only
                            (228 nodes), DEFORMED coordinates:
                              x, y        node position (m)
                              p           fluid pressure (Pa)
                              tau_x/y     wall shear stress vector (Pa)
                              tau_mag     WSS magnitude (Pa)
                              nx, ny      outward normal
                              tx, ty      traction applied to the solid
      g_field.npy           (22769,) per-cell growth stretch at this step
                             (>1 = grown; matches cell_centroids.npy order)
      solid_fields.h5/.xdmf Full-cushion solid mechanics fields (11615
                             nodes / 22769 cells), written by FEniCS. See
                             the docstring at the top of
                             load_and_plot_fields_example.py for the exact
                             field-name-to-index mapping (displacement,
                             von Mises, principal stresses, growth factor,
                             deformation gradient, Green-Lagrange strain,
                             Cauchy stress).
      fluid_velocity.h5/.xdmf, fluid_pressure.h5/.xdmf
                             Full fluid-domain fields for that step.
      fluid.msh              Fluid mesh used for that step's flow solve.

  NOTE ON extracted_fields / dynamic_inputs: only flow_U0p0360 has these
  (post-processed final-step and all-step exports made with
  extract_fsg_fields.py / process_fsg_results.py). The other two cases can
  be post-processed the same way by re-running those scripts with the case
  name changed — they weren't pre-run here to keep the folder smaller.


4. HOW TO LOAD AND PLOT RESULTS (NO SIMULATION SOFTWARE NEEDED)
-------------------------------------------------------------------
All the files above (.npz, .npy, .h5, .csv) are plain NumPy/HDF5/CSV — you
do NOT need FEniCS, gmsh, or any FSG-specific software installed to load
and plot them. You only need:

    numpy, scipy, h5py, matplotlib, pandas

Quick start:
    python3 load_and_plot_fields_example.py
    python3 load_and_plot_fields_example.py --case flow_U0p0180 --step 14

This loads one FSG step directly from the raw output files and plots WSS,
fluid pressure, von Mises stress, and the growth field g. Read the
docstring at the top of that script for a full explanation of every field
and how the HDF5 file is laid out — it's the best starting reference for
writing your own analysis.

Other scripts you can run/adapt once you're comfortable with the field
layout (all runnable with the same plain numpy/h5py/matplotlib/pandas
environment — none require FEniCS):
    plot_fsg_fields.py            example final-step scatter plots
                                   (currently pointed at flow_U0p0360;
                                   change STEP_DIR at the top to switch case)
    visualize_growth.py <run_dir> animates cushion growth across FSG steps,
                                   e.g.:
                                     python3 visualize_growth.py "FSG Results/flow_U0p0180"
    extract_fsg_fields.py         dumps final-step WSS + stress to
                                   .csv/.json/.npz/.txt (easier formats than
                                   raw HDF5, if you'd rather not use h5py)
    process_fsg_results.py        same idea but for EVERY step, and at
                                   multiple downsampled resolutions — this
                                   is what feeds the GRN model
    build_grn_inputs.py           regenerates grn_inputs/ from the raw
                                   mechanics output
    plot_grn_hypothesis_trends.py regenerates the hypothesis_figures/
                                   (reads AHA GRN Plots/3scenarios_results)

For full 3-D/mesh-level inspection (not just scatter plots), open the
.xdmf files in ParaView (free, https://www.paraview.org/) — it reads the
paired .h5 file automatically.


5. GENE REGULATORY NETWORK (GRN) OUTPUTS: AHA GRN Plots/
-------------------------------------------------------------
3scenarios_results/080_180, 080_360, 080_540 each hold one GRN run, driven
by the mechanical output of the correspondingly-named flow case (080 = the
80%-downsampled mechanical grid from grn_inputs/).

  results_spatial.csv   One row per spatial node. Columns:
      x_m, y_m                    node position (m)
      wss_mag_dyn_cm2             wall shear stress magnitude (dyn/cm^2)
      von_mises_Pa                tissue (von Mises) stress (Pa)
      wss_norm, mech_norm         normalized mechanical inputs [0,1] fed to
                                   the GRN
      <Gene>_shear_only           GRN node activity driven by WSS alone
      <Gene>_mech_only            GRN node activity driven by tissue
                                   stress alone
      <Gene>_combined             GRN node activity with both inputs active
    where <Gene> in {Snai1, Snai2, EndMT, NICD, YAP_TAZ} — EndMT is the
    endothelial-to-mesenchymal transition output node; the others are
    upstream regulators (Notch intracellular domain, YAP/TAZ, Snail family).

  plot_spatial_*.png     Per-node spatial maps (one image per gene x
                          condition, matching the CSV columns above).

hypothesis_figures/ (built by plot_grn_hypothesis_trends.py, see that
script's docstring for full details of each figure):
      mechanosensing_axes.png        WSS-driven vs stress-driven "axes" of
                                      gene activity, per node, across the
                                      three flow cases
      spatial_switch_profile.png     smooth mechanical input -> sharp EndMT
                                      output along the cushion
      lr_asymmetry_vs_flow.png       left- vs right-half activity vs. flow
      yap_taz_saturation.png         fraction of nodes near ceiling
                                      activity, per node, per case
      input_field_profiles.png       WSS + stress vs position, all 3 cases
      surface_expression_profiles.png gene activity vs position, all 3 cases
      grn_dashboard.png              all of the above combined in one panel

FSG Results/comparison_figures/metrics_summary.csv gives one row per flow
case with whole-cushion phenotype metrics (height_mm, length_mm, peak
stress/WSS location and magnitude, etc.) — a useful compact summary table
if you want case-level (rather than node-level) numbers to hand to an LLM.


5a. GRN MODEL SOURCE CODE: networkpoint.py + networkinput.py
------------------------------------------------------------------
These two scripts ARE the GRN model — the thing that turns mechanical
input (WSS, tissue stress) into the gene-activity columns described above.
Everything in section 5 is output that these scripts produced; use them if
you want to rerun the model, change its parameters, or drive it with a
different mechanical input than the ones already exported.

  networkpoint.py   The GRN itself: a ~23-node ODE network (ligand ->
                     receptor -> transcription factor -> output) covering
                     Notch (JAG/DLL/NICD/HEY...), Wnt (LRP/beta-catenin/
                     TCF-LEF), YAP/TAZ mechanosensing, TGFb, and BMP
                     signaling, converging on the three output nodes Snai1,
                     Snai2, and EndMT. Each node relaxes toward a Hill-
                     function "target" activation on its own timescale
                     (tau_signal / tau_tf / tau_output for fast/medium/slow
                     nodes) — see the NODES list and rhs() for the full
                     wiring, and the Params dataclass for tunable constants
                     (Hill coefficient n, EC50s, fixed ligand levels like
                     WNT/DKK/VEGF/Frizzled/Noggin_Chordin). This is the
                     natural place to test "what if this pathway were
                     stronger/weaker" hypotheses.

                     Two entry points do the actual integration:
                       simulate(shear, mech, ...)          normalized [0,1] inputs
                       simulate_physical(shear_dyn, mech_pa, ...)  physical units
                                                            (dyn/cm^2, Pa) —
                                                            matches the units
                                                            used throughout
                                                            this package
                     Both return the full time trajectory and the steady
                     -state node values.

                     Run directly (`python3 networkpoint.py`) to sanity
                     -check the network in isolation, independent of any
                     spatial mechanics: it sweeps shear alone, mech alone,
                     runs one time course, and builds a 2-D EndMT(shear,
                     mech) heatmap, saving plot_shear_sweep.png,
                     plot_mech_sweep.png, plot_timecourse.png, and
                     plot_endmt_heatmap.png into this folder.

  networkinput.py   Drives networkpoint.py's model across an ENTIRE
                     mechanical field instead of one point — this is what
                     generated AHA GRN Plots/3scenarios_results/. It reads
                     a mechanical-input CSV (x_m, y_m, wss_mag_dyn_cm2,
                     von_mises_Pa — the same schema as grn_inputs/ and
                     dynamic_inputs/step_XXX/mechanical_inputs_trimmed.csv,
                     see section 3), runs the GRN at every spatial point
                     under three scenarios (shear_only / mech_only /
                     combined, using joblib to parallelize across points),
                     and writes results_spatial.csv + plot_spatial_*.png.

                     BEFORE RUNNING: CSV_PATH near the top of the script is
                     still hardcoded to a path on the machine it was
                     written on and will not exist here. Point it at one of
                     the mechanical_inputs_trimmed.csv files already in
                     this package instead, e.g.:
                         FSG Results/flow_U0p0360/extracted_fields/mechanical_inputs_trimmed.csv
                     or any FSG Results/<case>/dynamic_inputs/step_XXX/...
                     variant if you want a different flow case or growth
                     step (or a downsampled_XXXpct/ subfolder for a coarser
                     grid). Also needs the `joblib` package, which isn't
                     required by anything else in this folder
                     (pip install joblib / conda install joblib).

                     Note: this copy of networkinput.py only computes
                     OUTPUT_NODES = [Snai1, Snai2, EndMT]. The
                     results_spatial.csv files already sitting in
                     3scenarios_results/ also have NICD and YAP_TAZ
                     columns, so those were produced by a slightly earlier
                     version of this pipeline — re-running the script as-is
                     will give you a narrower (3-node, not 5-node) CSV.
                     That's expected, not a bug; ask Daniel if you need the
                     5-node version.


6. RUNNING NEW SIMULATIONS (OPTIONAL — NOT NEEDED FOR ANALYSIS)
---------------------------------------------------------------------
Everything above works without re-running any mechanics. If you do want to
generate a new flow case yourself:

    python3 run_fsg.py --u_inlet 0.045   # e.g. a new speed, in m/s

This DOES require a FEniCS (legacy dolfin, Python 3.8) environment plus
gmsh — these runs were produced in a conda environment; ask Daniel for
environment setup help if you want to go this route. It also requires
mesh_builder.py, fluid_solver.py, solid_solver.py, and traction_mapper.py,
all included here since run_fsg.py imports them directly. A fresh run
writes into FSG Results/flow_U0p<speed>/, following the same structure
described in section 3.


7. QUESTIONS
------------
Ping Daniel (dpp48@cornell.edu) with any questions about the physics,
the field definitions, or the GRN model itself.
