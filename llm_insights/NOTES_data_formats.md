# NOTES_data_formats.md

What is actually readable out of `../one_way_fsg_model/FSG Results/` without h5py, where each
quantity lives, and the traps. Established 2026-09-01 by reading every `.py` in the model and
verifying against the shipped arrays. Written so this does not have to be rediscovered.

`RESOURCES.md` §6 covers the pipeline architecture. This file covers the **data on disk**.

---

## Case mapping

| folder | condition | inlet | source |
|---|---|---|---|
| `flow_U0p0180` | Underflow | 1.8 cm/s | `README.txt:38-42`, `build_grn_inputs.py:67-71` |
| `flow_U0p0360` | Healthy | 3.6 cm/s | same |
| `flow_U0p0540` | Overflow | 5.4 cm/s | same |

`FSG Results/comparison_figures/metrics_summary.csv` is a machine-readable version and a useful
independent cross-check. **The script that generated it is not in the handoff**, so the meaning
of its `peak_xnorm` column is unknown — see Open questions in `TASKS.md`.

---

## Availability matrix

| quantity | all 3 cases? | all 15 steps? | file |
|---|---|---|---|
| surface WSS, pressure, traction, arc geometry | yes | yes | `step_NNN/arc_data.npz` |
| full-mesh von Mises + surface WSS | yes | **step 14 only** | `grn_inputs/downsampled_100pct/grn_input.npz` |
| per-cell growth `g` | yes | yes | `step_NNN/g_field.npy` + `cell_centroids.npy` |
| GRN activities (5 nodes × 3 scenarios) | yes | step 14 only | `AHA GRN Plots/3scenarios_results/080_{180,360,540}/results_spatial.csv` |
| per-step full stress tensor, strain, displacement, `J_e`, `J_g` | **healthy only** | yes | `extracted_fields/`, `dynamic_inputs/` |
| everything else | — | — | **`.h5` only, needs h5py** |

`extracted_fields/` and `dynamic_inputs/` exist only for `flow_U0p0360` because
`process_fsg_results.py:57` hard-codes `FLOW_CASE = "flow_U0p0360"`.

---

## Schemas

**`step_NNN/arc_data.npz`** — 10 keys, shape `(N,)`, N drifts 219-259 by case and step (it is a
mesh-facet count, not `N_ARC_PTS`). Stored in ascending `x`.

`x`, `y` (m, deformed) · `p` (Pa, gauge) · `tx`, `ty` (Pa, **full traction** `σ·n`, not WSS) ·
`tau_x`, `tau_y`, `tau_mag` (Pa, tangential = WSS) · `nx`, `ny` (unit normal, pointing into the
fluid).

`osi` is referenced in `visualize_growth.py` but **is never written** — `run_fsg.py` does not
call the transient solver. Treat OSI as unavailable.

**`grn_inputs/downsampled_100pct/grn_input.npz`** — 5 keys, 11,615 rows (the full solid mesh;
"100pct" is literal here). `x_m`, `y_m` (m, deformed) · `von_mises_Pa` (Pa, every node) ·
`wss_mag_dyn_cm2` (dyn/cm², **NaN at interior nodes**) · `is_surface` (`uint8`).

**`results_spatial.csv`** — 21 columns, ~8,900 rows (the 80% grid). `x_m`, `y_m`,
`wss_mag_dyn_cm2`, `von_mises_Pa`, `wss_norm`, `mech_norm`, then `{Snai1, Snai2, EndMT, NICD,
YAP_TAZ} × {_shear_only, _mech_only, _combined}`. The other **18 of the 23 GRN nodes are
simulated and discarded**.

**`g_field.npy`** — `(22769,)` per cell, paired with `cell_centroids.npy` `(22769, 2)` in the
**reference** frame. Growth is the only per-cell quantity; everything else is per-node and
deformed, so the two are not co-registered.

---

## Traps

1. **Arc ordering is by angle, not by x.** `visualize_growth.py:62-71` sorts by descending polar
   angle about the base-line centroid. A plain `argsort(x)` disagrees on the underflow case at
   **every step** (up to 13 of 257 points) because the cushion leans and x stops being monotonic.
   `io/dataset.arc_order_and_s` reproduces the angle sort; reuse it, never reimplement it.
2. **`step_k/arc_data.npz` is the solid geometry at the end of step k-1.** Verified numerically
   to 5 significant figures. `step_000` is the pristine undeformed cap (height 0.250 mm in all
   three cases) and corresponds to no solid state.
3. **WSS has three names and two units.** `tau_mag` (Pa) in `arc_data.npz` becomes
   `wss_mag_dyn_cm2` (×10) downstream. `wss_magnitude` appears in `extract_fsg_fields.py` but
   that script's output is not on disk.
4. **`tx`/`ty` are the full traction, not shear.** `tx` reaches 35 Pa where `tau_x` peaks at
   4.4 Pa. Grabbing the wrong one is an easy and large error.
5. **Negative values in non-negative fields.** `von_mises_Pa` has min −0.82 Pa on one node
   (CG1 projection undershoot). `io/dataset` clips at 0 and records the raw minimum.
6. **Two "100pct" folders are different node sets.** `grn_inputs/downsampled_100pct` has 11,615
   rows; `extracted_fields/downsampled_100pct` has 11,094 — grid decimation runs even at
   fraction 1.0.
7. **Surface band membership is unstable.** It is a KDTree distance threshold recomputed per
   step from the deformed mesh, so the count drifts (632→736 within one case) and differs across
   cases. Arc-based summaries avoid it.
8. **`g_field` floor is 0.5; `solid_solver.py:51` default is 0.3.** The shipped runs were
   produced with a floor the current source no longer has, so they are **not reproducible** from
   the code as it stands. Raised in TASKS.md Open questions.
9. **`plot_grn_hypothesis_trends.py` crashes as shipped** — it reads
   `AHA GRN Plots/comparison_figures/metrics_summary.csv`, but the file lives under
   `FSG Results/`. Do not fix it; it is Dan's file.

---

## Saturation, measured

`networkpoint.py:112-113` clips both GRN inputs: `SHEAR_MAX = 30.0` dyn/cm², `MECH_MAX = 100.0`
Pa. Measured exceedance:

| case | von Mises > 100 Pa (full mesh, 11,615) | on 80% grid | peak vM |
|---|---|---|---|
| Underflow | 0.02% | 0.0% | 236 Pa |
| Healthy | 33.7% | 29.6% | 597 Pa |
| Overflow | 46.3% | 44.0% | 859 Pa |

**The two denominators differ**, so "44% of the domain saturates" is grid-dependent. Any claim
about a saturated fraction has to name its grid. `summary/metrics.py` exposes both
(`mech_clipped_fraction` on the full mesh, `grn_saturated_fraction` on the 80% grid) and says so.

Every clipped node returns the identical pinned activity, which compresses spatial variance —
this is why EndMT spatial SD is **higher in Healthy (0.128) than Overflow (0.070)** despite
Overflow having the larger raw stress spread. That inversion is the demo's headline card.
