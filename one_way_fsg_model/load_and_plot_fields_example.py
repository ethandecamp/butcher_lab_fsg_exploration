#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_and_plot_fields_example.py
--------------------------------
Minimal, self-contained example showing how to load and plot the mechanical
and growth fields written by run_fsg.py, straight from the raw per-step
output files (no FEniCS/dolfin installation required -- just numpy, h5py,
and matplotlib).

Run it as-is:
    python3 load_and_plot_fields_example.py

Or point it at a different flow case / step:
    python3 load_and_plot_fields_example.py --case flow_U0p0180 --step 14

Produces a 4-panel figure (saved as example_fields_<case>_step<k>.png):
    1. WSS magnitude along the cushion surface (arc_data.npz)
    2. Fluid pressure along the cushion surface (arc_data.npz)
    3. von Mises stress over the full cushion, deformed shape (solid_fields.h5)
    4. Growth stretch g, per mesh cell (g_field.npy + cell_centroids.npy)

WHERE THE DATA COMES FROM
--------------------------------
Each step_XXX/ folder is one FSG (fluid-solid-growth) iteration -- see the
docstring at the top of run_fsg.py for the full algorithm. Two files matter
most for post-processing:

  arc_data.npz
      1-D arrays, one entry per node on the fluid-facing cushion surface
      (the "arc"), in the DEFORMED (current, grown) configuration:
        x, y        -- node coordinates (m)
        p           -- fluid pressure (Pa)
        tau_x/y     -- wall shear stress vector components (Pa)
        tau_mag     -- wall shear stress magnitude (Pa)
        nx, ny      -- outward surface normal
        tx, ty      -- traction vector applied to the solid

  solid_fields.h5
      FEniCS-written HDF5 file covering the FULL cushion (not just the
      surface). Two groups matter:
        Mesh/0/mesh/geometry        -- (11615, 2) REFERENCE (undeformed)
                                        node coordinates, meters
        VisualisationVector/<i>     -- field i, written in this fixed order
                                        by _write_solid_fields() in run_fsg.py:
            0  displacement            (11615, 3) m   [node]
            1  von_mises               (11615, 1) Pa  [node]
            2  sigma_max_principal     (11615, 1) Pa  [node]
            3  sigma_min_principal     (11615, 1) Pa  [node]
            4  sigma_tension           (11615, 1) Pa  [node]
            5  sigma_comp_mag          (11615, 1) Pa  [node]  <- growth stimulus
            6  J_e                     (22769, 1) --  [cell] elastic volume change
            7  J_g                     (22769, 1) --  [cell] growth volume change
            8  growth_factor_g         (22769, 1) --  [cell] (same as g_field.npy)
            9  deformation_gradient_F  (11615, 9) --  [node] 3x3 tensor, flattened
            10 green_lagrange_strain_E (11615, 9) --  [node]
            11 cauchy_stress           (11615, 9) --  [node]
      "node" fields have 11615 rows (one per mesh vertex); "cell" fields have
      22769 rows (one per triangle, matching cell_centroids.npy / g_field.npy).
      Add the reference geometry to the displacement to get the DEFORMED
      (grown + elastically loaded) shape shown in the plots below.

  g_field.npy / cell_centroids.npy (in the case root, not per-step)
      g_field.npy: per-cell isotropic growth stretch at that step, indexed
      the same way as cell_centroids.npy (element centroid x,y in the
      REFERENCE configuration). g > 1 means that element has grown.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS_ROOT = HERE / "FSG Results"

# VisualisationVector index -> (field name, units)
NODE_FIELDS = {
    1: ("von_mises",           "Pa"),
    5: ("sigma_comp_mag",      "Pa"),  # growth stimulus (compression-homeostasis law)
}


def load_step(case: str, step: int):
    """Load arc (surface) and solid (full-cushion) fields for one FSG step."""
    case_dir = RESULTS_ROOT / case
    step_dir = case_dir / f"step_{step:03d}"

    arc = np.load(step_dir / "arc_data.npz")

    with h5py.File(step_dir / "solid_fields.h5", "r") as f:
        ref_geom = f["Mesh/0/mesh/geometry"][:]         # (11615, 2) reference coords
        disp = f["VisualisationVector/0"][:, :2]         # (11615, 2) displacement
        von_mises = f["VisualisationVector/1"][:, 0]      # (11615,)
        g_cell = f["VisualisationVector/8"][:, 0]          # (22769,) growth factor

    deformed_xy = (ref_geom + disp) * 1e3   # mm, deformed configuration

    cell_centroids = np.load(case_dir / "cell_centroids.npy") * 1e3  # mm, reference

    return {
        "arc_x_mm": arc["x"] * 1e3, "arc_y_mm": arc["y"] * 1e3,
        "wss_mag": arc["tau_mag"], "pressure": arc["p"],
        "deformed_xy_mm": deformed_xy, "von_mises": von_mises,
        "cell_centroids_mm": cell_centroids, "g_cell": g_cell,
    }


def plot_fields(data: dict, case: str, step: int, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    sc = ax.scatter(data["arc_x_mm"], data["arc_y_mm"], c=data["wss_mag"],
                     cmap="viridis", s=15)
    fig.colorbar(sc, ax=ax, label="WSS magnitude (Pa)")
    ax.set_title("Wall shear stress on cushion surface")

    ax = axes[0, 1]
    sc = ax.scatter(data["arc_x_mm"], data["arc_y_mm"], c=data["pressure"],
                     cmap="coolwarm", s=15)
    fig.colorbar(sc, ax=ax, label="Fluid pressure (Pa)")
    ax.set_title("Fluid pressure on cushion surface")

    ax = axes[1, 0]
    sc = ax.scatter(data["deformed_xy_mm"][:, 0], data["deformed_xy_mm"][:, 1],
                     c=data["von_mises"], cmap="jet", s=4)
    fig.colorbar(sc, ax=ax, label="von Mises stress (Pa)")
    ax.set_title("von Mises stress, full cushion (deformed shape)")

    ax = axes[1, 1]
    sc = ax.scatter(data["cell_centroids_mm"][:, 0], data["cell_centroids_mm"][:, 1],
                     c=data["g_cell"], cmap="plasma", s=4)
    fig.colorbar(sc, ax=ax, label="Growth stretch g (-)")
    ax.set_title("Growth factor g, per cell (reference shape)")

    for ax in axes.flat:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")

    fig.suptitle(f"{case} -- step {step:03d}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="flow_U0p0360",
                     help="flow_U0p0180 (1.8 cm/s) | flow_U0p0360 (3.6 cm/s) | flow_U0p0540 (5.4 cm/s)")
    ap.add_argument("--step", type=int, default=14, help="FSG step index (0-14 for these runs)")
    args = ap.parse_args()

    data = load_step(args.case, args.step)
    out_path = HERE / f"example_fields_{args.case}_step{args.step:03d}.png"
    plot_fields(data, args.case, args.step, out_path)


if __name__ == "__main__":
    main()
