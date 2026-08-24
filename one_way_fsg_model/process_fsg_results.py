"""
process_fsg_results.py

Single entry-point for post-processing one FSG flow case.

Static mode  (DYNAMIC = False)
    Processes the final FSG step only.  Outputs land in extracted_fields/.

Dynamic mode  (DYNAMIC = True)
    Processes every step_XXX folder so the student's GRN can be driven by
    the full time-history of mechanical loading.
    Final step  → extracted_fields/          (unchanged, for reference)
    Every step  → dynamic_inputs/step_XXX/

Output structure per processed step
-------------------------------------
  wss_arc_surface.{txt,npz,json,csv}
  cauchy_stress_cushion.{txt,npz,json,csv}
  mechanical_inputs_unified.{txt,npz,json,csv}
  mechanical_inputs_trimmed.{txt,npz,json,csv}
  wss_arc_surface.png
  growth_stimulus.png
  mechanical_inputs_unified.png
  downsampled_XXXpct/
      wss_arc_surface.{txt,npz,json,csv}
      growth_stimulus.{txt,npz,json,csv}
      mechanical_inputs_unified.{txt,npz,json,csv}
      mechanical_inputs_trimmed.{txt,npz,json,csv}
      wss_arc_surface.png
      growth_stimulus.png
      mechanical_inputs_unified.png

Usage
-----
  /path/to/.venv/bin/python process_fsg_results.py
"""

import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.spatial import KDTree

try:
    import h5py
except ImportError:
    raise ImportError(
        "h5py not found.  Run from the PyCharm venv: "
        "/Users/danielpearce/PyCharmMiscProject/.venv/bin/python process_fsg_results.py"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FLOW_CASE  = "flow_U0p0360"
DYNAMIC    = True            # True  → process every step for GRN time-series
                             # False → final step only
BAND_MULT  = 2.0             # surface band thickness (multiples of element spacing h)
FRACTIONS  = [1.0, 0.8, 0.6, 0.4, 0.2]
PA_TO_DYN  = 10.0            # 1 Pa = 10 dyn/cm²

# Custom fixed-count levels: arc and solid surface/interior sampled independently.
# n_arc     → target nodes in the wss_arc_surface file
# n_surface → target solid surface-band nodes  (get WSS + stress)
# n_interior→ target solid interior nodes      (stress only)
# label     → subfolder name: downsampled_<label>/
CUSTOM_LEVELS = [
    {"label": "custom_200pt", "n_arc": 200, "n_surface": 200, "n_interior": 200},
]
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":        "Arial",
    "font.weight":        "bold",
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "figure.titleweight": "bold",
})

BASE_DIR = Path(__file__).parent / "FSG Results" / FLOW_CASE

step_dirs = sorted(d for d in BASE_DIR.iterdir()
                   if d.is_dir() and d.name.startswith("step_"))
if not step_dirs:
    raise RuntimeError(f"No step_XXX folders found in {BASE_DIR}")

FINAL_STEP = step_dirs[-1]
print(f"Flow case : {FLOW_CASE}")
print(f"Steps found: {len(step_dirs)}  (final: {FINAL_STEP.name})")
print(f"Dynamic mode: {DYNAMIC}\n")


# ===========================================================================
# Shared helpers
# ===========================================================================

UNI_HEADER = [
    "x_m", "y_m",
    "sigma_xx_Pa", "sigma_xy_Pa", "sigma_yy_Pa",
    "von_mises_Pa",
    "sigma_max_principal_Pa", "sigma_min_principal_Pa",
    "sigma_tension_Pa", "sigma_comp_mag_Pa",
    "is_surface",
    "wss_mag_dyn_cm2", "wss_x_dyn_cm2", "wss_y_dyn_cm2",
    "pressure_Pa", "normal_x", "normal_y",
]
TRIM_HEADER = ["x_m", "y_m", "von_mises_Pa", "wss_mag_dyn_cm2"]
TRIM_COLS   = [UNI_HEADER.index(c) for c in TRIM_HEADER]


def write_all(stem, header, array, meta):
    """Write .txt, .csv, .npz, and .json for one dataset."""
    np.savetxt(str(stem) + ".txt", array,
               header=" ".join(header), fmt="%.10e", comments="")

    with open(str(stem) + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in array:
            w.writerow(["nan" if np.isnan(v) else f"{v:.10e}" for v in row])

    np.savez(str(stem) + ".npz",
             **{col: array[:, i] for i, col in enumerate(header)})

    obj = {**meta, "columns": header,
           "data": [[None if np.isnan(v) else v for v in row]
                    for row in array]}
    with open(str(stem) + ".json", "w") as fh:
        json.dump(obj, fh, indent=2)


def scatter_panel(ax, x, y, values, title, unit_label, pt_size=8):
    sc = ax.scatter(x * 1e3, y * 1e3, c=values, cmap="jet", s=pt_size,
                    linewidths=0, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(unit_label, fontsize=8, fontweight="bold")
    cb.ax.tick_params(labelsize=7)
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("x (mm)", fontsize=8, fontweight="bold")
    ax.set_ylabel("y (mm)", fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=7)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_unified_figure(path, sx, sy, s_vals, surf_mask,
                        wx, wy, w_vals, suptitle, n_sol, n_arc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(suptitle, fontsize=10, fontweight="bold")
    scatter_panel(ax1, sx, sy, s_vals,
                  r"Growth Stimulus $\sigma_{\rm comp}$ — All Nodes",
                  "Pa", pt_size=max(1, int(4 * n_sol / 11615)))
    if surf_mask.any():
        ax1.scatter(sx[surf_mask] * 1e3, sy[surf_mask] * 1e3,
                    s=max(2, int(6 * n_sol / 11615)),
                    facecolors="none", edgecolors="white",
                    linewidths=0.4, rasterized=True, label="surface band")
        ax1.legend(fontsize=7, frameon=False)
    scatter_panel(ax2, wx, wy, w_vals,
                  "WSS Magnitude — Surface-Band Nodes",
                  "dyn/cm²", pt_size=max(4, int(20 * n_arc / 228)))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Per-step processing function
# ===========================================================================

def grid_downsample_coords(coords, n_target):
    """
    Grid-based spatial decimation of a 2-D (N, 2) point cloud to ~n_target points.
    Returns an index array into coords.
    """
    tree             = KDTree(coords)
    x_min, y_min     = coords.min(axis=0)
    x_max, y_max     = coords.max(axis=0)
    dx, dy           = x_max - x_min, y_max - y_min
    aspect           = dx / (dy + 1e-30)
    ny               = max(2, int(np.sqrt(n_target / aspect)))
    nx               = max(2, int(n_target / ny))
    for _ in range(20):
        xs = np.linspace(x_min + dx/(2*nx), x_max - dx/(2*nx), nx)
        ys = np.linspace(y_min + dy/(2*ny), y_max - dy/(2*ny), ny)
        gx, gy    = np.meshgrid(xs, ys)
        _, nn_idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
        selected  = np.unique(nn_idx)
        if abs(len(selected) - n_target) / n_target < 0.05:
            break
        scale = n_target / max(len(selected), 1)
        ny = max(2, int(ny * np.sqrt(scale)))
        nx = max(2, int(nx * np.sqrt(scale)))
    return selected


def process_one_step(step_dir: Path, out_dir: Path):
    """
    Load raw data from step_dir, build all output files and figures,
    write everything to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    step_label = f"{FLOW_CASE} / {step_dir.name}"

    # -----------------------------------------------------------------------
    # Load arc (WSS) data
    # -----------------------------------------------------------------------
    arc      = np.load(step_dir / "arc_data.npz")
    arc_x    = arc["x"]
    arc_y    = arc["y"]
    arc_wmag = arc["tau_mag"] * PA_TO_DYN
    arc_wx   = arc["tau_x"]   * PA_TO_DYN
    arc_wy   = arc["tau_y"]   * PA_TO_DYN
    arc_p    = arc["p"]
    arc_nx   = arc["nx"]
    arc_ny   = arc["ny"]
    N_ARC    = len(arc_x)

    # -----------------------------------------------------------------------
    # Load solid stress data (deformed coordinates)
    # -----------------------------------------------------------------------
    with h5py.File(step_dir / "solid_fields.h5", "r") as f:
        ref_geom        = f["Mesh/0/mesh/geometry"][:]
        disp            = f["VisualisationVector/0"][:]
        cauchy_9        = f["VisualisationVector/11"][:]
        von_mises       = f["VisualisationVector/1"][:].ravel()
        sigma_max_princ = f["VisualisationVector/2"][:].ravel()
        sigma_min_princ = f["VisualisationVector/3"][:].ravel()
        sigma_tension   = f["VisualisationVector/4"][:].ravel()
        sigma_comp      = f["VisualisationVector/5"][:].ravel()

    sol_x    = ref_geom[:, 0] + disp[:, 0]
    sol_y    = ref_geom[:, 1] + disp[:, 1]
    sigma_xx = cauchy_9[:, 0]
    sigma_xy = cauchy_9[:, 1]
    sigma_yy = cauchy_9[:, 4]
    N_SOL    = len(sol_x)
    sol_xy   = np.column_stack([sol_x, sol_y])

    # -----------------------------------------------------------------------
    # Identify surface-band nodes
    # -----------------------------------------------------------------------
    sol_tree         = KDTree(sol_xy)
    nn_dists, _      = sol_tree.query(sol_xy, k=2)
    h_est            = np.median(nn_dists[:, 1])
    threshold        = BAND_MULT * h_est

    arc_tree         = KDTree(np.column_stack([arc_x, arc_y]))
    d_to_arc, nn_arc = arc_tree.query(sol_xy)
    is_surface       = (d_to_arc < threshold).astype(np.uint8)
    N_SURF           = int(is_surface.sum())

    # -----------------------------------------------------------------------
    # Builders
    # -----------------------------------------------------------------------
    def build_unified(sol_idx):
        n    = len(sol_idx)
        surf = is_surface[sol_idx]
        wm   = np.full(n, np.nan);  wx_  = np.full(n, np.nan)
        wy_  = np.full(n, np.nan);  wp   = np.full(n, np.nan)
        wnx  = np.full(n, np.nan);  wny  = np.full(n, np.nan)
        s_idx = np.where(surf)[0]
        nn    = nn_arc[sol_idx[s_idx]]
        wm[s_idx]  = arc_wmag[nn];  wx_[s_idx] = arc_wx[nn]
        wy_[s_idx] = arc_wy[nn];    wp[s_idx]  = arc_p[nn]
        wnx[s_idx] = arc_nx[nn];    wny[s_idx] = arc_ny[nn]
        return np.column_stack([
            sol_x[sol_idx], sol_y[sol_idx],
            sigma_xx[sol_idx], sigma_xy[sol_idx], sigma_yy[sol_idx],
            von_mises[sol_idx],
            sigma_max_princ[sol_idx], sigma_min_princ[sol_idx],
            sigma_tension[sol_idx], sigma_comp[sol_idx],
            surf.astype(float),
            wm, wx_, wy_, wp, wnx, wny,
        ])

    def downsample_arc(fraction):
        order   = np.argsort(arc_x)
        xy_s    = np.column_stack([arc_x[order], arc_y[order]])
        arc_len = np.concatenate([[0.0],
                                  np.cumsum(np.hypot(*np.diff(xy_s, axis=0).T))])
        n_tgt   = max(2, round(fraction * N_ARC))
        idx_s   = np.clip(
            np.searchsorted(arc_len, np.linspace(0, arc_len[-1], n_tgt)),
            0, len(order) - 1)
        orig    = order[idx_s]
        _, upos = np.unique(orig, return_index=True)
        return orig[np.sort(upos)]

    def downsample_solid_grid(fraction):
        return grid_downsample_coords(sol_xy, max(4, round(fraction * N_SOL)))

    def downsample_split(n_surf_tgt, n_int_tgt):
        """Sample surface and interior solid nodes independently, return combined indices."""
        surf_global = np.where(is_surface.astype(bool))[0]
        int_global  = np.where(~is_surface.astype(bool))[0]
        surf_sel    = grid_downsample_coords(sol_xy[surf_global], min(n_surf_tgt, len(surf_global)))
        int_sel     = grid_downsample_coords(sol_xy[int_global],  min(n_int_tgt,  len(int_global)))
        return np.sort(np.concatenate([surf_global[surf_sel], int_global[int_sel]]))

    # -----------------------------------------------------------------------
    # Full-resolution: separate WSS + stress files
    # -----------------------------------------------------------------------
    wss_header = ["x_m", "y_m",
                  "wss_mag_dyn_cm2", "wss_x_dyn_cm2", "wss_y_dyn_cm2",
                  "pressure_Pa", "normal_x", "normal_y"]
    write_all(out_dir / "wss_arc_surface", wss_header,
              np.column_stack([arc_x, arc_y, arc_wmag, arc_wx, arc_wy,
                               arc_p, arc_nx, arc_ny]), {
        "description": f"WSS on cushion arc surface — {step_label}",
        "n_nodes": N_ARC,
        "units": {"coordinates": "m", "wss": "dyn/cm2", "pressure": "Pa"},
    })

    sol_header = ["x_m", "y_m",
                  "sigma_xx_Pa", "sigma_xy_Pa", "sigma_yy_Pa",
                  "von_mises_Pa",
                  "sigma_max_principal_Pa", "sigma_min_principal_Pa",
                  "sigma_tension_Pa", "sigma_comp_mag_Pa"]
    write_all(out_dir / "cauchy_stress_cushion", sol_header,
              np.column_stack([sol_x, sol_y,
                               sigma_xx, sigma_xy, sigma_yy,
                               von_mises,
                               sigma_max_princ, sigma_min_princ,
                               sigma_tension, sigma_comp]), {
        "description": f"Cauchy stress — all solid cushion nodes, {step_label}",
        "n_nodes": N_SOL,
        "units": {"coordinates": "m", "stress": "Pa"},
    })

    # -----------------------------------------------------------------------
    # Full-resolution: unified + trimmed
    # -----------------------------------------------------------------------
    all_idx  = np.arange(N_SOL)
    uni_full = build_unified(all_idx)

    write_all(out_dir / "mechanical_inputs_unified", UNI_HEADER, uni_full, {
        "description": (f"Unified GRN mechanical inputs — {step_label}. "
                        f"All solid nodes with stress; surface-band nodes carry WSS."),
        "n_nodes_total":    N_SOL,
        "n_nodes_surface":  N_SURF,
        "n_nodes_interior": N_SOL - N_SURF,
        "surface_definition": (f"Within {BAND_MULT}h = {threshold*1e3:.4f} mm of arc. "
                               f"WSS by nearest arc node."),
        "units": {"coordinates": "m", "stress": "Pa", "wss": "dyn/cm2",
                  "is_surface": "1=surface, 0=interior (WSS=NaN)"},
    })

    write_all(out_dir / "mechanical_inputs_trimmed", TRIM_HEADER,
              uni_full[:, TRIM_COLS], {
        "description": (f"Trimmed GRN inputs — {step_label}. "
                        f"x, y, von Mises (all nodes), WSS magnitude (surface only / NaN interior)."),
        "n_nodes_total":    N_SOL,
        "n_nodes_surface":  N_SURF,
        "n_nodes_interior": N_SOL - N_SURF,
        "units": {"coordinates": "m", "stress": "Pa", "wss": "dyn/cm2"},
    })

    # -----------------------------------------------------------------------
    # Full-resolution figures
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(f"WSS Magnitude — Arc/Cushion Surface\n{step_label}  |  {N_ARC} nodes",
                 fontsize=10, fontweight="bold")
    scatter_panel(ax, arc_x, arc_y, arc_wmag, "WSS Magnitude", "dyn/cm²", pt_size=20)
    fig.tight_layout()
    fig.savefig(out_dir / "wss_arc_surface.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(r"Growth Stimulus $\sigma_{\rm comp}$ — Full Cushion Domain"
                 f"\n{step_label}  |  {N_SOL} nodes",
                 fontsize=10, fontweight="bold")
    scatter_panel(ax, sol_x, sol_y, sigma_comp,
                  r"$\sigma_{\rm comp}$ (Pa)", "Pa", pt_size=4)
    fig.tight_layout()
    fig.savefig(out_dir / "growth_stimulus.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    surf_mask = is_surface.astype(bool)
    save_unified_figure(
        out_dir / "mechanical_inputs_unified.png",
        sol_x, sol_y, sigma_comp, surf_mask,
        arc_x, arc_y, arc_wmag,
        suptitle=(f"Unified Mechanical Inputs — {step_label}\n"
                  f"({N_SOL} solid nodes, {N_SURF} surface-band nodes)"),
        n_sol=N_SOL, n_arc=N_ARC,
    )

    # -----------------------------------------------------------------------
    # Downsampled outputs
    # -----------------------------------------------------------------------
    gs_header = ["x_m", "y_m", "sigma_comp_mag_Pa"]

    for frac in FRACTIONS:
        pct     = round(frac * 100)
        ds_dir  = out_dir / f"downsampled_{pct:03d}pct"
        ds_dir.mkdir(exist_ok=True)

        arc_idx = downsample_arc(frac)
        sol_idx = downsample_solid_grid(frac)
        n_a, n_s = len(arc_idx), len(sol_idx)
        n_surf_ds = int(is_surface[sol_idx].sum())

        # WSS
        write_all(ds_dir / "wss_arc_surface", wss_header,
                  np.column_stack([arc_x[arc_idx], arc_y[arc_idx],
                                   arc_wmag[arc_idx], arc_wx[arc_idx],
                                   arc_wy[arc_idx], arc_p[arc_idx],
                                   arc_nx[arc_idx], arc_ny[arc_idx]]), {
            "description": f"WSS — {step_label}, {pct}% resolution",
            "n_nodes": n_a,
            "units": {"coordinates": "m", "wss": "dyn/cm2", "pressure": "Pa"},
        })

        # Growth stimulus
        write_all(ds_dir / "growth_stimulus", gs_header,
                  np.column_stack([sol_x[sol_idx], sol_y[sol_idx],
                                   sigma_comp[sol_idx]]), {
            "description": f"Growth stimulus — {step_label}, {pct}% resolution",
            "n_nodes": n_s,
            "units": {"coordinates": "m", "stress": "Pa"},
        })

        # Unified + trimmed
        uni_ds = build_unified(sol_idx)
        write_all(ds_dir / "mechanical_inputs_unified", UNI_HEADER, uni_ds, {
            "description": (f"Unified GRN inputs — {step_label}, {pct}% resolution. "
                            f"WSS from full arc by nearest neighbour."),
            "n_nodes_total": n_s, "n_nodes_surface": n_surf_ds,
            "n_nodes_interior": n_s - n_surf_ds,
            "units": {"coordinates": "m", "stress": "Pa", "wss": "dyn/cm2"},
        })
        write_all(ds_dir / "mechanical_inputs_trimmed", TRIM_HEADER,
                  uni_ds[:, TRIM_COLS], {
            "description": (f"Trimmed GRN inputs — {step_label}, {pct}% resolution."),
            "n_nodes_total": n_s, "n_nodes_surface": n_surf_ds,
            "n_nodes_interior": n_s - n_surf_ds,
            "units": {"coordinates": "m", "stress": "Pa", "wss": "dyn/cm2"},
        })

        pt_s = max(1, int(4  * frac))
        pt_a = max(4, int(20 * frac))

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle(f"WSS Magnitude\n{step_label}  |  {pct}%  ({n_a} nodes)",
                     fontsize=10, fontweight="bold")
        scatter_panel(ax, arc_x[arc_idx], arc_y[arc_idx], arc_wmag[arc_idx],
                      "WSS Magnitude", "dyn/cm²", pt_size=pt_a)
        fig.tight_layout()
        fig.savefig(ds_dir / "wss_arc_surface.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(r"Growth Stimulus $\sigma_{\rm comp}$"
                     f"\n{step_label}  |  {pct}%  ({n_s} nodes)",
                     fontsize=10, fontweight="bold")
        scatter_panel(ax, sol_x[sol_idx], sol_y[sol_idx], sigma_comp[sol_idx],
                      r"$\sigma_{\rm comp}$ (Pa)", "Pa", pt_size=pt_s)
        fig.tight_layout()
        fig.savefig(ds_dir / "growth_stimulus.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        surf_mask_ds  = is_surface[sol_idx].astype(bool)
        surf_sol_idx  = sol_idx[surf_mask_ds]
        save_unified_figure(
            ds_dir / "mechanical_inputs_unified.png",
            sol_x[sol_idx], sol_y[sol_idx], sigma_comp[sol_idx], surf_mask_ds,
            sol_x[surf_sol_idx], sol_y[surf_sol_idx],
            arc_wmag[nn_arc[surf_sol_idx]],
            suptitle=(f"Unified Mechanical Inputs — {step_label}\n"
                      f"{pct}% resolution  ({n_s} solid nodes, {n_surf_ds} surface-band)"),
            n_sol=n_s, n_arc=n_a,
        )

    # -----------------------------------------------------------------------
    # Custom fixed-count levels (surface/interior sampled independently)
    # -----------------------------------------------------------------------
    for level in CUSTOM_LEVELS:
        lbl    = level["label"]
        ds_dir = out_dir / f"downsampled_{lbl}"
        ds_dir.mkdir(exist_ok=True)

        arc_idx = downsample_arc(level["n_arc"] / N_ARC)
        sol_idx = downsample_split(level["n_surface"], level["n_interior"])
        n_a, n_s      = len(arc_idx), len(sol_idx)
        n_surf_ds     = int(is_surface[sol_idx].sum())
        n_int_ds      = n_s - n_surf_ds

        write_all(ds_dir / "wss_arc_surface", wss_header,
                  np.column_stack([arc_x[arc_idx], arc_y[arc_idx],
                                   arc_wmag[arc_idx], arc_wx[arc_idx],
                                   arc_wy[arc_idx], arc_p[arc_idx],
                                   arc_nx[arc_idx], arc_ny[arc_idx]]), {
            "description": f"WSS — {step_label}, {lbl}",
            "n_nodes": n_a,
            "units": {"coordinates": "m", "wss": "dyn/cm2", "pressure": "Pa"},
        })
        write_all(ds_dir / "growth_stimulus", gs_header,
                  np.column_stack([sol_x[sol_idx], sol_y[sol_idx],
                                   sigma_comp[sol_idx]]), {
            "description": f"Growth stimulus — {step_label}, {lbl}",
            "n_nodes": n_s,
            "units": {"coordinates": "m", "stress": "Pa"},
        })
        uni_ds = build_unified(sol_idx)
        write_all(ds_dir / "mechanical_inputs_unified", UNI_HEADER, uni_ds, {
            "description": (f"Unified GRN inputs — {step_label}, {lbl}. "
                            f"{n_surf_ds} surface + {n_int_ds} interior solid nodes."),
            "n_nodes_total": n_s, "n_nodes_surface": n_surf_ds,
            "n_nodes_interior": n_int_ds,
            "units": {"coordinates": "m", "stress": "Pa", "wss": "dyn/cm2"},
        })
        write_all(ds_dir / "mechanical_inputs_trimmed", TRIM_HEADER,
                  uni_ds[:, TRIM_COLS], {
            "description": f"Trimmed GRN inputs — {step_label}, {lbl}.",
            "n_nodes_total": n_s, "n_nodes_surface": n_surf_ds,
            "n_nodes_interior": n_int_ds,
            "units": {"coordinates": "m", "stress": "Pa", "wss": "dyn/cm2"},
        })

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle(f"WSS Magnitude\n{step_label}  |  {lbl}  ({n_a} arc nodes)",
                     fontsize=10, fontweight="bold")
        scatter_panel(ax, arc_x[arc_idx], arc_y[arc_idx], arc_wmag[arc_idx],
                      "WSS Magnitude", "dyn/cm²", pt_size=12)
        fig.tight_layout()
        fig.savefig(ds_dir / "wss_arc_surface.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(r"Growth Stimulus $\sigma_{\rm comp}$"
                     f"\n{step_label}  |  {lbl}  ({n_s} solid nodes)",
                     fontsize=10, fontweight="bold")
        scatter_panel(ax, sol_x[sol_idx], sol_y[sol_idx], sigma_comp[sol_idx],
                      r"$\sigma_{\rm comp}$ (Pa)", "Pa", pt_size=6)
        fig.tight_layout()
        fig.savefig(ds_dir / "growth_stimulus.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        surf_mask_ds = is_surface[sol_idx].astype(bool)
        surf_sol_idx = sol_idx[surf_mask_ds]
        save_unified_figure(
            ds_dir / "mechanical_inputs_unified.png",
            sol_x[sol_idx], sol_y[sol_idx], sigma_comp[sol_idx], surf_mask_ds,
            sol_x[surf_sol_idx], sol_y[surf_sol_idx],
            arc_wmag[nn_arc[surf_sol_idx]],
            suptitle=(f"Unified Mechanical Inputs — {step_label}  |  {lbl}\n"
                      f"({n_surf_ds} surface + {n_int_ds} interior solid nodes)"),
            n_sol=n_s, n_arc=n_a,
        )

    print(f"  {step_dir.name}  →  {N_SOL} nodes, {N_SURF} surface  "
          f"| h={h_est*1e3:.4f} mm, threshold={threshold*1e3:.4f} mm")


# ===========================================================================
# Run
# ===========================================================================

# Final step → extracted_fields/ (always produced)
extracted = BASE_DIR / "extracted_fields"
print("--- Final step ---")
process_one_step(FINAL_STEP, extracted)
print(f"  Wrote to: extracted_fields/\n")

# All steps → dynamic_inputs/step_XXX/ (when DYNAMIC = True)
if DYNAMIC:
    dyn_base = BASE_DIR / "dynamic_inputs"
    print(f"--- Dynamic mode: {len(step_dirs)} steps ---")
    for step_dir in step_dirs:
        process_one_step(step_dir, dyn_base / step_dir.name)
    print(f"\nDynamic outputs in: dynamic_inputs/")

print("\nAll done.")
