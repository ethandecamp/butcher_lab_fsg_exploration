"""
build_grn_inputs.py

Builds gene-regulatory-network (GRN) input fields for the cushion, for each of
the three developmental flow cases used in the grant application:

    flow_U0p0180  — underflow
    flow_U0p0360  — healthy (nominal)
    flow_U0p0540  — overflow

For every case this script:

  1. Reads the final FSG step (step_014):
       arc_data.npz      — WSS on the fluid-side cushion surface (deformed frame)
       solid_fields.h5   — solid stress fields for all cushion nodes
  2. Builds a table over ALL cushion nodes (deformed configuration):
       - von Mises stress is filled for every node (interior + surface)
       - WSS is assigned only to surface-band nodes (within BAND_MULT x h of the
         fluid arc, h = median solid nearest-neighbour spacing); interior nodes
         get NaN for WSS.
       - is_surface flags which nodes carry a real WSS value.
  3. Writes a uniform 5-column table for the GRN:
       x_m, y_m, von_mises_Pa, wss_mag_dyn_cm2, is_surface
  4. Repeats at several spatial downsampling levels (100/80/60/40/20 %) plus a
     fixed 140-node custom level, using 2-D grid decimation over the whole
     domain, with a two-panel plot (von Mises fill | WSS on surface) each.
  5. Writes a 3-case comparison figure at full resolution.

All CSV/npz/json share identical column labels across every case and level.

Usage
-----
    /Users/danielpearce/PyCharmMiscProject/.venv/bin/python3 build_grn_inputs.py

Output (per case): FSG Results/<case>/grn_inputs/
    downsampled_100pct/grn_input.{csv,txt,npz,json,png}
    downsampled_080pct/...
    downsampled_060pct/...
    downsampled_040pct/...
    downsampled_020pct/...
    custom_140node/grn_input.{csv,txt,npz,json,png}
Top level: FSG Results/grn_inputs_comparison.png
"""

import csv
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.spatial import KDTree

plt.rcParams.update({
    "font.family":        "Arial",
    "font.weight":        "bold",
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "figure.titleweight": "bold",
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "FSG Results"

CASES = [
    # (folder,          condition label)
    ("flow_U0p0180",    "Underflow"),
    ("flow_U0p0360",    "Healthy"),
    ("flow_U0p0540",    "Overflow"),
]

STEP_NAME   = "step_014"     # final growth step (all three cases have 000-014)
BAND_MULT   = 2.0            # surface-band thickness in units of element spacing h
PA_TO_DYN   = 10.0          # 1 Pa = 10 dyn/cm^2

FRACTIONS   = [1.0, 0.8, 0.6, 0.4, 0.2]
CUSTOM_N    = 140

# Uniform column schema shared across every file and case
COLS = ["x_m", "y_m", "von_mises_Pa", "wss_mag_dyn_cm2", "is_surface"]


# ---------------------------------------------------------------------------
# Load + build the full-resolution table (all cushion nodes) for one case
# ---------------------------------------------------------------------------
def build_full_table(case_dir):
    """Return (table (N,5), coords (N,2), meta) for all cushion nodes.

    table columns follow COLS. WSS is NaN on interior nodes; von Mises is
    filled everywhere. coords are deformed (x, y) for downsampling.
    """
    step_dir = case_dir / STEP_NAME

    # --- Arc / WSS (fluid side, deformed frame) --------------------------
    arc = np.load(step_dir / "arc_data.npz")
    arc_x   = arc["x"]
    arc_y   = arc["y"]
    arc_tau = arc["tau_mag"] * PA_TO_DYN            # Pa -> dyn/cm^2

    # --- Solid stress (all cushion nodes), deformed coordinates ----------
    with h5py.File(step_dir / "solid_fields.h5", "r") as f:
        ref_geom  = f["Mesh/0/mesh/geometry"][:]       # (N, 2) reference
        disp      = f["VisualisationVector/0"][:]       # (N, 3) displacement
        von_mises = f["VisualisationVector/1"][:].ravel()   # (N,) von Mises (Pa)

    sol_x = ref_geom[:, 0] + disp[:, 0]
    sol_y = ref_geom[:, 1] + disp[:, 1]
    sol_xy = np.column_stack([sol_x, sol_y])
    n_solid = len(sol_x)

    # --- Surface-band detection: solid nodes near the arc ----------------
    sol_tree      = KDTree(sol_xy)
    nn_dists, _   = sol_tree.query(sol_xy, k=2)
    h_est         = np.median(nn_dists[:, 1])
    threshold     = BAND_MULT * h_est

    arc_xy        = np.column_stack([arc_x, arc_y])
    arc_tree      = KDTree(arc_xy)
    d_to_arc, nn_arc = arc_tree.query(sol_xy)

    is_surface = (d_to_arc < threshold)

    # --- WSS: real value on surface nodes, NaN in the interior -----------
    wss = np.full(n_solid, np.nan)
    surf_idx = np.where(is_surface)[0]
    wss[surf_idx] = arc_tau[nn_arc[surf_idx]]

    table = np.column_stack([sol_x, sol_y, von_mises, wss,
                             is_surface.astype(float)])

    meta = dict(h_mm=h_est * 1e3, threshold_mm=threshold * 1e3,
                n_solid=n_solid, n_surface=int(is_surface.sum()),
                n_arc=len(arc_x))
    return table, sol_xy, meta


# ---------------------------------------------------------------------------
# 2-D grid decimation over the whole cushion domain
# ---------------------------------------------------------------------------
def downsample_grid(coords, n_target):
    """Return indices of ~n_target nodes with spatially uniform 2-D coverage.

    Partitions the bounding box into an aspect-ratio-corrected grid and keeps
    the node nearest each grid-cell centre (KDTree). Returns all nodes when
    n_target >= N.
    """
    N = len(coords)
    n_target = max(4, min(N, round(n_target)))
    if n_target >= N:
        return np.arange(N)

    tree = KDTree(coords)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    dx = x_max - x_min
    dy = y_max - y_min
    aspect = dx / (dy + 1e-30)

    ny = max(2, int(np.sqrt(n_target / aspect)))
    nx = max(2, int(n_target / ny))

    selected = np.arange(min(n_target, N))
    for _ in range(30):
        xs = np.linspace(x_min + dx / (2 * nx), x_max - dx / (2 * nx), nx)
        ys = np.linspace(y_min + dy / (2 * ny), y_max - dy / (2 * ny), ny)
        gx, gy = np.meshgrid(xs, ys)
        _, nn_idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
        selected = np.unique(nn_idx)
        n_sel = len(selected)
        if abs(n_sel - n_target) / n_target < 0.05:
            break
        scale = n_target / max(n_sel, 1)
        ny = max(2, int(ny * np.sqrt(scale)))
        nx = max(2, int(nx * np.sqrt(scale)))
    return selected


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_level(out_dir, table, level_desc, case_folder, condition):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(table)
    n_surf = int(table[:, 4].sum())

    # .csv (NaN written literally as 'nan')
    with open(out_dir / "grn_input.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for row in table:
            w.writerow([("nan" if np.isnan(v) else v) for v in row])
    # .txt (space-delimited)
    np.savetxt(out_dir / "grn_input.txt", table,
               header=" ".join(COLS), fmt="%.10e", comments="")
    # .npz
    np.savez(out_dir / "grn_input.npz",
             x_m=table[:, 0], y_m=table[:, 1],
             von_mises_Pa=table[:, 2], wss_mag_dyn_cm2=table[:, 3],
             is_surface=table[:, 4].astype(np.uint8))
    # .json
    with open(out_dir / "grn_input.json", "w") as fh:
        json.dump({
            "description": (f"GRN mechanical input — {case_folder} ({condition}), "
                            f"all cushion nodes, {level_desc}."),
            "n_nodes": int(n),
            "n_surface": n_surf,
            "n_interior": int(n - n_surf),
            "units": {"coordinates": "m", "von_mises": "Pa",
                      "wss_mag": "dyn/cm2",
                      "is_surface": "1 = surface band (WSS present), 0 = interior (WSS = NaN)"},
            "columns": COLS,
            "notes": {
                "von_mises": "solid VisualisationVector/1 — von Mises equivalent stress, filled for every node",
                "wss_mag": "nearest arc-node WSS on surface-band nodes; NaN in the interior",
                "coordinates": "deformed configuration (reference + displacement)",
            },
            "data": [[(None if np.isnan(v) else v) for v in row]
                     for row in table],
        }, fh, indent=2)


def scatter_panel(ax, x, y, values, unit_label, pt_size, cmap="jet",
                  vmin=None, vmax=None):
    sc = ax.scatter(x * 1e3, y * 1e3, c=values, cmap=cmap, s=pt_size,
                    linewidths=0, rasterized=True, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(unit_label, fontsize=8, fontweight="bold")
    cb.ax.tick_params(labelsize=7)
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontweight("bold")
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


def plot_level(out_dir, table, case_folder, condition, level_desc,
               vm_vmax, wss_vmax):
    n = len(table)
    surf = table[:, 4] > 0.5
    n_surf = int(surf.sum())
    pt = float(np.clip(3000.0 / max(n, 1), 3, 40))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(
        f"{condition}  ({case_folder}) — GRN Cushion Inputs\n"
        f"{level_desc}  ({n} nodes, {n_surf} surface)  |  shared color scale",
        fontsize=11, fontweight="bold",
    )
    # von Mises — every node (tissue fill)
    scatter_panel(a1, table[:, 0], table[:, 1], table[:, 2], "Pa", pt,
                  vmin=0.0, vmax=vm_vmax)
    a1.set_title(r"von Mises stress $\sigma_{\rm vM}$ (Pa) — all nodes",
                 fontsize=9, fontweight="bold")
    # WSS — surface-band nodes only (interior is NaN)
    pt_surf = float(np.clip(3000.0 / max(n_surf, 1), 4, 40))
    scatter_panel(a2, table[surf, 0], table[surf, 1], table[surf, 3],
                  "dyn/cm$^2$", pt_surf, vmin=0.0, vmax=wss_vmax)
    a2.set_title("Wall shear stress (dyn/cm²) — surface band",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "grn_input.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared color-scale limits
# ---------------------------------------------------------------------------
def nice_ceil(v):
    """Round v up to a clean 1 / 2 / 2.5 / 5 x 10^k value for colorbar limits."""
    if v <= 0:
        return 1.0
    exp = np.floor(np.log10(v))
    base = 10.0 ** exp
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * base:
            return m * base
    return 10 * base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Pass 1: build every full table, gather pooled values ------------
    cases = {}     # case_folder -> (table, coords, meta, condition)
    vm_pool, wss_pool = [], []
    for case_folder, condition in CASES:
        table, coords, meta = build_full_table(RESULTS_DIR / case_folder)
        cases[case_folder] = (table, coords, meta, condition)
        surf = table[:, 4] > 0.5
        vm_pool.append(table[:, 2])
        wss_pool.append(table[surf, 3])
        print(f"{case_folder} ({condition}): {meta['n_surface']} surface / "
              f"{meta['n_solid']} nodes | "
              f"vM max {table[:,2].max():.1f} Pa, "
              f"WSS max {table[surf,3].max():.2f} dyn/cm²")

    vm_all  = np.concatenate(vm_pool)
    wss_all = np.concatenate(wss_pool)
    # Shared color caps (anchored at 0). WSS auto from pooled 99th pct;
    # von Mises fixed at 300 Pa for readable low-to-mid contrast.
    VM_VMAX  = 300.0
    WSS_VMAX = nice_ceil(np.percentile(wss_all, 99))
    print(f"\nShared color scale (0..cap, pooled 99th pct):")
    print(f"  von Mises : 0 .. {VM_VMAX:g} Pa   "
          f"(true max {vm_all.max():.1f}, 99th pct {np.percentile(vm_all,99):.1f})")
    print(f"  WSS       : 0 .. {WSS_VMAX:g} dyn/cm²  "
          f"(true max {wss_all.max():.2f}, 99th pct {np.percentile(wss_all,99):.2f})")

    # --- Pass 2: downsample, write, and plot with shared limits ----------
    full_tables = {}
    for case_folder, condition in CASES:
        table, coords, meta, _ = cases[case_folder]
        out_root = RESULTS_DIR / case_folder / "grn_inputs"
        print(f"\n=== {case_folder}  ({condition}) ===")
        full_tables[case_folder] = (table, condition)

        # Percentage ladder — 2-D grid decimation over the full domain
        for frac in FRACTIONS:
            pct = round(frac * 100)
            sel = downsample_grid(coords, frac * meta["n_solid"])
            sub = table[sel]
            out_dir = out_root / f"downsampled_{pct:03d}pct"
            desc = f"{pct}% spatial resolution"
            write_level(out_dir, sub, desc, case_folder, condition)
            plot_level(out_dir, sub, case_folder, condition, desc,
                       VM_VMAX, WSS_VMAX)
            print(f"    {pct:>3d}%  -> {len(sub):5d} nodes "
                  f"({int(sub[:,4].sum())} surface)  ({out_dir.name}/)")

        # Custom fixed node count
        sel = downsample_grid(coords, CUSTOM_N)
        sub = table[sel]
        out_dir = out_root / f"custom_{CUSTOM_N}node"
        desc = f"custom ~{CUSTOM_N}-node sampling"
        write_level(out_dir, sub, desc, case_folder, condition)
        plot_level(out_dir, sub, case_folder, condition, desc,
                   VM_VMAX, WSS_VMAX)
        print(f"    ~{CUSTOM_N}n -> {len(sub):5d} nodes "
              f"({int(sub[:,4].sum())} surface)  ({out_dir.name}/)")

    # --- 3-case comparison figure (full resolution) ----------------------
    fig, axes = plt.subplots(len(CASES), 2, figsize=(11, 4.0 * len(CASES)))
    fig.suptitle("Cushion GRN Inputs — Developmental Flow Conditions\n"
                 f"Final FSG Step, Full Resolution  |  shared color scale "
                 f"(vM 0–{VM_VMAX:g} Pa, WSS 0–{WSS_VMAX:g} dyn/cm²)",
                 fontsize=13, fontweight="bold")
    for row, (case_folder, _) in enumerate(CASES):
        table, condition = full_tables[case_folder]
        n = len(table)
        surf = table[:, 4] > 0.5
        n_surf = int(surf.sum())
        pt = float(np.clip(3000.0 / max(n, 1), 3, 40))
        pt_surf = float(np.clip(3000.0 / max(n_surf, 1), 4, 40))
        scatter_panel(axes[row, 0], table[:, 0], table[:, 1], table[:, 2], "Pa", pt,
                      vmin=0.0, vmax=VM_VMAX)
        axes[row, 0].set_title(
            f"{condition} — von Mises $\\sigma_{{\\rm vM}}$ (Pa)   [{n} nodes]",
            fontsize=9, fontweight="bold")
        scatter_panel(axes[row, 1], table[surf, 0], table[surf, 1], table[surf, 3],
                      "dyn/cm$^2$", pt_surf, vmin=0.0, vmax=WSS_VMAX)
        axes[row, 1].set_title(f"{condition} — WSS (dyn/cm²)   [{n_surf} surface]",
                               fontsize=9, fontweight="bold")
    fig.tight_layout()
    cmp_path = RESULTS_DIR / "grn_inputs_comparison.png"
    fig.savefig(cmp_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison figure -> {cmp_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
