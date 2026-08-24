import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib as mpl
import os
from joblib import Parallel, delayed

from networkpoint import simulate, NODES, IDX, Params, looksmaxxed_plot_style

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "/Users/sophiewang/Desktop/Research/Inputs/dynamic_inputs/step_014/mechanical_inputs_trimmed.csv"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
T_END    = 400.0

OUTPUT_NODES = ["Snai1", "Snai2", "EndMT"]

custom_color_palette = {
    "Snai1": "#F72585",
    "Snai2": "#7209B7",
    "EndMT": "#3A0CA3",
}

# -----------------------------
# Normalization ranges (raw physical units)
# -----------------------------
WSS_MAX  = 30.0   # dyn/cm²
MECH_MAX = 100.0  # Pa

SHEAR_MID_NORM = 5.0  / WSS_MAX    # 5 dyn/cm²
MECH_MID_NORM  = 40.0 / MECH_MAX   # 40 Pa

# -----------------------------
# Load spatial data
# -----------------------------
df = pd.read_csv(CSV_PATH)

x_coords = df["x_m"].values
y_coords = df["y_m"].values
wss_raw  = df["wss_mag_dyn_cm2"].values
mech_raw = df["von_mises_Pa"].values

wss_norm  = np.clip(wss_raw  / WSS_MAX,  0.0, 1.0)
mech_norm = np.clip(mech_raw / MECH_MAX, 0.0, 1.0)

wss_has_data  = ~np.isnan(wss_raw)
mech_has_data = ~np.isnan(mech_raw)

inside_tissue = mech_has_data

n_points = len(df)
print(f"Loaded {n_points} spatial points")
print(f"  Points with valid WSS (surface):  {wss_has_data.sum()}")
print(f"  Points with valid Mech (tissue):  {mech_has_data.sum()}")
print(f"  Interior points (no WSS):         {(inside_tissue & ~wss_has_data).sum()}")
print(f"Running simulations per scenario...")

# -----------------------------
# Run model at each spatial point — three scenarios
# -----------------------------
p = Params()

scenarios = {
    "shear_only":  {"use_shear": True,  "use_mech": False},
    "mech_only":   {"use_shear": False, "use_mech": True},
    "combined":    {"use_shear": True,  "use_mech": True},
}

results = {
    scenario: {n: np.full(n_points, np.nan) for n in OUTPUT_NODES}
    for scenario in scenarios
}

def run_point(i, flags):
    surface_only = flags["use_shear"] and not flags["use_mech"]

    if flags["use_shear"]:
        if wss_has_data[i]:
            shear = float(wss_norm[i])
        elif surface_only:
            return {n: np.nan for n in OUTPUT_NODES}
        else:
            shear = 0.0
    else:
        shear = SHEAR_MID_NORM

    if flags["use_mech"]:
        mech = float(mech_norm[i]) if mech_has_data[i] else MECH_MID_NORM
    else:
        mech = MECH_MID_NORM

    _, _, yss = simulate(shear, mech, p=p, t_end=T_END)

    return {n: yss[IDX[n]] for n in OUTPUT_NODES}

for scenario, flags in scenarios.items():
    print(f"\nRunning scenario: {scenario}...")

    parallel_results = Parallel(
        n_jobs=-1,
        backend="loky",
        verbose=10
    )(
        delayed(run_point)(i, flags)
        for i in range(n_points)
    )

    for i, point_result in enumerate(parallel_results):
        for n in OUTPUT_NODES:
            results[scenario][n][i] = point_result[n]

print("\nSimulations complete. Saving results and generating plots...")

# -----------------------------
# Save results to CSV
# -----------------------------
out_df = pd.DataFrame({
    "x_m": x_coords,
    "y_m": y_coords,
    "wss_mag_dyn_cm2": wss_raw,
    "von_mises_Pa": mech_raw,
    "wss_norm": wss_norm,
    "mech_norm": mech_norm,
})
for scenario in scenarios:
    for node in OUTPUT_NODES:
        out_df[f"{node}_{scenario}"] = results[scenario][node]

csv_out = os.path.join(OUT_DIR, "results_spatial.csv")
out_df.to_csv(csv_out, index=False)
print(f"  Saved: {csv_out}")

# -----------------------------
# Plot helper
# -----------------------------
looksmaxxed_plot_style()

def plot_spatial(x, y, values, title, label, cmap, fname,
                 vmin=None, vmax=None, surface_only=False, surface_mask=None):
    mask = np.isfinite(values)

    xm = x[mask]
    ym = y[mask]
    vm = values[mask]

    if len(vm) < 3:
        print(f"Skipping {fname} — too few valid points")
        return

    if vmin is None:
        vmin = np.percentile(vm, 2)

    if vmax is None:
        vmax = np.percentile(vm, 98)

    if np.isclose(vmin, vmax):
        spread = max(abs(vmin) * 0.01, 1e-6)
        vmin -= spread
        vmax += spread

    fig, ax = plt.subplots(figsize=(6, 5))

    if surface_only:
        cf = ax.scatter(xm, ym, c=vm, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=8, edgecolors="none")
    else:
        levels = np.linspace(vmin, vmax, 100)

        triang = tri.Triangulation(xm, ym)

        xtri = xm[triang.triangles]
        ytri = ym[triang.triangles]

        edge1 = np.sqrt(
            (xtri[:, 0] - xtri[:, 1])**2 +
            (ytri[:, 0] - ytri[:, 1])**2
        )

        edge2 = np.sqrt(
            (xtri[:, 1] - xtri[:, 2])**2 +
            (ytri[:, 1] - ytri[:, 2])**2
        )

        edge3 = np.sqrt(
            (xtri[:, 2] - xtri[:, 0])**2 +
            (ytri[:, 2] - ytri[:, 0])**2
        )

        max_edge = np.max([edge1, edge2, edge3], axis=0)

        median_edge = np.median(max_edge)
        threshold   = median_edge * 8.0

        triang.set_mask(max_edge > threshold)

        cf = ax.tricontourf(
            triang,
            vm,
            levels=levels,
            cmap=cmap,
            extend="both",
        )

        if surface_mask is not None:
            sm = surface_mask & np.isfinite(values)
            if sm.any():
                ax.scatter(x[sm], y[sm], c=values[sm], cmap=cmap,
                           vmin=vmin, vmax=vmax, s=6, edgecolors="none")

    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(label, fontsize=14)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)

    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    print(f"Saved: {fname}")
    print(f"  value range = [{vm.min():.6f}, {vm.max():.6f}]")

plot_spatial(x_coords, y_coords, wss_norm,
             "Shear Stress (WSS)\nNormalized (surface)", "WSS (0-1)", "Blues",
             os.path.join(OUT_DIR, "plot_spatial_input_shear.png"),
             surface_only=True)

plot_spatial(x_coords, y_coords, mech_norm,
             "Mechanical Stress\nNormalized", "Stress (0-1)", "Oranges",
             os.path.join(OUT_DIR, "plot_spatial_input_mech.png"))

scenario_titles = {
    "shear_only": "Shear Only\n(Mech fixed at 40 Pa)",
    "mech_only":  "Mech Only\n(Shear fixed at 5 Dynes/cm²)",
    "combined":   "Combined\n(Shear + Mech)",
}

for scenario in scenarios:

    surface_only = (scenario == "shear_only")
    surf_overlay = wss_has_data if scenario == "combined" else None

    for node in OUTPUT_NODES:

        values = results[scenario][node].copy()

        plot_spatial(
            x_coords,
            y_coords,
            values,
            f"{node} — {scenario_titles[scenario]}",
            f"{node} Activity (0-1)",
            "plasma",
            os.path.join(OUT_DIR, f"plot_spatial_{node}_{scenario}.png"),
            surface_only=surface_only,
            surface_mask=surf_overlay,
        )

print("\nAll done!")
