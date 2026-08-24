"""
plot_fsg_fields.py

Scatter plots of FSG mechanical fields at the final time step, plotted in the
DEFORMED configuration. Each node is a point; color represents the field value;
colormap is jet.

Produces two figures:
    wss_arc_surface_final.png       -- WSS and fluid pressure on arc surface (228 nodes)
    cauchy_stress_cushion_final.png -- Growth stimulus (sigma_comp_mag) over full cushion (11615 nodes)

Growth metric: sigma_comp_mag = max(0, -sigma_min_principal)
    Drives the Buskohl-style compression-homeostasis growth law in solid_solver.py.

Deformed coordinates:
    Arc/WSS nodes: arc_data.npz x,y are already in the deformed frame
        (extracted via get_deformed_arc() in run_fsg.py).
    Solid nodes: reference geometry + displacement from VisualisationVector/0.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
from pathlib import Path

plt.rcParams.update({
    "font.family":        "Arial",
    "font.weight":        "bold",
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "figure.titleweight": "bold",
})

STEP_DIR  = (Path(__file__).parent
             / "FSG Results" / "flow_U0p0360" / "step_014")
EXTRACTED = (Path(__file__).parent
             / "FSG Results" / "flow_U0p0360" / "extracted_fields")

# ---------------------------------------------------------------------------
# Load WSS data (arc coords are already deformed)
# ---------------------------------------------------------------------------
wss = np.load(EXTRACTED / "wss_arc_surface_final.npz")
wss_x = wss["x"] * 1e3   # mm
wss_y = wss["y"] * 1e3

# ---------------------------------------------------------------------------
# Load solid data with deformed coordinates
# ---------------------------------------------------------------------------
sol = np.load(EXTRACTED / "cauchy_stress_cushion_final.npz")

with h5py.File(STEP_DIR / "solid_fields.h5", "r") as f:
    ref_geom = f["Mesh/0/mesh/geometry"][:]    # reference coordinates, (N, 2)
    disp     = f["VisualisationVector/0"][:]   # displacement, (N, 3); col 2 is zero (2D)

def_x = (ref_geom[:, 0] + disp[:, 0]) * 1e3  # deformed x, mm
def_y = (ref_geom[:, 1] + disp[:, 1]) * 1e3  # deformed y, mm

sigma_comp_mag = sol["sigma_comp_mag"]         # growth stimulus field (Pa)

# ---------------------------------------------------------------------------
# Shared scatter helper
# ---------------------------------------------------------------------------
def scatter_panel(ax, x, y, values, title, unit_label, pt_size=8):
    sc = ax.scatter(x, y, c=values, cmap="jet", s=pt_size,
                    linewidths=0, rasterized=True)
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

# ---------------------------------------------------------------------------
# Figure 1: WSS magnitude on arc surface — deformed arc coordinates
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 4))
fig1.suptitle("Wall Shear Stress Magnitude — Arc/Cushion Surface, Deformed Config (Final FSG Step)",
              fontsize=10, fontweight="bold")

scatter_panel(ax1, wss_x, wss_y, wss["wss_magnitude"],
              "WSS Magnitude", "dyn/cm²", pt_size=20)

fig1.tight_layout()
out1 = EXTRACTED / "wss_arc_surface_final.png"
fig1.savefig(out1, dpi=180, bbox_inches="tight")
print(f"Saved {out1}")
plt.close(fig1)

# ---------------------------------------------------------------------------
# Figure 2: Growth stimulus — sigma_comp_mag on deformed cushion
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
fig2.suptitle(
    r"Growth Stimulus: $\sigma_{\rm comp}$ = max$(0,\,-\sigma_{\rm min})$"
    "\nFull Cushion Domain, Deformed Config (Final FSG Step)",
    fontsize=11, fontweight="bold",
)

scatter_panel(ax2, def_x, def_y, sigma_comp_mag,
              title=r"Compressive Stress Magnitude $\sigma_{\rm comp}$ (Pa)",
              unit_label="Pa",
              pt_size=4)

fig2.tight_layout()
out2 = EXTRACTED / "cauchy_stress_cushion_final.png"
fig2.savefig(out2, dpi=180, bbox_inches="tight")
print(f"Saved {out2}")
plt.close(fig2)

print("Done.")
