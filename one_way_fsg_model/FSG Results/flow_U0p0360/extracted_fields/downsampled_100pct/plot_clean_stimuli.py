import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent

plt.rcParams.update({
    "font.family":      "Arial",
    "font.weight":      "bold",
    "axes.labelweight": "bold",
})

# ── sigma_comp_mag ──────────────────────────────────────────────────────────
mech = np.load(BASE / "mechanical_inputs_unified.npz")
x_mm = mech["x_m"] * 1e3
y_mm = mech["y_m"] * 1e3
sigma_comp = mech["sigma_comp_mag_Pa"]

fig, ax = plt.subplots(figsize=(7, 4))
sc = ax.scatter(x_mm, y_mm, c=sigma_comp, cmap="jet", s=4,
                linewidths=0, rasterized=True)
cb = plt.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Pa", fontsize=9, fontweight="bold")
cb.ax.tick_params(labelsize=8)
for lbl in cb.ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.set_xlabel("x (mm)", fontsize=9, fontweight="bold")
ax.set_ylabel("y (mm)", fontsize=9, fontweight="bold")
ax.tick_params(labelsize=8)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.set_aspect("equal")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(BASE / "sigma_comp_mag_clean.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved sigma_comp_mag_clean.png")

# ── WSS magnitude ────────────────────────────────────────────────────────────
wss = np.load(BASE / "wss_arc_surface.npz")
x_wss = wss["x_m"] * 1e3
y_wss = wss["y_m"] * 1e3
wss_mag = wss["wss_mag_dyn_cm2"]

fig, ax = plt.subplots(figsize=(7, 4))
sc = ax.scatter(x_wss, y_wss, c=wss_mag, cmap="jet", s=20,
                linewidths=0, rasterized=True)
cb = plt.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("dyn/cm²", fontsize=9, fontweight="bold")
cb.ax.tick_params(labelsize=8)
for lbl in cb.ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.set_xlabel("x (mm)", fontsize=9, fontweight="bold")
ax.set_ylabel("y (mm)", fontsize=9, fontweight="bold")
ax.tick_params(labelsize=8)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.set_aspect("equal")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(BASE / "wss_mag_clean.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved wss_mag_clean.png")
