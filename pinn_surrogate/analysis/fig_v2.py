"""figures/v2/*.png from eval.json files and figures/v2/fig_data.json (run fig_v2_data.py first)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False,
})
D = json.load(open("figures/v2/fig_data.json"))
med = lambda r: float(np.median([x["tau_rel_l2"] for x in json.load(open(f"{r}/eval.json"))["splits"]["test_B"]["per_solve"]]))

# ---- Fig 1: experiment summary (test_B median wall-shear error, both seeds) ----
rows = [  # label, run prefix, group
    ("v1 (no wall-shear loss)", "runs/grid/n160_physoff_s", "base"),
    ("mask width 40 µm", "runs/delta/d0.1_s", "neg"), ("mask width 80 µm", "runs/delta/d0.2_s", "neg"),
    ("mask width 200 µm", "runs/delta/d0.5_s", "neg"),
    ("gradient loss only", "runs/grad/t0g1_s", "neg"),
    ("wall shear + gradient", "runs/grad/t1g1_s", "mid"),
    ("wall-shear loss, weight 1", "runs/tau/w1_s", "win"), ("wall-shear loss, weight 10", "runs/tau/w10_s", "win"),
]
fig, ax = plt.subplots(figsize=(7.6, 4.3))
for y, (lab, pre, grp) in enumerate(rows[::-1]):
    v = [med(pre + s) for s in "01"]
    c = BLUE if grp == "win" else (MUTED if grp != "base" else INK)
    ax.plot(v, [y, y], color=GRID, lw=2, zorder=1)
    ax.scatter(v, [y, y], s=46, color=c, edgecolor=SURF, linewidth=2, zorder=3)
    ax.text(max(v) + 0.012, y, f"{np.mean(v):.2f}", va="center", color=INK2, fontsize=9)
ax.axvline(0.544, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(0.537, -0.45, "no-learning\nprior 0.54", color=INK2, fontsize=8.5, ha="right", va="bottom")
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
ax.set_xlim(0, 0.6); ax.set_ylim(-0.6, len(rows) - 0.2); ax.grid(axis="y", visible=False)
ax.set_xlabel("median wall-shear error on held-out FSG shapes (relative L2; lower is better)")
ax.set_title("Only the direct wall-shear loss moves the error\n(n160, physics off, 3,000 steps; dots = seeds 0 and 1, label = mean)",
             loc="left", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig("figures/v2/fig_experiments.png", dpi=160); plt.close(fig)

# ---- Fig 2: wall-shear curves, one FSG shape, 4 flows ----
A = D["A"]; order = np.argsort(A["Re"])
fig, axs = plt.subplots(1, 4, figsize=(12, 3.9), sharex=True)
for ax, i in zip(axs, order):
    s = np.array(A["s"][i])
    ax.plot(s, A["true"][i], color=INK, lw=2, label="Dan's FEM (truth)")
    ax.plot(s, A["v1"][i], color=ORANGE, lw=2, label="v1")
    ax.plot(s, A["v2"][i], color=BLUE, lw=2, label="v2: + wall-shear loss")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"Re = {A['Re'][i]:.1f}", loc="left", fontsize=10)
    ax.set_xlabel("position along cushion (arc fraction)")
axs[0].set_ylabel("wall shear τ  (units of μŪ/H)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=3, fontsize=9)
fig.suptitle("Held-out FSG shape g_00354: v2 closes most of v1's peak gap; at the highest Re it is still ~25% low", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig("figures/v2/fig_wall_curves_v2.png", dpi=160); plt.close(fig)

# ---- Fig 3: near-wall gradient profile ----
C = D["C"]; e = np.array(C["edges_um"], float); mid = np.sqrt(e[:-1] * e[1:])
fig, ax = plt.subplots(figsize=(7.2, 4.0))
series = [("Dan's FEM", INK, [1.0] + C["dan"]), ("v1", ORANGE, [C["v1"]["wall"]] + C["v1"]["bins"]),
          ("v2: wall-shear loss", BLUE, [C["tau"]["wall"]] + C["tau"]["bins"]),
          ("wall shear + gradient loss", AQUA, [C["tau_grad"]["wall"]] + C["tau_grad"]["bins"])]
xs = np.r_[0.0, mid]
for lab, col, ys in series:
    ax.plot(xs, ys, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=2, label=lab)
ax.axvspan(0, 2, color=GRID, alpha=0.6, lw=0)
ax.text(1.0, 0.52, "≈2 packed\nnodes/solve\nin here", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xlim(-1, 35); ax.set_ylim(0.5, 1.12)
ax.set_xlabel("distance from cushion wall (µm; 0 = wall, others = bin centres)")
ax.set_ylabel("du_t/dn  ÷  true wall shear")
ax.set_title("Peak-shear region, held-out FSG shapes: v1 is right ≥5 µm out but wrong at the wall",
             loc="left", fontsize=10.5)
ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout(); fig.savefig("figures/v2/fig_near_wall_gradient.png", dpi=160); plt.close(fig)
print("wrote figures/v2/{fig_experiments,fig_wall_curves_v2,fig_near_wall_gradient}.png")
