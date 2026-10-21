"""figures/v3/*.png: the big run vs v1 and v2-small. Needs fig_v3_data.py and a9_big_eval.py outputs."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
C = {"v1": "#eb6834", "v2": "#2a78d6", "big": "#1baf7a"}  # entity colours, same as figures/v2 for v1/v2
LAB = {"v1": "v1 (no wall-shear loss)", "v2": "v2 small (wall-shear loss, 160 shapes)", "big": "big run (920 shapes, 0.67M params, 68k steps)"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False,
})
A = json.load(open("figures/v3/fig_data.json"))["354"]
E = {k: json.load(open(f"analysis/a9_{f}.json")) for k, f in
     (("v1", "grid_n160_physoff_s0"), ("v2", "tau_w1_s0"), ("big", "big"))}

# 1. wall curves
order = np.argsort(A["Re"])
fig, axs = plt.subplots(1, 4, figsize=(12, 3.9), sharex=True)
for ax, i in zip(axs, order):
    s = np.array(A["s"][i])
    ax.plot(s, A["true"][i], color=INK, lw=2.4, label="Dan's FEM (truth)")
    for k in ("v1", "v2", "big"):
        ax.plot(s, A[k][i], color=C[k], lw=2, label=LAB[k])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"Re = {A['Re'][i]:.1f}", loc="left", fontsize=10)
    ax.set_xlabel("position along cushion (arc fraction)")
axs[0].set_ylabel("wall shear τ  (units of μŪ/H)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=4, fontsize=8.5)
fig.suptitle("Held-out FSG shape g_00354: the big run tracks the truth at every flow rate, including the high-Re backflow, but with ripples",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig("figures/v3/fig_wall_curves_big.png", dpi=160); plt.close(fig)

# 2. by flow rate (test_B and val)
re_lab = ["k0\nRe ≈ 0.3", "k1\nRe ≈ 1.5", "k2\nRe ≈ 6.6", "k3\nRe ≈ 33"]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, sp, title in zip(axs, ("test_B", "val"), ("held-out FSG shapes (test_B)", "held-out synthetic shapes (val)")):
    for k in ("v1", "v2", "big"):
        y = E[k][sp]["tau_by_k"]
        ax.plot(range(4), y, color=C[k], marker="o", ms=7, mec=SURF, mew=2, label=LAB[k])
        ax.text(3.12, y[3], f"{y[3]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(4), re_lab); ax.set_xlim(-0.2, 3.5); ax.set_ylim(0, 0.55)
    ax.set_title(title, loc="left", fontsize=10)
axs[0].set_ylabel("median wall-shear error (relative L2)")
axs[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle("Error by flow rate: the high-Re penalty is gone", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("figures/v3/fig_by_flow.png", dpi=160); plt.close(fig)

# 3. train vs held-out (generalization gap), log x
fig, ax = plt.subplots(figsize=(10, 3.4))
SPL = [("train", "training shapes", "o"), ("test_B", "held-out FSG shapes", "s"), ("val", "held-out synthetic shapes", "D")]
for y, k in enumerate(("big", "v2", "v1")):
    xs = [E[k][sp]["tau"] for sp, _, _ in SPL]
    ax.plot([min(xs), max(xs)], [y, y], color=GRID, lw=2, zorder=1)
    for j, ((sp, lab, mk), x) in enumerate(sorted(zip(SPL, xs), key=lambda t: t[1])):
        ax.scatter([x], [y], s=60, marker=mk, color=C[k], edgecolor=SURF, linewidth=1.5, zorder=3,
                   label=lab if k == "big" else None)
    ax.text(1.02, y, "train {:.3f}   FSG {:.3f}   synthetic {:.3f}".format(*xs), transform=ax.get_yaxis_transform(),
            va="center", fontsize=8.5, color=INK2)
ax.set_xscale("log"); ax.set_xlim(0.008, 0.7)
ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
ax.set_yticks(range(3), ["big run", "v2 small", "v1"]); ax.set_ylim(-0.5, 2.6); ax.grid(axis="y", visible=False)
leg = ax.legend(loc="upper left", fontsize=8.5)
for hnd in leg.legend_handles: hnd.set_color(INK2)
ax.set_xlabel("median wall-shear error (relative L2, log scale)")
ax.set_title("The big run fits its training shapes 5–9× better than held-out ones: now generalization-limited",
             loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("figures/v3/fig_generalization.png", dpi=160); plt.close(fig)

# 4. learning curve (quick val = the same 16 val solves in every run)
def qv(run):
    rows = [json.loads(l) for l in open(f"{run}/metrics.jsonl") if l.strip()]
    return [(r["step"], r["quick_val_tau_rel_l2"]) for r in rows if "quick_val_tau_rel_l2" in r]
fig, ax = plt.subplots(figsize=(7.6, 3.6))
q = qv("runs/big"); ax.plot([s for s, _ in q], [v for _, v in q], color=C["big"], lw=2)
ax.text(30000, 0.075, f"big run ({q[-1][1]:.2f} at 68k steps)", fontsize=8.5, color=INK2)
for k, run in (("v2", "runs/tau/w1_s0"), ("v1", "runs/grid/n160_physoff_s0")):
    final = qv(run)[-1][1]
    ax.axhline(final, color=C[k], lw=1.5, ls=(0, (4, 3)))
    ax.text(68000, final + 0.006, f"{LAB[k].split(' (')[0]} final ({final:.2f}, 3k steps)", ha="right", fontsize=8.5, color=INK2)
ax.set_xlim(0, 69000); ax.set_ylim(0, 0.42)
ax.set_xlabel("training step"); ax.set_ylabel("wall-shear error, 16 val solves")
ax.set_title("Big run learning curve: most of the gain by ~20k steps, flat after ~40k", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("figures/v3/fig_learning_curve.png", dpi=160); plt.close(fig)
print("wrote figures/v3/{fig_wall_curves_big,fig_by_flow,fig_generalization,fig_learning_curve}.png")
