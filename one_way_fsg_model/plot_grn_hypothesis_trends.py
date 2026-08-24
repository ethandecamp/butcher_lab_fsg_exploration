"""
plot_grn_hypothesis_trends.py

Figures supporting the hypothesis threads pulled from the AHA GRN spatial
results (three developmental flow conditions x GRN node activity, decomposed
into shear-only / mech-only / combined contributions).

Source data: AHA GRN Plots/3scenarios_results/<case>/results_spatial.csv
             AHA GRN Plots/comparison_figures/metrics_summary.csv

Figures (written to AHA GRN Plots/hypothesis_figures/):
  1. mechanosensing_axes.png   -- per-node correlation with local WSS vs local
                                   tissue stress, one panel per node, across
                                   the three cases. Shows the Notch/Snai2
                                   "shear axis" vs YAP-TAZ/Snai1 "stress axis"
                                   split, and whether it strengthens with flow.
  2. spatial_switch_profile.png -- normalized mechanical input vs EndMT output
                                   binned along x, input vs output steepness
                                   annotated. Smooth input -> sharp output.
  3. lr_asymmetry_vs_flow.png   -- left-half vs right-half mean activity, per
                                   node, across the three cases, surface-band
                                   vs full-mesh side by side (the full-mesh
                                   version is diluted by interior nodes that
                                   never see real WSS -- see docstring on
                                   fig_lr_asymmetry_vs_flow).
  4. yap_taz_saturation.png     -- fraction of nodes with combined activity
                                   above a near-ceiling threshold, per node,
                                   across the three cases.
  5. input_field_profiles.png   -- surface-band WSS and tissue-stress vs x,
                                   left to right, all three cases overlaid.
  6. surface_expression_profiles.png -- surface-band GRN node activity vs x,
                                   left to right, one panel per node, all
                                   three cases overlaid. The direct "gene
                                   expression intensity left-to-right" view.
  7. grn_dashboard.png          -- single combined panel: input fields, node
                                   expression profiles, mechanosensing-axis
                                   correlations, and LR asymmetry together.

Usage
-----
    /Users/danielpearce/PyCharmMiscProject/.venv/bin/python3 plot_grn_hypothesis_trends.py
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family":        "Arial",
    "font.weight":        "bold",
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "figure.titleweight": "bold",
})

ROOT       = Path(__file__).parent / "AHA GRN Plots"
SCEN_DIR   = ROOT / "3scenarios_results"
OUT_DIR    = ROOT / "hypothesis_figures"
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    ("080_180", "Underflow", "#2C6FBB"),
    ("080_360", "Healthy",   "#2E9E4F"),
    ("080_540", "Overflow",  "#C0392B"),
]

NODES = ["NICD", "Snai2", "YAP_TAZ", "Snai1", "EndMT"]
SATURATION_THRESH = 0.79
N_XBINS = 50


@lru_cache(maxsize=None)
def load_case(folder):
    return pd.read_csv(SCEN_DIR / folder / "results_spatial.csv")


def surface_only(df):
    """Nodes with a real WSS value (fluid-facing surface band, ~6% of the
    mesh). Interior nodes never see shear and sit at a fixed GRN default,
    so any left-right or WSS-correlation analysis needs this filter."""
    return df[df["wss_norm"].notna()]


def surface_binned_profile(df, col, n_bins=N_XBINS):
    """Surface-band values of `col`, binned along x_m, for a clean
    left-to-right line profile out of otherwise scattered node data."""
    surf = surface_only(df)
    bins = pd.cut(surf["x_m"], n_bins)
    g = surf.groupby(bins, observed=True).agg(
        x_m=("x_m", "mean"), val=(col, "mean")
    ).dropna()
    return g["x_m"].values * 1e6, g["val"].values


def compute_correlations():
    """{node: {case_label: (corr_with_wss, corr_with_stress)}}, surface-band."""
    out = {}
    for node in NODES:
        out[node] = {}
        for folder, label, _ in CASES:
            surf = surface_only(load_case(folder))
            comb = surf[f"{node}_combined"]
            out[node][label] = (
                np.corrcoef(surf["wss_norm"], comb)[0, 1],
                np.corrcoef(surf["mech_norm"], comb)[0, 1],
            )
    return out


def compute_asymmetry(restrict_surface=True):
    """{node: {case_label: pct_asymmetry}}. 100 x (left - right) / mean."""
    out = {}
    for node in NODES:
        out[node] = {}
        for folder, label, _ in CASES:
            df = load_case(folder)
            df = surface_only(df) if restrict_surface else df
            left = df[df["x_m"] < 0][f"{node}_combined"].mean()
            right = df[df["x_m"] > 0][f"{node}_combined"].mean()
            out[node][label] = 100 * (left - right) / ((left + right) / 2)
    return out


# ---------------------------------------------------------------------------
# 1. Mechanosensing axes: corr(node, WSS) vs corr(node, stress), per case
# ---------------------------------------------------------------------------
def fig_mechanosensing_axes():
    fig, axes = plt.subplots(1, len(NODES), figsize=(4 * len(NODES), 4.2),
                              sharey=True)
    case_labels = [label for _, label, _ in CASES]
    x = np.arange(len(CASES))
    width = 0.35
    corrs = compute_correlations()

    for ax, node in zip(axes, NODES):
        corr_wss  = [corrs[node][label][0] for label in case_labels]
        corr_mech = [corrs[node][label][1] for label in case_labels]

        ax.bar(x - width / 2, corr_wss, width, label="corr(WSS)", color="#1B9E77")
        ax.bar(x + width / 2, corr_mech, width, label="corr(stress)", color="#D95F02")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, rotation=20)
        ax.set_title(node)
        ax.set_ylim(-1, 1)

    axes[0].set_ylabel("correlation with node activity\n(surface nodes only)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Mechanosensing axis per GRN node: shear-driven vs stress-driven")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mechanosensing_axes.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Spatial switch profile: smooth input -> sharp EndMT output
# ---------------------------------------------------------------------------
def fig_spatial_switch_profile():
    fig, axes = plt.subplots(1, len(CASES), figsize=(5 * len(CASES), 4.5),
                              sharey=True)

    for ax, (folder, label, color) in zip(axes, CASES):
        df = load_case(folder)
        bins = pd.cut(df["x_m"], N_XBINS)
        binned = df.groupby(bins, observed=True).agg(
            x_m=("x_m", "mean"),
            mech=("mech_norm", "mean"),
            endmt=("EndMT_combined", "mean"),
        ).dropna()

        x_um = binned["x_m"] * 1e6
        mech_n = (binned["mech"] - binned["mech"].min()) / np.ptp(binned["mech"])
        endmt_n = (binned["endmt"] - binned["endmt"].min()) / np.ptp(binned["endmt"])

        ax.plot(x_um, mech_n, "--", color="gray", linewidth=2, label="mech input (norm.)")
        ax.plot(x_um, endmt_n, "-", color=color, linewidth=2.5, label="EndMT output (norm.)")

        max_slope_in = np.max(np.abs(np.diff(mech_n))) / np.mean(np.abs(np.diff(x_um)))
        max_slope_out = np.max(np.abs(np.diff(endmt_n))) / np.mean(np.abs(np.diff(x_um)))
        sharpness = max_slope_out / max_slope_in if max_slope_in > 0 else np.nan
        ax.text(0.03, 0.05, f"output/input\nmax slope: {sharpness:.1f}x",
                transform=ax.transAxes, fontsize=9, va="bottom")

        ax.set_title(label, color=color)
        ax.set_xlabel("x (µm)")
        if ax is axes[0]:
            ax.set_ylabel("normalized (0-1)")
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Mechanical input is smooth; EndMT output is a step (x-binned means)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_switch_profile.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Left-right asymmetry vs flow, per node -- test for a peak at Healthy
# ---------------------------------------------------------------------------
def fig_lr_asymmetry_vs_flow():
    """Surface-band nodes only (wss_norm defined). ~94% of the mesh is
    interior and never sees real WSS, so shear-driven nodes (NICD, Snai2)
    sit at a near-constant baseline there and swamp the real signal if the
    whole volumetric mesh is averaged -- this also matches biology, since
    EndMT is an endocardial (surface-cell) process to begin with."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    case_labels = [label for _, label, _ in CASES]
    x = np.arange(len(CASES))

    for ax, (subset_name, restrict) in zip(
        axes,
        [("surface nodes (real WSS)", True),
         ("full mesh (surface + interior)", False)],
    ):
        asym_by_node = compute_asymmetry(restrict_surface=restrict)
        for node in NODES:
            asym = [asym_by_node[node][label] for label in case_labels]
            lw = 3 if node == "EndMT" else 1.5
            ax.plot(x, asym, "-o", label=node, linewidth=lw)

        ax.set_xticks(x)
        ax.set_xticklabels(case_labels)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(subset_name)
    axes[0].set_ylabel("left-right asymmetry (%)\n100 x (left - right) / mean")
    axes[0].legend(fontsize=8)
    fig.suptitle("Left-right activity asymmetry vs flow condition")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lr_asymmetry_vs_flow.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. YAP/TAZ (and other node) saturation fraction vs flow
# ---------------------------------------------------------------------------
def fig_saturation_fraction():
    fig, ax = plt.subplots(figsize=(7, 5))
    case_labels = [label for _, label, _ in CASES]
    x = np.arange(len(CASES))
    width = 0.15

    for i, node in enumerate(NODES):
        frac = []
        for folder, _, _ in CASES:
            df = load_case(folder)
            frac.append(100 * (df[f"{node}_combined"] > SATURATION_THRESH).mean())
        ax.bar(x + (i - len(NODES) / 2) * width, frac, width, label=node)

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels)
    ax.set_ylabel(f"% of nodes with combined activity > {SATURATION_THRESH}")
    ax.set_title("Node saturation (activation ceiling) vs flow condition")
    ax.legend(fontsize=8, ncol=len(NODES))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yap_taz_saturation.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Input fields left-to-right: surface-band WSS and stress vs x
# ---------------------------------------------------------------------------
def fig_input_profiles():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in zip(
        axes, ["wss_norm", "mech_norm"],
        ["Wall shear stress (normalized)", "Tissue stress (normalized)"],
    ):
        for folder, label, color in CASES:
            x_um, val = surface_binned_profile(load_case(folder), col)
            ax.plot(x_um, val, "-o", ms=3, color=color, label=label)
        ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlabel("x (µm)")
        ax.set_title(title)
    axes[0].set_ylabel("normalized (0-1), surface band")
    axes[0].legend(fontsize=8)
    fig.suptitle("Mechanical input fields, left to right (surface band only)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "input_field_profiles.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Surface-band GRN node activity, left to right, per node
# ---------------------------------------------------------------------------
def fig_surface_expression_profiles():
    fig, axes = plt.subplots(1, len(NODES), figsize=(4.2 * len(NODES), 4.5))
    for ax, node in zip(axes, NODES):
        for folder, label, color in CASES:
            x_um, val = surface_binned_profile(load_case(folder), f"{node}_combined")
            ax.plot(x_um, val, "-o", ms=3, color=color, label=label)
        ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlabel("x (µm)")
        ax.set_title(node)
    axes[0].set_ylabel("combined activity (0-1), surface band")
    axes[0].legend(fontsize=8)
    fig.suptitle("GRN node activity, left to right (surface band only)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "surface_expression_profiles.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Combined dashboard: inputs, node profiles, correlations, asymmetry
# ---------------------------------------------------------------------------
def fig_dashboard():
    case_labels = [label for _, label, _ in CASES]
    corrs = compute_correlations()
    asym_surf = compute_asymmetry(restrict_surface=True)

    fig = plt.figure(figsize=(26, 20))
    gs = fig.add_gridspec(4, 5, height_ratios=[1, 1.1, 1, 1],
                           hspace=0.55, wspace=0.4)

    # Row 0: input fields (2 wide panels) + case metadata box
    for j, (col, title) in enumerate(
        [("wss_norm", "WSS (normalized)"), ("mech_norm", "Tissue stress (normalized)")]
    ):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        for folder, label, color in CASES:
            x_um, val = surface_binned_profile(load_case(folder), col)
            ax.plot(x_um, val, "-o", ms=3, color=color, label=label)
        ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlabel("x (µm)")
        ax.set_title(f"Input: {title}")
        if j == 0:
            ax.set_ylabel("normalized, surface band")
            ax.legend(fontsize=8)

    ax_meta = fig.add_subplot(gs[0, 4])
    ax_meta.axis("off")
    metrics = pd.read_csv(ROOT / "comparison_figures" / "metrics_summary.csv")
    lines = ["Case metadata\n"]
    for _, row in metrics.iterrows():
        lines.append(
            f"{row['condition']}: height {row['height_mm']:.3f} mm, "
            f"peak vM {row['vm_peak']:.0f} Pa, peak WSS {row['wss_peak']:.1f} dyn/cm2, "
            f"n_surf {int(row['n_surface'])}"
        )
    ax_meta.text(0, 1, "\n\n".join(lines), va="top", fontsize=9, transform=ax_meta.transAxes)

    # Row 1: surface-band node activity profiles, one column per node
    for j, node in enumerate(NODES):
        ax = fig.add_subplot(gs[1, j])
        for folder, label, color in CASES:
            x_um, val = surface_binned_profile(load_case(folder), f"{node}_combined")
            ax.plot(x_um, val, "-o", ms=3, color=color, label=label)
        ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlabel("x (µm)")
        ax.set_title(node)
        if j == 0:
            ax.set_ylabel("combined activity\n(surface band)")
            ax.legend(fontsize=7)

    # Row 2: mechanosensing-axis correlation heatmaps (WSS, stress)
    for j, (idx, title) in enumerate([(0, "corr(node, WSS)"), (1, "corr(node, stress)")]):
        ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
        mat = np.array([[corrs[node][label][idx] for label in case_labels] for node in NODES])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(case_labels)))
        ax.set_xticklabels(case_labels)
        ax.set_yticks(range(len(NODES)))
        ax.set_yticklabels(NODES)
        for yi in range(mat.shape[0]):
            for xi in range(mat.shape[1]):
                ax.text(xi, yi, f"{mat[yi, xi]:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_sat = fig.add_subplot(gs[2, 4])
    x = np.arange(len(CASES))
    width = 0.15
    for i, node in enumerate(NODES):
        frac = [100 * (load_case(folder)[f"{node}_combined"] > SATURATION_THRESH).mean()
                for folder, _, _ in CASES]
        ax_sat.bar(x + (i - len(NODES) / 2) * width, frac, width, label=node)
    ax_sat.set_xticks(x)
    ax_sat.set_xticklabels(case_labels, fontsize=8)
    ax_sat.set_ylabel(f"% nodes > {SATURATION_THRESH}", fontsize=8)
    ax_sat.set_title("Saturation (full mesh)", fontsize=10)
    ax_sat.legend(fontsize=6, ncol=2)

    # Row 3: left-right asymmetry (surface-band, corrected) across full width
    ax_asym = fig.add_subplot(gs[3, 0:3])
    for node in NODES:
        asym = [asym_surf[node][label] for label in case_labels]
        lw = 3 if node == "EndMT" else 1.5
        ax_asym.plot(x, asym, "-o", label=node, linewidth=lw)
    ax_asym.set_xticks(x)
    ax_asym.set_xticklabels(case_labels)
    ax_asym.axhline(0, color="black", linewidth=0.8)
    ax_asym.set_ylabel("left-right asymmetry (%)\n100 x (left - right) / mean")
    ax_asym.set_title("LR asymmetry vs flow (surface band, corrected)")
    ax_asym.legend(fontsize=8)

    ax_note = fig.add_subplot(gs[3, 3:5])
    ax_note.axis("off")
    note = (
        "Notes\n\n"
        "- NICD/Snai2 (shear axis) vs YAP_TAZ/Snai1 (stress axis) pull in\n"
        "  opposite directions and the split sharpens toward Overflow.\n"
        "- Only YAP_TAZ saturates (>0.79) at high flow; other nodes stay\n"
        "  far from ceiling across all three conditions.\n"
        "- LR asymmetry here is surface-band only. The full-mesh version\n"
        "  is diluted by interior nodes with no real WSS (94% of the mesh)\n"
        "  and gives a misleading 'peaks at Healthy' shape for EndMT --\n"
        "  see lr_asymmetry_vs_flow.png for the side-by-side comparison."
    )
    ax_note.text(0, 1, note, va="top", fontsize=9, transform=ax_note.transAxes)

    fig.suptitle("FSG -> GRN dashboard: mechanical inputs, node activity, "
                 "mechanosensing axes, and left-right asymmetry", fontsize=16)
    fig.savefig(OUT_DIR / "grn_dashboard.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_mechanosensing_axes()
    fig_spatial_switch_profile()
    fig_lr_asymmetry_vs_flow()
    fig_saturation_fraction()
    fig_input_profiles()
    fig_surface_expression_profiles()
    fig_dashboard()
    print(f"Wrote 7 figures to {OUT_DIR}")
