import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl


# -----------------------------
# Plot style
# -----------------------------
def looksmaxxed_plot_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 18,
        "axes.labelsize": 22,
        "axes.labelweight": "bold",
        "axes.linewidth": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "xtick.major.width": 2.5,
        "ytick.major.width": 2.5,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "lines.linewidth": 3,
        "legend.frameon": False,
        "legend.fontsize": 14,
        "figure.figsize": (7, 5),
        "figure.dpi": 150,
    })


custom_color_palette = {
    "JAG":          "#A8DADC",
    "DLL":          "#457B9D",
    "NICD":         "#1D3557",
    "HEY":          "#6A994E",
    "MAML":         "#BC4749",
    "RBPJ":         "#C77DFF",
    "LRP":          "#2A9D8F",
    "beta_catenin": "#E76F51",
    "TCF_LEF":      "#A8DADC",
    "YAP_TAZ":      "#F4D35E",
    "TGFb_TypeI":   "#98C1D9",
    "TGFb_123":     "#3D5A80",
    "TGFb_TypeII":  "#293241",
    "SMAD23":       "#56CFE1",
    "SMAD3":        "#C2DFE3",
    "SMAD4":        "#9DB4C0",
    "BMP_TypeI":    "#C9B1BD",
    "BMP_2456":     "#A0785A",
    "BMP_TypeII":   "#7D5A50",
    "SMAD67":       "#B5838D",
    "SMAD158":      "#6D6875",
    "Snai1":        "#F72585",
    "Snai2":        "#7209B7",
    "EndMT":        "#3A0CA3",
}


# -----------------------------
# Math helpers
# -----------------------------
def hill(x, n=2.0, ec50=0.5):
    x = np.clip(x, 0.0, 1.0)
    return (x**n) / (ec50**n + x**n + 1e-12)

def NOT(x):
    return 1.0 - np.clip(x, 0.0, 1.0)

def OR(*xs):
    y = 0.0
    for x in xs:
        y = y + x - y * x
    return np.clip(y, 0.0, 1.0)

def AND(*xs):
    y = 1.0
    for x in xs:
        y *= x
    return np.clip(y, 0.0, 1.0)


# -----------------------------
# Parameters
# -----------------------------
@dataclass
class Params:
    n: float    = 2.0
    ec50: float = 0.2

    ec50_input: float = 0.5

    # ---- Fixed external inputs ----
    Noggin_Chordin: float = 0.05
    VEGF:           float = 0.30
    DKK:            float = 0.1
    WNT:            float = 0.2
    Frizzled:       float = 0.5

    # ---- Time constants (hours) ----
    tau_signal:  float = 0.1
    tau_tf:      float = 1.0
    tau_output:  float = 10.0


# -----------------------------
# Physical input ranges
# -----------------------------
SHEAR_MAX = 30.0   # Dynes/cm²
MECH_MAX  = 100.0  # Pa

def normalize_shear(shear_dyn):
    return np.clip(shear_dyn / SHEAR_MAX, 0.0, 1.0)

def normalize_mech(mech_pa):
    return np.clip(mech_pa / MECH_MAX, 0.0, 1.0)

def simulate_physical(shear_dyn, mech_pa, p=None, t_end=300.0, y0=None):
    return simulate(normalize_shear(shear_dyn), normalize_mech(mech_pa), p=p, t_end=t_end, y0=y0)


# -----------------------------
# Node ordering
# -----------------------------
NODES = [
    # Notch
    "JAG", "DLL", "NOTCH_receptors", "NICD", "MAML", "RBPJ", "HEY",
    # Wnt
    "LRP", "beta_catenin", "TCF_LEF",
    # Mechanosensing
    "YAP_TAZ",
    # TGFb
    "TGFb_TypeI", "TGFb_TypeII", "TGFb_123",
    "SMAD23", "SMAD4",
    # BMP
    "BMP_TypeI", "BMP_TypeII", "BMP_2456",
    "SMAD67", "SMAD158",
    # Outputs
    "Snai1", "Snai2", "EndMT",
]
IDX = {name: i for i, name in enumerate(NODES)}


# -----------------------------
# RHS
# -----------------------------
def rhs(t, y, shear_stress, mech_stress, p: Params):

    JAG             = y[IDX["JAG"]]
    DLL             = y[IDX["DLL"]]
    NOTCH_receptors = y[IDX["NOTCH_receptors"]]
    NICD            = y[IDX["NICD"]]
    MAML            = y[IDX["MAML"]]
    RBPJ            = y[IDX["RBPJ"]]
    HEY             = y[IDX["HEY"]]
    LRP             = y[IDX["LRP"]]
    beta_catenin    = y[IDX["beta_catenin"]]
    TCF_LEF         = y[IDX["TCF_LEF"]]
    YAP_TAZ         = y[IDX["YAP_TAZ"]]
    TGFb_TypeI      = y[IDX["TGFb_TypeI"]]
    TGFb_TypeII     = y[IDX["TGFb_TypeII"]]
    TGFb_123        = y[IDX["TGFb_123"]]
    SMAD23          = y[IDX["SMAD23"]]
    SMAD4           = y[IDX["SMAD4"]]
    BMP_TypeI       = y[IDX["BMP_TypeI"]]
    BMP_TypeII      = y[IDX["BMP_TypeII"]]
    BMP_2456        = y[IDX["BMP_2456"]]
    SMAD67          = y[IDX["SMAD67"]]
    SMAD158         = y[IDX["SMAD158"]]
    Snai1           = y[IDX["Snai1"]]
    Snai2           = y[IDX["Snai2"]]
    EndMT           = y[IDX["EndMT"]]

    n    = p.n
    ec50 = p.ec50
    ei   = p.ec50_input

    # ==================================================
    # NOTCH PATHWAY
    # ==================================================
    JAG_inf  = hill(shear_stress, n, ei)
    DLL_inf  = NOT(hill(shear_stress, n, ei))

    notch_OR = OR(hill(JAG, n, ec50), hill(DLL, n, ec50))
    gamma_secretase = 1.0
    NOTCH_receptors_inf = hill(BMP_2456, n, ec50)

    notch_AND = AND(notch_OR, gamma_secretase, hill(NOTCH_receptors, n, ec50))
    NICD_inf = np.clip(
        0.4 * notch_AND
        + 0.35 * hill(shear_stress, n, ei)
        + 0.25 * hill(beta_catenin, n, ec50),
        0, 1
    )

    MAML_inf = 1.0
    RBPJ_inf = 0.3
    HEY_inf  = hill(NICD, n, ec50)

    # ==================================================
    # WNT PATHWAY
    # ==================================================
    LRP_inf = NOT(hill(p.DKK, n, ec50))

    klf2      = hill(shear_stress, n, ei)
    wnt_drive = OR(0.3 * klf2, hill(p.WNT, n, ei))
    wnt_gate  = AND(wnt_drive, hill(p.Frizzled, n, ec50), hill(LRP, n, ec50))
    beta_catenin_inf = np.clip(
        (0.7 * wnt_gate + 0.3 * hill(NICD, n, ec50)) * NOT(hill(p.VEGF, n, ec50)),
        0, 1
    )

    TCF_LEF_inf = 1.0
    wnt_AND_out = AND(hill(beta_catenin, n, ec50), hill(TCF_LEF, n, ec50))

    # ==================================================
    # YAP/TAZ
    # ==================================================
    YAP_TAZ_inf = hill(mech_stress, n, ei)

    # ==================================================
    # TGFb PATHWAY
    # ==================================================
    TGFb_123_inf    = AND(hill(BMP_2456, n, ec50), NOT(hill(p.VEGF, n, ec50)))
    TGFb_TypeII_inf = 1.0
    notch_AND_gate  = AND(hill(NICD, n, ec50), hill(MAML, n, ec50), hill(RBPJ, n, ec50))
    TGFb_TypeI_inf  = np.clip(notch_AND_gate * NOT(hill(p.VEGF, n, ec50)), 0, 1)

    smad23_base = AND(
        hill(TGFb_TypeI, n, ec50),
        hill(TGFb_123, n, ec50),
        hill(TGFb_TypeII, n, ec50),
        NOT(hill(SMAD67, n, ec50)),
        NOT(hill(SMAD158, n, ec50)),
        NOT(hill(RBPJ, n, ec50)),
    )
    nicd_mod_smad23 = 0.3 + 0.7 * hill(NICD, n, ec50)
    SMAD23_inf = np.clip(smad23_base * nicd_mod_smad23, 0, 1)

    SMAD4_inf = OR(hill(SMAD23, n, ec50), hill(SMAD158, n, ec50))

    # ==================================================
    # BMP PATHWAY
    # ==================================================
    BMP_TypeI_inf  = 1.0
    BMP_TypeII_inf = 1.0
    BMP_2456_inf   = AND(NOT(hill(HEY, n, ec50)), NOT(hill(p.Noggin_Chordin, n, ec50))) \
                     * (1.0 - 1.0 * hill(klf2, n, ec50))

    bmp_AND     = AND(hill(BMP_TypeI, n, ec50), hill(BMP_TypeII, n, ec50), hill(BMP_2456, n, ec50))
    SMAD158_inf = AND(bmp_AND, NOT(hill(SMAD67, n, ec50)))

    # ==================================================
    # OUTPUT GATES
    # ==================================================
    gate_A = AND(hill(beta_catenin, n, ec50), hill(TCF_LEF, n, ec50), hill(YAP_TAZ, n, ec50))

    gate_B     = AND(hill(SMAD23, n, ec50), hill(SMAD4, n, ec50))
    SMAD67_inf = OR(bmp_AND, gate_B) + 1.0 * hill(klf2, n, ec50)
    SMAD67_inf = np.clip(SMAD67_inf, 0, 1)

    gate_D = np.clip(0.5 * hill(YAP_TAZ, n, ec50) + 0.5 * AND(gate_B, hill(YAP_TAZ, n, ec50)), 0, 1)

    and_158_4  = AND(hill(SMAD158, n, ec50), hill(SMAD4, n, ec50))
    bmp_OR_out = OR(and_158_4, gate_B)
    gate_C     = AND(bmp_OR_out, hill(NICD, n, ec50))

    # --- Snai1 ---
    snai1_terms   = [gate_A, gate_D, gate_C]
    snai1_weights = [0.35,   0.55,   0.10  ]
    snai1_drive   = sum(w * v for w, v in zip(snai1_weights, snai1_terms))
    Snai1_inf = np.clip(snai1_drive * (1.0 - 0.2 * hill(Snai2, n, ec50)), 0, 1)

    # --- Snai2 ---
    lower_AND  = AND(wnt_AND_out, hill(YAP_TAZ, n, ec50), hill(NICD, n, ec50))
    snai2_self = hill(Snai2, n, ec50)

    snai2_terms   = [notch_AND_gate, wnt_AND_out, lower_AND, snai2_self]
    snai2_weights = [0.30,           0.25,        0.30,      0.15      ]
    snai2_drive   = sum(w * v for w, v in zip(snai2_weights, snai2_terms))
    nicd_mod  = 0.3 + 0.7 * hill(NICD, n, ec50)
    Snai2_inf = np.clip(
        snai2_drive
        * (1.0 - 0.2 * hill(Snai1, n, ec50))
        * (1.0 - 0.5 * hill(RBPJ, n, ec50))
        * nicd_mod,
        0, 1
    )

    # --- EndMT ---
    EndMT_inf = np.clip((hill(Snai1, n, ec50) + hill(Snai2, n, ec50)) / 2.0, 0, 1)

    # ==================================================
    # Assemble dy/dt
    # ==================================================
    dy = np.zeros_like(y)

    fast_nodes = {
        "JAG":             JAG_inf,
        "DLL":             DLL_inf,
        "NOTCH_receptors": NOTCH_receptors_inf,
        "LRP":             LRP_inf,
        "TGFb_TypeI":      TGFb_TypeI_inf,
        "TGFb_TypeII":     TGFb_TypeII_inf,
        "TGFb_123":        TGFb_123_inf,
        "BMP_TypeI":       BMP_TypeI_inf,
        "BMP_TypeII":      BMP_TypeII_inf,
        "BMP_2456":        BMP_2456_inf,
    }
    for name, inf in fast_nodes.items():
        dy[IDX[name]] = (inf - y[IDX[name]]) / p.tau_signal

    medium_nodes = {
        "NICD":         NICD_inf,
        "MAML":         MAML_inf,
        "RBPJ":         RBPJ_inf,
        "HEY":          HEY_inf,
        "beta_catenin": beta_catenin_inf,
        "TCF_LEF":      TCF_LEF_inf,
        "YAP_TAZ":      YAP_TAZ_inf,
        "SMAD23":       SMAD23_inf,
        "SMAD4":        SMAD4_inf,
        "SMAD67":       SMAD67_inf,
        "SMAD158":      SMAD158_inf,
    }
    for name, inf in medium_nodes.items():
        dy[IDX[name]] = (inf - y[IDX[name]]) / p.tau_tf

    slow_nodes = {
        "Snai1": Snai1_inf,
        "Snai2": Snai2_inf,
        "EndMT": EndMT_inf,
    }
    for name, inf in slow_nodes.items():
        dy[IDX[name]] = (inf - y[IDX[name]]) / p.tau_output

    return dy


# -----------------------------
# Simulator
# -----------------------------
def simulate(shear_stress, mech_stress, p=None, t_end=300.0, y0=None):
    if p is None:
        p = Params()
    if y0 is None:
        y0 = np.full(len(NODES), 0.05)
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, shear_stress=shear_stress, mech_stress=mech_stress, p=p),
        t_span=(0.0, t_end),
        y0=y0,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.t, sol.y, sol.y[:, -1]


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    import os
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    p = Params()
    looksmaxxed_plot_style()

    output_nodes  = ["Snai1", "Snai2", "EndMT"]
    output_colors = {n: custom_color_palette[n] for n in output_nodes}

    shear_vals_phys = np.linspace(0.0, SHEAR_MAX, 51)
    mech_vals_phys  = np.linspace(0.0, MECH_MAX,  51)

    shear_mid = SHEAR_MAX / 6   # 5 Dynes/cm²
    mech_mid  = MECH_MAX  * 0.4  # 40 Pa

    # PLOT 1: Shear sweep
    results_shear = {n: [] for n in output_nodes}
    for S in shear_vals_phys:
        _, _, yss = simulate_physical(shear_dyn=S, mech_pa=mech_mid, p=p, t_end=400.0)
        for n in output_nodes:
            results_shear[n].append(yss[IDX[n]])

    fig, ax = plt.subplots()
    for n in output_nodes:
        ax.plot(shear_vals_phys, results_shear[n], label=n, color=output_colors[n])
    ax.set_xlabel("Shear Stress (Dynes/cm²)")
    ax.set_ylabel("Node Activity")
    ax.set_title(f"Shear Stress Sweep\n(Mech Stress = {mech_mid:.0f} Pa)")
    ax.set_xlim(0, SHEAR_MAX); ax.set_ylim(0, 1)
    ax.margins(x=0, y=0)
    ax.spines["bottom"].set_zorder(10)
    ax.legend(fontsize=14, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(bottom=0.18, right=0.8)
    plt.savefig(os.path.join(OUT_DIR, "plot_shear_sweep.png"), bbox_inches="tight")
    plt.close()

    # PLOT 2: Mech sweep
    results_mech = {n: [] for n in output_nodes}
    for M in mech_vals_phys:
        _, _, yss = simulate_physical(shear_dyn=shear_mid, mech_pa=M, p=p, t_end=400.0)
        for n in output_nodes:
            results_mech[n].append(yss[IDX[n]])

    fig, ax = plt.subplots()
    for n in output_nodes:
        ax.plot(mech_vals_phys, results_mech[n], label=n, color=output_colors[n])
    ax.set_xlabel("Mechanical Stress/Strain (Pa)")
    ax.set_ylabel("Node Activity")
    ax.set_title(f"Mechanical Stress Sweep\n(Shear Stress = {shear_mid:.0f} Dynes/cm²)")
    ax.set_xlim(0, MECH_MAX); ax.set_ylim(0, 1)
    ax.margins(x=0, y=0)
    ax.spines["bottom"].set_zorder(10)
    ax.legend(fontsize=14, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(bottom=0.18, right=0.8)
    plt.savefig(os.path.join(OUT_DIR, "plot_mech_sweep.png"), bbox_inches="tight")
    plt.close()

    # PLOT 3: Timecourse
    shear_tc = 24.0
    mech_tc  = 80.0
    t, y_traj, _ = simulate_physical(shear_dyn=shear_tc, mech_pa=mech_tc, p=p, t_end=80.0)

    fig, ax = plt.subplots()
    for n in output_nodes:
        ax.plot(t, y_traj[IDX[n], :], label=n, color=output_colors[n])
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Node Activity")
    ax.set_title(f"Time Course\n(Shear = {shear_tc} Dynes/cm², Mech = {mech_tc} Pa)")
    ax.set_xlim(0, t[-1]); ax.set_ylim(0, 1)
    ax.margins(x=0, y=0)
    ax.spines["bottom"].set_zorder(10)
    ax.legend(fontsize=14, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(bottom=0.18, right=0.8)
    plt.savefig(os.path.join(OUT_DIR, "plot_timecourse.png"), bbox_inches="tight")
    plt.close()

    # PLOT 4: Heatmap
    N_grid     = 30
    ss_phys    = np.linspace(0.0, SHEAR_MAX, N_grid)
    ms_phys    = np.linspace(0.0, MECH_MAX,  N_grid)
    endmt_grid = np.zeros((N_grid, N_grid))

    for i, S in enumerate(ss_phys):
        for j, M in enumerate(ms_phys):
            _, _, yss = simulate_physical(shear_dyn=S, mech_pa=M, p=p, t_end=400.0)
            endmt_grid[j, i] = yss[IDX["EndMT"]]

    vmin, vmax = endmt_grid.min(), endmt_grid.max()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        endmt_grid, origin="lower",
        extent=[0, SHEAR_MAX, 0, MECH_MAX],
        aspect="auto", cmap="plasma", vmin=vmin, vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("EndMT Activity", fontsize=14)
    ax.set_xlabel("Shear Stress (Dynes/cm²)")
    ax.set_ylabel("Mechanical Stress/Strain (Pa)")
    ax.set_title("EndMT Steady State\n(2D Input Sweep)")
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(os.path.join(OUT_DIR, "plot_endmt_heatmap.png"), bbox_inches="tight")
    plt.close()

    print(f"\nAll plots saved to: {OUT_DIR}")
    print(f"\nSteady-state at Shear=24 Dynes/cm², Mech=80 Pa:")
    _, _, yss_hi = simulate_physical(shear_dyn=24.0, mech_pa=80.0, p=p, t_end=400.0)
    for n in output_nodes:
        print(f"  {n:10s}: {yss_hi[IDX[n]]:.4f}")
