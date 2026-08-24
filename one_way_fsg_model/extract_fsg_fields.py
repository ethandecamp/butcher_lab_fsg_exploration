"""
extract_fsg_fields.py

Extracts WSS (wall shear stress) on the arc/cushion surface and Cauchy stress
for all cushion nodes from the final FSG time step, then writes both datasets
to .txt, .npy (.npz), .json, and .csv output files.

Usage:
    python extract_fsg_fields.py

Outputs (written to OUTPUT_DIR):
    wss_arc_surface_final.{txt,npz,json,csv}
    cauchy_stress_cushion_final.{txt,npz,json,csv}
"""

import json
import csv
import numpy as np
import h5py
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FSG_DIR   = Path(__file__).parent / "FSG Results" / "flow_U0p0360"
STEP_DIR  = FSG_DIR / "step_014"          # final growth step
OUTPUT_DIR = FSG_DIR / "extracted_fields"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load WSS arc-surface data (fluid side, cushion boundary)
# ---------------------------------------------------------------------------
arc = np.load(STEP_DIR / "arc_data.npz")

wss_x   = arc["x"]          # node x-coordinate (m)
wss_y   = arc["y"]          # node y-coordinate (m)
tau_mag = arc["tau_mag"]     # WSS magnitude (Pa)
tau_x   = arc["tau_x"]      # WSS x-component (Pa)
tau_y   = arc["tau_y"]      # WSS y-component (Pa)
pressure = arc["p"]          # fluid pressure at surface (Pa)
nx      = arc["nx"]          # surface unit normal, x-component
ny      = arc["ny"]          # surface unit normal, y-component

n_wss = len(wss_x)
print(f"WSS arc surface: {n_wss} nodes")
print(f"  tau_mag range: {tau_mag.min():.4f} – {tau_mag.max():.4f} Pa")

# ---------------------------------------------------------------------------
# 2. Load Cauchy stress and scalar stress fields (solid side, all cushion nodes)
# ---------------------------------------------------------------------------
with h5py.File(STEP_DIR / "solid_fields.h5", "r") as f:
    geom              = f["Mesh/0/mesh/geometry"][:]          # (N, 2)
    cauchy_9          = f["VisualisationVector/11"][:]        # (N, 9) full tensor
    von_mises         = f["VisualisationVector/1"][:]         # (N, 1)
    sigma_max_princ   = f["VisualisationVector/2"][:]         # (N, 1)
    sigma_min_princ   = f["VisualisationVector/3"][:]         # (N, 1)
    sigma_tension     = f["VisualisationVector/4"][:]         # (N, 1)
    sigma_comp_mag    = f["VisualisationVector/5"][:]         # (N, 1)

solid_x = geom[:, 0]
solid_y = geom[:, 1]

# 2D in-plane Cauchy stress components from the 9-component row-major tensor:
#   index layout: [s11, s12, s13, s21, s22, s23, s31, s32, s33]
sigma_xx = cauchy_9[:, 0]   # normal stress, x-direction (Pa)
sigma_xy = cauchy_9[:, 1]   # shear stress (Pa)
sigma_yy = cauchy_9[:, 4]   # normal stress, y-direction (Pa)

# Flatten scalar fields
von_mises      = von_mises.ravel()
sigma_max_princ = sigma_max_princ.ravel()
sigma_min_princ = sigma_min_princ.ravel()
sigma_tension  = sigma_tension.ravel()
sigma_comp_mag = sigma_comp_mag.ravel()

n_solid = len(solid_x)
print(f"Solid cushion nodes: {n_solid}")
print(f"  sigma_xx range: {sigma_xx.min():.2f} – {sigma_xx.max():.2f} Pa")
print(f"  sigma_yy range: {sigma_yy.min():.2f} – {sigma_yy.max():.2f} Pa")
print(f"  von_mises range: {von_mises.min():.2f} – {von_mises.max():.2f} Pa")

# ---------------------------------------------------------------------------
# Helper: write a 2-D array to CSV
# ---------------------------------------------------------------------------
def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# 3. Write WSS outputs
# ---------------------------------------------------------------------------
PA_TO_DYN_CM2 = 10.0   # 1 Pa = 10 dyn/cm²

wss_header = ["x_m", "y_m", "wss_mag_dyn_cm2", "wss_x_dyn_cm2", "wss_y_dyn_cm2",
               "pressure_Pa", "normal_x", "normal_y"]

wss_array = np.column_stack([wss_x, wss_y,
                              tau_mag * PA_TO_DYN_CM2,
                              tau_x   * PA_TO_DYN_CM2,
                              tau_y   * PA_TO_DYN_CM2,
                              pressure, nx, ny])

# .txt  (space-delimited with header)
txt_path = OUTPUT_DIR / "wss_arc_surface_final.txt"
np.savetxt(txt_path, wss_array,
           header=" ".join(wss_header),
           fmt="%.10e", comments="")
print(f"Wrote {txt_path}")

# .npz  (named arrays)
npz_path = OUTPUT_DIR / "wss_arc_surface_final.npz"
np.savez(npz_path,
         x=wss_x, y=wss_y,
         wss_magnitude=tau_mag * PA_TO_DYN_CM2,
         wss_x=tau_x * PA_TO_DYN_CM2,
         wss_y=tau_y * PA_TO_DYN_CM2,
         pressure=pressure, normal_x=nx, normal_y=ny)
print(f"Wrote {npz_path}")

# .json
json_path = OUTPUT_DIR / "wss_arc_surface_final.json"
wss_json = {
    "description": "WSS and fluid pressure on cushion arc surface, final FSG step (step_014)",
    "n_nodes": n_wss,
    "units": {"coordinates": "m", "wss": "dyn/cm2", "pressure": "Pa"},
    "columns": wss_header,
    "data": wss_array.tolist(),
}
with open(json_path, "w") as fh:
    json.dump(wss_json, fh, indent=2)
print(f"Wrote {json_path}")

# .csv
csv_path = OUTPUT_DIR / "wss_arc_surface_final.csv"
write_csv(csv_path, wss_header, wss_array.tolist())
print(f"Wrote {csv_path}")

# ---------------------------------------------------------------------------
# 4. Write Cauchy stress outputs
# ---------------------------------------------------------------------------
solid_header = ["x_m", "y_m",
                "sigma_xx_Pa", "sigma_xy_Pa", "sigma_yy_Pa",
                "von_mises_Pa",
                "sigma_max_principal_Pa", "sigma_min_principal_Pa",
                "sigma_tension_Pa", "sigma_comp_mag_Pa"]

solid_array = np.column_stack([solid_x, solid_y,
                                sigma_xx, sigma_xy, sigma_yy,
                                von_mises,
                                sigma_max_princ, sigma_min_princ,
                                sigma_tension, sigma_comp_mag])

# .txt
txt_path = OUTPUT_DIR / "cauchy_stress_cushion_final.txt"
np.savetxt(txt_path, solid_array,
           header=" ".join(solid_header),
           fmt="%.10e", comments="")
print(f"Wrote {txt_path}")

# .npz
npz_path = OUTPUT_DIR / "cauchy_stress_cushion_final.npz"
np.savez(npz_path,
         x=solid_x, y=solid_y,
         sigma_xx=sigma_xx, sigma_xy=sigma_xy, sigma_yy=sigma_yy,
         von_mises=von_mises,
         sigma_max_principal=sigma_max_princ,
         sigma_min_principal=sigma_min_princ,
         sigma_tension=sigma_tension,
         sigma_comp_mag=sigma_comp_mag,
         cauchy_stress_full=cauchy_9)   # full 9-component tensor also saved
print(f"Wrote {npz_path}")

# .json
json_path = OUTPUT_DIR / "cauchy_stress_cushion_final.json"
solid_json = {
    "description": "Cauchy stress and scalar stress fields for all solid cushion nodes, final FSG step (step_014)",
    "n_nodes": n_solid,
    "units": {"coordinates": "m", "stress": "Pa"},
    "columns": solid_header,
    "notes": {
        "sigma_xx": "Cauchy normal stress, x-direction",
        "sigma_xy": "Cauchy shear stress, xy",
        "sigma_yy": "Cauchy normal stress, y-direction",
        "von_mises": "von Mises equivalent stress",
        "sigma_max_principal": "Maximum principal Cauchy stress",
        "sigma_min_principal": "Minimum principal Cauchy stress",
        "sigma_tension": "Tensile stress measure",
        "sigma_comp_mag": "Compressive stress magnitude",
        "cauchy_stress_full_9comp": "Available in .npz as cauchy_stress_full, shape (N,9), row-major [s11,s12,s13,s21,s22,s23,s31,s32,s33]",
    },
    "data": solid_array.tolist(),
}
with open(json_path, "w") as fh:
    json.dump(solid_json, fh, indent=2)
print(f"Wrote {json_path}")

# .csv
csv_path = OUTPUT_DIR / "cauchy_stress_cushion_final.csv"
write_csv(csv_path, solid_header, solid_array.tolist())
print(f"Wrote {csv_path}")

print("\nDone. All files written to:", OUTPUT_DIR)
