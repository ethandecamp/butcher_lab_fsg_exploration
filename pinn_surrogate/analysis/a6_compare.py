"""Compare runs from eval.json (no torch): medians on val and test_B, per-flow tau, paired win share."""
import json, sys
import numpy as np
runs = sys.argv[1:]
base = runs[0]
def rows(r, sp): return json.load(open(f"{r}/eval.json"))["splits"][sp]["per_solve"]
keys = ["tau_rel_l2", "p_rel_l2", "u_rel_l2", "v_rel_l2", "p_field_rel_l2", "peak_loc_err", "pde_res"]
for sp in ("val", "test_B"):
    print(f"\n[{sp}] medians" + " " * 24 + "  ".join(f"{k[:9]:>9s}" for k in keys) + "   tau by k0..k3            tau better than base")
    b = {x["solve"]: x["tau_rel_l2"] for x in rows(base, sp)}
    for r in runs:
        R = rows(r, sp)
        med = [np.median([x[k] for x in R]) for k in keys]
        byk = [np.median([x["tau_rel_l2"] for x in R if x["k"] == k]) for k in range(4)]
        win = np.mean([x["tau_rel_l2"] < b[x["solve"]] for x in R])
        print(f"  {r:36s}" + "  ".join(f"{m:9.3f}" for m in med) + "   " + " ".join(f"{v:.3f}" for v in byk) + f"   {win:5.0%}")
