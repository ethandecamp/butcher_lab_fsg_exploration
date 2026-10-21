"""Wall-curve data for figures/v3 (torch): v1, v2-small (tau w1 s0), big, on test_B shape g_00354."""
import json, os, sys, pathlib
import numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
from pinn_surrogate.data import load_packed
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.evaluate import frozen, predict_wall
from pinn_surrogate.metrics import arc_fraction
from pinn_surrogate.model import load_checkpoint
torch.set_num_threads(4)
pk = load_packed(); nd = pk.nondim(); a = pk.arrays
out = {}
for gid in (354,):
    sol = np.array([i for i in pk.split_indices("test_B") if a["geom_id"][i] == gid])
    s = SolveSet.from_packed(pk, nd, sol)
    runs = {"v1": "runs/grid/n160_physoff_s0", "v2": "runs/tau/w1_s0", "big": "runs/big"}
    A = {"Re": [float(x) for x in s.re], "s": [arc_fraction(s.outline[i].numpy()).tolist() for i in range(len(s))],
         "true": [s.wall_tau_t[i].numpy().tolist() for i in range(len(s))]}
    for k, r in runs.items():
        m, _ = load_checkpoint(pathlib.Path(r) / "model.pt")
        with frozen(m):
            A[k] = [predict_wall(m, s, i)["tau_t"].tolist() for i in range(len(s))]
    out[str(gid)] = A
pathlib.Path("figures/v3/fig_data.json").write_text(json.dumps(out))
print("ok")
