"""Ripple check: roughness of predicted wall shear along the arc vs the truth's.

roughness = RMS of the second difference of tau_t over the 256 arc points; reported as the
ratio pred / true per solve (1 = as smooth as Dan's FEM, > 1 = spurious ripple).
"""
import os, sys, pathlib
import numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
from pinn_surrogate.data import load_packed
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.evaluate import frozen, predict_wall
from pinn_surrogate.model import load_checkpoint
torch.set_num_threads(4)
pk = load_packed(); nd = pk.nondim()
rough = lambda t: np.sqrt(np.mean(np.diff(t, 2) ** 2))
for run in sys.argv[1:]:
    m, _ = load_checkpoint(pathlib.Path(run) / "model.pt")
    for sp in ("test_B", "val"):
        s = SolveSet.from_packed(pk, nd, pk.split_indices(sp)); r = []
        with frozen(m):
            for i in range(len(s)):
                r.append(rough(predict_wall(m, s, i)["tau_t"]) / rough(s.wall_tau_t[i].numpy()))
        r = np.array(r)
        print(f"{run:28s} {sp:6s} roughness ratio median {np.median(r):.2f}  IQR {np.percentile(r,25):.2f}-{np.percentile(r,75):.2f}  share >1.5: {np.mean(r > 1.5):.0%}")
