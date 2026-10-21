"""v1 diagnostics, part 3: near-wall tangential-velocity profile in the peak-shear region.

For interior nodes within 60 um of the cushion whose nearest outline vertex j sits where
|tau_true| > 50% of its max (the peak region that dominates the error), report, per distance
bin, true and predicted u_t normalised by the TRUE wall shear times distance (u_t / (tau d)):
a linear wall layer gives 1 near the wall. Also U = u/m (the network's free field under the
tanh(d/delta) mask) normalised by tau*delta: the value the network must output at the wall is 1.
"""
import os, sys, pathlib
import numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
from pinn_surrogate.data import load_packed
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.evaluate import predict_wall, frozen
from pinn_surrogate.model import load_checkpoint
torch.set_num_threads(4)
pk = load_packed(); nd = pk.nondim(); Hum = pk.h * 1e6
run = sys.argv[1]; split = sys.argv[2]
model, _ = load_checkpoint(pathlib.Path(run) / "model.pt"); dl = model.cfg.delta
idx = pk.split_indices(split) if split != "train" else pk.train_subset(160)[::4]
s = SolveSet.from_packed(pk, nd, idx)
edges = np.array([2, 4, 7, 10, 15, 20, 30, 45, 60]) / Hum
R_t, R_p, U_t, U_p = [[[] for _ in range(len(edges) - 1)] for _ in range(4)]
with frozen(model):
    for i in range(len(s)):
        sl = slice(i, i + 1)
        xy = s.xy[i].numpy(); ol = s.outline[i].numpy(); n = s.normals[i].numpy()
        n = n / np.linalg.norm(n, axis=1, keepdims=True); t = np.stack([n[:, 1], -n[:, 0]], 1)
        tau = s.wall_tau_t[i].numpy(); tau_p = predict_wall(model, s, i)["tau_t"]
        with torch.no_grad():
            u, v, _ = (q[0].numpy() for q in model(s.xy[sl], s.outline[sl], s.log_re[sl]))
        dd = np.linalg.norm(xy[:, None, :] - ol[None], axis=-1); j = dd.argmin(1); d = dd.min(1)
        keep = (s.tag[i].numpy() == 0) & (d < edges[-1]) & (np.abs(tau[j]) > 0.5 * np.abs(tau).max())
        for k in np.flatnonzero(keep):
            b = np.searchsorted(edges, d[k]) - 1
            if b < 0: continue
            jj = j[k]; ut = s.uvp[i, k, :2].numpy() @ t[jj]; up = np.array([u[k], v[k]]) @ t[jj]
            R_t[b].append(ut / (tau[jj] * d[k])); R_p[b].append(up / (tau[jj] * d[k]))
            m = np.tanh(d[k] / dl)
            U_t[b].append(ut / m / (tau[jj] * dl)); U_p[b].append(up / m / (tau[jj] * dl))
print(f"{run} [{split}]  delta = {dl} H = {dl*Hum:.0f} um")
print(" d (um)      n    u_t/(tau d): true  pred   |  U/(tau delta): true  pred")
print(f" wall         -                1.00   {np.nan:4}   |                 1.00   (pred tau/true tau at wall, see a2)")
for b in range(len(edges) - 1):
    if R_t[b]:
        print(f" {edges[b]*Hum:4.0f}-{edges[b+1]*Hum:<4.0f} {len(R_t[b]):6d}           {np.median(R_t[b]):5.2f}  {np.median(R_p[b]):5.2f}   |       {np.median(U_t[b]):5.2f}  {np.median(U_p[b]):5.2f}")
