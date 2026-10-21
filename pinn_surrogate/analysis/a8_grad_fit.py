"""Does the model's near-wall du_t/dn match Dan's? Peak-shear region of test_B, by distance bin.

Prints median of (model du_t/dn) / tau_wall and (Dan du_t/dn) / tau_wall per bin, per run.
"""
import os, sys, pathlib
import numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
from pinn_surrogate import physics
from pinn_surrogate.data import load_packed
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.evaluate import frozen
from pinn_surrogate.model import load_checkpoint
torch.set_num_threads(4)
pk = load_packed(); nd = pk.nondim(); a = pk.arrays
G = np.load("data/packed_v1_grads.npz")["grad"].astype(np.float64) * (pk.h / pk.u_bar)[:, None, None]
idx = pk.split_indices("test_B")[::2]; s = SolveSet.from_packed(pk, nd, idx)
edges = np.array([2, 5, 10, 20, 50])
def dudn(g, t, n):
    return t[:, 0] * (g[:, 0] * n[:, 0] + g[:, 1] * n[:, 1]) + t[:, 1] * (g[:, 2] * n[:, 0] + g[:, 3] * n[:, 1])
for run in sys.argv[1:]:
    model, _ = load_checkpoint(pathlib.Path(run) / "model.pt")
    rt, rp = [[] for _ in edges[:-1]], [[] for _ in edges[:-1]]
    with frozen(model):
        for i in range(len(s)):
            gi = s.index[i]; xy = nd["xy"][gi]; ol = nd["outline"][gi]
            n = np.stack([a["wall_nx"][gi], a["wall_ny"][gi]], 1); n /= np.linalg.norm(n, axis=1, keepdims=True)
            t = np.stack([n[:, 1], -n[:, 0]], 1); tau = nd["wall_tau_t"][gi]; dist = a["dist_arc"][gi] * 1e6
            k = np.flatnonzero((a["tag"][gi] == 0) & (dist < 50) & (dist >= 2))
            j = np.linalg.norm(xy[k, None] - ol[None], axis=-1).argmin(1)
            pk_ = np.abs(tau[j]) > 0.5 * np.abs(tau).max(); k, j = k[pk_], j[pk_]
            xt = s.xy[i:i + 1, k].clone().requires_grad_(True)
            u, v, _ = model(xt, s.outline[i:i + 1], s.log_re[i:i + 1])
            gp = torch.cat([physics.grad(u, xt), physics.grad(v, xt)], -1)[0].detach().numpy()
            b = np.searchsorted(edges, dist[k], side="right") - 1
            dt, dp = dudn(G[gi, k], t[j], n[j]) / tau[j], dudn(gp, t[j], n[j]) / tau[j]
            for bb in range(len(edges) - 1):
                rt[bb] += list(dt[b == bb]); rp[bb] += list(dp[b == bb])
    print(f"{run:24s}", "  ".join(f"{edges[bb]}-{edges[bb+1]}um: Dan {np.median(rt[bb]):.2f} model {np.median(rp[bb]):.2f}" for bb in range(len(edges) - 1)))
