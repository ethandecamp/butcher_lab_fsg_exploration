"""Compute the data behind figures/v2/*.png -> figures/v2/fig_data.json (torch needed)."""
import json, os, sys, pathlib
import numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
from pinn_surrogate import physics
from pinn_surrogate.data import load_packed
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.evaluate import frozen, predict_wall
from pinn_surrogate.metrics import arc_fraction
from pinn_surrogate.model import load_checkpoint
torch.set_num_threads(4)
pk = load_packed(); nd = pk.nondim(); a = pk.arrays
out = {}
# A: wall curves for one FSG-derived test shape, all 4 flows
GID = 354
sol = [i for i in pk.split_indices("test_B") if a["geom_id"][i] == GID]
sA = SolveSet.from_packed(pk, nd, np.array(sol))
runs_A = {"v1": "runs/grid/n160_physoff_s0", "v2": "runs/tau/w1_s0"}
A = {"Re": [float(x) for x in sA.re], "s": [], "true": [], **{k: [] for k in runs_A}}
models = {k: load_checkpoint(pathlib.Path(r) / "model.pt")[0] for k, r in runs_A.items()}
for i in range(len(sA)):
    A["s"].append(arc_fraction(sA.outline[i].numpy()).tolist()); A["true"].append(sA.wall_tau_t[i].numpy().tolist())
    for k, m in models.items():
        with frozen(m):
            A[k].append(predict_wall(m, sA, i)["tau_t"].tolist())
out["A"] = A
# C: du_t/dn / tau_wall vs distance in the peak region (test_B, every 2nd solve)
G = np.load("data/packed_v1_grads.npz")["grad"].astype(np.float64) * (pk.h / pk.u_bar)[:, None, None]
idx = pk.split_indices("test_B")[::2]; s = SolveSet.from_packed(pk, nd, idx)
edges = np.array([2, 5, 10, 20, 50])
def dudn(g, t, n):
    return t[:, 0] * (g[:, 0] * n[:, 0] + g[:, 1] * n[:, 1]) + t[:, 1] * (g[:, 2] * n[:, 0] + g[:, 3] * n[:, 1])
runs_C = {"v1": "runs/grid/n160_physoff_s0", "tau": "runs/tau/w1_s0", "tau_grad": "runs/grad/t1g1_s0"}
C = {"edges_um": edges.tolist(), "dan": None}
for name, run in runs_C.items():
    m, _ = load_checkpoint(pathlib.Path(run) / "model.pt")
    rt, rp, wall = [[] for _ in edges[:-1]], [[] for _ in edges[:-1]], []
    with frozen(m):
        for i in range(len(s)):
            gi = s.index[i]; xy = nd["xy"][gi]; ol = nd["outline"][gi]
            n = np.stack([a["wall_nx"][gi], a["wall_ny"][gi]], 1); n /= np.linalg.norm(n, axis=1, keepdims=True)
            t = np.stack([n[:, 1], -n[:, 0]], 1); tau = nd["wall_tau_t"][gi]; dist = a["dist_arc"][gi] * 1e6
            pkm = np.abs(tau) > 0.5 * np.abs(tau).max()
            wall += list(predict_wall(m, s, i)["tau_t"][pkm] / tau[pkm])
            k = np.flatnonzero((a["tag"][gi] == 0) & (dist < 50) & (dist >= 2))
            j = np.linalg.norm(xy[k, None] - ol[None], axis=-1).argmin(1)
            keep = pkm[j]; k, j = k[keep], j[keep]
            xt = s.xy[i:i + 1, k].clone().requires_grad_(True)
            u, v, _ = m(xt, s.outline[i:i + 1], s.log_re[i:i + 1])
            gp = torch.cat([physics.grad(u, xt), physics.grad(v, xt)], -1)[0].detach().numpy()
            b = np.searchsorted(edges, dist[k], side="right") - 1
            dt, dp = dudn(G[gi, k], t[j], n[j]) / tau[j], dudn(gp, t[j], n[j]) / tau[j]
            for bb in range(len(edges) - 1):
                rt[bb] += list(dt[b == bb]); rp[bb] += list(dp[b == bb])
    C[name] = {"wall": float(np.median(wall)), "bins": [float(np.median(x)) for x in rp]}
    C["dan"] = [float(np.median(x)) for x in rt]
out["C"] = C
pathlib.Path("figures/v2/fig_data.json").write_text(json.dumps(out))
print("ok", {k: (v["wall"], [round(x, 2) for x in v["bins"]]) for k, v in C.items() if isinstance(v, dict)}, [round(x, 2) for x in C["dan"]])
