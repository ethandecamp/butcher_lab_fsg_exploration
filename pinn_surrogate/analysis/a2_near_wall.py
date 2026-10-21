"""v1 diagnostics, part 2: where near the wall does the trained model go wrong?"""
import os, sys
import numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
from pinn_surrogate.data import load_packed
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.evaluate import predict_wall, frozen
from pinn_surrogate.model import load_checkpoint
from pinn_surrogate.geometry import polyline_distance
torch.set_num_threads(4)
pk = load_packed(); nd = pk.nondim(); H = pk.h
runs = sys.argv[1:] or ["runs/grid/n160_physoff_s0"]
rng = np.random.default_rng(0)
tr = pk.train_subset(160); tr = np.sort(rng.choice(tr, 80, replace=False))
sets = {"train": SolveSet.from_packed(pk, nd, tr), "test_B": SolveSet.from_packed(pk, nd, pk.split_indices("test_B"))}
bins_um = [0, 2, 5, 10, 20, 50, 1e9]
for run in runs:
    model, _ = load_checkpoint(__import__("pathlib").Path(run) / "model.pt")
    print(f"\n######## {run}")
    for name, s in sets.items():
        dist = pk.arrays["dist_arc"][s.index] * 1e6  # um
        tag = pk.arrays["tag"][s.index]
        acc = {b: [[], [], []] for b in range(len(bins_um) - 1)}
        peak, scale, tau_e, Ut_ratio = [], [], [], []
        with frozen(model):
            for i in range(len(s)):
                sl = slice(i, i + 1)
                with torch.no_grad():
                    u, v, p = (t[0].numpy() for t in model(s.xy[sl], s.outline[sl], s.log_re[sl]))
                    m = torch.tanh(polyline_distance(s.xy[sl], s.outline[sl]) / model.cfg.delta)[0].numpy()
                ut, vt = s.uvp[i, :, 0].numpy(), s.uvp[i, :, 1].numpy()
                sp_t, sp_p = np.hypot(ut, vt), np.hypot(u, v)
                for b in acc:
                    k = (tag == 0)[i] & (dist[i] >= bins_um[b]) & (dist[i] < bins_um[b + 1])
                    if k.sum():
                        acc[b][0].append(np.median(sp_p[k] / np.maximum(sp_t[k], 1e-9)))
                        acc[b][1].append(np.sqrt(np.mean((sp_p[k] - sp_t[k]) ** 2)) / np.sqrt(np.mean(sp_t[k] ** 2)))
                        acc[b][2].append(k.sum())
                w = predict_wall(model, s, i); tp, tt = w["tau_t"], s.wall_tau_t[i].numpy()
                peak.append(np.abs(tp).max() / np.abs(tt).max()); scale.append(tp @ tt / (tt @ tt))
                tau_e.append(np.linalg.norm(tp - tt) / np.linalg.norm(tt))
        print(f"-- {name}: tau relL2 {np.median(tau_e):.3f}  peak ratio {np.median(peak):.2f}  best-fit scale {np.median(scale):.2f}")
        print("   dist bin (um)   nodes/solve  median(|u|pred/|u|true)  relL2 |u|")
        for b, (r, e, n) in acc.items():
            print(f"   {bins_um[b]:>4}-{bins_um[b+1]:<6g}   {np.mean(n):8.0f}     {np.median(r):.3f}                 {np.median(e):.3f}")
