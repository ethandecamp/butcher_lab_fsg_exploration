"""Compare runs on the v1 pack (same nodes for every model, so field errors are comparable).

Per split (train = 128 solves (every 5th, all flows) of the 160 v1 train shapes, val, test_B): median wall-shear relL2,
peak ratio, error after best-fit rescale, wall-pressure relL2 (outlet-gauged), u / v field
relL2, and wall shear by flow index. Writes analysis/a9_<run name>.json.
"""
import json, os, sys, pathlib
import numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
from pinn_surrogate.data import load_packed
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.evaluate import frozen, predict_wall, outlet_mean
from pinn_surrogate.metrics import rel_l2
from pinn_surrogate.model import load_checkpoint
torch.set_num_threads(4)
pk = load_packed(); nd = pk.nondim()
splits = {"train": pk.train_subset(160)[::5], "val": pk.split_indices("val"), "test_B": pk.split_indices("test_B")}
for run in sys.argv[1:]:
    model, _ = load_checkpoint(pathlib.Path(run) / "model.pt")
    res = {}
    for name, idx in splits.items():
        s = SolveSet.from_packed(pk, nd, idx); R = []
        with frozen(model):
            for i in range(len(s)):
                sl = slice(i, i + 1)
                with torch.no_grad():
                    u, v, p = (t[0].numpy() for t in model(s.xy[sl], s.outline[sl], s.log_re[sl]))
                w = predict_wall(model, s, i); tt = s.wall_tau_t[i].numpy(); tp = w["tau_t"]
                true = s.uvp[i].numpy()
                pw = w["p"] - outlet_mean(p, s, i); pt = s.wall_p[i].numpy() - outlet_mean(true[:, 2], s, i)
                c = tp @ tt / (tt @ tt)
                R.append(dict(k=int(s.flow_k[i]), tau=rel_l2(tp, tt), peak=np.abs(tp).max() / np.abs(tt).max(),
                              resc=rel_l2(tp / c, tt), p=rel_l2(pw, pt), u=rel_l2(u, true[:, 0]), v=rel_l2(v, true[:, 1])))
        med = lambda key, rows=R: float(np.median([r[key] for r in rows]))
        res[name] = {k: med(k) for k in ("tau", "peak", "resc", "p", "u", "v")}
        res[name]["tau_by_k"] = [med("tau", [r for r in R if r["k"] == k]) for k in range(4)]
    tag = run.strip("/").replace("runs/", "").replace("/", "_")
    pathlib.Path(f"analysis/a9_{tag}.json").write_text(json.dumps(res, indent=1))
    print(run)
    for name, r in res.items():
        print(f"  {name:6s} tau {r['tau']:.3f}  peak {r['peak']:.2f}  rescaled {r['resc']:.3f}  wall p {r['p']:.3f}  u {r['u']:.3f}  v {r['v']:.3f}  tau by k " + " ".join(f"{x:.3f}" for x in r["tau_by_k"]))
