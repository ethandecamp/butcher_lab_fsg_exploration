"""v1 diagnostics, part 4: error by arc region and Re, abs-vs-rel error, trust weights, unused data."""
import os, sys, pathlib, glob
import numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
from pinn_surrogate.data import load_packed, node_weights
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.evaluate import predict_wall, frozen
from pinn_surrogate.model import load_checkpoint
torch.set_num_threads(4)
pk = load_packed(); nd = pk.nondim(); a = pk.arrays
s = SolveSet.from_packed(pk, nd, pk.split_indices("test_B"))
model, _ = load_checkpoint(pathlib.Path("runs/grid/n160_physoff_s0/model.pt"))
prior, _ = load_checkpoint(pathlib.Path("runs/grid/n160_physoff_s0/model.pt"))
with torch.no_grad():
    prior.trunk[-1].weight.zero_(); prior.trunk[-1].bias.zero_()
fr = np.linspace(0, 1, 11); sh = np.zeros(10); tot = 0; resc = {k: [] for k in range(4)}; raw = {k: [] for k in range(4)}
pr_peak = []
with frozen(model), frozen(prior):
    for i in range(len(s)):
        tp = predict_wall(model, s, i)["tau_t"]; tt = s.wall_tau_t[i].numpy()
        t0 = predict_wall(prior, s, i)["tau_t"]; pr_peak.append(np.abs(t0).max() / np.abs(tt).max())
        ol = s.outline[i].numpy(); sf = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ol, axis=0), axis=1))]; sf /= sf[-1]
        e2 = (tp - tt) ** 2; tot += e2.sum()
        for b in range(10): sh[b] += e2[(sf >= fr[b]) & (sf < fr[b + 1] + (b == 9))].sum()
        c = tp @ tt / (tt @ tt); k = int(s.flow_k[i])
        raw[k].append(np.linalg.norm(tp - tt) / np.linalg.norm(tt)); resc[k].append(np.linalg.norm(tp / c - tt) / np.linalg.norm(tt))
print("prior (N=0) median peak ratio on test_B:", round(float(np.median(pr_peak)), 2))
print("share of test_B squared tau error by arc fraction (0 = upstream base, 1 = downstream base):")
print("  " + "  ".join(f"{fr[b]:.1f}-{fr[b+1]:.1f}:{sh[b]/tot:4.0%}" for b in range(10)))
print("tau relL2 by flow k, raw -> after best-fit rescale:", {k: (round(float(np.median(raw[k])), 3), round(float(np.median(resc[k])), 3)) for k in range(4)})
# trust weights in the near-wall layer
dist = a["dist_arc"] * 1e6; w = node_weights(a["r_rel"], a["tag"]); tag = a["tag"]
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]:
    k = (tag == 0) & (dist >= lo) & (dist < hi)
    print(f"trust weight, interior nodes {lo}-{hi:g} um: mean {w[k].mean():.2f}, share <1: {np.mean(w[k] < 1):.0%}, share <0.1: {np.mean(w[k] < 0.1):.0%}")
# what is unused: full mesh near-wall nodes vs packed
for f in sorted(glob.glob("data/full/*.npz")):
    z = np.load(f); keys = z.files
    dk = "dist_arc" if "dist_arc" in keys else None
    print(os.path.basename(f), "keys:", keys[:40])
    if dk:
        d = z[dk] * 1e6; print("   full-mesh nodes:", d.size, " <5um:", (d < 5).sum(), " <20um:", (d < 20).sum(), " <50um:", (d < 50).sum())
k = (tag == 0); print("packed per solve: <5um", ((dist < 5) & k).sum() / len(a['Re']), " <20um", ((dist < 20) & k).sum() / len(a['Re']), " <50um", ((dist < 50)).sum() / len(a['Re']))
print("packed: geoms", len(np.unique(a['geom_id'])), "of 1000; families per split:",
      {sp: np.unique(a['families'][a['family'][a['split'] == c]], return_counts=True)[1].tolist() for c, sp in enumerate(a['split_names'])})
