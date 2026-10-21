"""v1 diagnostics, part 5: absolute (not relative) velocity error by wall distance, in units of U_bar."""
import os, sys, pathlib
import numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
from pinn_surrogate.data import load_packed
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.model import load_checkpoint
torch.set_num_threads(4)
pk = load_packed(); nd = pk.nondim()
s = SolveSet.from_packed(pk, nd, pk.split_indices("test_B"))
model, _ = load_checkpoint(pathlib.Path(sys.argv[1]) / "model.pt")
dist = pk.arrays["dist_arc"][s.index] * 1e6; tag = pk.arrays["tag"][s.index]
with torch.no_grad():
    u, v, _ = model(s.xy, s.outline, s.log_re)
eu = np.hypot(u.numpy() - s.uvp[..., 0].numpy(), v.numpy() - s.uvp[..., 1].numpy()); sp = np.hypot(s.uvp[..., 0].numpy(), s.uvp[..., 1].numpy())
for lo, hi in [(2, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]:
    k = (tag == 0) & (dist >= lo) & (dist < hi)
    print(f"{lo:>3}-{hi:<5g} um: RMS |err| {np.sqrt((eu[k]**2).mean()):.3f}   RMS |u| {np.sqrt((sp[k]**2).mean()):.3f}   ratio {np.sqrt((eu[k]**2).mean()/(sp[k]**2).mean()):.2f}")
