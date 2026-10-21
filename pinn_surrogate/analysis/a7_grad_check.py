"""Sanity check of Dan's projected velocity gradients near the wall: du_t/dn vs his wall tau.

In the peak-shear region (|tau| > 50 % max at the nearest outline vertex), du_t/dn at interior
nodes should approach tau_wall as d -> 0 (a3 shows the profile is ~linear out to ~20 um).
"""
import os, sys
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
from pinn_surrogate.data import load_packed
pk = load_packed(); nd = pk.nondim(); a = pk.arrays
G = np.load("data/packed_v1_grads.npz")["grad"].astype(np.float64) * (pk.h / pk.u_bar)[:, None, None]
edges = np.array([0, 2, 5, 10, 20, 50]); res = {b: [] for b in range(5)}
for i in range(0, 960, 3):
    xy = nd["xy"][i]; ol = nd["outline"][i]; n = np.stack([a["wall_nx"][i], a["wall_ny"][i]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True); t = np.stack([n[:, 1], -n[:, 0]], 1)
    tau = nd["wall_tau_t"][i]; dist = a["dist_arc"][i] * 1e6
    k = (a["tag"][i] == 0) & (dist < 50)
    j = np.linalg.norm(xy[k, None] - ol[None], axis=-1).argmin(1)
    g = G[i, k]; tt, nn = t[j], n[j]
    dudn = tt[:, 0] * (g[:, 0] * nn[:, 0] + g[:, 1] * nn[:, 1]) + tt[:, 1] * (g[:, 2] * nn[:, 0] + g[:, 3] * nn[:, 1])
    peak = np.abs(tau[j]) > 0.5 * np.abs(tau).max()
    for b in range(5):
        m = peak & (dist[k] >= edges[b]) & (dist[k] < edges[b + 1])
        res[b] += list(dudn[m] / tau[j][m])
for b in range(5):
    r = np.array(res[b])
    if len(r):
        print(f"{edges[b]:>2}-{edges[b+1]:<2} um  n={len(r):6d}  median du_t/dn / tau_wall = {np.median(r):.3f}  IQR {np.percentile(r,25):.3f}-{np.percentile(r,75):.3f}")
