# LOG

Every number below is pasted command output (run in the cloud container, python 3.13, numpy 2.5.3).

## 2026-10-14 — Stage 1: data pipeline (`data.py`)

Lint/format (the one finding is in the pre-existing `pack.py`, which this stage must not modify):

```
$ ruff format --check src tests && ruff check src tests
6 files already formatted
PLC0206 Extracting value from dictionary without calling `.items()`
   --> src/pinn_surrogate/pack.py:150:17
    |
148 |             tag[i] = idx_tag
149 |             outline[i] = out256
150 |             for key in wall:
    |                 ^^^^^^^^^^^
151 |                 wall[key][i] = a[f"{key}_256"]
    |                 ---------
152 |             scal["Re"][i] = meta["flow"][k]["Re"]
153 |             scal["u_inlet_max"][i] = meta["flow"][k]["u_inlet_max"]
    |
help: Use `for key, value in wall.items()` instead

Found 1 error.
```

Unit-conversion round trip, Re, Poiseuille gradient (dp*/dx* = -12 upstream), trust weights, split integrity:

```
$ PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_data.py' -v
test_poiseuille_pressure_gradient_is_minus_12 (test_data.TestNondim.test_poiseuille_pressure_gradient_is_minus_12)
Upstream of the cushion the flow is plane Poiseuille, where dp*/dx* = -12. ... ok
test_reynolds_matches_stored (test_data.TestNondim.test_reynolds_matches_stored)
Re = rho U H / mu with U = (2/3) u_inlet_max reproduces the stored Re. ... ok
test_round_trip_every_kind (test_data.TestNondim.test_round_trip_every_kind)
Every unit kind round-trips on real full-dump fields. ... ok
test_round_trip_packed (test_data.TestNondim.test_round_trip_packed)
to_dim(to_nondim(q)) == q for every packed SI quantity. ... ok
test_trust_weights (test_data.TestNondim.test_trust_weights)
w = min(1, r0/r_rel) is in (0, 1] and is 1 at or below r0. ... ok
test_no_geometry_in_two_splits (test_data.TestSplits.test_no_geometry_in_two_splits)
Each geometry id belongs to exactly one split. ... ok
test_packed_order_matches_splits_json (test_data.TestSplits.test_packed_order_matches_splits_json)
The packed geometry order equals splits.json (train, val, test_B). ... ok
test_test_b_is_all_family_b (test_data.TestSplits.test_test_b_is_all_family_b)
test_B holds only family B, and family B appears nowhere else. ... ok
test_train_prefixes_are_family_balanced (test_data.TestSplits.test_train_prefixes_are_family_balanced)
train_subset(n) for n = 20, 80, 160 holds n/4 geometries of each of A, C, D, E. ... ok

----------------------------------------------------------------------
Ran 9 tests in 2.825s

OK
```

Family balance of the train prefixes used by the data-size ablation (claim in `pack.py` verified):

```
$ PYTHONPATH=src python3 -c "
import numpy as np
from pinn_surrogate.data import load_packed
pk = load_packed(); fam = pk.arrays['family']; names = pk.arrays['families']
for n in (20, 80, 160):
    f, c = np.unique(fam[pk.train_subset(n)][::4], return_counts=True)
    print(n, 'geoms:', dict(zip(names[f].tolist(), c.tolist())))
"
20 geoms: {'A': 5, 'C': 5, 'D': 5, 'E': 5}
80 geoms: {'A': 20, 'C': 20, 'D': 20, 'E': 20}
160 geoms: {'A': 40, 'C': 40, 'D': 40, 'E': 40}
```

Trust weights w = min(1, 1e-2 / r_rel) by boundary tag over all packed nodes:

```
$ PYTHONPATH=src python3 -c "
import numpy as np
from pinn_surrogate.data import load_packed, trust_weights
pk = load_packed(); a = pk.arrays; w = trust_weights(a['r_rel'])
for t in (0, 10, 11, 12, 13, 14):
    m = a['tag'] == t
    print(f'tag {t:2d}: n={m.sum():8d}  r_rel median={np.median(a[\"r_rel\"][m]):.2e}  frac w<1={np.mean(w[m] < 1):.3f}  mean w={w[m].mean():.3f}')
near = (a['tag'] == 0) & (a['dist_arc'] < 5e-6)
print(f'interior <5um: n={near.sum()}  r_rel median={np.median(a[\"r_rel\"][near]):.2e}  frac w<1={np.mean(w[near] < 1):.3f}')
"
tag  0: n= 2396160  r_rel median=1.34e-04  frac w<1=0.035  mean w=0.984
tag 10: n=   30720  r_rel median=2.13e-05  frac w<1=0.000  mean w=1.000
tag 11: n=   30720  r_rel median=3.75e-02  frac w<1=0.797  mean w=0.428
tag 12: n=  122880  r_rel median=7.48e-05  frac w<1=0.090  mean w=0.957
tag 13: n=  245760  r_rel median=2.93e-02  frac w<1=0.800  mean w=0.458
tag 14: n=  122880  r_rel median=1.63e-04  frac w<1=0.066  mean w=0.964
interior <5um: n=103860  r_rel median=8.32e-03  frac w<1=0.440
```

Nondimensional magnitudes (written to `data/nondim_stats.json`; p_span = per-solve max-min of p*):

```
$ PYTHONPATH=src python3 -m pinn_surrogate.stats
loaded /mnt/attach/outputs/pinn_surrogate/data/packed_v1.npz: 960 solves
quantity                 p0         p1         p5        p50        p95        p99       p100
x                       -10        -10      -7.61    -0.0128       7.61         10         10
y                         0          0    0.00889      0.288      0.957          1          1
u                     -1.12    -0.0146          0      0.474        1.5       2.51       9.52
v                     -2.85      -0.57     -0.212          0      0.288       0.76       2.89
p                      -625     -0.781       23.2        128        608    1.5e+03   2.29e+03
p_span                  244        245        247        300   1.07e+03   1.78e+03   2.63e+03
wall_p                 -627       -8.6       69.8        128        547   1.27e+03   2.21e+03
wall_tau_t            -30.1      -3.44     -0.528       9.12       54.4        125        407
wall_tau_mag        7.9e-12   0.000368      0.047       9.16       54.4        125        407
Re                    0.163      0.173      0.234       3.24       47.8       61.6       64.4
k=0 p                 -2.38       1.87       28.1        134        596   1.46e+03    2.1e+03
k=0 p_span              245        245        248        297        884   1.71e+03    2.1e+03
k=0 wall_tau_t        -2.55     -0.667     -0.203       9.39       51.8        119        326
k=0 wall_tau_mag    7.9e-12   0.000386     0.0501       9.39       51.8        119        326
k=0 Re                0.163      0.166      0.176      0.357      0.656      0.706      0.718
k=1 p                 -2.32        1.8       28.1        133        596   1.45e+03    2.1e+03
k=1 p_span              245        245        247        297        884   1.72e+03   2.11e+03
k=1 wall_tau_t        -2.51     -0.673     -0.209       9.39       51.8        119        326
k=1 wall_tau_mag    9.5e-12   0.000396     0.0502       9.39       51.8        119        326
k=1 Re                0.725      0.734      0.767       1.42       3.06       3.18       3.21
k=2 p                 -54.3       1.55         28        128        596   1.41e+03   2.11e+03
k=2 p_span              244        245        247        297        895   1.72e+03   2.12e+03
k=2 wall_tau_t        -16.4      -1.37     -0.373       9.15       52.7        121        340
k=2 wall_tau_mag   1.13e-11   0.000389     0.0499       9.15       52.7        121        340
k=2 Re                 3.27       3.29       3.51       6.42       13.1       13.9       14.4
k=3 p                  -625      -48.6       12.1        110        657   1.56e+03   2.29e+03
k=3 p_span              244        245        247        309   1.28e+03    2.4e+03   2.63e+03
k=3 wall_tau_t        -30.1      -7.65      -2.73       8.33       60.8        139        407
k=3 wall_tau_mag   1.72e-11   0.000305     0.0388       8.54       60.8        139        407
k=3 Re                 14.5       14.7       15.9         32       61.3       63.8       64.4
wrote /mnt/attach/outputs/pinn_surrogate/data/nondim_stats.json
```

## 2026-10-14 — Stage 2: reference physics (`reference.py`) vs Dan's saved arrays

Agreement table for the three full dumps (residual sign/normalization, wall conventions, outlet BC):

```
$ PYTHONPATH=src python3 -m pinn_surrogate.reference
metric                    full_g_00071_k0    full_g_00071_k3    full_g_00354_k2
Re                              3.822e-01          3.271e+01          5.167e+00
r_mom_x_relL2                   1.156e-07          1.051e-07          8.192e-07
r_mom_y_relL2                   6.615e-08          7.487e-08          4.445e-07
r_mom_maxerr/scale              6.491e-08          6.251e-08          5.032e-08
r_div_relL2                     1.249e-06          1.294e-06          2.227e-06
r_rel_relL2                     7.448e-07          1.124e-06          1.520e-06
r_rel_maxabs_err                4.815e-08          5.830e-08          4.514e-08
nondim_vs_dim_relL2             4.710e-16          3.828e-16          2.510e-15
mid_vs_arc_xy_max_m             1.581e-11          1.581e-11          7.175e-12
normal_min_cos                  1.000e+00          1.000e+00          1.000e+00
p_relL2                         1.622e-08          1.474e-08          1.549e-08
traction_relL2                  7.630e-04          1.652e-03          7.573e-05
tau_t_relL2                     7.010e-03          1.193e-02          7.729e-04
tau_t_maxabs_Pa                 1.900e-02          3.018e+00          1.032e-02
tau_t_max_Pa                    8.011e-01          1.174e+02          6.907e+00
tau_t_worst_at_s/L              1.595e-01          1.595e-01          4.094e-01
tau_t_flipped_relL2             1.998e+00          1.994e+00          2.001e+00
tau_mag_relL2                   7.010e-03          1.193e-02          7.729e-04
outlet_sym_rms/scale            4.439e-02          3.577e-02          3.878e-02
outlet_dn_rms/scale             4.960e-01          4.488e-01          4.938e-01
outlet_max|v|/u_max             1.753e-01          9.013e-02          1.521e-01
```


```
$ PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_reference.py' -v
test_outlet_is_symmetric_traction_free (test_reference.TestOutlet.test_outlet_is_symmetric_traction_free)
sigma n = 0 with the symmetric stress holds; the do-nothing form does not. ... ok
test_momentum_and_divergence_match_dan (test_reference.TestResidual.test_momentum_and_divergence_match_dan)
Pointwise agreement within 2 EPS32 of the local term scale. ... ok
test_nondim_residual_rescales_to_dim (test_reference.TestResidual.test_nondim_residual_rescales_to_dim)
Nondim residual * (mu U / H^2) equals the dimensional residual (float64). ... ok
test_relative_residual_matches_dan (test_reference.TestResidual.test_relative_residual_matches_dan)
r_rel = |r_mom|_2 / sum|terms| reproduces Dan's r_rel to float32 precision. ... ok
test_facets_are_dan_arc_points (test_reference.TestWallTraction.test_facets_are_dan_arc_points)
Arc points are the tag-13 facet midpoints in lines order; normals point the same way. ... ok
test_pressure_matches (test_reference.TestWallTraction.test_pressure_matches)
arc_p is the fluid pressure at the facet midpoint (not -t.n). ... ok
test_signed_shear_matches (test_reference.TestWallTraction.test_signed_shear_matches)
tau_t = t . (n_y, -n_x) within discretization; the flipped sign fails. ... ok
test_traction_vector_matches (test_reference.TestWallTraction.test_traction_vector_matches)
(arc_tx, arc_ty) = sigma n with n into the fluid. ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.583s

OK
```

Full suite (the command in INSTRUCTIONS.md):

```
$ PYTHONPATH=src python3 -m unittest discover -s tests -v
test_poiseuille_pressure_gradient_is_minus_12 (test_data.TestNondim.test_poiseuille_pressure_gradient_is_minus_12)
Upstream of the cushion the flow is plane Poiseuille, where dp*/dx* = -12. ... ok
test_reynolds_matches_stored (test_data.TestNondim.test_reynolds_matches_stored)
Re = rho U H / mu with U = (2/3) u_inlet_max reproduces the stored Re. ... ok
test_round_trip_every_kind (test_data.TestNondim.test_round_trip_every_kind)
Every unit kind round-trips on real full-dump fields. ... ok
test_round_trip_packed (test_data.TestNondim.test_round_trip_packed)
to_dim(to_nondim(q)) == q for every packed SI quantity. ... ok
test_trust_weights (test_data.TestNondim.test_trust_weights)
w = min(1, r0/r_rel) is in (0, 1] and is 1 at or below r0. ... ok
test_no_geometry_in_two_splits (test_data.TestSplits.test_no_geometry_in_two_splits)
Each geometry id belongs to exactly one split. ... ok
test_packed_order_matches_splits_json (test_data.TestSplits.test_packed_order_matches_splits_json)
The packed geometry order equals splits.json (train, val, test_B). ... ok
test_test_b_is_all_family_b (test_data.TestSplits.test_test_b_is_all_family_b)
test_B holds only family B, and family B appears nowhere else. ... ok
test_train_prefixes_are_family_balanced (test_data.TestSplits.test_train_prefixes_are_family_balanced)
train_subset(n) for n = 20, 80, 160 holds n/4 geometries of each of A, C, D, E. ... ok
test_outlet_is_symmetric_traction_free (test_reference.TestOutlet.test_outlet_is_symmetric_traction_free)
sigma n = 0 with the symmetric stress holds; the do-nothing form does not. ... ok
test_momentum_and_divergence_match_dan (test_reference.TestResidual.test_momentum_and_divergence_match_dan)
Pointwise agreement within 2 EPS32 of the local term scale. ... ok
test_nondim_residual_rescales_to_dim (test_reference.TestResidual.test_nondim_residual_rescales_to_dim)
Nondim residual * (mu U / H^2) equals the dimensional residual (float64). ... ok
test_relative_residual_matches_dan (test_reference.TestResidual.test_relative_residual_matches_dan)
r_rel = |r_mom|_2 / sum|terms| reproduces Dan's r_rel to float32 precision. ... ok
test_facets_are_dan_arc_points (test_reference.TestWallTraction.test_facets_are_dan_arc_points)
Arc points are the tag-13 facet midpoints in lines order; normals point the same way. ... ok
test_pressure_matches (test_reference.TestWallTraction.test_pressure_matches)
arc_p is the fluid pressure at the facet midpoint (not -t.n). ... ok
test_signed_shear_matches (test_reference.TestWallTraction.test_signed_shear_matches)
tau_t = t . (n_y, -n_x) within discretization; the flipped sign fails. ... ok
test_traction_vector_matches (test_reference.TestWallTraction.test_traction_vector_matches)
(arc_tx, arc_ty) = sigma n with n into the fluid. ... ok

----------------------------------------------------------------------
Ran 17 tests in 3.229s

OK
```

## 2026-10-14 — Stage 3–5: code written, not yet run

**No torch in the cloud container, so nothing below has been trained or timed.** All torch
code (`physics.py`, `geometry.py`, `model.py`, `dataset.py`, `losses.py`, `train.py`,
`evaluate.py`, `single_case.py`) has been compiled and linted only. Its 23 tests
(`test_*_torch.py`) skip here and must be run on the VM first (INSTRUCTIONS.md §1).

Design values (justified in docstrings/comments): δ = 0.05 (hard no-slip mask
m = tanh(d/δ)); S_u, S_v, S_p = 8, 3, 200; Fourier σ = 2 on (x*/10, y*), 32 frequencies;
trunk 5 × 96 SiLU, 133,443 parameters by hand count (encoder 86,304 + trunk 47,139; the
torch count is printed by `test_model_torch`); momentum residual divided by the solve's
data-derived mean pressure gradient c = (p̄_inlet − p̄_outlet)/20 (range 12.07–114.3, median
14.85 over the packed solves), continuity by c/2, outlet traction by 6.

Torch-free evidence for the design numbers (numpy only, real output):

```
$ PYTHONPATH=src python3 -m pinn_surrogate.geometry_np
loaded /mnt/attach/outputs/pinn_surrogate/data/packed_v1.npz: 960 solves
inlet: min distance to any outline 6.7867 H; worst 1 - m (delta 0.05) = 2.54e-118
required S*N over 640 train solves (delta 0.05):
  u: n 1445680  std 8.8  mean -1.94  |.| p50 0.753  p95 12.4  p99 26.7  max 385
  v: n 1445680  std 2.93  mean 0.29  |.| p50 0.169  p95 5.38  p99 12.5  max 94
  p: n 1597440  std 223  mean 83.2  |.| p50 12.6  p95 482  p99 1.21e+03  max 1.96e+03
```

Finding while checking inputs: Dan's packed wall normals (`wall_nx, wall_ny` = `nx_256, ny_256`)
all point into the fluid, but 13,148 of 245,760 (5 %) are not unit length (min |n| = 0.915;
they are interpolated). τ_t is quadratic in n, so `physics.wall_quantities` now normalizes
them (test `test_non_unit_normals_are_normalized`).

```
$ PYTHONPATH=src python3 -m unittest discover -s tests
Ran 63 tests in 6.161s

OK (skipped=23)
$ ruff check src tests     # only the known pack.py PLC0206 finding (Stage 1)
Found 1 error.
$ ruff format --check src tests
27 files already formatted
```

Still unknown until the VM run: whether the torch tests pass, seconds/step (the 10–15 min per
3000-step physics-on run target is an estimate), memory with two runs in parallel, and every
accuracy number.

## 2026-10-14 — Stages 3–5 run on the Cowork VM (python 3.10, torch 2.14.1+cpu, 4 cores)

All commands from `pinn_surrogate/` with `PYTHONPATH=src`. Outputs pasted, trimmed to the
relevant lines.

### Torch tests

One failure on first run: `test_no_slip_on_cushion_and_walls` asserted u = 0 *exactly* at
segment midpoints, which float32 rounding puts ~6e-8 off the segment (mask ~1e-6 there; the
test's randomized weights inflated u to 0.13). The test now asserts u = v = 0 exactly at the
vertices and walls, and mask < 1e-5 at midpoints.

```
$ python3 -m unittest discover -s tests
Ran 63 tests in 7.197s
OK
```

### Timing probe (physics on, 20 geometries, 4 threads)

```
step    40  loss 8.662  data 3.21 (x1)  pde 5.95 (x0.901)  out 0.103 (x0.9)  0.079 s/step
```

### Stage 3 — data-free single-case PINN (`full_g_00071_k0`, Re 0.38)

```
$ python3 -m pinn_surrogate.single_case --steps 3000 --threads 4 --out runs/single_case
step  3000  pde 0.0117  out 3.86e-06 (x42.5)  tau relL2 1.0019  p relL2 0.4507  0.037 s/step
done: {... "tau_rel_l2": 1.0019, "p_rel_l2": 0.4507, "peak_loc_err": 0.0039, "sign_agree": 0.2918}
```

Diagnostic (`/tmp/diag.py`, model vs the full dump): field relL2 u 0.714, v 1.004; at the shear
peak the tangential velocity at distance 0.1 H from the wall was 9.9e-3 (true flow O(1)); wall
tau true 28.7 vs predicted 0.099. The residual is low because the network found a wrong,
nearly stagnant solution near the cushion — a training failure, not an evaluation bug
(the predicted tau is consistent with the predicted field).

### Stage 4 pilot (n = 160, physics on, original settings: share 1, lambda EMA from 1, per-solve residual scale)

```
val    median: tau 0.9644  p 0.3516  peak 0.1216  sign 0.981  pde 0.334
test_B median: tau 0.9622  p 0.1869  peak 0.1667  sign 0.994  pde 0.303
```

Data loss never fell (1–5 throughout). On training solves the model was *worse* than the
ansatz with N = 0 (Poiseuille prior only), e.g. solve 9: u relL2 0.494 vs 0.143. Indexing of
batches vs targets checked and correct. Physics-off control (same code, 20 geometries,
1500 steps): data loss 1.04 -> 0.13, test_B tau 0.49. So the physics term was overriding the
data. Three changes, each in the code with a comment pointing here:

1. `--physics-share` (gradient budget of each physics term relative to data, Wang et al. use 1).
2. AdaptiveWeights jumps to its target on the first update (it took ~2000 steps to get there
   from lambda = 1 with alpha 0.1; with share 0.1/0.01 lambda was still ~0.5 at step 700).
3. The momentum residual is normalized per node by the local momentum-term size from the data
   (Dan's r_rel denominator; packed as `mom_scale`, `div_scale` by a re-run of `pack.py` with the
   same seeded subset), not by one per-solve constant. With the constant, a few hundred
   near-wall nodes whose terms are ~100x the constant dominated the loss. Dan's FEM in this
   metric: interior median 1.3e-4.

After 1+2 (still per-solve scale) the data fit recovered (data loss 0.2) but lambda_pde fell to
its 1e-3 floor and the residual stayed 40–110: physics effectively off. After 3, 20 geometries,
1500 steps, 8 eval solves (noisy):

```
share 0.1: val tau 0.4694 p 0.2191 pde 1.85 | test_B tau 0.5559 p 0.1301 pde 1.15
share 1.0: val tau 0.5408 p 0.2978 pde 1.65 | test_B tau 0.6582 p 0.1771 pde 1.27
```

Share 0.1 chosen for the grid (now the config default). The pilot and debug runs are kept in
`runs/_debug/`.

### Stage 5 — grid: n in {20, 80, 160} x physics {on, off} x seeds {0, 1}, 3000 steps

Background processes in the Cowork VM die when the launching shell call returns (nohup and
setsid both tested; a resumed run reproduced the step-450 losses exactly before dying again),
so the grid ran as 70–170 s chunks via `ablation_chunked.sh`, resuming from checkpoints every
100 steps. Seed 1 ran partly at 1 thread per run, so its wall-clock times are not comparable.
One chunk had a process killed by the VM (4 runs + an evaluation in 3.9 GB); it resumed.

`runs/grid/ablation_summary.md` (per run) and `runs/grid/comparison.md` (pooled, paired) are
the canonical outputs. Pooled test_B, median over 320 paired solves:

```
| n | split | metric | median on | median off | on better | sign-test p |
| 20 | test_B | tau_rel_l2 | 0.477 | 0.481 | 44% of 320 | 0.029 |
| 20 | test_B | p_rel_l2 | 0.104 | 0.135 | 62% of 320 | 2.5e-05 |
| 20 | test_B | pde_res | 0.925 | 1.42 | 98% of 320 | 2.4e-81 |
| 80 | test_B | tau_rel_l2 | 0.363 | 0.374 | 53% of 320 | 0.24 |
| 80 | test_B | p_rel_l2 | 0.0958 | 0.0778 | 30% of 320 | 6.2e-13 |
| 80 | test_B | pde_res | 0.838 | 1.34 | 82% of 320 | 4.1e-32 |
| 160 | test_B | tau_rel_l2 | 0.369 | 0.362 | 48% of 320 | 0.4 |
| 160 | test_B | p_rel_l2 | 0.108 | 0.07 | 17% of 320 | 8.4e-35 |
| 160 | test_B | pde_res | 0.865 | 1.44 | 96% of 320 | 1e-72 |
```

Seed 0 alone showed physics-on better on test_B wall shear at every n (0.425/0.446,
0.361/0.407, 0.366/0.385); seed 1 reversed it (0.535/0.524, 0.370/0.332, 0.380/0.343). Not
claimed.

### Wall-shear amplitude bias (`python3 -m pinn_surrogate.wall_bias <run dirs>`; first run as an ad-hoc script with the same logic; test_B, 160 solves per model)

```
n160_physon_s0: median peak ratio 0.63 (IQR 0.55-0.71); median best-fit scale 0.65; frac under-predicted peak 0.98
n160_physoff_s0: median peak ratio 0.62 (IQR 0.52-0.76); median best-fit scale 0.63; frac under-predicted peak 0.97
n160_physon_s1: median peak ratio 0.60 (IQR 0.52-0.69); median best-fit scale 0.63; frac under-predicted peak 0.98
n160_physoff_s1: median peak ratio 0.63 (IQR 0.54-0.76); median best-fit scale 0.67; frac under-predicted peak 1.00
n20_physoff_s0: median peak ratio 0.64 (IQR 0.53-0.81); median best-fit scale 0.58; frac under-predicted peak 0.97
```

Peak ratio = max|tau_pred| / max|tau_true| per solve; best-fit scale = argmin_a ||a tau_true −
tau_pred|| (= tau_pred·tau_true / tau_true·tau_true).
