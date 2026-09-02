# AV cushion FSG + GRN model — summary briefing

A one-way fluid-solid-growth model of a developing heart atrioventricular cushion,
coupled to a 23-node gene regulatory network. Blood flows over a cushion that grows
in response to mechanical stimulus; the GRN reads local wall shear stress and tissue
stress and returns gene activities.

Three cases differ only in inlet velocity: Underflow = 1.8 cm/s, Healthy = 3.6 cm/s, Overflow = 5.4 cm/s. Fifteen growth steps each.
The cushion is a shallow cap, so its surface is described by normalized arc length
s in [0, 1], running from the atrial (upstream) end at s=0 to the ventricular
(downstream) end at s=1.

## 1. Scalar metrics (all three cases, final growth step)

| metric                       | units         |  Underflow |    Healthy |   Overflow |
| ---------------------------- | ------------- | ---------- | ---------- | ---------- |
| arc_length_mm                | mm            |      1.020 |     0.9055 |     0.8741 |
| cushion_height_mm            | mm            |     0.2500 |     0.1686 |     0.1244 |
| endmt_mean_combined          | dimensionless |     0.0715 |     0.3066 |     0.3773 |
| endmt_sd_combined            | dimensionless |     0.0311 |     0.1281 |     0.0697 |
| grn_saturated_fraction       | dimensionless |   0.00e+00 |     0.2964 |     0.4404 |
| growth_mean                  | dimensionless |      1.198 |     0.6879 |     0.5899 |
| growth_pinned_low_fraction   | dimensionless |   5.71e-04 |     0.6127 |     0.7912 |
| mech_clipped_fraction        | dimensionless |   1.72e-04 |     0.3372 |     0.4626 |
| von_mises_lr_asymmetry       | dimensionless |    -0.4580 |    -0.8224 |    -0.7381 |
| von_mises_mean_pa            | Pa            |      6.978 |     83.888 |      108.7 |
| von_mises_peak_pa            | Pa            |      235.8 |      596.7 |      859.0 |
| von_mises_peak_s             | dimensionless |      1.000 |   0.00e+00 |   0.00e+00 |
| wss_lr_asymmetry             | dimensionless |     0.1071 |     0.2283 |     0.0361 |
| wss_mean_dyn_cm2             | dyn/cm^2      |     12.767 |     16.871 |     23.393 |
| wss_peak_dyn_cm2             | dyn/cm^2      |     22.607 |     33.551 |     43.227 |
| wss_peak_s                   | dimensionless |     0.5117 |     0.6210 |     0.7716 |

### What each metric means, and where it comes from

- **arc_length_mm** (mm) — Traced length of the cushion surface outline at the final step.
  - source: cumulative polyline length of the angle-ordered arc; NOT the x-extent chord, which is what metrics_summary.csv's length_mm reports
- **cushion_height_mm** (mm) — Maximum cushion height above the channel floor at the final step.
  - source: max y of step_014/arc_data.npz, converted m -> mm
- **endmt_mean_combined** (dimensionless) — Mean EndMT activity under the combined shear+stress scenario.
  - source: mean of EndMT_combined in results_spatial.csv (80% grid)
- **endmt_sd_combined** (dimensionless) — Spatial standard deviation of EndMT activity under the combined scenario.
  - source: population SD of EndMT_combined in results_spatial.csv (80% grid)
- **grn_saturated_fraction** (dimensionless) — Fraction of GRN grid nodes whose mechanical input has hit its normalization ceiling.
  - source: count(mech_norm >= 1.0) / rows of results_spatial.csv (80% grid denominator; differs from mech_clipped_fraction, which uses the full mesh)
- **growth_mean** (dimensionless) — Mean per-cell growth multiplier at the final step.
  - source: mean of step_014/g_field.npy over 22,769 cells
- **growth_pinned_low_fraction** (dimensionless) — Fraction of cells pinned at the growth floor, i.e. not growing at all.
  - source: count(g <= 0.5 + 1e-9) / 22,769 cells in step_014/g_field.npy; note the shipped floor is 0.5 while solid_solver.py's current default is 0.3, so these runs are not reproducible from the source as it stands
- **mech_clipped_fraction** (dimensionless) — Fraction of solid nodes whose von Mises exceeds the 100 Pa GRN normalization ceiling.
  - source: count(von_mises_Pa > MECH_MAX_PA) / 11,615 solid nodes (full mesh denominator)
- **von_mises_lr_asymmetry** (dimensionless) — Ventricular-minus-atrial mean von Mises stress on the surface, over the overall mean.
  - source: surface-band nodes projected onto arc, halves split at s=0.5
- **von_mises_mean_pa** (Pa) — Mean von Mises stress over the whole cushion at the final step.
  - source: mean of von_mises_Pa over all 11,615 solid nodes (grn_input.npz), clipped at 0
- **von_mises_peak_pa** (Pa) — Maximum von Mises stress anywhere in the cushion at the final step.
  - source: max of von_mises_Pa over all 11,615 solid nodes (grn_input.npz), clipped at 0
- **von_mises_peak_s** (dimensionless) — Normalized arc position of the peak von Mises stress among surface-band nodes.
  - source: argmax over surface-band nodes projected onto the angle-ordered arc
- **wss_lr_asymmetry** (dimensionless) — Ventricular-minus-atrial mean wall shear stress, divided by the overall mean.
  - source: halves split at s=0.5 on the angle-ordered arc from step_014/arc_data.npz
- **wss_mean_dyn_cm2** (dyn/cm^2) — Mean wall shear stress magnitude along the cushion surface at the final step.
  - source: mean of tau_mag from step_014/arc_data.npz, converted Pa -> dyn/cm^2 (x10)
- **wss_peak_dyn_cm2** (dyn/cm^2) — Maximum wall shear stress magnitude on the cushion surface at the final step.
  - source: max of tau_mag from step_014/arc_data.npz, converted Pa -> dyn/cm^2 (x10)
- **wss_peak_s** (dimensionless) — Normalized arc position (0 atrial, 1 ventricular) of the wall shear stress maximum.
  - source: argmax of tau_mag on the angle-ordered arc from step_014/arc_data.npz

## 2. Arc-length profiles (25 points, atrial s=0 to ventricular s=1)

Values are linearly resampled onto a uniform grid in normalized arc length. No
smoothing is applied, so saturation plateaus appear as plateaus.

### wss (dyn/cm^2)

```
s        0.000   0.042   0.083   0.125   0.167   0.208   0.250   0.292   0.333   0.375   0.417   0.458   0.500   0.542   0.583   0.625   0.667   0.708   0.750   0.792   0.833   0.875   0.917   0.958   1.000
Underflow  0.033   2.127   4.564   7.408   9.551  11.166  12.393  13.241  13.325  20.112  21.306  22.156  22.580  22.475  21.820  20.664  19.107  17.261  15.208  12.949  10.371   7.272   3.645   0.483   0.002
Healthy    0.492   7.213  14.123   9.985  14.145  16.465  17.818  18.675  19.110  19.071  18.372  16.667  13.482   7.475  25.451  33.183  30.551  28.359  26.049  23.257  19.588  14.664   8.536   2.479   0.071
Overflow   1.028  10.490  19.545  14.553  19.977  23.455  25.610  27.089  28.164  28.940  29.451  29.666  29.496  28.764  27.113  23.746  17.900  30.925  34.630  39.056  31.068  23.116  14.233   5.253   0.036
```
- source: step_014/arc_data.npz, angle-ordered, linearly resampled from 257 facets onto 25 uniform points
- peak locations: Underflow peaks at s=0.500, Healthy peaks at s=0.625, Overflow peaks at s=0.792

### von_mises (Pa)

```
s        0.000   0.042   0.083   0.125   0.167   0.208   0.250   0.292   0.333   0.375   0.417   0.458   0.500   0.542   0.583   0.625   0.667   0.708   0.750   0.792   0.833   0.875   0.917   0.958   1.000
Underflow 77.459  19.824   8.308   5.473   4.333   3.624   3.213   3.070   2.445  19.679   9.841   8.679   7.379   6.356   5.592   4.592   3.684   3.181   2.823   2.655   2.657   2.788   2.690   4.031 235.837
Healthy  596.660  89.618  59.824  47.432  44.097  35.640  33.510  35.042  32.214  29.817  25.105  21.475  13.427   3.580  11.063  30.476  21.037  21.690  19.702  16.357  14.619  14.019  15.674  21.723 118.096
Overflow 858.976 147.520  89.414  76.478  64.904  54.411  46.948  46.102  42.883  40.646  40.672  37.415  32.402  26.476  18.876  13.031   9.935  25.285  35.820  34.744  28.415  29.418  30.883  42.283 296.247
```
- source: grn_input.npz surface-band nodes (779 of 11615), angle-ordered, linearly resampled onto 25 uniform points
- peak locations: Underflow peaks at s=1.000, Healthy peaks at s=0.000, Overflow peaks at s=0.000

### height (mm)

```
s        0.000   0.042   0.083   0.125   0.167   0.208   0.250   0.292   0.333   0.375   0.417   0.458   0.500   0.542   0.583   0.625   0.667   0.708   0.750   0.792   0.833   0.875   0.917   0.958   1.000
Underflow  0.002   0.041   0.075   0.105   0.132   0.155   0.175   0.192   0.207   0.226   0.238   0.246   0.249   0.250   0.246   0.240   0.229   0.216   0.198   0.177   0.150   0.119   0.084   0.044   0.002
Healthy    0.001   0.026   0.048   0.060   0.075   0.089   0.101   0.110   0.118   0.123   0.127   0.130   0.133   0.139   0.157   0.167   0.168   0.163   0.153   0.139   0.120   0.097   0.068   0.036   0.002
Overflow   0.001   0.024   0.045   0.057   0.070   0.084   0.095   0.104   0.111   0.117   0.121   0.124   0.124   0.124   0.121   0.118   0.115   0.120   0.123   0.118   0.104   0.084   0.060   0.031   0.001
```
- source: step_014/arc_data.npz, angle-ordered, linearly resampled from 257 facets onto 25 uniform points
- peak locations: Underflow peaks at s=0.542, Healthy peaks at s=0.667, Overflow peaks at s=0.500

## 3. Morphology across growth steps

```
step          0      1      2      3      4      5      6      7      8      9     10     11     12     13     14
Underflow 0.250  0.235  0.247  0.249  0.249  0.250  0.250  0.250  0.250  0.250  0.250  0.250  0.250  0.250  0.250
Healthy   0.250  0.166  0.163  0.163  0.162  0.162  0.163  0.165  0.166  0.167  0.168  0.168  0.168  0.169  0.169
Overflow  0.250  0.139  0.133  0.131  0.130  0.128  0.127  0.126  0.126  0.125  0.124  0.124  0.124  0.124  0.124
```
- units: mm. source: max y of each step's arc_data.npz.
- **step 0 is the pristine undeformed cap**, identical across cases by construction.
  It is not a simulation result; growth claims should start from step 1.

## 4. Caveats you must account for

1. **Mechanical input saturates.** The GRN normalizes von Mises stress against a
   ceiling of 100 Pa and clips. Peak stress reaches 597 Pa (healthy) and
   859 Pa (overflow), so the clipped fraction is: Underflow 0.0%, Healthy 33.7%, Overflow 46.3%. Every gene activity at a
   clipped node is pinned at the same value regardless of how high the real stress is.
   A pattern in GRN output across cases may therefore be a property of the clip rather
   than of the biology. Any claim about gene expression in the overflow case has to
   survive this.
2. **Shear input saturates too**, against 30 dyn/cm^2, though far
   less often.
3. **Grid dependence.** von Mises statistics use the full 11,615-node mesh; GRN
   statistics use the 80% grid the GRN was evaluated on. The same physical saturation
   reads differently on the two denominators. Name the grid in any fraction claim.
4. **Only 5 of 23 GRN nodes were written out** (Snai1, Snai2, EndMT, NICD, YAP_TAZ).
   The other 18 were simulated and discarded, so nothing can be claimed about them.
5. **Interior nodes have no shear.** About 94% of the mesh never touches the fluid;
   WSS is NaN there, and the combined GRN scenario feeds those nodes shear = 0, which
   is not the same thing as 'no flow measured'.
6. **Geometry lag.** step_k/arc_data.npz is the solid shape at the end of step k-1.
7. **Growth floor.** The shipped runs pin g at 0.5, while the current source default
   is 0.3, so these runs are not reproducible from the code as it stands.

## 5. What you can test

A claim is admissible only if it compiles into one of the harness primitives with a
decision rule fixed before the test runs. Claims are qualitative and plain-English;
tests are mechanical. The simulator decides the outcome — no language model grades
any claim, including its own.

Registered metrics usable in `compare_metric` and `metric_ordering`:

- `arc_length_mm` (mm)
- `cushion_height_mm` (mm)
- `endmt_mean_combined` (dimensionless)
- `endmt_sd_combined` (dimensionless)
- `grn_saturated_fraction` (dimensionless)
- `growth_mean` (dimensionless)
- `growth_pinned_low_fraction` (dimensionless)
- `mech_clipped_fraction` (dimensionless)
- `von_mises_lr_asymmetry` (dimensionless)
- `von_mises_mean_pa` (Pa)
- `von_mises_peak_pa` (Pa)
- `von_mises_peak_s` (dimensionless)
- `wss_lr_asymmetry` (dimensionless)
- `wss_mean_dyn_cm2` (dyn/cm^2)
- `wss_peak_dyn_cm2` (dyn/cm^2)
- `wss_peak_s` (dimensionless)

Full arrays are not in this briefing but are on disk and can be queried directly:
11,615 solid nodes per case at the final step, 219-259 arc facets per case per step
across 15 steps, 22,769 growth cells per case per step, and roughly 8,900 GRN grid
nodes per case. Ask for a specific array rather than assuming this summary is all
there is.
