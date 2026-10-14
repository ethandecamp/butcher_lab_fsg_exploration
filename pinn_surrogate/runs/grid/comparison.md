## Physics on vs off, pooled over seeds (paired per solve)

| n | split | metric | median on | median off | on better | sign-test p |
|---|---|---|---|---|---|---|
| 20 | val | tau_rel_l2 | 0.525 | 0.546 | 52% of 320 | 0.4 |
| 20 | val | p_rel_l2 | 0.177 | 0.211 | 65% of 320 | 1.6e-07 |
| 20 | val | peak_loc_err | 0.0157 | 0.0196 | 51% of 320 | 0.7 |
| 20 | val | pde_res | 1.11 | 1.72 | 95% of 320 | 7e-69 |
| 20 | test_B | tau_rel_l2 | 0.477 | 0.481 | 44% of 320 | 0.029 |
| 20 | test_B | p_rel_l2 | 0.104 | 0.135 | 62% of 320 | 2.5e-05 |
| 20 | test_B | peak_loc_err | 0.0588 | 0.0628 | 50% of 320 | 0.95 |
| 20 | test_B | pde_res | 0.925 | 1.42 | 98% of 320 | 2.4e-81 |
| 80 | val | tau_rel_l2 | 0.409 | 0.407 | 47% of 320 | 0.34 |
| 80 | val | p_rel_l2 | 0.172 | 0.116 | 17% of 320 | 4.1e-34 |
| 80 | val | peak_loc_err | 0.00784 | 0.00784 | 50% of 320 | 0.95 |
| 80 | val | pde_res | 1.16 | 1.61 | 80% of 320 | 2.4e-28 |
| 80 | test_B | tau_rel_l2 | 0.363 | 0.374 | 53% of 320 | 0.24 |
| 80 | test_B | p_rel_l2 | 0.0958 | 0.0778 | 30% of 320 | 6.2e-13 |
| 80 | test_B | peak_loc_err | 0.0314 | 0.0353 | 49% of 320 | 0.81 |
| 80 | test_B | pde_res | 0.838 | 1.34 | 82% of 320 | 4.1e-32 |
| 160 | val | tau_rel_l2 | 0.409 | 0.391 | 44% of 320 | 0.038 |
| 160 | val | p_rel_l2 | 0.203 | 0.11 | 14% of 320 | 1.2e-40 |
| 160 | val | peak_loc_err | 0.00978 | 0.00784 | 49% of 320 | 0.79 |
| 160 | val | pde_res | 1.25 | 1.87 | 94% of 320 | 2.8e-65 |
| 160 | test_B | tau_rel_l2 | 0.369 | 0.362 | 48% of 320 | 0.4 |
| 160 | test_B | p_rel_l2 | 0.108 | 0.07 | 17% of 320 | 8.4e-35 |
| 160 | test_B | peak_loc_err | 0.0314 | 0.0294 | 50% of 320 | 1 |
| 160 | test_B | pde_res | 0.865 | 1.44 | 96% of 320 | 1e-72 |

## Per seed, test_B wall shear (median relL2)

| n | seed | on | off | on better | sign-test p | by k (on/off), k=0..3 low->high Re |
|---|---|---|---|---|---|---|
| 20 | 0 | 0.425 | 0.446 | 55% | 0.24 | 0.40/0.43, 0.41/0.44, 0.43/0.47, 0.49/0.53 |
| 20 | 1 | 0.535 | 0.524 | 32% | 1.1e-05 | 0.49/0.49, 0.52/0.50, 0.56/0.52, 0.59/0.60 |
| 80 | 0 | 0.361 | 0.407 | 62% | 0.002 | 0.35/0.36, 0.33/0.37, 0.34/0.40, 0.42/0.49 |
| 80 | 1 | 0.370 | 0.332 | 44% | 0.18 | 0.31/0.29, 0.33/0.31, 0.37/0.36, 0.46/0.47 |
| 160 | 0 | 0.366 | 0.385 | 62% | 0.0033 | 0.37/0.34, 0.35/0.37, 0.34/0.41, 0.41/0.51 |
| 160 | 1 | 0.380 | 0.343 | 33% | 2.4e-05 | 0.33/0.31, 0.35/0.32, 0.38/0.35, 0.48/0.48 |

'on better' = share of solves (ties excluded) where the physics-on model had the lower error. Sign test: two-sided exact binomial against 50%.
