| run | n geoms | physics | train time | val tau relL2 | val p relL2 | val peak loc | val sign agree | val PDE res | test_B tau relL2 | test_B p relL2 | test_B peak loc | test_B sign agree | test_B PDE res |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n20_physon_s0 | 20 | on | 9.9 min | 0.491 | 0.182 | 0.0118 | 0.962 | 1.06 | 0.425 | 0.0935 | 0.0608 | 0.992 | 0.902 |
| n20_physon_s1 | 20 | on | 11.2 min | 0.545 | 0.17 | 0.0176 | 0.946 | 1.19 | 0.535 | 0.111 | 0.0588 | 0.960 | 0.947 |
| n20_physoff_s0 | 20 | off | 1.5 min | 0.492 | 0.183 | 0.0196 | 0.961 | 2.08 | 0.446 | 0.107 | 0.0608 | 0.992 | 1.58 |
| n20_physoff_s1 | 20 | off | 1.4 min | 0.583 | 0.237 | 0.0196 | 0.978 | 1.53 | 0.524 | 0.166 | 0.0745 | 0.992 | 1.19 |
| n80_physon_s0 | 80 | on | 9.7 min | 0.417 | 0.152 | 0.00784 | 0.984 | 0.978 | 0.361 | 0.0909 | 0.0314 | 1.000 | 0.693 |
| n80_physon_s1 | 80 | on | 11.3 min | 0.404 | 0.189 | 0.00784 | 0.991 | 1.31 | 0.37 | 0.0985 | 0.0412 | 1.000 | 1.03 |
| n80_physoff_s0 | 80 | off | 1.4 min | 0.418 | 0.117 | 0.00784 | 0.982 | 1.85 | 0.407 | 0.0727 | 0.0392 | 1.000 | 1.55 |
| n80_physoff_s1 | 80 | off | 2.0 min | 0.394 | 0.114 | 0.00979 | 0.990 | 1.36 | 0.332 | 0.0815 | 0.0314 | 0.996 | 1.09 |
| n160_physon_s0 | 160 | on | 5.2 min | 0.392 | 0.181 | 0.00784 | 0.971 | 1.11 | 0.366 | 0.0984 | 0.0314 | 1.000 | 0.807 |
| n160_physon_s1 | 160 | on | 11.0 min | 0.416 | 0.215 | 0.0118 | 0.986 | 1.36 | 0.38 | 0.112 | 0.0314 | 1.000 | 0.929 |
| n160_physoff_s0 | 160 | off | 1.4 min | 0.375 | 0.123 | 0.00784 | 0.974 | 2.17 | 0.385 | 0.0703 | 0.0294 | 1.000 | 1.65 |
| n160_physoff_s1 | 160 | off | 2.1 min | 0.399 | 0.097 | 0.00784 | 0.990 | 1.56 | 0.343 | 0.0696 | 0.0294 | 1.000 | 1.22 |

Medians over solves. tau/p relL2 at the 256 wall points (p gauge-free: each relative to its own outlet mean); peak loc = |arc-length error of peak |tau||/L; sign agree = fraction of points with |tau| > 1% max where sign matches; PDE res = median per solve of the normalized interior residual sqrt((|r_mom| / s_mom)^2 + (r_div / s_div)^2), s = local term size from the data; the momentum part is Dan's r_rel (FEM median 1.3e-4).
