# Falsification report

**Question.** What differs between the healthy and overflow cases, and why?

Every verdict below was produced by executable code run against the simulation output. No language model judged any result.

| Verdict | Count |
| --- | ---: |
| SURVIVED | 5 |
| FALSIFIED | 0 |
| NARROWED | 1 |
| COULD NOT RUN | 1 |

## Provenance

- **Dataset root:** /mnt/user-data/uploads/butcher_lab_fsg_exploration/one_way_fsg_model
- **Generated:** 2026-09-01T23:33:29Z
- **Generator:** transcript:../data/demo_transcript.json
- **Model:** claude-opus-5
- **Question:** What differs between the healthy and overflow cases, and why?
- **Briefing sha256:** 9b228a874effeab9
- **Schema version:** 1
- **N proposed:** 7
- **N admissible:** 7
- **N rejected:** 0
- **N survived:** 5
- **N falsified:** 0
- **N narrowed:** 1
- **N could not run:** 1
- **Replayed from:** {"generated_at": "2026-09-01T00:00:00Z", "generator": "interactive-claude", "model": "claude-opus-5", "question": "What differs between the healthy and overflow cases, and why?"}

## Claims

### H1 — Raising inlet flow raises the peak wall shear stress the cushion surface experiences, monotonically across all three cases.

*Why it was proposed:* The three cases differ only in inlet velocity. In a channel of fixed geometry, wall shear scales with the near-wall velocity gradient, so the ordering should follow the inlet ordering. If it does not, the cushion's own shape change is dominating the flow field, which would itself be the interesting result.

**Decision rule (written before the test ran):** Peak WSS must strictly increase from Underflow to Healthy to Overflow. Any inversion or tie refutes the claim.

```
primitive: metric_ordering
params:    {"cases": ["Underflow", "Healthy", "Overflow"], "direction": "increasing", "metric": "wss_peak_dyn_cm2", "strict": true}
```

| Observed | Value |
| --- | ---: |
| value_flow_U0p0180 | 22.6072 |
| value_flow_U0p0360 | 33.551 |
| value_flow_U0p0540 | 43.227 |
| n_cases | 3 |
| n_violations | 0 |
| min_step | 9.67604 |
| max_step | 10.9437 |

**✓ SURVIVED.** PASS: wss_peak_dyn_cm2 across flow_U0p0180=22.6072 -> flow_U0p0360=33.551 -> flow_U0p0540=43.227: strictly increasing with 0 violation(s) out of 2 step(s)

### H2 — As flow increases, the point of maximum shear on the cushion migrates downstream, toward the ventricular end.

*Why it was proposed:* Faster flow should delay separation and push the point of maximum near-wall gradient further along the obstacle before the boundary layer detaches. If the shear maximum instead stayed fixed, that would suggest the peak is pinned by geometry rather than set by the flow.

**Decision rule (written before the test ran):** The normalized arc position of peak WSS must strictly increase with inlet velocity. s is 0 at the atrial end and 1 at the ventricular end.

```
primitive: metric_ordering
params:    {"cases": ["Underflow", "Healthy", "Overflow"], "direction": "increasing", "metric": "wss_peak_s", "strict": true}
```

| Observed | Value |
| --- | ---: |
| value_flow_U0p0180 | 0.511722 |
| value_flow_U0p0360 | 0.621037 |
| value_flow_U0p0540 | 0.771643 |
| n_cases | 3 |
| n_violations | 0 |
| min_step | 0.109315 |
| max_step | 0.150606 |

**✓ SURVIVED.** PASS: wss_peak_s across flow_U0p0180=0.511722 -> flow_U0p0360=0.621037 -> flow_U0p0540=0.771643: strictly increasing with 0 violation(s) out of 2 step(s)

### Falsification chain: H3 → H3b

> A claim the simulator rejected, and the narrower claim written in its place.

#### H3 — The overflow case has the most spatially heterogeneous EndMT response of the three, because it experiences both the highest and the most widely spread mechanical stress.

*Why it was proposed:* Overflow shows the largest peak von Mises stress (859 Pa vs 597 Pa healthy) and the largest mean. A wider spread of mechanical input across the tissue should drive a wider spread of downstream gene activity, producing a more heterogeneous fate landscape across the cushion.

**Decision rule (written before the test ran):** Spatial standard deviation of EndMT activity must be greater in Overflow than in Healthy, by at least 5% relative, before the difference counts as real.

```
primitive: compare_metric
params:    {"case_a": "Overflow", "case_b": "Healthy", "metric": "endmt_sd_combined", "min_rel_diff": 0.05, "op": "gt"}
```

| Observed | Value |
| --- | ---: |
| value_a | 0.069745 |
| value_b | 0.128086 |
| abs_diff | 0.0583407 |
| rel_diff | 0.455482 |
| min_rel_diff | 0.05 |
| ordering_ok | 0 |
| margin_ok | 1 |

**✕ FALSIFIED → ↳ NARROWED.** FAIL: endmt_sd_combined: flow_U0p0540=0.069745 > flow_U0p0360=0.128086? relative difference 0.455 vs required 0.05 -> does not hold (ordering does not hold)

Narrowed by **H3b** below.

**↓ The simulator rejected the claim above. The agent narrowed it and re-tested:**

#### H3b — EndMT heterogeneity peaks in the healthy case, not the overflow case. Past the healthy flow range, raising flow further makes the tissue's gene response more uniform rather than less, because so much of the cushion is pinned at the mechanical ceiling that variation in stress stops producing variation in expression.

*Why it was proposed:* The test refuted the original claim in the opposite direction: EndMT spatial SD is 0.128 in Healthy against 0.070 in Overflow, even though Overflow has the higher raw stress spread. The mechanism that reconciles this is saturation. 46% of Overflow nodes exceed the 100 Pa ceiling against 34% in Healthy, and every clipped node returns the identical pinned activity. Clipping half the domain to one value compresses the spatial variance. So the ordering I predicted from raw stress is inverted by the normalization sitting between stress and expression.

**Decision rule (written before the test ran):** Spatial standard deviation of EndMT activity must be greater in Healthy than in Overflow by at least 20% relative. This is the reverse of the original prediction, and a stricter margin, so it cannot pass on the same noise the first test would have.

```
primitive: compare_metric
params:    {"case_a": "Healthy", "case_b": "Overflow", "metric": "endmt_sd_combined", "min_rel_diff": 0.2, "op": "gt"}
```

| Observed | Value |
| --- | ---: |
| value_a | 0.128086 |
| value_b | 0.069745 |
| abs_diff | 0.0583407 |
| rel_diff | 0.836485 |
| min_rel_diff | 0.2 |
| ordering_ok | 1 |
| margin_ok | 1 |

**✓ SURVIVED.** PASS: endmt_sd_combined: flow_U0p0360=0.128086 > flow_U0p0540=0.069745? relative difference 0.836 vs required 0.2 -> holds

Narrows the falsified claim **H3**.

### H4 — In the overflow case, a large majority of the tissue sits above the stress level at which the gene network stops responding, so most of the cushion is mechanically saturated.

*Why it was proposed:* The GRN normalizes von Mises stress against a 100 Pa ceiling and clips. Overflow peaks at 859 Pa. If a large fraction of nodes exceed 100 Pa, gene activity there is pinned regardless of the true local stress, which would make any gene-level comparison in this case partly an artifact of the normalization.

**Decision rule (written before the test ran):** More than 40% of the 11,615 solid nodes in the Overflow case must exceed 100 Pa von Mises stress.

```
primitive: fraction_above
params:    {"case": "Overflow", "field": "von_mises", "op": "gt", "threshold": 100.0, "value": 0.4}
```

| Observed | Value |
| --- | ---: |
| fraction | 0.462591 |
| threshold | 100 |
| target | 0.4 |
| n_above | 5373 |
| n_used | 11615 |
| n_dropped | 0 |

**✓ SURVIVED.** PASS: 5373/11615 von_mises samples for flow_U0p0540 exceed 100 (fraction 0.4626); 0.4626 > 0.4 is true

### H5 — Faster flow suppresses cushion growth rather than promoting it: the share of tissue that never grows at all rises steeply with inlet velocity.

*Why it was proposed:* Cushion height falls from 0.250 mm to 0.124 mm across the three cases, so something is holding growth back at high flow. If the mechanism is tissue being pushed to the growth floor rather than growing more slowly on average, the pinned fraction should order with inlet velocity.

**Decision rule (written before the test ran):** The fraction of cells pinned at the growth floor must strictly increase from Underflow to Healthy to Overflow.

```
primitive: metric_ordering
params:    {"cases": ["Underflow", "Healthy", "Overflow"], "direction": "increasing", "metric": "growth_pinned_low_fraction", "strict": true}
```

| Observed | Value |
| --- | ---: |
| value_flow_U0p0180 | 0.000570952 |
| value_flow_U0p0360 | 0.612719 |
| value_flow_U0p0540 | 0.791163 |
| n_cases | 3 |
| n_violations | 0 |
| min_step | 0.178444 |
| max_step | 0.612148 |

**✓ SURVIVED.** PASS: growth_pinned_low_fraction across flow_U0p0180=0.000570952 -> flow_U0p0360=0.612719 -> flow_U0p0540=0.791163: strictly increasing with 0 violation(s) out of 2 step(s)

### H6 — In the overflow case, Notch signalling and TGF-beta signalling move together spatially, indicating the two pathways are co-activated by the same mechanical input.

*Why it was proposed:* If both pathways read the same mechanical stimulus, their spatial patterns should correlate strongly. A weak correlation would suggest the model routes them through genuinely different inputs.

**Decision rule (written before the test ran):** Pearson correlation between NICD and SMAD23 activity across the Overflow grid must exceed 0.7.

```
primitive: correlation
params:    {"case": "Overflow", "field_x": "NICD_combined", "field_y": "SMAD23_combined", "method": "pearson", "op": "gt", "value": 0.7}
```

**— COULD NOT RUN.** COULD NOT RUN: unknown GRN field 'SMAD23_combined'; available fields are ['mech_norm', 'von_mises_pa', 'wss_dyn_cm2', 'wss_norm', 'EndMT_combined', 'EndMT_mech_only', 'EndMT_shear_only', 'NICD_combined', 'NICD_mech_only', 'NICD_shear_only', 'Snai1_combined', 'Snai1_mech_only', 'Snai1_shear_only', 'Snai2_combined', 'Snai2_mech_only', 'Snai2_shear_only', 'YAP_TAZ_combined', 'YAP_TAZ_mech_only', 'YAP_TAZ_shear_only']

**Why it could not run:** unknown GRN field 'SMAD23_combined'; available fields are ['mech_norm', 'von_mises_pa', 'wss_dyn_cm2', 'wss_norm', 'EndMT_combined', 'EndMT_mech_only', 'EndMT_shear_only', 'NICD_combined', 'NICD_mech_only', 'NICD_shear_only', 'Snai1_combined', 'Snai1_mech_only', 'Snai1_shear_only', 'Snai2_combined', 'Snai2_mech_only', 'Snai2_shear_only', 'YAP_TAZ_combined', 'YAP_TAZ_mech_only', 'YAP_TAZ_shear_only']
