# NOTES_metrics_glossary.md

Every scalar metric `llm_insights` computes (`summary.metrics.metric_value` in the card
fixtures), grouped by how it relates to Dan's own `comparison_figures/metrics_summary.csv`.
Values below are the three flow cases in order: Underflow (`flow_U0p0180`) → Healthy
(`flow_U0p0360`) → Overflow (`flow_U0p0540`). See [[metric-reconciliation]] project-memory note
for the reconciliation methodology; this file is the catalog.

---

## Reproduces Dan's own numbers exactly

- `wss_peak_dyn_cm2` — the hardest the blood scrapes anywhere on the cushion surface, rising
  22.607 → 33.551 → 43.227 dyn/cm² with flow.
- `von_mises_peak_pa` — the highest tissue stress anywhere in the cushion, 235.8 → 596.7 →
  859.0 Pa.
- `von_mises_mean_pa` — average tissue stress across all 11,615 nodes, 6.978 → 83.888 →
  108.7 Pa, showing the healthy case is already an order of magnitude above underflow.
- `cushion_height_mm` — how tall the cushion has grown, 0.2500 → 0.1686 → 0.1244 mm, shrinking
  as flow rises.

## Same name as Dan's, different convention — worth asking about

- `wss_mean_dyn_cm2` — average surface shear, 12.767 → 16.871 → 23.393 dyn/cm², computed over
  arc facets where Dan computes it over surface nodes (his: 12.4235 / 16.6812 / 24.4906).
- `arc_length_mm` — the traced length of the cushion outline, 1.020 → 0.9055 → 0.8741 mm, which
  shrinks as the cushion flattens whereas Dan's `length_mm` stays near-constant at ~0.80.
- `wss_peak_s` — where along the cushion the shear maximum sits on a 0-to-1 scale from atrial
  to ventricular end, moving steadily downstream at 0.5117 → 0.6210 → 0.7716, unlike Dan's
  `peak_xnorm` which jumps around (0.100 / 0.378 / 0.035).

## New — no counterpart in Dan's file

- `mech_clipped_fraction` — the share of the full mesh whose stress exceeds the model's 100 Pa
  ceiling and is therefore capped, 0.0002 → 0.3372 → 0.4626.
- `grn_saturated_fraction` — the same saturation measured on the coarser GRN grid instead,
  0.0 → 0.2964 → 0.4404, which is why any "N% of the domain saturates" claim has to say which
  grid it means.
- `endmt_mean_combined` — average EndMT activity across the tissue, 0.0715 → 0.3066 → 0.3773,
  rising with flow as you'd expect.
- `endmt_sd_combined` — how unevenly EndMT is distributed in space, 0.0311 → 0.1281 → 0.0697,
  where the healthy case is roughly twice as heterogeneous as overflow and that inversion is the
  demo's headline.
- `growth_mean` — the average growth multiplier per cell, 1.198 → 0.6879 → 0.5899, so only the
  underflow case is net growing.
- `growth_pinned_low_fraction` — the share of cells stuck at the growth floor and not growing
  at all, 0.0006 → 0.6127 → 0.7912.
- `wss_lr_asymmetry` — how lopsided surface shear is between the ventricular and atrial halves,
  0.1071 → 0.2283 → 0.0361.
- `von_mises_lr_asymmetry` — the same lopsidedness for tissue stress, −0.4580 → −0.8224 →
  −0.7381, negative meaning the atrial half carries more.
- `von_mises_peak_s` — where peak surface stress sits along the arc, 1.000 → 0.0 → 0.0, which is
  degenerate because the maximum lands on the clamped endpoints and so shouldn't carry a claim.
