"""Section 4b of the briefing: the gene network's structure, not just its outputs.

Before this section the briefing carried GRN *outputs* -- three aggregate EndMT scalars
and a saturation fraction -- and no structure at all: no nodes, no pathways, no
parameters. A model reading it could say "EndMT is higher in the overflow case" and
could not say why, because it had not been told that mechanical stress reaches the
outputs through exactly one node. Correlational claims were the only kind available.

Three things go in, and the third is the one with a budget problem:

1. **Topology.** 24 nodes grouped by pathway, and the input wiring in words: which nodes
   read shear, which reads mechanical stress, which three are the outputs.
2. **Parameters.** The ``Params`` values every shipped run used, the integration
   settings, and the two normalization ceilings, pointed back at the saturation caveat.
3. **Per-node statistics** from TASK-013's re-run.

On (3): 19 newly available nodes x 3 cases x 2 statistics is 114 numbers, and dumping
them would roughly double the prompt and bury everything else. So the section reports
the constant nodes as a single list (they are constant *by construction*, which is one
fact rather than N numbers), then names only the k=5 most spatially variable and the
k=5 most flow-sensitive nodes with numbers, and says plainly that the full arrays are
one harness call away. k=5 was chosen because it is the smallest k that still covers
every pathway that varies at all -- see TASK-014's Log.

**This section supersedes caveat 4** ("only 5 of 23 GRN nodes were written out"), which
was true until TASK-013 re-ran the network. The caveats section is frozen, so the
correction is stated here rather than made there.

The statistics degrade rather than break. ``data/grn_full/`` is regenerable output and
is gitignored; without it this section still renders topology and parameters, and says
what command produces the numbers.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from llm_insights.grn import store
from llm_insights.grn.runner import T_END
from llm_insights.io.dataset import MECH_MAX_PA, SHEAR_MAX_DYN_CM2
from llm_insights.summary.metrics import register_metric
from llm_insights.summary.registry import briefing_section

LOG = logging.getLogger(__name__)

#: Rendered between the caveats (40) and the protocol (50), because it corrects one of
#: the caveats and has to be read after it.
ORDER: int = 45

#: How many nodes are named with numbers in each of the two rankings. See the module
#: docstring and TASK-014's Log for why 5.
TOP_K: Final[int] = 5

#: The 24 nodes grouped the way ``networkpoint.py`` groups them in its own comments.
#: Order within a pathway is state-vector order, so this doubles as the node roster.
PATHWAYS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("Notch", ("JAG", "DLL", "NOTCH_receptors", "NICD", "MAML", "RBPJ", "HEY")),
    ("Wnt", ("LRP", "beta_catenin", "TCF_LEF")),
    ("Mechanosensing", ("YAP_TAZ",)),
    ("TGFb", ("TGFb_TypeI", "TGFb_TypeII", "TGFb_123", "SMAD23", "SMAD4")),
    ("BMP", ("BMP_TypeI", "BMP_TypeII", "BMP_2456", "SMAD67", "SMAD158")),
    ("Outputs", ("Snai1", "Snai2", "EndMT")),
)

#: ``networkpoint.Params`` defaults, which every shipped run used unchanged. Restated
#: here so the briefing can quote them; ``tests/test_summary_grn.py`` checks them
#: against the dataclass rather than trusting this tuple.
PARAMS: Final[tuple[tuple[str, float, str], ...]] = (
    ("n", 2.0, "Hill coefficient, every edge"),
    ("ec50", 0.2, "half-activation for internal (node-to-node) edges"),
    ("ec50_input", 0.5, "half-activation for the two physical drives"),
    ("Noggin_Chordin", 0.05, "fixed external BMP antagonist"),
    ("VEGF", 0.30, "fixed external input"),
    ("DKK", 0.1, "fixed external Wnt antagonist"),
    ("WNT", 0.2, "fixed external Wnt ligand"),
    ("Frizzled", 0.5, "fixed external Wnt receptor"),
    ("tau_signal", 0.1, "time constant, hours, for the 10 fast signalling nodes"),
    ("tau_tf", 1.0, "time constant, hours, for the 11 transcription-factor nodes"),
    ("tau_output", 10.0, "time constant, hours, for Snai1, Snai2, EndMT"),
)


# --- Metrics ----------------------------------------------------------------------
#
# Eight, not twenty-one. Each one is a node that sits at a distinct place in the wiring
# and that actually varies over the grid: the two hubs the physical inputs reach first
# (NICD, YAP_TAZ), the transducer demo card H6 was refused over (SMAD23, mean and
# spread), the Wnt hub (beta_catenin), the Notch output that represses BMP (HEY), the
# shear-repressed BMP ligand (BMP_2456), and the inhibitory SMAD that shear raises
# directly (SMAD67). Nodes that are pinned by construction get no metric: a metric whose
# value is the same number in all three cases cannot discriminate between hypotheses.

_SOURCE = "data/grn_full/<case>/activities.csv column {column}, 80% GRN grid (TASK-013 re-run)"


def _register_node_metric(metric: str, node: str, statistic: str, description: str) -> None:
    """Register one per-node scalar over the full-node GRN tables.

    Args:
        metric: Registry key.
        node: Node name, e.g. ``"SMAD23"``.
        statistic: ``"mean"`` or ``"sd"``.
        description: One-line meaning for the briefing's metric glossary.
    """
    column = f"{node}_{store.REFERENCE_SCENARIO}"

    @register_metric(metric, "dimensionless", description, _SOURCE.format(column=column))
    def _fn(ds: Any, case: str, _node: str = node, _stat: str = statistic) -> float:
        del ds
        return store.node_statistic(case, _node, _stat)


_register_node_metric(
    "nicd_mean_combined",
    "NICD",
    "mean",
    "Mean NICD activity, combined scenario. The Notch hub; shear drives it directly.",
)
_register_node_metric(
    "yap_taz_mean_combined",
    "YAP_TAZ",
    "mean",
    "Mean YAP/TAZ activity, combined scenario. The only node mechanical stress drives.",
)
_register_node_metric(
    "smad23_mean_combined",
    "SMAD23",
    "mean",
    "Mean SMAD2/3 activity, combined scenario. The TGFb transducer.",
)
_register_node_metric(
    "smad23_sd_combined",
    "SMAD23",
    "sd",
    "Spatial standard deviation of SMAD2/3 activity, combined scenario.",
)
_register_node_metric(
    "beta_catenin_mean_combined",
    "beta_catenin",
    "mean",
    "Mean beta-catenin activity, combined scenario. The Wnt hub.",
)
_register_node_metric(
    "hey_mean_combined",
    "HEY",
    "mean",
    "Mean HEY activity, combined scenario. Notch's output and the repressor of BMP_2456.",
)
_register_node_metric(
    "bmp2456_mean_combined",
    "BMP_2456",
    "mean",
    "Mean BMP-2/4/5/6 ligand activity, combined scenario. Repressed by HEY and by shear.",
)
_register_node_metric(
    "smad67_mean_combined",
    "SMAD67",
    "mean",
    "Mean inhibitory SMAD6/7 activity, combined scenario. Shear raises it, damping TGFb.",
)


# --- The section ------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format an activity in [0, 1] compactly.

    Args:
        value: The number.

    Returns:
        Four decimal places, or ``"n/a"`` for NaN.
    """
    if value != value:  # NaN
        return "n/a"
    return f"{value:.4f}"


def _statistics_lines(ds: Any) -> list[str]:
    """Render the per-node statistics, or say what would produce them.

    Args:
        ds: The dataset, used only for its case list and condition labels.

    Returns:
        The lines.
    """
    cases = tuple(ds.cases())
    summaries = store.summarize_nodes(cases)
    if not summaries:
        return [
            "**Per-node statistics are not on this machine.** The full 24-node tables are",
            "regenerable output and are gitignored. Build them once with",
            "`python3 -m llm_insights.grn.runner --out data/grn_full` (~30 min on 10 cores);",
            "until then the per-node metrics above read `n/a`.",
            "",
        ]

    constant = [s for s in summaries.values() if s.is_constant]
    varying = [s for s in summaries.values() if not s.is_constant]
    by_space = sorted(varying, key=lambda s: -s.spatial_sd)[:TOP_K]
    by_flow = sorted(varying, key=lambda s: -s.flow_span)[:TOP_K]
    labels = [ds.condition(c) for c in cases]

    lines = [
        f"**Pinned by construction ({len(constant)} of {len(summaries)}).** Identical at every",
        "grid point in every case, so they cannot explain any spatial or cross-case pattern:",
        "  " + ", ".join(f"{s.node} = {s.constant_value:g}" for s in constant) + ".",
        "",
        f"**Most spatially variable ({len(varying)} nodes vary at all).** Standard deviation",
        "across grid points, combined scenario, averaged over the three cases:",
        "",
        "| node | spatial SD | " + " | ".join(f"mean, {x}" for x in labels) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    for s in by_space:
        means = " | ".join(_fmt(s.mean_by_case[c]) for c in cases)
        lines.append(f"| {s.node} | {_fmt(s.spatial_sd)} | {means} |")

    lines += [
        "",
        "**Most flow-sensitive.** Spread of the domain-mean activity across the three inlet",
        "velocities (largest minus smallest), combined scenario. **n = 3**, so this ranks",
        "nodes; it does not establish a trend:",
        "",
        "| node | spread over cases | " + " | ".join(f"mean, {x}" for x in labels) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    for s in by_flow:
        means = " | ".join(_fmt(s.mean_by_case[c]) for c in cases)
        lines.append(f"| {s.node} | {_fmt(s.flow_span)} | {means} |")

    lines += [
        "",
        "Full per-point arrays for **any** of the 24 nodes under **any** of the three",
        "scenarios are on disk at `data/grn_full/<case>/activities.csv` (columns",
        "`<Node>_<scenario>`) and can be requested through the harness. Nothing here is the",
        "only view of a node; these two rankings are a way in, not the whole story.",
        "",
    ]
    return lines


@briefing_section(order=ORDER, name="grn")
def grn_section(ds: Any, n_profile_points: int) -> list[str]:
    """Render the gene-network structure section.

    Args:
        ds: The dataset to describe.
        n_profile_points: Unused; the signature is uniform across sections.

    Returns:
        The section's lines.
    """
    del n_profile_points
    n_nodes = sum(len(nodes) for _, nodes in PATHWAYS)
    out = [
        "## 4b. The gene network: nodes, wiring, parameters",
        "",
        f"A **{n_nodes}-state** ODE system (`networkpoint.py`): each node is an activity in",
        "[0, 1], each edge a Hill function. All 24 are now on disk for all three cases and all",
        "three scenarios, which **supersedes caveat 4 above**.",
        "",
        "**Nodes by pathway**, in state-vector order:",
        "",
    ]
    out += [f"- **{name}** ({len(nodes)}): {', '.join(nodes)}" for name, nodes in PATHWAYS]
    out += [
        "",
        "**Where the two physical inputs enter.** Each point hands the network two numbers:",
        "normalized shear and normalized mechanical stress.",
        "",
        "- **Shear** reaches six nodes directly: `JAG` rises with it and `DLL` falls",
        "  (`DLL = 1 - hill(shear)`), so shear sets the JAG/DLL balance; `NICD` gets a direct",
        "  0.35-weighted shear term; and a KLF2 term (`hill(shear)`) drives `beta_catenin`,",
        "  represses `BMP_2456` and raises `SMAD67`.",
        "- **Mechanical stress** reaches exactly one node directly: `YAP_TAZ`. Every mechanical",
        "  effect in this network is downstream of that one node, so a mechanical claim not",
        "  routed through YAP/TAZ is claiming something the model cannot do.",
        "- **Outputs**: `Snai1` and `Snai2` cross-repress; `EndMT = (hill(Snai1)+hill(Snai2))/2`",
        "  is a readout of those two with no dynamics of its own.",
        "- Five inputs are **fixed scalars, not fields**: VEGF 0.30, WNT 0.20, DKK 0.10,",
        "  Frizzled 0.50, Noggin_Chordin 0.05. Nothing in the FSG model varies them.",
        "",
        "**Parameters** (`networkpoint.Params`, unchanged in every shipped run): "
        + ", ".join(f"`{name}`={value:g}" for name, value, _ in PARAMS)
        + ".",
        "`ec50` governs internal edges and `ec50_input` the two physical drives; the three",
        "taus are hours, for the 10 fast signalling nodes, the 11 transcription factors, and",
        "the 3 outputs. The rest are fixed external ligand levels.",
        "",
        f"Integrated to t = {T_END:g} h (LSODA, rtol 1e-8, atol 1e-10) = 40 output time",
        "constants, so every number below is a steady state, not a transient.",
        "",
        f"**Normalization ceilings**: shear against {SHEAR_MAX_DYN_CM2:g} dyn/cm^2, mechanical",
        f"stress against {MECH_MAX_PA:g} Pa, both clipped to [0, 1]. Read with caveat 1:",
        f"`ec50_input`=0.5 puts half-maximal drive at half the ceiling ({SHEAR_MAX_DYN_CM2/2:g}",
        f"dyn/cm^2, {MECH_MAX_PA/2:g} Pa), and every point above a ceiling is pinned at exactly",
        "the same activity as every other point above it.",
        "",
    ]
    out += _statistics_lines(ds)
    return out


__all__ = ["ORDER", "PARAMS", "PATHWAYS", "TOP_K", "grn_section"]
