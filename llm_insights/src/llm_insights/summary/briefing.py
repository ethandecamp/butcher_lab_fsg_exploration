"""The prompt-sized view of the simulation.

This module answers the question the whole project is really about: what does a
faithful, small description of 27 million field values look like?

The briefing is three nested zoom levels, coarsest first:

1. **Scalars** — every registered metric for all three cases, with units and the
   provenance of each number.
2. **Profiles** — a handful of fields collapsed onto normalized arc length at 25
   points each, which is where spatial claims come from.
3. **Everything else, on request** — the full arrays stay on disk. The briefing tells
   the reader what exists and how to ask for it rather than pretending it is all here.

It also carries a caveats section. Several of the most striking patterns in this
dataset are artifacts of the model's normalization rather than of its biology, and a
summary that surfaced the pattern while hiding the artifact would be actively
misleading — it would invite exactly the confident, wrong hypothesis this project
exists to catch.
"""

from __future__ import annotations

import logging
from typing import Final

from llm_insights.io.dataset import (
    INLET_VELOCITY_M_S,
    MECH_MAX_PA,
    SHEAR_MAX_DYN_CM2,
    Dataset,
)
from llm_insights.summary.metrics import METRICS, METRIC_NAMES, metric_value
from llm_insights.summary.profiles import DEFAULT_N_POINTS, height_trajectory, profile

LOG = logging.getLogger(__name__)

#: Fields profiled in the briefing by default.
BRIEFING_FIELDS: Final[tuple[str, ...]] = ("wss", "von_mises", "height")


def _fmt(value: float) -> str:
    """Format a number compactly without losing meaningful precision."""
    if value != value:  # NaN
        return "n/a"
    a = abs(value)
    if a >= 100:
        return f"{value:.1f}"
    if a >= 1:
        return f"{value:.3f}"
    if a >= 0.001:
        return f"{value:.4f}"
    return f"{value:.2e}"


def _metric_section(ds: Dataset) -> list[str]:
    """Render the scalar metric table with units and provenance."""
    cases = ds.cases()
    labels = [ds.condition(c) for c in cases]
    out = ["## 1. Scalar metrics (all three cases, final growth step)", ""]
    header = f"| {'metric':<28} | {'units':<13} | " + " | ".join(f"{x:>10}" for x in labels) + " |"
    out.append(header)
    out.append(
        f"| {'-' * 28} | {'-' * 13} | " + " | ".join("-" * 10 for _ in labels) + " |"
    )
    for name in METRIC_NAMES:
        md = METRICS[name]
        values = [_fmt(metric_value(ds, name, c)) for c in cases]
        out.append(
            f"| {name:<28} | {md.units:<13} | " + " | ".join(f"{v:>10}" for v in values) + " |"
        )
    out += ["", "### What each metric means, and where it comes from", ""]
    for name in METRIC_NAMES:
        md = METRICS[name]
        out.append(f"- **{name}** ({md.units}) — {md.description}")
        out.append(f"  - source: {md.source}")
    return out


def _profile_section(ds: Dataset, n_points: int) -> list[str]:
    """Render the arc-length profiles for each field and case."""
    out = [
        "",
        f"## 2. Arc-length profiles ({n_points} points, atrial s=0 to ventricular s=1)",
        "",
        "Values are linearly resampled onto a uniform grid in normalized arc length. No",
        "smoothing is applied, so saturation plateaus appear as plateaus.",
        "",
    ]
    for field in BRIEFING_FIELDS:
        first = profile(ds, ds.cases()[0], field, n=n_points)
        out.append(f"### {field} ({first.units})")
        out.append("")
        out.append("```")
        grid = "s      " + " ".join(f"{v:>7.3f}" for v in first.s)
        out.append(grid)
        for case in ds.cases():
            pr = profile(ds, case, field, n=n_points)
            label = f"{pr.condition:<9}"
            row = label + " ".join(f"{v:>7.3f}" for v in pr.values)
            out.append(row)
        out.append("```")
        out.append(f"- source: {first.source}")
        peaks = ", ".join(
            f"{ds.condition(c)} peaks at s={profile(ds, c, field, n=n_points).peak_s():.3f}"
            for c in ds.cases()
        )
        out.append(f"- peak locations: {peaks}")
        out.append("")
    return out


def _trajectory_section(ds: Dataset) -> list[str]:
    """Render cushion height across all 15 growth steps."""
    out = ["## 3. Morphology across growth steps", "", "```"]
    steps, _ = height_trajectory(ds, ds.cases()[0])
    out.append("step     " + " ".join(f"{int(s):>6d}" for s in steps))
    for case in ds.cases():
        _, h = height_trajectory(ds, case)
        out.append(f"{ds.condition(case):<9}" + " ".join(f"{v:>6.3f}" for v in h))
    out += [
        "```",
        "- units: mm. source: max y of each step's arc_data.npz.",
        "- **step 0 is the pristine undeformed cap**, identical across cases by construction.",
        "  It is not a simulation result; growth claims should start from step 1.",
        "",
    ]
    return out


def _caveats_section(ds: Dataset) -> list[str]:
    """Render the caveats that would otherwise produce confident, wrong hypotheses."""
    clipped = ", ".join(
        f"{ds.condition(c)} {metric_value(ds, 'mech_clipped_fraction', c) * 100:.1f}%"
        for c in ds.cases()
    )
    return [
        "## 4. Caveats you must account for",
        "",
        f"1. **Mechanical input saturates.** The GRN normalizes von Mises stress against a",
        f"   ceiling of {MECH_MAX_PA:g} Pa and clips. Peak stress reaches 597 Pa (healthy) and",
        f"   859 Pa (overflow), so the clipped fraction is: {clipped}. Every gene activity at a",
        "   clipped node is pinned at the same value regardless of how high the real stress is.",
        "   A pattern in GRN output across cases may therefore be a property of the clip rather",
        "   than of the biology. Any claim about gene expression in the overflow case has to",
        "   survive this.",
        f"2. **Shear input saturates too**, against {SHEAR_MAX_DYN_CM2:g} dyn/cm^2, though far",
        "   less often.",
        "3. **Grid dependence.** von Mises statistics use the full 11,615-node mesh; GRN",
        "   statistics use the 80% grid the GRN was evaluated on. The same physical saturation",
        "   reads differently on the two denominators. Name the grid in any fraction claim.",
        "4. **Only 5 of 23 GRN nodes were written out** (Snai1, Snai2, EndMT, NICD, YAP_TAZ).",
        "   The other 18 were simulated and discarded, so nothing can be claimed about them.",
        "5. **Interior nodes have no shear.** About 94% of the mesh never touches the fluid;",
        "   WSS is NaN there, and the combined GRN scenario feeds those nodes shear = 0, which",
        "   is not the same thing as 'no flow measured'.",
        "6. **Geometry lag.** step_k/arc_data.npz is the solid shape at the end of step k-1.",
        "7. **Growth floor.** The shipped runs pin g at 0.5, while the current source default",
        "   is 0.3, so these runs are not reproducible from the code as it stands.",
        "",
    ]


def _protocol_section() -> list[str]:
    """Render the rules a hypothesis has to satisfy, and what can be asked for."""
    return [
        "## 5. What you can test",
        "",
        "A claim is admissible only if it compiles into one of the harness primitives with a",
        "decision rule fixed before the test runs. Claims are qualitative and plain-English;",
        "tests are mechanical. The simulator decides the outcome — no language model grades",
        "any claim, including its own.",
        "",
        "Registered metrics usable in `compare_metric` and `metric_ordering`:",
        "",
        *(f"- `{n}` ({METRICS[n].units})" for n in METRIC_NAMES),
        "",
        "Full arrays are not in this briefing but are on disk and can be queried directly:",
        "11,615 solid nodes per case at the final step, 219-259 arc facets per case per step",
        "across 15 steps, 22,769 growth cells per case per step, and roughly 8,900 GRN grid",
        "nodes per case. Ask for a specific array rather than assuming this summary is all",
        "there is.",
        "",
    ]


def build_briefing(ds: Dataset, n_profile_points: int = DEFAULT_N_POINTS) -> str:
    """Assemble the full prompt-sized briefing for one dataset.

    Args:
        ds: The dataset to describe.
        n_profile_points: Points per arc-length profile.

    Returns:
        The briefing as markdown text.
    """
    velocities = ", ".join(
        f"{ds.condition(c)} = {INLET_VELOCITY_M_S[c] * 100:.1f} cm/s" for c in ds.cases()
    )
    head = [
        "# AV cushion FSG + GRN model — summary briefing",
        "",
        "A one-way fluid-solid-growth model of a developing heart atrioventricular cushion,",
        "coupled to a 23-node gene regulatory network. Blood flows over a cushion that grows",
        "in response to mechanical stimulus; the GRN reads local wall shear stress and tissue",
        "stress and returns gene activities.",
        "",
        f"Three cases differ only in inlet velocity: {velocities}. Fifteen growth steps each.",
        "The cushion is a shallow cap, so its surface is described by normalized arc length",
        "s in [0, 1], running from the atrial (upstream) end at s=0 to the ventricular",
        "(downstream) end at s=1.",
        "",
    ]
    body = (
        head
        + _metric_section(ds)
        + _profile_section(ds, n_profile_points)
        + _trajectory_section(ds)
        + _caveats_section(ds)
        + _protocol_section()
    )
    text = "\n".join(body)
    LOG.info("briefing built: %d characters, %d profile points", len(text), n_profile_points)
    return text
