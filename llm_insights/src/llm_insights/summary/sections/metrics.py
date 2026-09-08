"""Section 1 of the briefing: the scalar metric table, with units and provenance.

Moved verbatim out of ``briefing.py`` by TASK-012. The rendering is unchanged; only
where it lives and how it is reached changed.
"""

from __future__ import annotations

from typing import Any

from llm_insights.summary.metrics import METRIC_REGISTRY, metric_value
from llm_insights.summary.registry import briefing_section

#: Rendered first.
ORDER: int = 10


def fmt(value: float) -> str:
    """Format a number compactly without losing meaningful precision.

    Shared by several sections, which is why it is public here rather than private to
    one of them. The thresholds are the ones the briefing has always used, and changing
    any of them changes the briefing's bytes.

    Args:
        value: The number to render.

    Returns:
        The formatted string, or ``"n/a"`` for NaN.
    """
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


@briefing_section(order=ORDER, name="metrics")
def metric_section(ds: Any, n_profile_points: int) -> list[str]:
    """Render the scalar metric table with units and provenance.

    Args:
        ds: The dataset to describe.
        n_profile_points: Unused; every section takes the same two arguments so that
            adding one never means changing the composer.

    Returns:
        The section's lines.
    """
    del n_profile_points
    cases = ds.cases()
    labels = [ds.condition(c) for c in cases]
    names = sorted(METRIC_REGISTRY)
    out = ["## 1. Scalar metrics (all three cases, final growth step)", ""]
    header = f"| {'metric':<28} | {'units':<13} | " + " | ".join(f"{x:>10}" for x in labels) + " |"
    out.append(header)
    out.append(f"| {'-' * 28} | {'-' * 13} | " + " | ".join("-" * 10 for _ in labels) + " |")
    for name in names:
        md = METRIC_REGISTRY[name]
        values = [fmt(metric_value(ds, name, c)) for c in cases]
        out.append(
            f"| {name:<28} | {md.units:<13} | " + " | ".join(f"{v:>10}" for v in values) + " |"
        )
    out += ["", "### What each metric means, and where it comes from", ""]
    for name in names:
        md = METRIC_REGISTRY[name]
        out.append(f"- **{name}** ({md.units}) — {md.description}")
        out.append(f"  - source: {md.source}")
    return out


__all__ = ["ORDER", "fmt", "metric_section"]
