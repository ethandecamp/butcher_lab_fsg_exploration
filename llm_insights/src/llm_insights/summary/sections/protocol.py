"""Section 5 of the briefing: the rules a claim has to satisfy, and what can be asked.

Moved verbatim out of ``briefing.py`` by TASK-012. The wording is unchanged; the metric
list is now read from :data:`llm_insights.summary.metrics.METRIC_REGISTRY` in sorted
order, so a metric another module registers appears here without this file changing.
"""

from __future__ import annotations

from typing import Any

from llm_insights.summary.metrics import METRIC_REGISTRY
from llm_insights.summary.registry import briefing_section

#: Rendered last.
ORDER: int = 50


@briefing_section(order=ORDER, name="protocol")
def protocol_section(ds: Any, n_profile_points: int) -> list[str]:
    """Render the rules a hypothesis has to satisfy, and what can be asked for.

    Args:
        ds: Unused; the signature is uniform across sections.
        n_profile_points: Unused; the signature is uniform across sections.

    Returns:
        The section's lines.
    """
    del ds, n_profile_points
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
        *(f"- `{n}` ({METRIC_REGISTRY[n].units})" for n in sorted(METRIC_REGISTRY)),
        "",
        "Full arrays are not in this briefing but are on disk and can be queried directly:",
        "11,615 solid nodes per case at the final step, 219-259 arc facets per case per step",
        "across 15 steps, 22,769 growth cells per case per step, and roughly 8,900 GRN grid",
        "nodes per case. Ask for a specific array rather than assuming this summary is all",
        "there is.",
        "",
    ]


__all__ = ["ORDER", "protocol_section"]
