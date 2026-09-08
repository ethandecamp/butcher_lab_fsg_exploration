"""Section 3 of the briefing: cushion height across all fifteen growth steps.

Moved verbatim out of ``briefing.py`` by TASK-012. The rendering is unchanged.
"""

from __future__ import annotations

from typing import Any

from llm_insights.summary.profiles import height_trajectory
from llm_insights.summary.registry import briefing_section

#: Rendered third.
ORDER: int = 30


@briefing_section(order=ORDER, name="trajectory")
def trajectory_section(ds: Any, n_profile_points: int) -> list[str]:
    """Render cushion height across all 15 growth steps.

    Args:
        ds: The dataset to describe.
        n_profile_points: Unused; the signature is uniform across sections.

    Returns:
        The section's lines.
    """
    del n_profile_points
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


__all__ = ["ORDER", "trajectory_section"]
