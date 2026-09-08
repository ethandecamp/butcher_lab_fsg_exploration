"""Section 2 of the briefing: arc-length profiles, where spatial claims come from.

Moved verbatim out of ``briefing.py`` by TASK-012, along with the field list it reads.
The rendering is unchanged.
"""

from __future__ import annotations

from typing import Any, Final

from llm_insights.summary.profiles import profile
from llm_insights.summary.registry import briefing_section

#: Rendered second.
ORDER: int = 20

#: Fields profiled in the briefing by default. Adding one changes the briefing's bytes
#: and therefore has to come with a re-run of the answer-preservation suite.
BRIEFING_FIELDS: Final[tuple[str, ...]] = ("wss", "von_mises", "height")


@briefing_section(order=ORDER, name="profiles")
def profile_section(ds: Any, n_profile_points: int) -> list[str]:
    """Render the arc-length profiles for each field and case.

    Args:
        ds: The dataset to describe.
        n_profile_points: Points per profile.

    Returns:
        The section's lines.
    """
    n_points = n_profile_points
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


__all__ = ["BRIEFING_FIELDS", "ORDER", "profile_section"]
