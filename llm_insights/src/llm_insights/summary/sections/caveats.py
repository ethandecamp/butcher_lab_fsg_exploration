"""Section 4 of the briefing: the artifacts that would otherwise look like biology.

Several of the most striking patterns in this dataset come from the model's
normalization ceilings rather than from its biology, and a summary that showed the
pattern while hiding the ceiling would invite exactly the confident, wrong hypothesis
this project exists to catch.

Moved verbatim out of ``briefing.py`` by TASK-012. The wording is unchanged.
"""

from __future__ import annotations

from typing import Any

from llm_insights.io.dataset import MECH_MAX_PA, SHEAR_MAX_DYN_CM2
from llm_insights.summary.metrics import metric_value
from llm_insights.summary.registry import briefing_section

#: Rendered fourth.
ORDER: int = 40


@briefing_section(order=ORDER, name="caveats")
def caveats_section(ds: Any, n_profile_points: int) -> list[str]:
    """Render the caveats that would otherwise produce confident, wrong hypotheses.

    Args:
        ds: The dataset to describe.
        n_profile_points: Unused; the signature is uniform across sections.

    Returns:
        The section's lines.
    """
    del n_profile_points
    clipped = ", ".join(
        f"{ds.condition(c)} {metric_value(ds, 'mech_clipped_fraction', c) * 100:.1f}%"
        for c in ds.cases()
    )
    return [
        "## 4. Caveats you must account for",
        "",
        "1. **Mechanical input saturates.** The GRN normalizes von Mises stress against a",
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


__all__ = ["ORDER", "caveats_section"]
