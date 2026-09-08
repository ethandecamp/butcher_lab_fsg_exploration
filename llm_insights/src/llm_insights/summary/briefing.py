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

**This file is now a composer and nothing else.** Every section lives in its own module
under :mod:`llm_insights.summary.sections` and registers itself through
:func:`llm_insights.summary.registry.briefing_section`; this file builds the header,
iterates the registry in sorted order, and joins. TASK-012 made it that way so that two
people adding sections at once add two files rather than conflicting on one, and
**nothing after Phase 0 edits this file.** A new section is a new module.
"""

from __future__ import annotations

import logging

from llm_insights.io.dataset import INLET_VELOCITY_M_S, Dataset
from llm_insights.summary import sections as _sections  # noqa: F401 - registers sections
from llm_insights.summary.profiles import DEFAULT_N_POINTS
from llm_insights.summary.registry import render_sections

LOG = logging.getLogger(__name__)


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
    body = head + render_sections(ds, n_profile_points)
    text = "\n".join(body)
    LOG.info("briefing built: %d characters, %d profile points", len(text), n_profile_points)
    return text


__all__ = ["build_briefing"]
