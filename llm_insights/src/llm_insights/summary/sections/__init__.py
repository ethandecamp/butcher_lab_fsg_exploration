"""The briefing's sections, one per file, each registering itself on import.

Importing this package is what populates
:data:`llm_insights.summary.registry._SECTIONS`. :mod:`llm_insights.summary.briefing`
imports it for exactly that side effect and then iterates the registry; it never names
a section directly, which is what lets a new section be a new file rather than an edit
to a shared one.

The imports below are in alphabetical order and that is deliberate: it is *not* the
render order, and anyone tempted to reorder them to change the briefing should find
that it does nothing. Render order comes from the ``order`` each section declares, and
:func:`llm_insights.summary.registry.registered_sections` sorts by it.
"""

from __future__ import annotations

from llm_insights.summary.sections import (
    caveats,
    metrics,
    profiles,
    protocol,
    trajectory,
)

__all__ = ["caveats", "metrics", "profiles", "protocol", "trajectory"]
