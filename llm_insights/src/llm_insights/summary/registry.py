"""The briefing's extension point: sections register themselves, in a fixed order.

Before this module, every section of the briefing was a private function inside
``briefing.py``, and adding one meant editing that file. With several people adding
sections at once that is a merge conflict whose failure mode is *silent*: a resolution
that drops one call to one section builder produces a briefing that is shorter by a
section and wrong in no way a test would notice.

So sections register instead. A new section is a new file under
:mod:`llm_insights.summary.sections`, decorated with :func:`briefing_section`, and
``briefing.py`` never changes again.

Two properties are load-bearing:

* **Iteration is sorted by ``(order, name)``, never by import order.** Registration
  happens as a side effect of importing a module, and import order is not something a
  test can rely on -- it depends on which lane's module happened to be imported first.
  A briefing whose section order varied that way would make the answer-preservation
  suites flaky for reasons that had nothing to do with the data.
* **A duplicate name raises.** Two sections silently sharing a key would mean one of
  them vanishing from every briefing, which is the exact failure this module exists to
  prevent.

Usage::

    from llm_insights.summary.registry import briefing_section


    @briefing_section(order=60, name="grn")
    def grn_section(ds: Dataset, n_profile_points: int) -> list[str]:
        del n_profile_points
        return ["## 6. The gene network", "", ...]
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger(__name__)

#: Signature every registered section obeys: it is handed the dataset and the profile
#: resolution, and returns the lines it contributes. Both arguments are passed to every
#: section whether or not it wants them, so that adding a section never means changing
#: the composer -- a section that does not need one simply ``del``\\ s it.
SectionFn = Callable[..., list[str]]


@dataclass(frozen=True)
class SectionDef:
    """One registered briefing section.

    Attributes:
        order: Sort key. Lower comes first. The five original sections use 10, 20, 30,
            40 and 50, leaving room to interleave without renumbering.
        name: Registry key, unique across the whole briefing.
        fn: The builder, matching :data:`SectionFn`.
    """

    order: int
    name: str
    fn: SectionFn


_SECTIONS: dict[str, SectionDef] = {}


def briefing_section(order: int, name: str) -> Callable[[SectionFn], SectionFn]:
    """Register a briefing section builder.

    Args:
        order: Sort key. Lower sorts earlier. Ties are broken by ``name``, so the
            output is deterministic even when two sections claim the same order.
        name: Unique registry key.

    Returns:
        A decorator that registers the wrapped function and returns it unchanged.

    Raises:
        ValueError: If ``name`` is already registered. Registering the same
            ``(order, name)`` twice therefore raises too, rather than the second
            registration silently replacing the first.
    """

    def wrap(fn: SectionFn) -> SectionFn:
        existing = _SECTIONS.get(name)
        if existing is not None:
            raise ValueError(
                f"briefing section {name!r} is already registered at order "
                f"{existing.order}; a second registration would drop one of the two "
                "from every briefing"
            )
        _SECTIONS[name] = SectionDef(order=int(order), name=str(name), fn=fn)
        LOG.debug("registered briefing section %s at order %d", name, order)
        return fn

    return wrap


def registered_sections() -> tuple[SectionDef, ...]:
    """Return every registered section, sorted by ``(order, name)``.

    Returns:
        The sections in the order the briefing renders them. Sorted, never in import
        order: see this module's docstring for why that matters.
    """
    return tuple(sorted(_SECTIONS.values(), key=lambda s: (s.order, s.name)))


def section_names() -> tuple[str, ...]:
    """Return the registered section names in render order.

    Returns:
        The names, sorted by ``(order, name)``.
    """
    return tuple(section.name for section in registered_sections())


def render_sections(ds: Any, n_profile_points: int) -> list[str]:
    """Build every registered section and concatenate the lines, in order.

    Args:
        ds: The dataset every section describes.
        n_profile_points: Points per arc-length profile, passed to every section.

    Returns:
        The concatenated lines.
    """
    lines: list[str] = []
    for section in registered_sections():
        lines.extend(section.fn(ds, n_profile_points))
    return lines


__all__ = [
    "SectionDef",
    "SectionFn",
    "briefing_section",
    "registered_sections",
    "render_sections",
    "section_names",
]
