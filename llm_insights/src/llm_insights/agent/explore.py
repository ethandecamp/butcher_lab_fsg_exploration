"""Autonomous exploration: the system picks the questions, and says what that was worth.

Every other entry point in this package starts from a question a human typed. This one
does not: it proposes its own questions from the schema, tests them through the
existing :class:`~llm_insights.api.Study` path, and reports what survived.

That is a low bar on its own -- "a language model writes a report about data" is what
every AI-for-science demo already does. Two properties are what make this defensible,
and both of them are in here rather than in the prose:

* **Every claim was pre-registered and executed.** Nothing new happens to verification.
  A question becomes hypotheses through :meth:`llm_insights.api.Study.ask`, the harness
  runs them, and arithmetic decides. No model in this module judges any result.
* **Every claim is labelled by how hard its answer was to get.** :func:`classify_test`
  sorts a claim into ``LOOKUP`` / ``DERIVED`` / ``COMPUTED`` by *string and numeric
  matching against the briefing text* -- never by asking a model. The survival rate is
  then reported split by that label, because "12 of 15 claims survived" means something
  very different when 12 of them only re-read a table the model was handed.

Three rules hold throughout, and each of them is a test:

1. **No language model makes a stopping decision.** The loop halts on a hard budget
   (:class:`Budget`, snapshotted into locals before the first call) and on a coverage
   plateau. Nothing a generator returns can raise either. See
   :class:`_BudgetedGenerator`.
2. **The classifier is mechanical.** :class:`BriefingIndex` parses the briefing once --
   its scalar table, its printed arc profiles, its printed peak locations and every
   numeral it spells -- and :func:`classify_test` decides from the test spec and that
   index alone.
3. **Ranking is deterministic.** :func:`rank_cards` scores from numbers already on the
   cards and breaks every tie on the card id, so the same cards in give the same order
   out.

**Why there is no coverage section in the briefing.** TASK-015 allows
``summary/sections/coverage.py`` "only if a coverage summary genuinely needs to reach
the prompt". It does not. Coverage is state belonging to one exploration run, not a
fact about the dataset, and a briefing that changed as a run progressed would stop
being reproducible -- ``briefing_sha256`` is recorded on every card set precisely so a
reader can tell that two runs saw the same facts. The proposer is given the covered
cells in its own prompt (:func:`build_question_prompt`) instead, which is where run
state belongs.

Typical use::

    from llm_insights.api import open_study
    from llm_insights.agent.explore import Budget, explore

    with open_study(generator=backend) as study:
        result = explore(study, budget=Budget(max_questions=4), out="data/explore")
    print(result.survival["by_class"])
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, NamedTuple

from llm_insights.cards.card import Card, cards_to_json
from llm_insights.harness import primitives as _primitives
from llm_insights.io.dataset import (
    CASE_BY_CONDITION,
    CONDITION_BY_CASE,
    GRN_NODES,
    GRN_SCENARIOS,
    INLET_VELOCITY_M_S,
)

LOG = logging.getLogger(__name__)

# --- The three classes -------------------------------------------------------------

#: The answer is already printed in the briefing the model was handed.
LOOKUP: Final[str] = "LOOKUP"
#: The answer is one arithmetic step from numbers printed in the briefing.
DERIVED: Final[str] = "DERIVED"
#: The answer requires touching arrays the briefing does not contain.
COMPUTED: Final[str] = "COMPUTED"
#: The three classes, cheapest evidence first. Iterated in this order everywhere.
CLASSES: Final[tuple[str, ...]] = (LOOKUP, DERIVED, COMPUTED)

#: How much a class is worth when ranking. A result that had to touch the arrays is
#: worth more than one that re-read the table, and this is the only place that opinion
#: is expressed as a number.
CLASS_WEIGHT: Final[dict[str, float]] = {LOOKUP: 0.0, DERIVED: 0.5, COMPUTED: 1.0}

#: The growth step everything scalar in the briefing is reported at. A test at any
#: other step is asking about numbers the briefing does not print.
BRIEFING_STEP: Final[int] = 14

#: GRN table columns that are not gene activities.
GRN_BASE_COLUMNS: Final[tuple[str, ...]] = ("von_mises_pa", "wss_dyn_cm2", "mech_norm", "wss_norm")


def _vocabulary(name: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """Read a primitive's parameter vocabulary out of the primitives module.

    The vocabularies are module-private constants over there, so this mirrors what
    :func:`llm_insights.agent.generator._allowed_values` already does: read them by
    name and warn loudly if a rename has made the name stale, rather than hard-coding a
    copy that would drift silently.

    Args:
        name: The constant's name in :mod:`llm_insights.harness.primitives`.
        fallback: What to use, with a warning, if the constant is gone.

    Returns:
        The vocabulary.
    """
    values = getattr(_primitives, name, None)
    if not values:
        LOG.warning("primitives has no %s; falling back to %r for coverage", name, fallback)
        return fallback
    return tuple(values)


#: Fields ``fraction_above`` whose threshold reproduces a number the briefing already
#: prints, and under which class. ``von_mises`` above 100 Pa over the full solid mesh is
#: *exactly* ``mech_clipped_fraction``: same numerator, same 11,615-node denominator, so
#: the test re-measures a printed number. ``growth`` above the 0.5 floor is one minus
#: the printed ``growth_pinned_low_fraction`` -- one arithmetic step, and only up to
#: that metric's 1e-9 pin tolerance, so it is DERIVED rather than LOOKUP.
FRACTION_EQUIVALENTS: Final[dict[tuple[str, float], tuple[str, str, str]]] = {
    ("von_mises", 100.0): (
        "mech_clipped_fraction",
        LOOKUP,
        "count(von_mises > 100 Pa) over the full 11,615-node mesh is exactly the "
        "printed mech_clipped_fraction row",
    ),
    ("growth", 0.5): (
        "growth_pinned_low_fraction",
        DERIVED,
        "one minus the printed growth_pinned_low_fraction, up to that metric's 1e-9 pin tolerance",
    ),
}


@dataclass(frozen=True)
class Classification:
    """How much new information one claim's test actually costs.

    Attributes:
        label: One of :data:`CLASSES`.
        reason: One line a reader can check by looking at the briefing.
        source: What in the briefing the decision was matched against -- a metric row,
            a printed profile, a printed peak -- or the empty string for COMPUTED.
    """

    label: str
    reason: str
    source: str = ""


# --- Reading the briefing ----------------------------------------------------------

#: Numeric literals, matching :mod:`llm_insights.agent.synthesis`'s reading of one: an
#: optional sign, digits, an optional fractional part and an optional exponent.
_NUMERAL_RE: Final[re.Pattern[str]] = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _displayed_tolerance(token: str) -> float:
    """Return half a unit in a printed numeral's last displayed place.

    ``"0.3372"`` was printed to four decimals, so it stands for anything in
    ``[0.33715, 0.33725)``; ``"1.72e-04"`` stands for ``[1.715e-4, 1.725e-4)``. This is
    the same rule :func:`llm_insights.agent.synthesis._tolerance` applies, deliberately
    re-derived here rather than imported: that one guards model prose against the
    cards, this one asks whether a computed value is already printed in the briefing,
    and tying the two together would mean a change for one silently retuning the other.

    **A bare integer gets no window at all.** Half a unit either side of ``0`` would
    make every value below 0.5 "already printed", and the briefing is full of bare
    integers that are not measurements -- section numbers, ``s=0``, ``100 Pa``, node
    counts. Treating those as ranges reported a lookup wherever a small number appeared,
    which flatters the result in exactly the direction this module exists to avoid, so
    an integer has to match exactly.

    Args:
        token: The numeral as the briefing spells it.

    Returns:
        The half-width of the interval the token stands for, 0.0 for a bare integer.
    """
    body = token.lstrip("+-")
    if "." not in body and "e" not in body.lower():
        return 0.0
    exponent = 0
    if "e" in body.lower():
        mantissa, _, exp = body.lower().partition("e")
        try:
            exponent = int(exp)
        except ValueError:  # pragma: no cover - the pattern cannot produce one
            exponent = 0
        body = mantissa
    decimals = len(body.split(".")[1]) if "." in body else 0
    return 0.5 * (10.0**-decimals) * (10.0**exponent)


def _numerals(text: str) -> list[tuple[str, float]]:
    """Extract ``(token, value)`` for every numeral in ``text``.

    Args:
        text: Any text.

    Returns:
        The pairs in order of appearance. Non-finite results are dropped.
    """
    out: list[tuple[str, float]] = []
    for match in _NUMERAL_RE.finditer(text):
        token = match.group(0)
        try:
            value = float(token)
        except ValueError:  # pragma: no cover - the pattern cannot produce one
            continue
        if math.isfinite(value):
            out.append((token, value))
    return out


def condition_label(case: str) -> str:
    """Resolve a case folder name or condition label to the label the briefing prints.

    Args:
        case: ``"flow_U0p0360"`` or ``"Healthy"``, in any capitalization.

    Returns:
        The condition label, or the input unchanged when it names neither.
    """
    text = str(case)
    if text in CONDITION_BY_CASE:
        return CONDITION_BY_CASE[text]
    for label in CASE_BY_CONDITION:
        if label.lower() == text.lower():
            return label
    return text


class BriefingIndex:
    """Everything the briefing prints, parsed once so matching against it is cheap.

    The briefing is markdown built by :mod:`llm_insights.summary.briefing`, and this
    reads back the three shapes a claim can be matched against: the scalar metric table
    (section 1), the resampled arc profiles and their printed peak locations
    (section 2), and every numeral anywhere in the text. Parsing is tolerant -- a
    section that is absent simply yields nothing, and every consumer treats "not
    printed" as COMPUTED, which is the safe direction to be wrong in.

    Args:
        text: The briefing.
    """

    def __init__(self, text: str) -> None:  # noqa: D107 - documented on the class
        self.text: str = str(text or "")
        self.metric_rows: dict[str, dict[str, str]] = {}
        self.case_labels: tuple[str, ...] = ()
        self.profiles: dict[str, dict[str, list[float]]] = {}
        self.profile_grid: dict[str, list[float]] = {}
        self.printed_peaks: dict[str, dict[str, float]] = {}
        self._tokens: list[tuple[str, float]] = _numerals(self.text)
        self._parse_metric_table()
        self._parse_profiles()
        LOG.debug(
            "briefing index: %d metric row(s), %d printed profile(s), %d numeral(s)",
            len(self.metric_rows),
            len(self.profiles),
            len(self._tokens),
        )

    def __repr__(self) -> str:
        """Return a representation naming what was parsed out."""
        return (
            f"BriefingIndex(metrics={len(self.metric_rows)}, "
            f"profiles={sorted(self.profiles)}, chars={len(self.text)})"
        )

    def _parse_metric_table(self) -> None:
        """Read section 1's markdown table into ``metric_rows``."""
        for line in self.text.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            if cells[0] == "metric":
                self.case_labels = tuple(cells[2:])
                continue
            if set(cells[0]) <= {"-"}:
                continue
            if not self.case_labels or len(cells) - 2 != len(self.case_labels):
                continue
            self.metric_rows[cells[0]] = dict(zip(self.case_labels, cells[2:], strict=True))

    def _parse_profiles(self) -> None:
        """Read section 2's resampled profiles and their printed peak locations."""
        field_name: str | None = None
        in_block = False
        for line in self.text.splitlines():
            heading = re.match(r"^###\s+([A-Za-z_]+)\s*\(", line)
            if heading:
                field_name = heading.group(1)
                continue
            if field_name is None:
                continue
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                parts = line.split()
                if len(parts) < 2:
                    continue
                label = parts[0]
                try:
                    values = [float(p) for p in parts[1:]]
                except ValueError:
                    continue
                if label == "s":
                    self.profile_grid[field_name] = values
                elif label in CASE_BY_CONDITION:
                    self.profiles.setdefault(field_name, {})[label] = values
                continue
            if line.strip().startswith("- peak locations:"):
                peaks: dict[str, float] = {}
                for label, where in re.findall(r"(\w+) peaks at s=([\d.]+)", line):
                    if label in CASE_BY_CONDITION:
                        peaks[label] = float(where)
                if peaks:
                    self.printed_peaks[field_name] = peaks

    # --- the questions a classifier asks ------------------------------------------

    def prints_metric(self, metric: Any, case: Any) -> bool:
        """Report whether the scalar table prints this metric for this case.

        Args:
            metric: A metric name.
            case: A case folder name or condition label.

        Returns:
            True when the row exists and carries a value for that case's column.
        """
        row = self.metric_rows.get(str(metric))
        return bool(row) and condition_label(case) in row

    def prints_peak(self, field_name: Any, case: Any) -> bool:
        """Report whether section 2 prints this field's peak location for this case.

        Args:
            field_name: ``"wss"``, ``"von_mises"`` or ``"height"``.
            case: A case folder name or condition label.

        Returns:
            True when a printed peak location exists.
        """
        peaks = self.printed_peaks.get(str(field_name), {})
        return condition_label(case) in peaks

    def profile_points_in(self, field_name: Any, case: Any, s_min: float, s_max: float) -> int:
        """Count printed profile samples inside an arc window.

        Args:
            field_name: The profile's field name.
            case: A case folder name or condition label.
            s_min: Lower edge of the window, inclusive.
            s_max: Upper edge, inclusive.

        Returns:
            How many of the printed grid points fall inside the window, or 0 when this
            field and case are not printed at all.
        """
        grid = self.profile_grid.get(str(field_name))
        rows = self.profiles.get(str(field_name), {})
        if not grid or condition_label(case) not in rows:
            return 0
        return sum(1 for s in grid if s_min <= s <= s_max)

    def prints_value(self, value: float) -> str | None:
        """Find a printed numeral this value would round to.

        Args:
            value: Any number.

        Returns:
            The briefing's token, or None when no printed numeral covers this value at
            the precision it was printed to.
        """
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        target = float(value)
        for token, printed in self._tokens:
            if abs(printed - target) <= _displayed_tolerance(token):
                return token
        return None


# --- The lookup filter -------------------------------------------------------------


def _params_of(spec: Any) -> dict[str, Any]:
    """Read a test spec's parameters, whether it is an object or a mapping."""
    params = getattr(spec, "params", None)
    if params is None and isinstance(spec, Mapping):
        params = spec.get("params")
    return dict(params) if isinstance(params, Mapping) else {}


def _primitive_of(spec: Any) -> str:
    """Read a test spec's primitive name, whether it is an object or a mapping."""
    name = getattr(spec, "primitive", None)
    if name is None and isinstance(spec, Mapping):
        name = spec.get("primitive")
    return str(name or "")


def _cases_of(primitive: str, params: Mapping[str, Any]) -> list[str]:
    """List every case a test touches, in the order the spec names them."""
    if primitive == "compare_metric":
        return [str(params.get("case_a", "")), str(params.get("case_b", ""))]
    if primitive == "metric_ordering":
        cases = params.get("cases")
        return (
            [str(c) for c in cases]
            if isinstance(cases, Sequence) and not isinstance(cases, str)
            else []
        )
    return [str(params.get("case", ""))]


def classify_test(spec: Any, index: BriefingIndex) -> Classification:
    """Sort one test into LOOKUP, DERIVED or COMPUTED. No model is consulted.

    The decision is made from the test spec and the parsed briefing alone, before the
    test runs, so it is a property of the *question* rather than of the answer. The
    rules, in the order they are tried:

    * ``compare_metric`` / ``metric_ordering`` on a metric whose row the scalar table
      prints, for cases the table has columns for, is **LOOKUP**: TASK-015 defines the
      class that way, and rightly -- both operands are sitting in the prompt.
    * ``peak_location`` at step 14 on a field whose peak location section 2 prints is
      **LOOKUP** for the same reason.
    * ``fraction_above`` is **LOOKUP** or **DERIVED** when its ``(field, threshold)``
      pair reproduces a printed fraction (:data:`FRACTION_EQUIVALENTS`), and
      **COMPUTED** otherwise -- an arbitrary threshold needs the array.
    * ``profile_monotonic`` at step 14 over a printed profile, with at least two printed
      samples in its window, is **DERIVED**: the sign of consecutive differences of
      printed numbers is one arithmetic step. It is *not* LOOKUP, and the distinction is
      real -- the primitive measures over the full 219-259 facet arc while the briefing
      prints a 25-point resample, so the fraction it observes is not in the prompt and a
      briefing-only reader can disagree with it near a threshold.
    * ``correlation`` is always **COMPUTED**: the briefing carries no GRN table column
      and no per-node pairs at all.
    * Anything at a step other than 14, on a field the briefing does not print, or on a
      metric outside the printed table, is **COMPUTED**.

    Args:
        spec: A :class:`~llm_insights.harness.spec.TestSpec`, or a mapping with
            ``primitive`` and ``params``.
        index: The parsed briefing.

    Returns:
        The :class:`Classification`.
    """
    primitive = _primitive_of(spec)
    params = _params_of(spec)
    cases = _cases_of(primitive, params)

    if primitive in ("compare_metric", "metric_ordering"):
        metric = params.get("metric")
        if cases and all(index.prints_metric(metric, c) for c in cases):
            labels = ", ".join(condition_label(c) for c in cases)
            return Classification(
                LOOKUP,
                f"{metric!r} is printed in the briefing's scalar table for {labels}",
                f"scalar table row {metric!r}",
            )
        return Classification(
            COMPUTED,
            f"{metric!r} is not printed in the scalar table for every case this test names",
        )

    if primitive == "peak_location":
        field_name = str(params.get("field", ""))
        step = params.get("step", BRIEFING_STEP)
        case = cases[0] if cases else ""
        if step == BRIEFING_STEP and index.prints_peak(field_name, case):
            printed = index.printed_peaks[field_name][condition_label(case)]
            return Classification(
                LOOKUP,
                f"the briefing prints {field_name} peaking at s={printed:g} for "
                f"{condition_label(case)}",
                f"printed peak location for {field_name}",
            )
        return Classification(
            COMPUTED,
            f"no printed peak location for {field_name!r} at step {step} in this case",
        )

    if primitive == "fraction_above":
        field_name = str(params.get("field", ""))
        threshold = params.get("threshold")
        step = params.get("step", BRIEFING_STEP)
        case = cases[0] if cases else ""
        key = (field_name, float(threshold)) if isinstance(threshold, (int, float)) else None
        entry = FRACTION_EQUIVALENTS.get(key) if key is not None else None
        growth_step_ok = field_name != "growth" or step == BRIEFING_STEP
        if entry is not None and growth_step_ok and index.prints_metric(entry[0], case):
            return Classification(entry[1], entry[2], f"scalar table row {entry[0]!r}")
        return Classification(
            COMPUTED,
            f"the fraction of {field_name!r} above {threshold!r} is not a printed number",
        )

    if primitive == "profile_monotonic":
        field_name = str(params.get("field", ""))
        step = params.get("step", BRIEFING_STEP)
        case = cases[0] if cases else ""
        s_min = params.get("s_min", 0.0)
        s_max = params.get("s_max", 1.0)
        n_points = 0
        if (
            step == BRIEFING_STEP
            and isinstance(s_min, (int, float))
            and isinstance(s_max, (int, float))
        ):
            n_points = index.profile_points_in(field_name, case, float(s_min), float(s_max))
        if n_points >= 2:
            return Classification(
                DERIVED,
                f"{n_points} printed {field_name} samples fall in [{s_min}, {s_max}], so the "
                "trend is the sign of differences of printed numbers; the fraction this test "
                "measures over the full arc is not printed",
                f"printed {field_name} profile",
            )
        return Classification(
            COMPUTED,
            f"the briefing prints no usable {field_name!r} profile for this case, step "
            f"{step} and window",
        )

    if primitive == "correlation":
        return Classification(
            COMPUTED,
            "the briefing carries no GRN table columns, so a correlation needs the array",
        )

    return Classification(COMPUTED, f"primitive {primitive!r} has no printed counterpart")


def classify_card(card: Card, index: BriefingIndex) -> Classification:
    """Classify a finished card by the test it ran.

    Args:
        card: The card.
        index: The parsed briefing.

    Returns:
        The :class:`Classification` its test spec earns. Identical to what
        :func:`classify_test` returned before the test ran -- the class is a property of
        the question, and nothing observed may change it after the fact.
    """
    return classify_test({"primitive": card.primitive, "params": card.params}, index)


def observed_in_briefing(card: Card, index: BriefingIndex) -> tuple[str, ...]:
    """Name the card's observed values that the briefing already prints.

    This is the *post-hoc* half of TASK-015's LOOKUP definition -- "the observed value
    appears in the briefing text at the printed precision" -- and it is reported
    alongside the class rather than folded into it. Folding it in would let a result
    change class after it ran, which would make the survival split unfalsifiable.

    Args:
        card: A finished card.
        index: The parsed briefing.

    Returns:
        The observation names whose values are printed in the briefing, sorted.
    """
    return tuple(sorted(name for name, value in card.observed.items() if index.prints_value(value)))


# --- Coverage bookkeeping ----------------------------------------------------------


class Cell(NamedTuple):
    """One thing that can be asked about one case with one test.

    Attributes:
        primitive: The executable test.
        target: The metric, field or GRN column it is applied to.
        case: The condition label.
    """

    primitive: str
    target: str
    case: str

    def key(self) -> str:
        """Return a stable flat key for JSON output."""
        return f"{self.primitive}|{self.target}|{self.case}"


def grn_columns() -> tuple[str, ...]:
    """Return every GRN table column ``correlation`` accepts, sorted.

    Returns:
        The four mechanical columns plus one per (node, scenario) activity.
    """
    activities = tuple(f"{node}_{scen}" for node in GRN_NODES for scen in GRN_SCENARIOS)
    return tuple(sorted(GRN_BASE_COLUMNS + activities))


def primitive_targets() -> dict[str, tuple[str, ...]]:
    """Return the admissible target vocabulary of each primitive, sorted.

    This is what makes the coverage denominator meaningful. A flat
    ``targets x cases x primitives`` grid would count cells no hypothesis could ever
    occupy -- ``correlation`` over ``arc_length_mm``, say -- and a coverage percentage
    over impossible cells is not a measurement of anything.

    The metric vocabulary is read from the registry at call time rather than captured at
    import, so a metric another module registers later is counted.

    Returns:
        Primitive name to the targets it accepts.
    """
    from llm_insights.summary.metrics import METRIC_REGISTRY

    metrics = tuple(sorted(METRIC_REGISTRY))
    return {
        "compare_metric": metrics,
        "correlation": grn_columns(),
        "fraction_above": _vocabulary(
            "_FRACTION_FIELDS", ("von_mises", "wss", "mech_norm", "growth")
        ),
        "metric_ordering": metrics,
        "peak_location": _vocabulary("_PEAK_FIELDS", ("wss", "von_mises")),
        "profile_monotonic": _vocabulary(
            "_PROFILE_FIELDS", ("wss", "von_mises", "pressure", "height")
        ),
    }


def cells_for_test(spec: Any) -> set[Cell]:
    """List the coverage cells one test touches.

    A ``correlation`` touches two cells, one per column; a ``metric_ordering`` touches
    one per case it names. Cases are normalized to condition labels so that
    ``"flow_U0p0360"`` and ``"Healthy"`` are the same cell.

    Args:
        spec: A test spec, or a mapping with ``primitive`` and ``params``.

    Returns:
        The cells, possibly empty when the spec names nothing recognizable.
    """
    primitive = _primitive_of(spec)
    params = _params_of(spec)
    cases = [condition_label(c) for c in _cases_of(primitive, params) if c]
    if primitive in ("compare_metric", "metric_ordering"):
        targets: list[str] = [str(params.get("metric", ""))]
    elif primitive == "correlation":
        targets = [str(params.get("field_x", "")), str(params.get("field_y", ""))]
    else:
        targets = [str(params.get("field", ""))]
    return {Cell(primitive, target, case) for target in targets if target for case in cases if case}


class Coverage:
    """Which (target, case, primitive) cells an exploration has opened.

    Args:
        targets: Primitive name to its admissible targets. Defaults to
            :func:`primitive_targets`.
        cases: The condition labels. Defaults to the three shipped cases.
    """

    def __init__(  # noqa: D107 - documented on the class
        self,
        targets: Mapping[str, Sequence[str]] | None = None,
        cases: Sequence[str] | None = None,
    ) -> None:
        source = primitive_targets() if targets is None else targets
        self.targets: dict[str, tuple[str, ...]] = {
            str(k): tuple(str(t) for t in v) for k, v in sorted(source.items())
        }
        self.cases: tuple[str, ...] = tuple(
            str(c) for c in (sorted(CASE_BY_CONDITION) if cases is None else cases)
        )
        self.touched: set[Cell] = set()
        self.ignored: set[Cell] = set()

    def __repr__(self) -> str:
        """Return a representation naming how much of the grid is open."""
        return f"Coverage({len(self.touched)}/{self.total} cells)"

    @property
    def total(self) -> int:
        """How many admissible cells exist in total."""
        return sum(len(t) for t in self.targets.values()) * len(self.cases)

    def is_admissible(self, cell: Cell) -> bool:
        """Report whether a cell is inside the counted grid.

        Args:
            cell: The cell to check.

        Returns:
            True when the primitive, target and case are all in the vocabulary.
        """
        return cell.target in self.targets.get(cell.primitive, ()) and cell.case in self.cases

    def record(self, spec: Any) -> set[Cell]:
        """Mark the cells one test opens, and return the ones that were new.

        Args:
            spec: A test spec, or a mapping with ``primitive`` and ``params``.

        Returns:
            The cells this test opened that had not been touched before. A cell outside
            the admissible grid is remembered in :attr:`ignored` and never counted, so
            an out-of-vocabulary claim cannot inflate coverage.
        """
        opened: set[Cell] = set()
        for cell in cells_for_test(spec):
            if not self.is_admissible(cell):
                self.ignored.add(cell)
                LOG.debug("cell %s is outside the admissible grid; not counted", cell.key())
                continue
            if cell not in self.touched:
                self.touched.add(cell)
                opened.add(cell)
        return opened

    @property
    def fraction(self) -> float:
        """The share of admissible cells touched so far, 0.0 when the grid is empty."""
        return len(self.touched) / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the coverage state.

        Returns:
            A JSON-ready mapping: the grid's shape, the totals, the touched cells in
            sorted order, and the per-primitive breakdown.
        """
        by_primitive = {
            name: {
                "targets": len(targets),
                "cells": len(targets) * len(self.cases),
                "touched": sum(1 for c in self.touched if c.primitive == name),
            }
            for name, targets in self.targets.items()
        }
        return {
            "cases": list(self.cases),
            "total_cells": self.total,
            "touched_cells": len(self.touched),
            "fraction": self.fraction,
            "by_primitive": by_primitive,
            "touched": [c.key() for c in sorted(self.touched)],
            "ignored": [c.key() for c in sorted(self.ignored)],
        }


# --- The budget, which nothing a model says can raise -------------------------------


class BudgetStopError(RuntimeError):
    """Raised inside the loop when a hard ceiling is reached.

    Caught by :func:`explore`, which turns it into a stop reason. It is an exception
    rather than a return value because it has to interrupt a call the generator is in
    the middle of making, which is the only place a runaway backend can be stopped.
    """


@dataclass(frozen=True)
class Budget:
    """The hard ceilings on one exploration. Frozen, and read only by the loop.

    Nothing a generator returns is ever consulted when building or checking one of
    these. :func:`explore` copies every field into a local before the first model call,
    so even a backend that reaches back and reassigns attributes mid-run cannot move a
    ceiling -- which is exactly what ``tests/test_explore.py`` asserts.

    Attributes:
        max_questions: How many questions may be investigated.
        max_calls: How many generator calls -- proposals, narrowings and narrations
            together -- the whole run may make.
        max_cost_usd: Cumulative spend ceiling, read from the backend's own
            ``cost_usd`` when it keeps one. Backends that cost nothing report 0.0 and
            never trip it.
        plateau_questions: Stop when this many consecutive questions each open no new
            coverage cell. 0 disables the plateau rule and leaves only the ceilings.
        hypotheses_per_question: How many hypotheses each question asks for.

    Raises:
        ValueError: If any ceiling is negative, or ``max_questions`` or
            ``hypotheses_per_question`` is not at least 1.
    """

    max_questions: int = 5
    max_calls: int = 24
    max_cost_usd: float = 5.0
    plateau_questions: int = 2
    hypotheses_per_question: int = 5

    def __post_init__(self) -> None:
        """Reject a budget that could not stop a run.

        Raises:
            ValueError: On a ceiling that is negative or below one.
        """
        if self.max_questions < 1:
            raise ValueError(f"max_questions must be at least 1, got {self.max_questions}")
        if self.max_calls < 1:
            raise ValueError(f"max_calls must be at least 1, got {self.max_calls}")
        if self.max_cost_usd < 0:
            raise ValueError(f"max_cost_usd must not be negative, got {self.max_cost_usd}")
        if self.plateau_questions < 0:
            raise ValueError(
                f"plateau_questions must not be negative, got {self.plateau_questions}"
            )
        if self.hypotheses_per_question < 1:
            raise ValueError(
                f"hypotheses_per_question must be at least 1, got {self.hypotheses_per_question}"
            )


class _BudgetedGenerator:
    """Counts every call a backend makes and refuses the one that would exceed the cap.

    Wrapping is how the ceiling becomes *hard*. The count lives here, on an object the
    generator cannot see, and the check happens before the call rather than after it, so
    a backend that returns "please make twenty more calls" has already been stopped by
    the time anything reads what it returned.

    Everything other than the three counted methods is forwarded, so provenance still
    names the real backend.

    Args:
        inner: The real generator.
        max_calls: The ceiling.
    """

    def __init__(self, inner: Any, max_calls: int) -> None:
        """Wrap a backend behind a call counter.

        Args:
            inner: The real generator.
            max_calls: The hard ceiling on calls through this wrapper.
        """
        self._inner = inner
        self._max_calls = int(max_calls)
        self.calls: int = 0

    def __repr__(self) -> str:
        """Return a representation naming the wrapped backend and the count."""
        return f"_BudgetedGenerator({self._inner!r}, {self.calls}/{self._max_calls} calls)"

    def __getattr__(self, item: str) -> Any:
        """Forward any attribute this wrapper does not define to the backend.

        ``synthesize`` is forwarded as a *counting* closure rather than directly,
        because narration is a model call and must be charged like one. It is reached
        through here rather than defined as a method so that
        :func:`llm_insights.agent.synthesis.supports_synthesis`, which duck-types on
        the attribute existing, still reports False for a backend that cannot narrate.

        Args:
            item: The attribute name.

        Returns:
            The backend's attribute, or a counting wrapper around ``synthesize``.

        Raises:
            AttributeError: If the backend does not have it either.
        """
        attr = getattr(self._inner, item)
        if item == "synthesize" and callable(attr):

            def counted(prompt: str) -> str | None:
                """Charge one call, then narrate through the backend.

                Declines rather than raises when the ceiling is reached, for the same
                reason :meth:`narrow` does: narration runs after every verdict is fixed,
                and losing the cards to save a call would be a bad trade.
                """
                if not self._charge("synthesize"):
                    return None
                return attr(prompt)

            return counted
        return attr

    def _charge(self, what: str) -> bool:
        """Charge one call against the ceiling, if there is room for it.

        Args:
            what: The kind of call, for the log line.

        Returns:
            True when the call may proceed, False when the ceiling is reached.
        """
        if self.calls >= self._max_calls:
            LOG.info("declining a %s call: the ceiling of %d is reached", what, self._max_calls)
            return False
        self.calls += 1
        return True

    def _spend(self, what: str) -> None:
        """Charge one call against the ceiling, or stop the run.

        Args:
            what: The kind of call, for the message.

        Raises:
            BudgetStopError: If the ceiling has already been reached.
        """
        if not self._charge(what):
            raise BudgetStopError(
                f"the call ceiling of {self._max_calls} was reached before this {what} call"
            )

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Charge a call and delegate the proposal.

        Args:
            briefing: The factual summary.
            question: The question.
            n: How many hypotheses to request.

        Returns:
            Whatever the backend returned, unmodified.
        """
        self._spend("propose")
        return self._inner.propose(briefing, question, n)

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Charge a call and delegate the narrowing, or decline it.

        A narrowing that cannot be afforded is *declined*, not raised: declining is
        already a legitimate outcome the loop handles, and raising here would throw away
        a whole question's worth of verified cards to save one call. Only ``propose``
        raises, because there is nothing to lose at the start of a question.

        Args:
            briefing: The factual summary.
            failed: The falsified hypothesis.
            outcome: Its outcome.

        Returns:
            The successor mapping, or None when the backend declined or the ceiling was
            reached.
        """
        if not self._charge("narrow"):
            return None
        return self._inner.narrow(briefing, failed, outcome)


# --- Proposing the questions --------------------------------------------------------

#: Question wording per primitive. Deliberately plain English and deliberately
#: qualitative: the proposer picks *what to ask about*, and the hypothesis generator
#: still has to compile the question into a pre-registered test.
_QUESTION_TEMPLATES: Final[dict[str, str]] = {
    "compare_metric": (
        "Does {target} differ between the {case} case and the other flow conditions by "
        "more than numerical noise?"
    ),
    "correlation": (
        "How strongly is {target} coupled to the other gene-network fields in the {case} case?"
    ),
    "fraction_above": (
        "How much of the {case} domain carries a high {target}, and is that share large "
        "enough to matter?"
    ),
    "metric_ordering": (
        "Does {target} change monotonically as inlet flow increases across all three cases "
        "(n=3, so say so)?"
    ),
    "peak_location": "Where along the cushion arc does {target} peak in the {case} case?",
    "profile_monotonic": (
        "Does {target} trend consistently in one direction along the cushion arc in the "
        "{case} case?"
    ),
}


def schema_catalogue() -> str:
    """Describe the schema a question must be answerable within.

    This is the grounding TASK-015 asks for: the metric registry, the case list, the
    GRN nodes and the six primitives, and nothing else. It is deliberately small --
    about a tenth of the briefing -- so a proposer can be given it without the briefing,
    which keeps a short-context backend from silently truncating the thing it is
    reasoning about.

    Returns:
        The catalogue text.
    """
    targets = primitive_targets()
    lines = [
        "SCHEMA",
        "cases (n=3, inlet velocity only): " + ", ".join(sorted(CASE_BY_CONDITION)),
        "gene-network nodes written out: " + ", ".join(GRN_NODES),
        "gene-network scenarios: " + ", ".join(GRN_SCENARIOS),
        "",
        "tests, and what each may be applied to:",
    ]
    for name in sorted(targets):
        lines.append(f"- {name}: " + ", ".join(targets[name]))
    return "\n".join(lines)


def build_question_prompt(k: int, coverage: Coverage) -> str:
    """Build the user turn asking a model for questions worth investigating.

    Coverage reaches the model here, in a per-run prompt, rather than through the
    briefing. See this module's docstring for why: the briefing has to stay a fixed
    description of the dataset, or ``briefing_sha256`` stops meaning anything.

    Args:
        k: How many questions to ask for.
        coverage: What the run has already opened.

    Returns:
        The prompt text.
    """
    touched = sorted(c.key() for c in coverage.touched)
    lines = [
        schema_catalogue(),
        "",
        "ALREADY INVESTIGATED (test|target|case)",
        "\n".join(f"- {t}" for t in touched) if touched else "- nothing yet",
        "",
        f"Propose exactly {k} question(s) about this simulation that can be answered with "
        "the tests and targets above, and that open ground the list of already-investigated "
        "cells does not cover. Each question must be one line of plain English a biologist "
        "would ask. Return one question per line and nothing else: no numbering, no "
        "commentary, no JSON.",
    ]
    return "\n".join(lines)


class SchemaQuestionProposer:
    """Enumerates questions from the schema, aimed at cells nothing has opened yet.

    Deterministic and model-free. It exists so an exploration can run with no model in
    the question-proposing seat at all, which is what makes the offline end-to-end
    evidence for this task possible, and so that a run's provenance can always say
    truthfully who wrote the questions.

    The walk is a rotation over primitives in sorted order, taking each primitive's
    lowest-sorted untouched cell in turn, so a run spreads across the six tests instead
    of exhausting one of them.
    """

    #: Provenance string. A report says this rather than naming a model.
    name: str = "schema-enumerator"

    def __repr__(self) -> str:
        """Return a short representation."""
        return "SchemaQuestionProposer()"

    def propose_questions(self, k: int, coverage: Coverage) -> list[str]:
        """Return up to ``k`` questions aimed at untouched cells.

        Args:
            k: How many questions to return.
            coverage: What has already been opened.

        Returns:
            The questions, at most one per untouched cell, possibly fewer than ``k``
            when the grid is nearly full and empty when it is full.
        """
        wanted = max(int(k), 0)
        pending: dict[str, list[Cell]] = {}
        for primitive, targets in coverage.targets.items():
            pending[primitive] = [
                Cell(primitive, target, case)
                for target in targets
                for case in coverage.cases
                if Cell(primitive, target, case) not in coverage.touched
            ]
        out: list[str] = []
        claimed: set[Cell] = set()
        while len(out) < wanted:
            progressed = False
            for primitive in sorted(pending):
                if len(out) >= wanted:
                    break
                for cell in pending[primitive]:
                    if cell in claimed:
                        continue
                    claimed.add(cell)
                    out.append(
                        _QUESTION_TEMPLATES[primitive].format(target=cell.target, case=cell.case)
                        if primitive in _QUESTION_TEMPLATES
                        else f"What does {primitive} say about {cell.target} in {cell.case}?"
                    )
                    progressed = True
                    break
            if not progressed:
                break
        LOG.info("schema proposer offered %d question(s)", len(out))
        return out


class ModelQuestionProposer:
    """Asks a backend for the questions, over its free-text channel.

    A backend advertises the capability by having ``propose_questions`` (preferred) or
    ``synthesize``, which every narration-capable backend already has. Nothing about the
    :class:`~llm_insights.agent.generator.Generator` protocol changes, and a backend
    with neither is rejected at construction rather than mid-run.

    Note what this does and does not decide: it chooses *what to ask*. It never decides
    whether a claim is true and never decides whether the run is finished -- those stay
    with the harness and with :class:`Budget` respectively.

    Args:
        generator: The backend to ask.
        fallback: A proposer used when the backend returns nothing usable. None lets an
            empty reply end the run.

    Raises:
        TypeError: If the backend has no free-text channel at all.
    """

    def __init__(self, generator: Any, fallback: Any | None = None) -> None:  # noqa: D107
        channel = getattr(generator, "propose_questions", None) or getattr(
            generator, "synthesize", None
        )
        if not callable(channel):
            raise TypeError(
                f"the {getattr(generator, 'name', type(generator).__name__)} backend has "
                "neither propose_questions nor synthesize, so it cannot be asked for "
                "questions; pass SchemaQuestionProposer() instead"
            )
        self._generator = generator
        self._fallback = fallback
        self.name = f"model:{getattr(generator, 'name', type(generator).__name__)}"

    def __repr__(self) -> str:
        """Return a representation naming the backend."""
        return f"ModelQuestionProposer({self._generator!r})"

    def propose_questions(self, k: int, coverage: Coverage) -> list[str]:
        """Ask the backend for ``k`` questions, and fall back if it will not.

        Args:
            k: How many questions to ask for.
            coverage: What has already been opened, put in front of the model.

        Returns:
            The parsed questions, at most ``k`` of them.
        """
        prompt = build_question_prompt(int(k), coverage)
        native = getattr(self._generator, "propose_questions", None)
        try:
            raw = native(prompt, int(k)) if callable(native) else self._generator.synthesize(prompt)
        except Exception as exc:  # a proposer failure must not discard verified work
            LOG.warning("the question proposer failed: %s: %s", type(exc).__name__, exc)
            raw = None
        questions = parse_questions(raw)[: max(int(k), 0)]
        if not questions and self._fallback is not None:
            LOG.warning("%s returned no usable question; falling back", self.name)
            return list(self._fallback.propose_questions(k, coverage))
        return questions


def parse_questions(raw: Any) -> list[str]:
    """Pull one question per line out of a backend's free-text reply.

    Tolerates the packaging a model adds unasked: markdown fences, bullets, numbering
    and a stray blank line. A line that carries no letters is dropped.

    Args:
        raw: Whatever the backend returned. A list is accepted as already-split lines.

    Returns:
        The cleaned questions, in order, without duplicates.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        lines = [str(item) for item in raw]
    else:
        text = str(raw).strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text.strip("`")
        lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not any(ch.isalpha() for ch in cleaned):
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out


# --- Ranking ------------------------------------------------------------------------

#: Weight per ranking component. Equal by default, and exposed as an argument, because
#: the relative worth of "a big effect" against "a result that was not already in the
#: prompt" is a judgement rather than a measurement. Every component is reported
#: separately on the ranked card so a reader can re-weigh them without rerunning.
DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    "effect": 0.25,
    "margin": 0.25,
    "class": 0.25,
    "naive": 0.25,
}

#: Denominator floor, mirroring :data:`llm_insights.harness.primitives.EPS`.
_EPS: Final[float] = 1e-12


def _squash(value: float) -> float:
    """Map a non-negative unbounded quantity into ``[0, 1)`` monotonically.

    Args:
        value: Any number; negatives are clamped to 0.

    Returns:
        ``x / (1 + x)``, which is 0 at 0, 0.5 at 1 and approaches 1.
    """
    x = max(float(value), 0.0)
    return x / (1.0 + x)


def _ordered_values(card: Card) -> list[float]:
    """Return a ``metric_ordering`` card's per-case values, in the asserted order."""
    return [v for k, v in card.observed.items() if k.startswith("value_")]


def effect_size(card: Card) -> float:
    """Score how large the effect a surviving card reports is, in ``[0, 1]``.

    Each primitive contributes the quantity its own decision consumed, normalized so the
    six are comparable. Unbounded quantities are squashed rather than clipped, so a
    tenfold difference still outranks a twofold one.

    Args:
        card: A finished card.

    Returns:
        The effect size, 0.0 when the card carries nothing to measure.
    """
    obs = card.observed
    primitive = card.primitive
    if primitive == "compare_metric":
        return _squash(abs(obs.get("rel_diff", 0.0)))
    if primitive == "metric_ordering":
        values = _ordered_values(card)
        if len(values) < 2:
            return 0.0
        return _squash(abs(values[-1] - values[0]) / max(abs(values[0]), _EPS))
    if primitive == "correlation":
        return min(abs(obs.get("r", 0.0)), 1.0)
    if primitive == "fraction_above":
        return min(abs(obs.get("fraction", 0.0) - obs.get("target", 0.0)), 1.0)
    if primitive == "peak_location":
        # A narrower pre-registered window that still held is a stronger statement than
        # a window covering most of the arc, so the effect is how much the window ruled
        # out rather than how high the peak was.
        width = abs(obs.get("s_max", 1.0) - obs.get("s_min", 0.0))
        return min(max(1.0 - width, 0.0), 1.0)
    if primitive == "profile_monotonic":
        return min(max(obs.get("fraction_monotonic", 0.0), 0.0), 1.0)
    return 0.0


def threshold_margin(card: Card) -> float:
    """Score how far a card cleared the threshold fixed before it ran, in ``[0, 1]``.

    Args:
        card: A finished card.

    Returns:
        The margin, 0.0 when it only just held or when there is nothing to measure.
    """
    obs = card.observed
    primitive = card.primitive
    if primitive == "compare_metric":
        return _squash(obs.get("rel_diff", 0.0) - obs.get("min_rel_diff", 0.0))
    if primitive == "metric_ordering":
        values = _ordered_values(card)
        scale = max(abs(values[0]), _EPS) if values else 1.0
        weakest = min(abs(obs.get("min_step", 0.0)), abs(obs.get("max_step", 0.0)))
        return _squash(weakest / scale)
    if primitive == "correlation":
        return min(abs(obs.get("r", 0.0) - obs.get("target", 0.0)), 1.0)
    if primitive == "fraction_above":
        return min(abs(obs.get("fraction", 0.0) - obs.get("target", 0.0)), 1.0)
    if primitive == "peak_location":
        peak = obs.get("s_at_peak", 0.0)
        inside = min(peak - obs.get("s_min", 0.0), obs.get("s_max", 1.0) - peak)
        return min(max(inside, 0.0) * 2.0, 1.0)
    if primitive == "profile_monotonic":
        return min(max(obs.get("fraction_monotonic", 0.0) - obs.get("min_fraction", 0.0), 0.0), 1.0)
    return 0.0


def _flow_of(case: Any) -> float | None:
    """Return a case's inlet velocity, or None when it names no known case."""
    label = condition_label(case)
    folder = CASE_BY_CONDITION.get(label)
    return INLET_VELOCITY_M_S.get(folder) if folder else None


def contradicts_naive(card: Card) -> bool:
    """Report whether a card's claim runs against the obvious expectation.

    The naive expectation, and the only one this can check mechanically, is that a
    quantity rises with inlet flow -- the single variable that differs between the three
    cases. A surviving claim that a quantity *falls* with flow is more informative than
    one that it rises, so ranking rewards it.

    This is a crude proxy and is meant to read as one. It says nothing about the four
    primitives that involve a single case, which return False rather than a guess.

    Args:
        card: A finished card.

    Returns:
        True when the claim asserts the opposite of "rises with flow".
    """
    params = card.params
    if card.primitive == "metric_ordering":
        flows = [_flow_of(c) for c in params.get("cases", []) or []]
        if len(flows) < 2 or any(f is None for f in flows):
            return False
        rising = all(b > a for a, b in pairwise(flows))
        falling = all(b < a for a, b in pairwise(flows))
        direction = str(params.get("direction", ""))
        return bool(
            (rising and direction == "decreasing") or (falling and direction == "increasing")
        )
    if card.primitive == "compare_metric":
        a, b = _flow_of(params.get("case_a")), _flow_of(params.get("case_b"))
        op = str(params.get("op", ""))
        if a is None or b is None or a == b:
            return False
        return bool((a > b and op in ("lt", "le")) or (a < b and op in ("gt", "ge")))
    return False


@dataclass(frozen=True)
class RankedCard:
    """One surviving card with the four components its rank was built from.

    Attributes:
        card: The card itself.
        score: The weighted total, in ``[0, 1]``.
        effect: :func:`effect_size`.
        margin: :func:`threshold_margin`.
        classification: The class :func:`classify_test` gave the card's test.
        contradicts: :func:`contradicts_naive`.
    """

    card: Card
    score: float
    effect: float
    margin: float
    classification: Classification
    contradicts: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ranked card.

        Returns:
            A JSON-ready mapping carrying every component, so a reader can re-rank
            under different weights without rerunning anything.
        """
        return {
            "id": self.card.id,
            "claim": self.card.claim,
            "primitive": self.card.primitive,
            "params": dict(self.card.params),
            "score": self.score,
            "effect": self.effect,
            "margin": self.margin,
            "class": self.classification.label,
            "class_reason": self.classification.reason,
            "contradicts_naive_expectation": self.contradicts,
            "summary": self.card.summary,
        }


def rank_cards(
    cards: Sequence[Card],
    classifications: Mapping[str, Classification],
    *,
    weights: Mapping[str, float] | None = None,
) -> list[RankedCard]:
    """Rank surviving cards, deterministically.

    Only cards that ran and survived are ranked; a falsified or unrunnable claim has no
    effect size to report. The order is by descending score with the card id as the
    final tie-break, so the same cards in give the same order out however the input was
    ordered.

    Args:
        cards: Every card from the run.
        classifications: Card id to its :class:`Classification`.
        weights: Component weights, defaulting to :data:`DEFAULT_WEIGHTS`. Missing keys
            take the default's value.

    Returns:
        The ranked surviving cards, best first.
    """
    w = dict(DEFAULT_WEIGHTS)
    w.update(dict(weights or {}))
    unknown = Classification(COMPUTED, "not classified")
    ranked: list[RankedCard] = []
    for card in cards:
        if card.error is not None or not card.passed:
            continue
        classification = classifications.get(card.id, unknown)
        effect = effect_size(card)
        margin = threshold_margin(card)
        contradicts = contradicts_naive(card)
        score = (
            w["effect"] * effect
            + w["margin"] * margin
            + w["class"] * CLASS_WEIGHT.get(classification.label, 0.0)
            + w["naive"] * (1.0 if contradicts else 0.0)
        )
        ranked.append(RankedCard(card, score, effect, margin, classification, contradicts))
    ranked.sort(key=lambda r: (-r.score, r.card.id))
    return ranked


# --- The number this task exists to produce ------------------------------------------


def stratified_survival(
    cards: Sequence[Card], classifications: Mapping[str, Classification]
) -> dict[str, Any]:
    """Split the survival rate by how hard each claim's answer was to get.

    This is the honest core of TASK-015. "Eleven of thirteen claims survived" is not a
    result if eleven of them only re-read the table the model was handed, so the rate is
    reported per class and the LOOKUP share of the survivors is stated outright.

    Claims that could not run are excluded from every survival denominator and counted
    separately: a test that did not execute produced no evidence in either direction,
    and folding it in would understate the rate for reasons that have nothing to do with
    the claims.

    Args:
        cards: Every card from the run.
        classifications: Card id to its :class:`Classification`.

    Returns:
        A JSON-ready mapping with the overall figures, a per-class block for each of
        :data:`CLASSES`, and the LOOKUP share of the survivors.
    """
    unknown = Classification(COMPUTED, "not classified")
    buckets: dict[str, dict[str, int]] = {
        label: {"claims": 0, "ran": 0, "survived": 0, "falsified": 0, "could_not_run": 0}
        for label in CLASSES
    }
    for card in cards:
        label = classifications.get(card.id, unknown).label
        bucket = buckets.setdefault(
            label, {"claims": 0, "ran": 0, "survived": 0, "falsified": 0, "could_not_run": 0}
        )
        bucket["claims"] += 1
        if card.error is not None:
            bucket["could_not_run"] += 1
            continue
        bucket["ran"] += 1
        if card.passed:
            bucket["survived"] += 1
        else:
            bucket["falsified"] += 1

    def rate(bucket: Mapping[str, int]) -> float | None:
        """Survivors over claims that actually ran, or None when none ran."""
        return bucket["survived"] / bucket["ran"] if bucket["ran"] else None

    total = {
        key: sum(b[key] for b in buckets.values())
        for key in ("claims", "ran", "survived", "falsified", "could_not_run")
    }
    survivors = total["survived"]
    lookup_survivors = buckets.get(LOOKUP, {}).get("survived", 0)
    computed_survivors = buckets.get(COMPUTED, {}).get("survived", 0)
    return {
        "n_claims": total["claims"],
        "n_ran": total["ran"],
        "n_survived": survivors,
        "n_falsified": total["falsified"],
        "n_could_not_run": total["could_not_run"],
        "survival_rate": rate(total) if total["ran"] else None,
        "by_class": {
            label: {**buckets[label], "survival_rate": rate(buckets[label])}
            for label in sorted(buckets)
        },
        "lookup_share_of_survivors": (lookup_survivors / survivors) if survivors else None,
        "computed_share_of_survivors": (computed_survivors / survivors) if survivors else None,
        "note": (
            "survival_rate denominators are claims that ran; claims that could not run "
            "are counted separately because they produced no evidence either way. A high "
            "overall rate carried by LOOKUP claims means the run mostly re-read its own "
            "briefing."
        ),
    }


# --- The loop ------------------------------------------------------------------------


@dataclass(frozen=True)
class StopReason:
    """Why the exploration stopped.

    Attributes:
        code: A stable machine-readable reason: ``"max_questions"``, ``"max_calls"``,
            ``"max_cost"``, ``"coverage_plateau"``, ``"no_questions"`` or
            ``"generator_error"``.
        detail: One line naming the numbers involved.
    """

    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the stop reason."""
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class QuestionRecord:
    """One question the run asked, and what it opened.

    Attributes:
        index: 1-based position in the run.
        question: The question as it was asked.
        n_cards: How many cards it produced.
        n_survived: How many of them survived.
        new_cells: The coverage cells it opened that nothing had opened before.
        error: The failure that stopped it, or None.
    """

    index: int
    question: str
    n_cards: int
    n_survived: int
    new_cells: tuple[str, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {
            "index": self.index,
            "question": self.question,
            "n_cards": self.n_cards,
            "n_survived": self.n_survived,
            "new_cells": list(self.new_cells),
            "error": self.error,
        }


@dataclass(frozen=True)
class ExplorationResult:
    """Everything one exploration produced.

    Attributes:
        questions: One :class:`QuestionRecord` per question attempted, in order.
        cards: Every card, in the order the questions produced them.
        classifications: Card id to its :class:`Classification`.
        coverage: The final :class:`Coverage`.
        stop: Why the run ended.
        ranking: Surviving cards, best first.
        survival: :func:`stratified_survival`'s mapping.
        calls: How many generator calls the run made.
        cost_usd: What the backend reported spending, 0.0 when it reports nothing.
        proposer: Provenance of whatever wrote the questions.
        headline: The checked one-line lede, or None.
        blurb: The checked plain-English summary, or None.
        synthesis_status: What the narration attempt did, always set.
        paths: Files written, keyed by a short label.
    """

    questions: tuple[QuestionRecord, ...]
    cards: tuple[Card, ...]
    classifications: dict[str, Classification]
    coverage: Coverage
    stop: StopReason
    ranking: tuple[RankedCard, ...]
    survival: dict[str, Any]
    calls: int
    cost_usd: float
    proposer: str
    headline: str | None = None
    blurb: str | None = None
    synthesis_status: str = "not attempted"
    paths: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the run record, without the card bodies.

        Returns:
            A JSON-ready mapping. Cards are written separately by
            :meth:`llm_insights.api.Study.ask`, so this carries only the classification
            of each one.
        """
        return {
            "stop": self.stop.to_dict(),
            "questions": [q.to_dict() for q in self.questions],
            "calls": self.calls,
            "cost_usd": self.cost_usd,
            "questions_from": self.proposer,
            "synthesis": self.synthesis_status,
            "classifications": {
                card_id: {"class": c.label, "reason": c.reason, "source": c.source}
                for card_id, c in sorted(self.classifications.items())
            },
        }


def _cost_of(generator: Any) -> float:
    """Read a backend's cumulative reported spend, or 0.0 when it keeps none."""
    value = getattr(generator, "cost_usd", 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def explore(
    study: Any,
    *,
    budget: Budget | None = None,
    proposer: Any = None,
    out: str | Path | None = None,
    seed_questions: Sequence[str] = (),
    narrow_failures: bool = True,
    synthesize_report: bool = True,
    skip_classes: Iterable[str] = (),
) -> ExplorationResult:
    """Run one autonomous exploration over an open study.

    The loop is: check the ceilings, check the plateau, take the next question (asking
    the proposer for more when the queue runs dry), investigate it through
    :meth:`llm_insights.api.Study.ask`, and record which coverage cells the resulting
    tests opened. Nothing in that sequence asks a model whether to continue.

    The ceilings are copied into locals before the first call, and the backend is
    wrapped in a :class:`_BudgetedGenerator` whose counter lives outside anything a
    generator can reach. A backend that reassigns ``budget`` on this module, on the
    study, or on itself changes nothing.

    Args:
        study: An open :class:`llm_insights.api.Study` with a generator. Its
            ``generator`` attribute is wrapped for the duration and restored on the way
            out, so the dataset and briefing are still built only once.
        budget: The hard ceilings. Defaults to :class:`Budget`'s defaults.
        proposer: Anything with ``propose_questions(k, coverage)``. Defaults to
            :class:`SchemaQuestionProposer`, which needs no model.
        out: Directory for the report, the coverage and the survival split. Each
            question also writes its own cards and report into a ``qNN`` subdirectory.
            None writes nothing.
        seed_questions: Questions to investigate before the proposer is consulted.
        narrow_failures: Passed through to :meth:`llm_insights.api.Study.ask`.
        synthesize_report: Ask the backend for a checked plain-English summary of the
            whole run. It costs one call, is checked by
            :func:`llm_insights.agent.synthesis.check_numeric_containment`, and can only
            add prose.
        skip_classes: Classes to *not* investigate. Empty by default, and that is
            deliberate: dropping LOOKUP questions would make the LOOKUP survival rate
            unmeasurable, and measuring it is the point of this task. An operator who
            already trusts the classifier can pass ``{"LOOKUP"}`` to spend the budget
            only on claims the briefing cannot answer.

    Returns:
        The :class:`ExplorationResult`.

    Raises:
        RuntimeError: If the study has no generator.
    """
    budget = budget or Budget()
    # Snapshot every ceiling. From here on nothing re-reads `budget`, so nothing a model
    # returns -- and no attribute anything reassigns -- can move one of them.
    max_questions = int(budget.max_questions)
    max_calls = int(budget.max_calls)
    max_cost = float(budget.max_cost_usd)
    plateau_after = int(budget.plateau_questions)
    per_question = int(budget.hypotheses_per_question)

    inner = study.generator
    if inner is None:
        raise RuntimeError(
            "this Study has no generator, so it cannot explore anything; pass one to "
            "open_study(generator=...)"
        )
    skip = {str(s).upper() for s in skip_classes}
    index = BriefingIndex(study.briefing)
    coverage = Coverage()
    proposer = proposer or SchemaQuestionProposer()
    wrapped = _BudgetedGenerator(inner, max_calls)
    out_dir = Path(out) if out is not None else None

    queue: list[str] = [str(q) for q in seed_questions]
    records: list[QuestionRecord] = []
    classifications: dict[str, Classification] = {}
    barren = 0
    stop: StopReason | None = None

    study.generator = wrapped
    try:
        while stop is None:
            if len(records) >= max_questions:
                stop = StopReason(
                    "max_questions", f"the budget allowed {max_questions} question(s)"
                )
                break
            if wrapped.calls >= max_calls:
                stop = StopReason("max_calls", f"the budget allowed {max_calls} model call(s)")
                break
            spent = _cost_of(inner)
            if spent >= max_cost:
                stop = StopReason(
                    "max_cost", f"the backend reported ${spent:.4f} against a ${max_cost:.2f} cap"
                )
                break
            if plateau_after and barren >= plateau_after:
                stop = StopReason(
                    "coverage_plateau",
                    f"the last {barren} question(s) opened no new coverage cell",
                )
                break

            if not queue:
                try:
                    offered = proposer.propose_questions(max_questions - len(records), coverage)
                except BudgetStopError as exc:
                    stop = StopReason("max_calls", str(exc))
                    break
                queue = [str(q) for q in offered]
            if not queue:
                stop = StopReason("no_questions", f"{proposer!r} offered no further question")
                break

            question = queue.pop(0)
            position = len(records) + 1
            LOG.info("question %d of at most %d: %s", position, max_questions, question)
            try:
                answer = study.ask(
                    question,
                    n=per_question,
                    out=(out_dir / f"q{position:02d}") if out_dir else None,
                    narrow_failures=narrow_failures,
                )
            except BudgetStopError as exc:
                stop = StopReason("max_calls", str(exc))
                break
            except Exception as exc:  # a backend failure must not discard verified work
                message = f"{type(exc).__name__}: {exc}"
                LOG.warning("question %d could not be investigated: %s", position, message)
                records.append(QuestionRecord(position, question, 0, 0, (), message))
                stop = StopReason("generator_error", message)
                break

            opened: set[Cell] = set()
            survived = 0
            for card in answer.cards:
                spec = {"primitive": card.primitive, "params": card.params}
                classification = classify_test(spec, index)
                classifications[card.id] = classification
                if classification.label in skip:
                    LOG.info(
                        "card %s is %s and was skipped by request", card.id, classification.label
                    )
                    continue
                opened |= coverage.record(spec)
                if card.error is None and card.passed:
                    survived += 1
            barren = 0 if opened else barren + 1
            records.append(
                QuestionRecord(
                    position,
                    question,
                    len(answer.cards),
                    survived,
                    tuple(sorted(c.key() for c in opened)),
                )
            )
            LOG.info(
                "question %d produced %d card(s) and opened %d new cell(s)",
                position,
                len(answer.cards),
                len(opened),
            )

        cards = tuple(study.cards)
        survival = stratified_survival(cards, classifications)
        ranking = tuple(rank_cards(cards, classifications))

        headline: str | None = None
        blurb: str | None = None
        status = "not attempted"
        if synthesize_report and cards:
            from llm_insights.agent.synthesis import synthesize

            outcome = synthesize(wrapped, _run_question(records), list(cards))
            headline, blurb, status = outcome.headline, outcome.blurb, outcome.status

        result = ExplorationResult(
            questions=tuple(records),
            cards=cards,
            classifications=classifications,
            coverage=coverage,
            stop=stop or StopReason("no_questions", "the loop ended without asking anything"),
            ranking=ranking,
            survival=survival,
            calls=wrapped.calls,
            cost_usd=_cost_of(inner),
            proposer=str(getattr(proposer, "name", type(proposer).__name__)),
            headline=headline,
            blurb=blurb,
            synthesis_status=status,
        )
    finally:
        study.generator = inner

    if out_dir is not None:
        result.paths.update(write_outputs(result, out_dir))
    return result


def _run_question(records: Sequence[QuestionRecord]) -> str:
    """Describe what the whole run set out to answer, for the narration prompt.

    Args:
        records: The questions asked.

    Returns:
        The single question when there was one, or a line naming all of them.
    """
    questions = [r.question for r in records]
    if len(questions) == 1:
        return questions[0]
    if not questions:
        return "(no question was investigated)"
    return "This run investigated several questions at once: " + " ".join(questions)


# --- Output --------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    """Render a rate as a percentage, or an em dash when there is no denominator."""
    return "—" if value is None else f"{100.0 * value:.1f}%"


def build_exploration_report(result: ExplorationResult) -> str:
    """Render the exploration report as Markdown.

    The stratified survival table comes before the findings on purpose. A reader who
    sees "12 of 13 survived" and stops has been misled; a reader who sees that eleven of
    the twelve were LOOKUP has the actual result of the run.

    Args:
        result: The finished exploration.

    Returns:
        The Markdown text.
    """
    survival = result.survival
    lines = [
        "# Autonomous exploration",
        "",
        f"- questions asked: **{len(result.questions)}** "
        f"(stopped: {result.stop.code} — {result.stop.detail})",
        f"- questions written by: {result.proposer}",
        f"- generator calls: {result.calls}; reported spend: ${result.cost_usd:.4f}",
        f"- coverage: {len(result.coverage.touched)} of {result.coverage.total} cells "
        f"({_pct(result.coverage.fraction)})",
        "",
        "Every claim below was written before its test ran, and every verdict is "
        "arithmetic. No language model graded a result or decided when to stop.",
        "",
    ]
    if result.headline:
        lines += [f"**{result.headline}**", ""]
    if result.blurb:
        lines += [result.blurb, ""]
    lines += [f"_narrative summary: {result.synthesis_status}_", ""]

    lines += [
        "## Survival, split by how hard the answer was to get",
        "",
        "`LOOKUP` means the answer was already printed in the briefing the model was "
        "handed. `DERIVED` means one arithmetic step from printed numbers. `COMPUTED` "
        "means the test had to touch arrays the briefing does not contain. A high "
        "overall survival rate carried by LOOKUP claims is not a finding about the "
        "simulation.",
        "",
        "| class | claims | ran | survived | falsified | could not run | survival rate |",
        "| ----- | -----: | --: | -------: | --------: | ------------: | ------------: |",
    ]
    for label in CLASSES:
        bucket = survival["by_class"].get(label)
        if not bucket:
            continue
        lines.append(
            f"| {label} | {bucket['claims']} | {bucket['ran']} | {bucket['survived']} | "
            f"{bucket['falsified']} | {bucket['could_not_run']} | "
            f"{_pct(bucket['survival_rate'])} |"
        )
    lines.append(
        f"| **all** | {survival['n_claims']} | {survival['n_ran']} | "
        f"{survival['n_survived']} | {survival['n_falsified']} | "
        f"{survival['n_could_not_run']} | {_pct(survival['survival_rate'])} |"
    )
    lines += [
        "",
        f"LOOKUP share of the surviving claims: **{_pct(survival['lookup_share_of_survivors'])}**. "
        f"COMPUTED share: **{_pct(survival['computed_share_of_survivors'])}**.",
        "",
        "## Questions",
        "",
        "| # | question | cards | survived | new coverage cells |",
        "| - | -------- | ----: | -------: | -----------------: |",
    ]
    for record in result.questions:
        note = f" (stopped: {record.error})" if record.error else ""
        lines.append(
            f"| {record.index} | {record.question}{note} | {record.n_cards} | "
            f"{record.n_survived} | {len(record.new_cells)} |"
        )

    lines += [
        "",
        "## Surviving claims, ranked",
        "",
        "Ranked by effect size, margin over the pre-registered threshold, class, and "
        "whether the result contradicts the naive 'it rises with flow' expectation — "
        "equally weighted, with every component printed so the ranking can be redone.",
        "",
        "| rank | id | class | score | effect | margin | against expectation | claim |",
        "| ---: | -- | ----- | ----: | -----: | -----: | ------------------- | ----- |",
    ]
    if result.ranking:
        for position, ranked in enumerate(result.ranking, start=1):
            lines.append(
                f"| {position} | {ranked.card.id} | {ranked.classification.label} | "
                f"{ranked.score:.3f} | {ranked.effect:.3f} | {ranked.margin:.3f} | "
                f"{'yes' if ranked.contradicts else 'no'} | {ranked.card.claim} |"
            )
    else:
        lines.append("| — | — | — | — | — | — | — | nothing survived |")

    lines += ["", "## How each claim was classified", ""]
    for card_id, classification in sorted(result.classifications.items()):
        lines.append(f"- **{card_id}** — {classification.label}: {classification.reason}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(result: ExplorationResult, out: str | Path) -> dict[str, Path]:
    """Write the report, the coverage and the survival split.

    Args:
        result: The finished exploration.
        out: The directory to write into. Created if it does not exist.

    Returns:
        The written files, keyed by a short label.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report:": out_dir / "exploration_report.md",
        "coverage:": out_dir / "coverage.json",
        "survival:": out_dir / "stratified_survival.json",
        "ranking:": out_dir / "ranking.json",
        "run:": out_dir / "exploration.json",
        "cards:": out_dir / "cards.json",
    }
    paths["report:"].write_text(build_exploration_report(result), encoding="utf-8")
    _write_json(paths["coverage:"], result.coverage.to_dict())
    _write_json(paths["survival:"], result.survival)
    _write_json(paths["ranking:"], [r.to_dict() for r in result.ranking])
    _write_json(paths["run:"], result.to_dict())
    cards_to_json(list(result.cards), paths["cards:"])
    LOG.info("wrote %d exploration artifact(s) to %s", len(paths), out_dir)
    return paths


def _write_json(path: Path, payload: Any) -> None:
    """Write a JSON payload with a trailing newline.

    Args:
        path: Destination file.
        payload: Any JSON-serializable value.
    """
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --- CLI ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m llm_insights.agent.explore",
        description=(
            "Explore the FSG + GRN model without being given a question, and report the "
            "survival rate split by whether the answer was already in the briefing."
        ),
    )
    parser.add_argument("--root", default=None, help="dataset root; defaults to the checkout")
    parser.add_argument(
        "--generator",
        default="echo",
        help="hypothesis backend, as accepted by llm_insights.agent.generator.make_generator",
    )
    parser.add_argument(
        "--transcript", default=None, help="transcript file, for --generator transcript"
    )
    parser.add_argument("--model", default=None, help="model id, for backends that take one")
    parser.add_argument("--briefing-file", default=None, help="read the briefing from this file")
    parser.add_argument("--out", default=None, help="directory for the report and the JSON")
    parser.add_argument("--max-questions", type=int, default=Budget.max_questions)
    parser.add_argument("--max-calls", type=int, default=Budget.max_calls)
    parser.add_argument("--max-cost", type=float, default=Budget.max_cost_usd)
    parser.add_argument("--plateau", type=int, default=Budget.plateau_questions)
    parser.add_argument("-n", "--hypotheses", type=int, default=Budget.hypotheses_per_question)
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="seed question, investigated before the proposer is consulted; repeatable",
    )
    parser.add_argument(
        "--ask-model-for-questions",
        action="store_true",
        help="let the backend propose the questions instead of the schema enumerator",
    )
    parser.add_argument(
        "--skip-class",
        action="append",
        default=[],
        choices=list(CLASSES),
        help="do not investigate claims of this class; off by default, see explore()",
    )
    parser.add_argument("--no-synthesis", action="store_true", help="skip the narrative summary")
    parser.add_argument("--verbose", action="store_true", help="log at DEBUG")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one exploration from the command line.

    Args:
        argv: Arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 when the backend could not be built.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from llm_insights.agent.generator import make_generator
    from llm_insights.api import open_study

    try:
        generator = make_generator(args.generator, transcript=args.transcript, model=args.model)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"could not build the {args.generator} backend: {exc}")
        return 2

    budget = Budget(
        max_questions=args.max_questions,
        max_calls=args.max_calls,
        max_cost_usd=args.max_cost,
        plateau_questions=args.plateau,
        hypotheses_per_question=args.hypotheses,
    )
    with open_study(args.root, generator=generator, briefing_file=args.briefing_file) as study:
        proposer: Any = SchemaQuestionProposer()
        if args.ask_model_for_questions:
            proposer = ModelQuestionProposer(generator, fallback=proposer)
        result = explore(
            study,
            budget=budget,
            proposer=proposer,
            out=args.out,
            seed_questions=args.question,
            synthesize_report=not args.no_synthesis,
            skip_classes=args.skip_class,
        )

    survival = result.survival
    print(f"\nasked {len(result.questions)} question(s); stopped: {result.stop.code}")
    print(f"coverage: {len(result.coverage.touched)}/{result.coverage.total} cells")
    print(
        f"survival: {survival['n_survived']}/{survival['n_ran']} claims that ran "
        f"({_pct(survival['survival_rate'])})"
    )
    for label in CLASSES:
        bucket = survival["by_class"].get(label, {})
        if bucket.get("claims"):
            print(
                f"  {label:<8} {bucket['survived']}/{bucket['ran']} "
                f"({_pct(bucket['survival_rate'])})"
            )
    print(f"LOOKUP share of survivors: {_pct(survival['lookup_share_of_survivors'])}")
    for label, path in sorted(result.paths.items()):
        print(f"{label:<12} {path}")
    return 0


__all__ = [
    "CLASSES",
    "CLASS_WEIGHT",
    "COMPUTED",
    "DEFAULT_WEIGHTS",
    "DERIVED",
    "FRACTION_EQUIVALENTS",
    "LOOKUP",
    "BriefingIndex",
    "Budget",
    "BudgetStopError",
    "Cell",
    "Classification",
    "Coverage",
    "ExplorationResult",
    "ModelQuestionProposer",
    "QuestionRecord",
    "RankedCard",
    "SchemaQuestionProposer",
    "StopReason",
    "build_exploration_report",
    "build_parser",
    "build_question_prompt",
    "cells_for_test",
    "classify_card",
    "classify_test",
    "condition_label",
    "contradicts_naive",
    "effect_size",
    "explore",
    "grn_columns",
    "main",
    "observed_in_briefing",
    "parse_questions",
    "primitive_targets",
    "rank_cards",
    "schema_catalogue",
    "stratified_survival",
    "threshold_margin",
    "write_outputs",
]


if __name__ == "__main__":  # pragma: no cover - exercised by running the module
    raise SystemExit(main())
