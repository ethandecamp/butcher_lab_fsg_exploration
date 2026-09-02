"""The hypothesis card: one claim, one executed test, one verdict.

A :class:`Card` is the frozen, serializable record of a single falsification attempt.
It is deliberately *flat* and *self-describing*: everything a reader needs to judge the
claim — what was asserted, what rule was fixed in advance, which executable primitive
ran, and every number that rule consumed — is present on the card itself. Nothing is
looked up later, and no language model is consulted at read time.

Two fields carry the part of the story that matters most:

* ``revision_of`` — the id of a falsified claim that this card narrows.
* ``revised_by`` — the id of the claim that replaced this one after it was falsified.

Together they form a *chain*: a claim the simulator rejected, followed by the tighter
claim the agent wrote in its place. :mod:`llm_insights.cards.render` renders that chain
as a single connected unit, because a falsified claim that produced a better one is the
system working, not the system failing.

The harness types (``llm_insights.harness.spec.Hypothesis`` and ``Outcome``) are
imported lazily inside :func:`cards_from_run`, and every field is read through
duck-typed accessors, so this module keeps working while that module is still moving.
Mappings are accepted wherever objects are, for the same reason.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - annotations only, never imported at runtime
    from llm_insights.harness.spec import Hypothesis, Outcome

LOG = logging.getLogger(__name__)

#: Sentinel distinguishing "field absent" from "field present and None".
_MISSING: Final[object] = object()

#: Attribute names tried, in order, when reading each field off a harness object.
#: The first name is the expected one; the rest are tolerated aliases so that a
#: rename in the harness does not silently produce empty cards.
_HYPOTHESIS_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "id": ("id", "hypothesis_id", "hid"),
    "claim": ("claim", "statement", "text"),
    "rationale": ("rationale", "reason", "why"),
    "decision_rule": ("decision_rule", "rule", "criterion"),
    # "test" is deliberately NOT an alias here: on Hypothesis it names the nested
    # TestSpec, and matching it would stringify the whole spec into this field.
    "primitive": ("primitive", "primitive_name"),
    "params": ("params", "parameters", "kwargs"),
    "parent_id": ("parent_id", "revision_of", "parent"),
}
#: Fields that live on ``Hypothesis.test`` (a ``TestSpec``) rather than on the
#: hypothesis itself. They are looked for in both places, hypothesis first.
_TEST_SPEC_FIELDS: Final[frozenset[str]] = frozenset({"decision_rule", "primitive", "params"})
#: Attribute names tried when reaching for the nested test spec.
_TEST_SPEC_ATTRS: Final[tuple[str, ...]] = ("test", "spec", "test_spec")

_OUTCOME_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "passed": ("passed", "survived", "ok"),
    "observed": ("observed", "observations", "values", "metrics"),
    "summary": ("summary", "message", "explanation"),
    "error": ("error", "err", "exception"),
    "hypothesis_id": ("hypothesis_id", "id", "hid", "claim_id"),
}


@dataclass(frozen=True)
class Card:
    """One claim, the rule fixed before testing it, and what the simulator returned.

    Attributes:
        id: Stable short identifier, unique within a run (e.g. ``"H3"``).
        claim: The plain-English qualitative claim being tested.
        rationale: Why the agent proposed the claim.
        decision_rule: The pass/fail rule, stated *before* the test ran.
        primitive: Name of the executable test that was run.
        params: Arguments the primitive was called with.
        passed: True if the decision rule was satisfied. Always False when ``error``
            is set — a test that could not run did not pass.
        observed: Every number the decision rule consumed, keyed by name.
        summary: One line describing what happened.
        error: Why the test could not run, or None if it ran.
        revision_of: Id of the falsified claim this card narrows, if any.
        revised_by: Id of the claim that replaced this one after falsification, if any.
    """

    id: str
    claim: str
    rationale: str
    decision_rule: str
    primitive: str
    params: dict[str, Any]
    passed: bool
    observed: dict[str, float]
    summary: str
    error: str | None = None
    revision_of: str | None = None
    revised_by: str | None = None

    @property
    def could_not_run(self) -> bool:
        """True when the primitive raised and no verdict was reached."""
        return self.error is not None

    @property
    def was_narrowed(self) -> bool:
        """True when this claim was falsified and a successor claim replaced it."""
        return not self.passed and self.error is None and self.revised_by is not None

    @property
    def is_revision(self) -> bool:
        """True when this claim was written to narrow an earlier falsified one."""
        return self.revision_of is not None


# --- Duck-typed reads over the harness objects -----------------------------------


def _read(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """Read the first present attribute or mapping key from ``names``.

    Args:
        obj: A harness object or a mapping standing in for one.
        names: Candidate field names, most-expected first.
        default: Returned when none of the names is present.

    Returns:
        The first value found, or ``default``.
    """
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _field(obj: Any, key: str, aliases: Mapping[str, tuple[str, ...]], default: Any = None) -> Any:
    """Read one logical field off a harness object using its alias table."""
    return _read(obj, aliases.get(key, (key,)), default)


def _hyp_field(hyp: Any, key: str, default: Any = None) -> Any:
    """Read a hypothesis field, falling through to its nested test spec.

    ``Hypothesis`` keeps the executable half of a claim on a nested ``TestSpec``
    (``primitive``, ``params``, ``decision_rule``), so those three are looked for on
    the hypothesis first and on ``hypothesis.test`` second. Anything that flattens
    them onto the hypothesis itself keeps working unchanged.

    Args:
        hyp: A hypothesis object or a mapping standing in for one.
        key: The logical field name.
        default: Returned when the field is present in neither place.

    Returns:
        The field value, or ``default``.
    """
    value = _field(hyp, key, _HYPOTHESIS_ALIASES, _MISSING)
    if value is not _MISSING:
        return value
    if key in _TEST_SPEC_FIELDS:
        spec = _read(hyp, _TEST_SPEC_ATTRS)
        if spec is not None:
            value = _field(spec, key, _HYPOTHESIS_ALIASES, _MISSING)
            if value is not _MISSING:
                return value
    return default


def _as_str(value: Any) -> str:
    """Coerce a field to a string, mapping None to the empty string."""
    return "" if value is None else str(value)


def _as_float_map(value: Any) -> dict[str, float]:
    """Coerce an ``observed`` payload to ``{name: float}``, dropping non-numerics.

    Args:
        value: A mapping of observation names to numbers, or None.

    Returns:
        A plain dict of floats. Entries whose value will not convert to float are
        skipped with a warning rather than failing the whole run.
    """
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            LOG.warning("observed value %r for key %r is not numeric; dropped", raw, key)
    return out


def cards_from_run(hypotheses: Iterable[Hypothesis], outcomes: Iterable[Outcome]) -> list[Card]:
    """Join proposed hypotheses to their executed outcomes and link revision chains.

    Outcomes are matched to hypotheses by id when they carry one, and positionally
    otherwise. A hypothesis with no outcome becomes a ``COULD NOT RUN`` card, so a
    crashed primitive can never silently vanish from the report.

    ``Hypothesis.parent_id`` drives the linking: the child gets ``revision_of`` set to
    its parent, and the parent gets ``revised_by`` set to the child.

    Args:
        hypotheses: The claims the agent proposed, in proposal order.
        outcomes: The results of executing them.

    Returns:
        One card per hypothesis, in the order the hypotheses were given, with
        revision links filled in on both ends of every chain.
    """
    # Lazily touch the harness module so its import errors surface here, named, rather
    # than at module import time. Absence is not fatal: every read below is duck-typed.
    try:  # pragma: no cover - depends on a module written in parallel
        import llm_insights.harness.spec  # noqa: F401
    except ImportError:
        LOG.debug("llm_insights.harness.spec unavailable; using duck-typed access only")

    hyps = list(hypotheses)
    outs = list(outcomes)
    by_id, leftover = _index_outcomes(outs)

    cards: list[Card] = []
    parent_of: dict[str, str] = {}
    for position, hyp in enumerate(hyps):
        hyp_id = _as_str(_hyp_field(hyp, "id", f"H{position + 1}"))
        outcome = by_id.get(hyp_id)
        if outcome is None and not by_id and position < len(leftover):
            outcome = leftover[position]
        parent_id = _hyp_field(hyp, "parent_id")
        if parent_id:
            parent_of[hyp_id] = str(parent_id)
        cards.append(_build_card(hyp, outcome, hyp_id, parent_id))

    return _link_revisions(cards, parent_of)


def _index_outcomes(outcomes: Sequence[Any]) -> tuple[dict[str, Any], list[Any]]:
    """Split outcomes into an id-keyed index and a positional fallback list."""
    by_id: dict[str, Any] = {}
    for outcome in outcomes:
        oid = _field(outcome, "hypothesis_id", _OUTCOME_ALIASES)
        if oid:
            by_id[str(oid)] = outcome
    return by_id, list(outcomes)


def _build_card(hyp: Any, outcome: Any | None, hyp_id: str, parent_id: Any) -> Card:
    """Assemble one card from a hypothesis and its (possibly missing) outcome."""
    if outcome is None:
        LOG.warning("hypothesis %s has no outcome; recording it as COULD NOT RUN", hyp_id)
        passed = False
        observed: dict[str, float] = {}
        summary = "No outcome was recorded for this claim."
        error = "The primitive produced no outcome record."
    else:
        error_value = _field(outcome, "error", _OUTCOME_ALIASES)
        error = _as_str(error_value) or None
        passed = bool(_field(outcome, "passed", _OUTCOME_ALIASES, False)) and error is None
        observed = _as_float_map(_field(outcome, "observed", _OUTCOME_ALIASES))
        summary = _as_str(_field(outcome, "summary", _OUTCOME_ALIASES))

    params = _hyp_field(hyp, "params", {})
    return Card(
        id=hyp_id,
        claim=_as_str(_hyp_field(hyp, "claim")),
        rationale=_as_str(_hyp_field(hyp, "rationale")),
        decision_rule=_as_str(_hyp_field(hyp, "decision_rule")),
        primitive=_as_str(_hyp_field(hyp, "primitive")),
        params=dict(params) if isinstance(params, Mapping) else {},
        passed=passed,
        observed=observed,
        summary=summary,
        error=error,
        revision_of=str(parent_id) if parent_id else None,
        revised_by=None,
    )


def _link_revisions(cards: Sequence[Card], parent_of: Mapping[str, str]) -> list[Card]:
    """Fill ``revised_by`` on every card that a later card narrows.

    Args:
        cards: Cards with ``revision_of`` already set from ``parent_id``.
        parent_of: Maps a child card id to the id of the card it narrows.

    Returns:
        A new list of cards; parents whose successor exists gain ``revised_by``.
    """
    child_of: dict[str, str] = {}
    known = {card.id for card in cards}
    for child_id, parent_id in parent_of.items():
        if parent_id not in known:
            LOG.warning("card %s claims parent %s, which is not in this run", child_id, parent_id)
            continue
        child_of[parent_id] = child_id

    linked: list[Card] = []
    for card in cards:
        successor = child_of.get(card.id)
        linked.append(replace_links(card, revised_by=successor) if successor else card)
    return linked


def replace_links(
    card: Card, *, revision_of: str | None = None, revised_by: str | None = None
) -> Card:
    """Return a copy of ``card`` with its revision links overridden.

    Args:
        card: The card to copy.
        revision_of: New ``revision_of`` value, or None to keep the existing one.
        revised_by: New ``revised_by`` value, or None to keep the existing one.

    Returns:
        A new frozen :class:`Card`.
    """
    data = asdict(card)
    if revision_of is not None:
        data["revision_of"] = revision_of
    if revised_by is not None:
        data["revised_by"] = revised_by
    return Card(**data)


# --- Serialization ----------------------------------------------------------------


def cards_to_json(cards: Sequence[Card], path: str | Path) -> None:
    """Write cards to a JSON file, one object per card, in order.

    Args:
        cards: The cards to serialize.
        path: Destination file. Parent directories are created if needed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(card) for card in cards]
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("wrote %d cards to %s", len(payload), target)


def cards_from_json(path: str | Path) -> list[Card]:
    """Read cards back from a file written by :func:`cards_to_json`.

    Unknown keys are ignored so an older file still loads after a field is added;
    missing optional keys fall back to their dataclass defaults.

    Args:
        path: The JSON file to read.

    Returns:
        The cards, in file order.

    Raises:
        ValueError: If the file does not contain a list of objects, or a card object
            is missing a required field.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of cards, got {type(raw).__name__}")

    known = {f.name for f in fields(Card)}
    required = {f.name for f in fields(Card) if f.default is MISSING}
    cards: list[Card] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: card {index} is {type(item).__name__}, not an object")
        missing = required - set(item)
        if missing:
            raise ValueError(f"{path}: card {index} is missing {sorted(missing)}")
        cards.append(Card(**{k: v for k, v in item.items() if k in known}))
    LOG.info("read %d cards from %s", len(cards), path)
    return cards
