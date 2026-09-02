"""Declarative description of a falsifiable claim and the record of testing it.

The point of this module is that a hypothesis carries its own decision rule, written
*before* the test runs. Nothing downstream is allowed to decide after the fact what
"passing" would have meant, so :class:`TestSpec` is frozen and the runner only ever
reads it.

Three small dataclasses and three serialization helpers live here:

* :class:`TestSpec` names a primitive in :data:`llm_insights.harness.primitives.PRIMITIVES`
  and the exact parameters it is called with.
* :class:`Hypothesis` wraps that with the plain-English claim it encodes. ``parent_id``
  links a narrowed revision back to the broader claim that was falsified.
* :class:`Outcome` is what the runner returns. ``error`` is set only when the test could
  not be evaluated at all, which is a different event from a claim being refuted.

:func:`hypothesis_from_dict` is deliberately strict: an LLM proposing hypotheses as JSON
should be told precisely which field it got wrong, not have a typo silently accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

LOG = logging.getLogger(__name__)

#: Keys accepted at the top level of a serialized hypothesis.
_HYPOTHESIS_KEYS: frozenset[str] = frozenset({"id", "claim", "rationale", "test", "parent_id"})
#: Keys accepted inside a serialized ``test`` block.
_TEST_KEYS: frozenset[str] = frozenset({"primitive", "params", "decision_rule"})


@dataclass(frozen=True)
class TestSpec:
    """The executable half of a hypothesis: which primitive, with which parameters.

    Attributes:
        primitive: Key into :data:`llm_insights.harness.primitives.PRIMITIVES`.
        params: Keyword arguments passed to that primitive, minus the dataset.
        decision_rule: Human-readable statement of what counts as passing. Written
            before the test runs; the runner never consults it, it exists so a reader
            can check that the code and the intent agree.
    """

    primitive: str
    params: dict[str, Any]
    decision_rule: str


@dataclass(frozen=True)
class Hypothesis:
    """A qualitative claim about the simulation plus the test that could refute it.

    Attributes:
        id: Stable identifier, unique within a run.
        claim: The claim in plain English, as proposed.
        rationale: Why the claim is worth testing.
        test: The :class:`TestSpec` that decides it.
        parent_id: Id of the claim this one narrows, when this hypothesis is a revision
            of something that failed. ``None`` for an original claim.
    """

    id: str
    claim: str
    rationale: str
    test: TestSpec
    parent_id: str | None = None


@dataclass(frozen=True)
class Outcome:
    """The result of running one hypothesis against the data.

    Attributes:
        hypothesis_id: The :attr:`Hypothesis.id` this outcome belongs to.
        passed: True only if the test ran and its decision rule was satisfied.
        observed: Every number the decision used, so the verdict can be re-derived by
            hand. Empty when the test could not run.
        summary: One line stating what was measured and what happened. Prefixed
            ``PASS``, ``FAIL`` or ``COULD NOT RUN`` so the three are never confused.
        error: Non-``None`` only when the test could not be evaluated. A test that ran
            and refuted its claim has ``passed=False`` and ``error=None``.
        error_kind: What sort of failure ``error`` describes. ``"environment"`` means
            the machine could not run the test -- a missing dependency, an unreadable
            file -- which is evidence about the setup and about nothing else. ``None``
            covers both a test that ran and a test defeated by the hypothesis itself, a
            bad parameter or an empty selection. The distinction matters because an
            environment fault reported as an ordinary failure reads like a fact about
            the model; on 2026-09-02 a missing scipy did exactly that.
    """

    hypothesis_id: str
    passed: bool
    observed: dict[str, float]
    summary: str
    error: str | None = None
    error_kind: str | None = None

    @property
    def ran(self) -> bool:
        """True if the test was actually evaluated, whatever its verdict."""
        return self.error is None

    @property
    def is_environment_fault(self) -> bool:
        """True when the failure was the machine's, not the hypothesis's."""
        return self.error is not None and self.error_kind == "environment"


def _require(d: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    """Fetch a required key and check its type.

    Args:
        d: The mapping being validated.
        key: The key that must be present.
        kind: Acceptable type(s) for the value.
        where: Human-readable location, used in error messages.

    Returns:
        The value at ``key``.

    Raises:
        ValueError: If the key is missing or the value has the wrong type.
    """
    if key not in d:
        raise ValueError(f"{where}: missing required field {key!r}")
    value = d[key]
    if not isinstance(value, kind):
        names = kind.__name__ if isinstance(kind, type) else "/".join(k.__name__ for k in kind)
        raise ValueError(
            f"{where}: field {key!r} must be {names}, got {type(value).__name__} ({value!r})"
        )
    return value


def _require_nonempty_str(d: dict[str, Any], key: str, where: str) -> str:
    """Fetch a required non-empty string field.

    Args:
        d: The mapping being validated.
        key: The key that must be present.
        where: Human-readable location, used in error messages.

    Returns:
        The stripped-non-empty string value.

    Raises:
        ValueError: If the field is missing, not a string, or blank.
    """
    value = _require(d, key, str, where)
    if not value.strip():
        raise ValueError(f"{where}: field {key!r} must not be empty")
    return value


def _reject_unknown(d: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    """Raise if the mapping carries keys outside ``allowed``.

    Args:
        d: The mapping being validated.
        allowed: The permitted key names.
        where: Human-readable location, used in error messages.

    Raises:
        ValueError: If any unexpected key is present.
    """
    extra = sorted(set(d) - allowed)
    if extra:
        raise ValueError(
            f"{where}: unexpected field(s) {extra}; allowed fields are {sorted(allowed)}"
        )


def test_spec_from_dict(d: Any, where: str = "test") -> TestSpec:
    """Build a :class:`TestSpec` from a plain mapping, strictly.

    Args:
        d: The mapping to validate, normally decoded from JSON.
        where: Human-readable location, used in error messages.

    Returns:
        The validated :class:`TestSpec`.

    Raises:
        ValueError: If a field is missing, mistyped, empty, unexpected, or names a
            primitive that is not registered.
    """
    from llm_insights.harness.primitives import PRIMITIVES  # local: avoids import cycle

    if not isinstance(d, dict):
        raise ValueError(f"{where}: must be an object/dict, got {type(d).__name__}")
    _reject_unknown(d, _TEST_KEYS, where)

    primitive = _require_nonempty_str(d, "primitive", where)
    if primitive not in PRIMITIVES:
        raise ValueError(
            f"{where}: field 'primitive' names {primitive!r}, which is not a known test; "
            f"known tests are {sorted(PRIMITIVES)}"
        )

    params = _require(d, "params", dict, where)
    bad_keys = sorted(k for k in params if not isinstance(k, str))
    if bad_keys:
        raise ValueError(f"{where}: field 'params' has non-string key(s) {bad_keys}")

    decision_rule = _require_nonempty_str(d, "decision_rule", where)
    return TestSpec(primitive=primitive, params=dict(params), decision_rule=decision_rule)


def hypothesis_from_dict(d: Any) -> Hypothesis:
    """Build a :class:`Hypothesis` from a plain mapping, strictly.

    Every failure names the offending field, because the usual producer of these
    mappings is a language model that must be told exactly what to fix.

    Args:
        d: The mapping to validate, normally decoded from JSON.

    Returns:
        The validated :class:`Hypothesis`.

    Raises:
        ValueError: If any field is missing, mistyped, empty, or unexpected, or if the
            nested test block is invalid.
    """
    if not isinstance(d, dict):
        raise ValueError(f"hypothesis: must be an object/dict, got {type(d).__name__}")
    _reject_unknown(d, _HYPOTHESIS_KEYS, "hypothesis")

    hid = _require_nonempty_str(d, "id", "hypothesis")
    claim = _require_nonempty_str(d, "claim", "hypothesis")
    rationale = _require_nonempty_str(d, "rationale", "hypothesis")

    parent_id = d.get("parent_id")
    if parent_id is not None and not isinstance(parent_id, str):
        raise ValueError(
            "hypothesis: field 'parent_id' must be str or null, got "
            f"{type(parent_id).__name__} ({parent_id!r})"
        )

    spec = test_spec_from_dict(d.get("test"), where=f"hypothesis {hid!r} test")
    LOG.debug("parsed hypothesis %s -> primitive %s", hid, spec.primitive)
    return Hypothesis(id=hid, claim=claim, rationale=rationale, test=spec, parent_id=parent_id)


def hypothesis_to_dict(h: Hypothesis) -> dict[str, Any]:
    """Serialize a hypothesis to a JSON-ready mapping.

    Args:
        h: The hypothesis to serialize.

    Returns:
        A mapping that :func:`hypothesis_from_dict` accepts unchanged.
    """
    return {
        "id": h.id,
        "claim": h.claim,
        "rationale": h.rationale,
        "parent_id": h.parent_id,
        "test": {
            "primitive": h.test.primitive,
            "params": dict(h.test.params),
            "decision_rule": h.test.decision_rule,
        },
    }


def outcome_to_dict(o: Outcome) -> dict[str, Any]:
    """Serialize an outcome to a JSON-ready mapping.

    Args:
        o: The outcome to serialize.

    Returns:
        A mapping with plain Python scalars only.
    """
    return {
        "hypothesis_id": o.hypothesis_id,
        "passed": bool(o.passed),
        "observed": {k: float(v) for k, v in o.observed.items()},
        "summary": o.summary,
        "error": o.error,
        "error_kind": o.error_kind,
    }


def narrow(h: Hypothesis, new_id: str, claim: str, rationale: str, test: TestSpec) -> Hypothesis:
    """Create a narrowed revision of a hypothesis, linked by ``parent_id``.

    Args:
        h: The hypothesis being revised, normally one that was falsified.
        new_id: Identifier for the revision.
        claim: The narrowed claim.
        rationale: Why the narrowing is justified.
        test: The new, tighter test.

    Returns:
        A new :class:`Hypothesis` whose ``parent_id`` points at ``h``.
    """
    return replace(h, id=new_id, claim=claim, rationale=rationale, test=test, parent_id=h.id)
