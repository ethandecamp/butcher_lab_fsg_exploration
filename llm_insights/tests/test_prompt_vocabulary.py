"""The prompt's vocabulary is a compression of the validator, and is checked against it.

A primitive's ``field``/``metric`` parameter draws on one of several disjoint
namespaces, and the same physical quantity is spelled differently in each:
``wss_mean_dyn_cm2`` is a metric, ``wss_dyn_cm2`` is a GRN table column, and ``wss`` is
an arc-resolved profile field. On 2026-09-09 a live run spent four of five hypotheses on
that confusion in a single question, because the prompt named the metrics and the GRN
columns as two undifferentiated lists and never said which primitive took which.

Whatever the prompt offers is therefore a *claim about the validator*, and this module
falsifies it in both directions:

* **Forward.** Every name :func:`~llm_insights.agent.generator.primitive_vocabularies`
  offers for a primitive is passed to that primitive. The call is free to fail -- the
  stub dataset here has three usable rows and no metric table -- but it must not fail
  with a *name* error. A name the prompt offers and the code rejects is a defect.
* **Reverse.** A deliberately bogus name is passed to each primitive, and the valid set
  is parsed out of **the validator's own error message**, which enumerates it. Every
  name recovered that way must appear in the prompt. This is the direction that catches
  drift, and it reads the answer from the exception text rather than from the constant
  the prompt reads, so the two sides are not the same object compared with itself.

A third test covers the stale-registry defect Lane B reported: the eight GRN metrics
register as an import side effect of :mod:`llm_insights.summary.sections.grn`, so
``build_system_prompt()`` in a process that has not built a briefing used to offer 16
metric names out of 24. It is checked in a *subprocess*, because a test that has already
imported the briefing machinery cannot observe it.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_insights.agent.generator import (
    build_system_prompt,
    invalid_name_examples,
    primitive_vocabularies,
)
from llm_insights.harness.primitives import PRIMITIVES
from llm_insights.io.dataset import GRNField

#: The source tree, handed to the subprocess that checks the metric vocabulary is not
#: stale. That check has to run in a process which has not imported the briefing.
SRC = Path(__file__).resolve().parents[1] / "src"

#: A name no vocabulary contains, used to make each primitive print its valid set.
BOGUS = "definitely_not_a_field_name"

#: The four shapes a name rejection takes across the primitives. Each one enumerates the
#: set it would have accepted, and that enumeration is what the reverse direction reads.
NAME_ERROR = re.compile(r"unknown metric |unknown GRN field |unknown case |must be one of ")

#: Pulls every Python list literal out of a rejection message. ``resolve_case`` prints
#: two -- the folder names and the condition labels -- and both are accepted spellings.
LIST_IN_MESSAGE = re.compile(r"\[[^]]*\]", re.DOTALL)

#: A value for each parameter of each primitive that is valid, boring and *not* under
#: test, so a call reaches the name check instead of tripping over something else on the
#: way. Written from the signatures by hand rather than read from the prompt's own
#: tables: a fixture that borrowed the thing under test would agree with it by
#: construction. A primitive that gains a parameter fails :func:`_params` loudly here.
DEFAULTS: dict[str, Any] = {
    "case": "Healthy",
    "case_a": "Healthy",
    "case_b": "Overflow",
    "cases": ["Healthy", "Overflow"],
    "direction": "increasing",
    "field": "wss",
    "field_x": "wss_dyn_cm2",
    "field_y": "von_mises_pa",
    "method": "pearson",
    "metric": "cushion_height_mm",
    "min_fraction": 0.5,
    "min_rel_diff": 0.0,
    "op": "gt",
    "s_max": 1.0,
    "s_min": 0.0,
    "step": 14,
    "strict": True,
    "threshold": 1.0,
    "value": 0.0,
}


def _grn_field() -> GRNField:
    """Build a GRN field carrying every activity column the loader would write.

    Returns:
        A three-row :class:`~llm_insights.io.dataset.GRNField`. Three rows is the
        minimum ``correlation`` accepts, which keeps a valid name from failing on row
        count rather than on the name itself.
    """
    from llm_insights.io.dataset import GRN_NODES, GRN_SCENARIOS

    values = np.array([1.0, 2.0, 3.0], dtype=float)
    return GRNField(
        case="flow_U0p0360",
        condition="Healthy",
        x_mm=values,
        y_mm=values,
        von_mises_pa=values,
        wss_dyn_cm2=values,
        mech_norm=values,
        wss_norm=values,
        activities={
            f"{node}_{scen}": values.copy() for node in GRN_NODES for scen in GRN_SCENARIOS
        },
    )


class _StubDataset:
    """The smallest dataset a name check has to get past.

    Nothing here is expected to produce a verdict. Every method exists so that a
    *valid* name fails on data rather than on the name, which is exactly the
    distinction these tests turn on.
    """

    root = "/stub"

    @staticmethod
    def resolve_case(name: str) -> str:
        """Resolve a case through the real dataset's own table.

        Args:
            name: A case folder name or condition label.

        Returns:
            The canonical case folder name.
        """
        from llm_insights.io.dataset import Dataset

        return Dataset.resolve_case(name)

    def grn(self, case: str) -> GRNField:
        """Return the stub GRN table for any case.

        Args:
            case: Ignored.

        Returns:
            The stub field.
        """
        del case
        return _grn_field()

    def __getattr__(self, item: str) -> Any:
        """Fail loudly for anything else, as a missing capability, not a name error.

        Args:
            item: The attribute asked for.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"the stub dataset does not implement {item!r}")


def _params(primitive: str) -> dict[str, Any]:
    """Build a full, boring parameter set for one primitive from its signature.

    Args:
        primitive: Registered primitive name.

    Returns:
        Every parameter of that primitive except ``ds``, set to its :data:`DEFAULTS`
        value.

    Raises:
        KeyError: If the primitive has a parameter :data:`DEFAULTS` does not cover,
            which means the fixture has fallen behind the code and must be updated.
    """
    import inspect

    signature = inspect.signature(PRIMITIVES[primitive])
    return {param: DEFAULTS[param] for param in signature.parameters if param != "ds"}


def _call(primitive: str, param: str, value: Any) -> Exception | None:
    """Call one primitive with ``param`` set to ``value`` and everything else valid.

    Args:
        primitive: Registered primitive name.
        param: The parameter carrying the vocabulary under test.
        value: The name to try.

    Returns:
        The exception raised, or None if the call returned.
    """
    params = _params(primitive)
    params[param] = value
    try:
        PRIMITIVES[primitive](_StubDataset(), **params)
    except Exception as exc:  # every failure mode is data for the caller
        return exc
    return None


def _valid_set_from_rejection(primitive: str, param: str) -> tuple[str, ...]:
    """Recover a primitive's accepted names from its own rejection message.

    Args:
        primitive: Registered primitive name.
        param: The parameter carrying the vocabulary.

    Returns:
        The names the message enumerated.

    Raises:
        AssertionError: If the primitive did not reject the bogus name, or rejected it
            without saying what it would have accepted.
    """
    exc = _call(primitive, param, BOGUS)
    assert exc is not None, f"{primitive}.{param} accepted {BOGUS!r}"
    message = str(exc)
    assert NAME_ERROR.search(message), f"{primitive}.{param} rejected {BOGUS!r} as: {message}"
    found = LIST_IN_MESSAGE.findall(message)
    assert found, f"{primitive}.{param} rejected {BOGUS!r} without listing what it takes"
    names: list[str] = []
    for literal in found:
        names.extend(ast.literal_eval(literal))
    return tuple(names)


class TestVocabularyMatchesValidator(unittest.TestCase):
    """Every offered name is accepted, and every accepted name is offered."""

    def test_offered_names_are_all_accepted(self) -> None:
        """Forward direction: the prompt never names something the code rejects."""
        offered = primitive_vocabularies()
        self.assertTrue(offered, "no vocabularies were offered at all")
        for (primitive, param), names in sorted(offered.items()):
            self.assertTrue(names, f"{primitive}.{param} offers an empty vocabulary")
            for name in names:
                with self.subTest(primitive=primitive, param=param, name=name):
                    exc = _call(primitive, param, name)
                    if exc is None:
                        continue
                    self.assertFalse(
                        isinstance(exc, ValueError) and NAME_ERROR.search(str(exc)),
                        f"the prompt offers {name!r} for {primitive}.{param}, but the "
                        f"primitive rejects it: {exc}",
                    )

    def test_accepted_names_are_all_offered(self) -> None:
        """Reverse direction: nothing the code takes is missing from the prompt."""
        offered = primitive_vocabularies()
        for (primitive, param), names in sorted(offered.items()):
            with self.subTest(primitive=primitive, param=param):
                accepted = _valid_set_from_rejection(primitive, param)
                missing = sorted(set(accepted) - set(names))
                self.assertEqual(
                    missing,
                    [],
                    f"{primitive}.{param} accepts {missing} but the prompt never offers them",
                )

    def test_every_constrained_parameter_has_a_vocabulary_in_the_prompt(self) -> None:
        """No parameter may enumerate a valid set to the code and none to the model.

        Which parameters those are is discovered rather than declared: every parameter
        of every primitive is handed a bogus name, and any that answers by listing what
        it would have accepted is a parameter the prompt owes the model a vocabulary
        for. A primitive that gains such a parameter fails here without anyone
        remembering to add it to a list.
        """
        offered = primitive_vocabularies()
        for primitive in sorted(PRIMITIVES):
            for param in _params(primitive):
                exc = _call(primitive, param, BOGUS)
                if exc is None or not NAME_ERROR.search(str(exc)):
                    continue
                if not LIST_IN_MESSAGE.search(str(exc)):
                    continue
                with self.subTest(primitive=primitive, param=param):
                    self.assertIn(
                        (primitive, param),
                        offered,
                        f"{primitive}.{param} tells the code exactly what it accepts "
                        f"({exc}) but the prompt offers the model nothing",
                    )

    def test_every_offered_name_appears_in_the_prompt_text(self) -> None:
        """The accessor is not allowed to promise more than the prompt actually says."""
        prompt = build_system_prompt()
        for (primitive, param), names in sorted(primitive_vocabularies().items()):
            for name in names:
                with self.subTest(primitive=primitive, param=param, name=name):
                    self.assertIn(
                        json.dumps(name),
                        prompt,
                        f"{name!r} is offered for {primitive}.{param} but never "
                        "written into the prompt",
                    )

    def test_the_prompt_names_the_primitive_each_vocabulary_serves(self) -> None:
        """A list of names with no owner is what caused the confusion in the first place."""
        prompt = build_system_prompt()
        for primitive, param in sorted(primitive_vocabularies()):
            with self.subTest(primitive=primitive, param=param):
                self.assertIn(
                    f"{primitive}.{param}",
                    prompt,
                    f"the prompt lists a vocabulary without saying it is for {primitive}.{param}",
                )


class TestNamedTrapsAreRealTraps(unittest.TestCase):
    """The prompt's *negative* claims are checked against the validator too.

    Listing what a parameter takes turned out not to be enough. In the live run of
    2026-09-09 eight of the ten claims that could not run passed ``correlation`` a name
    from the wrong register -- five the bare node name ``YAP_TAZ``, three the metric
    ``yap_taz_mean_combined`` -- while the prompt sat there with all nineteen valid GRN
    fields listed under that exact parameter. So the prompt now names the traps
    explicitly, and a named trap is a claim like any other: if the prompt says a name is
    rejected, the primitive had better reject it.
    """

    def test_every_name_called_invalid_really_is(self) -> None:
        """A trap the code would happily accept is worse than no trap at all."""
        examples = invalid_name_examples()
        self.assertTrue(examples, "the prompt warns against nothing")
        for (primitive, param), names in sorted(examples.items()):
            self.assertTrue(names, f"{primitive}.{param} has an empty warning list")
            for name in names:
                with self.subTest(primitive=primitive, param=param, name=name):
                    exc = _call(primitive, param, name)
                    self.assertIsNotNone(
                        exc, f"the prompt calls {name!r} invalid but {primitive} took it"
                    )
                    self.assertTrue(
                        NAME_ERROR.search(str(exc)),
                        f"the prompt calls {name!r} invalid for {primitive}.{param}, but "
                        f"the primitive failed for another reason: {exc}",
                    )

    def test_no_name_is_both_offered_and_called_invalid(self) -> None:
        """The prompt must not offer a name under a parameter it also warns against."""
        offered = primitive_vocabularies()
        for key, names in sorted(invalid_name_examples().items()):
            overlap = sorted(set(names) & set(offered.get(key, ())))
            with self.subTest(primitive=key[0], param=key[1]):
                self.assertEqual(
                    overlap, [], f"{key[0]}.{key[1]} both offers and warns off {overlap}"
                )

    def test_every_warned_name_appears_in_the_prompt(self) -> None:
        """A warning the accessor knows and the prompt never prints is not a warning."""
        prompt = build_system_prompt()
        for (primitive, param), names in sorted(invalid_name_examples().items()):
            for name in names:
                with self.subTest(primitive=primitive, param=param, name=name):
                    self.assertIn(json.dumps(name), prompt)


class TestMetricVocabularyIsNotStale(unittest.TestCase):
    """``build_system_prompt`` must offer every registered metric, not a stale subset."""

    def test_a_bare_process_offers_every_metric(self) -> None:
        """The GRN metrics register on import of a module the generator never imports."""
        import llm_insights.summary.sections  # noqa: F401 - registers the GRN metrics
        from llm_insights.summary import metrics as metrics_module

        expected = sorted(metrics_module.METRIC_NAMES)

        script = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(SRC)!r})\n"
            "from llm_insights.agent.generator import primitive_vocabularies\n"
            "print(json.dumps(sorted("
            "primitive_vocabularies()[('compare_metric', 'metric')])))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        offered = json.loads(completed.stdout.strip().splitlines()[-1])

        self.assertEqual(
            offered,
            expected,
            f"a bare process offers {len(offered)} metric(s); {len(expected)} are "
            "registered once every provider module is imported",
        )


if __name__ == "__main__":
    unittest.main()
