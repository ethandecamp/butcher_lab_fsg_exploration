"""Environment checks, and the verdict channel that keeps setup faults out of results.

Both halves of TASK-009's reporting fix are tested here, because they answer the same
incident from two ends. On 2026-09-02 a live run produced a card reading
``COULD NOT RUN: ModuleNotFoundError: No module named 'scipy'``: the run used an
interpreter that was not the virtualenv's, the import sat deep inside a primitive so it
failed only when a hypothesis reached it, and the failure was then displayed in the same
channel a genuinely unevaluable claim uses.

:mod:`llm_insights.preflight` stops that run before it starts. ``Outcome.error_kind``
makes sure that if one ever does slip through, it is labelled as the machine's fault
rather than presented as a verdict on the claim.
"""

from __future__ import annotations

import logging
import unittest

from llm_insights import preflight
from llm_insights.cards.card import cards_from_run
from llm_insights.cards.render import (
    COULD_NOT_RUN,
    ENVIRONMENT_BANNER,
    ENVIRONMENT_FAULT,
    counts,
    render_html,
    render_markdown,
    verdict_of,
)
from llm_insights.harness.primitives import PRIMITIVES
from llm_insights.harness.runner import ENVIRONMENT_PREFIX, ERROR_PREFIX, classify_error, run
from llm_insights.harness.spec import Hypothesis, TestSpec


def setUpModule() -> None:
    """Silence the warnings the error paths deliberately emit."""
    logging.getLogger("llm_insights").setLevel(logging.CRITICAL)


class TestPreflightChecks(unittest.TestCase):
    """The check itself."""

    def test_a_healthy_environment_reports_no_fatal_problem(self):
        """numpy and pandas import here, so nothing may be fatal."""
        self.assertFalse([p for p in preflight.check() if p.fatal])

    def test_a_missing_module_is_fatal_and_names_itself(self):
        """The whole point is to say which module, not just that something is wrong."""
        problems = preflight.check(required=("numpy", "a_module_that_does_not_exist"))
        fatal = [p for p in problems if p.fatal]
        self.assertEqual(len(fatal), 1)
        self.assertIn("a_module_that_does_not_exist", fatal[0].message)
        self.assertNotIn("numpy", fatal[0].message)

    def test_a_fatal_problem_carries_an_actionable_fix(self):
        """A diagnosis without a command is half a message."""
        (problem,) = [p for p in preflight.check(required=("nope_not_here",)) if p.fatal]
        self.assertIn("pip install", problem.fix)

    def test_the_message_names_the_interpreter(self):
        """The 2026-09-02 incident was entirely about which python was running."""
        (problem,) = [p for p in preflight.check(required=("nope_not_here",)) if p.fatal]
        self.assertIn("interpreter:", problem.message)

    def test_fatal_problems_sort_first(self):
        """Errors before warnings, so the thing that stops the run is read first."""
        problems = preflight.check(required=("nope_not_here",))
        self.assertTrue(problems[0].fatal)

    def test_format_is_empty_when_there_is_nothing_to_say(self):
        """A clean environment must print nothing at all."""
        self.assertEqual(preflight.format_problems([]), "")

    def test_format_shows_both_the_problem_and_the_fix(self):
        """Rendered output carries each message and each remedy."""
        text = preflight.format_problems(preflight.check(required=("nope_not_here",)))
        self.assertIn("ERROR", text)
        self.assertIn("nope_not_here", text)
        self.assertIn("fix:", text)

    def test_report_refuses_to_continue_on_a_fatal_problem(self):
        """False means 'do not spend a model call'."""
        problems = [preflight.Problem(fatal=True, message="broken", fix="do the thing")]
        self.assertFalse(preflight.report(problems))

    def test_report_allows_a_warning_through(self):
        """A warning informs; it must never stop a demo."""
        problems = [preflight.Problem(fatal=False, message="odd", fix="maybe")]
        self.assertTrue(preflight.report(problems))

    def test_report_allows_a_clean_environment(self):
        """Nothing to say, nothing to stop."""
        self.assertTrue(preflight.report([]))

    def test_in_virtualenv_answers_without_raising(self):
        """It is read on every startup, so it must be total."""
        self.assertIsInstance(preflight.in_virtualenv(), bool)


class TestErrorClassification(unittest.TestCase):
    """Which exceptions count as the machine's fault."""

    def test_a_missing_module_is_environmental(self):
        self.assertEqual(classify_error(ModuleNotFoundError("No module named 'scipy'")), "environment")

    def test_a_missing_file_is_environmental(self):
        self.assertEqual(classify_error(FileNotFoundError("no such file")), "environment")

    def test_a_bad_parameter_is_not(self):
        """A hypothesis the harness rejected is evidence about the hypothesis."""
        self.assertIsNone(classify_error(ValueError("unknown metric 'nope'")))

    def test_an_arbitrary_bug_is_not_excused_as_environmental(self):
        """Widening this would launder harness bugs as somebody's setup problem."""
        self.assertIsNone(classify_error(TypeError("boom")))


def _hypothesis(primitive: str) -> Hypothesis:
    """Build a minimal hypothesis pointing at ``primitive``."""
    return Hypothesis(
        id="H1",
        claim="a claim",
        rationale="a reason",
        test=TestSpec(primitive=primitive, params={}, decision_rule="passes if it runs"),
    )


class TestRunnerLabelsEnvironmentFaults(unittest.TestCase):
    """End to end through the real runner, with a primitive that misbehaves on purpose."""

    def setUp(self) -> None:
        """Register throwaway primitives, one per failure mode."""

        def missing_dependency(ds):
            raise ModuleNotFoundError("No module named 'scipy'")

        def bad_claim(ds):
            raise ValueError("there is no metric called 'invented_metric'")

        self._added = ("_t9_missing_dependency", "_t9_bad_claim")
        PRIMITIVES[self._added[0]] = missing_dependency
        PRIMITIVES[self._added[1]] = bad_claim

    def tearDown(self) -> None:
        """Leave the registry as it was found."""
        for name in self._added:
            PRIMITIVES.pop(name, None)

    def test_a_missing_dependency_is_labelled_an_environment_fault(self):
        """The exact 2026-09-02 failure, now named for what it is."""
        outcome = run(None, _hypothesis("_t9_missing_dependency"))
        self.assertEqual(outcome.error_kind, "environment")
        self.assertTrue(outcome.is_environment_fault)
        self.assertTrue(outcome.summary.startswith(f"{ENVIRONMENT_PREFIX}:"))
        self.assertFalse(outcome.passed)

    def test_a_hypothesis_level_failure_keeps_the_old_label(self):
        """A claim the harness could not evaluate is still COULD NOT RUN."""
        outcome = run(None, _hypothesis("_t9_bad_claim"))
        self.assertIsNone(outcome.error_kind)
        self.assertFalse(outcome.is_environment_fault)
        self.assertTrue(outcome.summary.startswith(f"{ERROR_PREFIX}:"))

    def test_an_unknown_primitive_is_not_an_environment_fault(self):
        """Naming a test that does not exist is the proposal's problem."""
        outcome = run(None, _hypothesis("_t9_bad_claim").__class__(
            id="H2",
            claim="c",
            rationale="r",
            test=TestSpec(primitive="_t9_bad_claim", params={"nope": 1}, decision_rule="d"),
        ))
        self.assertIsNone(outcome.error_kind)


class TestEnvironmentFaultsRender(unittest.TestCase):
    """How the report presents a fault that is nobody's hypothesis's fault."""

    def setUp(self) -> None:
        """One environment fault and one ordinary survivor."""

        def missing_dependency(ds):
            raise ModuleNotFoundError("No module named 'scipy'")

        def fine(ds):
            return True, {"r": 0.5}, "it ran"

        PRIMITIVES["_t9_missing"] = missing_dependency
        PRIMITIVES["_t9_fine"] = fine
        broken = _hypothesis("_t9_missing")
        ok = Hypothesis(
            id="H2",
            claim="a survivor",
            rationale="r",
            test=TestSpec(primitive="_t9_fine", params={}, decision_rule="d"),
        )
        self.cards = cards_from_run([broken, ok], [run(None, broken), run(None, ok)])
        self.clean = cards_from_run([ok], [run(None, ok)])

    def tearDown(self) -> None:
        """Leave the registry as it was found."""
        for name in ("_t9_missing", "_t9_fine"):
            PRIMITIVES.pop(name, None)

    def test_the_card_gets_its_own_verdict(self):
        """Not COULD NOT RUN: that channel is for claims, this one is for machines."""
        self.assertIs(verdict_of(self.cards[0]), ENVIRONMENT_FAULT)
        self.assertIsNot(verdict_of(self.cards[0]), COULD_NOT_RUN)

    def test_the_tally_counts_it_separately_and_still_adds_up(self):
        """Every card lands in exactly one row."""
        tally = counts(self.cards)
        self.assertEqual(tally[ENVIRONMENT_FAULT.label], 1)
        self.assertEqual(tally[COULD_NOT_RUN.label], 0)
        self.assertEqual(sum(tally.values()), len(self.cards))

    def test_markdown_warns_before_the_claims(self):
        """A skimmer must meet the warning before the verdicts."""
        text = render_markdown(self.cards, "Q", {})
        self.assertIn(ENVIRONMENT_BANNER, text)
        self.assertLess(text.index(ENVIRONMENT_BANNER), text.index("## Claims"))

    def test_html_warns_before_the_cards(self):
        """Same contract in the other renderer."""
        doc = render_html(self.cards, "Q", {})
        # The CSS rule ships unconditionally, so match the element, not the class name.
        self.assertIn('<p class="envbanner">', doc)
        # The renderer escapes the apostrophe in "machine's", so match a clause without
        # one rather than the constant verbatim.
        self.assertIn("carry no evidence either way", doc)
        self.assertLess(doc.index('<p class="envbanner">'), doc.index('<main class="blocks">'))

    def test_a_clean_run_carries_no_banner(self):
        """The warning must appear only when it is true."""
        self.assertNotIn(ENVIRONMENT_BANNER, render_markdown(self.clean, "Q", {}))
        self.assertNotIn('<p class="envbanner">', render_html(self.clean, "Q", {}))

    def test_the_banner_says_it_carries_no_evidence(self):
        """The whole point is that a reader draws nothing from these cards."""
        self.assertIn("carry no evidence either way", ENVIRONMENT_BANNER)
        self.assertIn("not because of anything in the data or the claims", ENVIRONMENT_BANNER)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
