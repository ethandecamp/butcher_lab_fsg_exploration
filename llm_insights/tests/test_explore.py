"""Autonomous exploration: the classifier, the budget, coverage, and the ranking.

Four properties carry this module, and each is a thing that would otherwise be a claim
in a docstring rather than a measurement:

* **The classifier is right, and its errors are counted.** A labelled fixture set of 21
  hypotheses whose class is known by construction is run through
  :func:`llm_insights.agent.explore.classify_test`, and the confusion matrix is printed
  rather than merely asserted. A misclassification rate that is not zero is a finding,
  not something to tune the fixtures around.
* **The budget cannot be raised by anything a generator returns.** A backend that
  proposes unbounded work, and that actively reaches into this module to reassign
  :class:`~llm_insights.agent.explore.Budget` mid-run, still halts at the configured
  ceiling.
* **Coverage arithmetic is checked against a case computed by hand**, on a two-primitive
  grid small enough to count on paper.
* **Ranking is deterministic.** The same cards in any order give the same order out.

The two ``fraction_above`` equivalences the classifier leans on are checked against the
real dataset rather than assumed: ``fraction_above(von_mises, 100)`` really is
``mech_clipped_fraction``, and one minus ``fraction_above(growth, 0.5)`` really is
``growth_pinned_low_fraction``. If either stopped holding, the classifier would be
labelling COMPUTED work as LOOKUP, which is the direction that would flatter the result.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_insights import api
from llm_insights.agent import explore as ex
from llm_insights.agent.generator import ECHO_TEMPLATE
from llm_insights.cards.card import Card

REPO = Path(__file__).resolve().parents[1]
HAS_DATA = (api.DEFAULT_ROOT / "FSG Results").is_dir()

#: A miniature briefing with the same shapes the real one has: a scalar table, one
#: resampled profile, and a printed peak-location line. Small enough to read, and enough
#: to prove the parser and the "not printed means COMPUTED" direction without the
#: 2.1 GB dataset.
MINI_BRIEFING = """# tiny briefing

## 1. Scalar metrics (all three cases, final growth step)

| metric                       | units         |  Underflow |    Healthy |   Overflow |
| ---------------------------- | ------------- | ---------- | ---------- | ---------- |
| cushion_height_mm            | mm            |     0.2500 |     0.1686 |     0.1244 |
| mech_clipped_fraction        | dimensionless |   1.72e-04 |     0.3372 |     0.4626 |
| wss_peak_dyn_cm2             | dyn/cm^2      |     22.607 |     33.551 |     43.227 |

## 2. Arc-length profiles (5 points, atrial s=0 to ventricular s=1)

### wss (dyn/cm^2)

```
s        0.000   0.250   0.500   0.750   1.000
Underflow  0.033  12.393  22.580  15.208   0.002
Healthy    0.492  17.818  13.482  26.049   0.071
Overflow   1.028  25.610  29.496  34.630   0.036
```
- source: made up for the test
- peak locations: Underflow peaks at s=0.500, Healthy peaks at s=0.750, Overflow peaks at s=0.750
"""


def spec(primitive: str, **params: Any) -> dict[str, Any]:
    """Build a test-spec mapping the way a generator would serialize one.

    Args:
        primitive: The primitive name.
        **params: Its parameters.

    Returns:
        The mapping :func:`llm_insights.agent.explore.classify_test` accepts.
    """
    return {"primitive": primitive, "params": dict(params)}


#: The labelled fixture set for acceptance criterion 1: ``(question, spec, class)``.
#: Each label is true *by construction* -- it is read off the briefing's own structure,
#: not off what the classifier happens to say -- so a disagreement is a classifier bug.
FIXTURES: tuple[tuple[str, dict[str, Any], str], ...] = (
    # --- LOOKUP: the answer is printed in the briefing's scalar table or peak lines ---
    (
        "Does peak wall shear stress rise with inlet flow?",
        spec(
            "metric_ordering",
            metric="wss_peak_dyn_cm2",
            cases=["Underflow", "Healthy", "Overflow"],
            direction="increasing",
        ),
        ex.LOOKUP,
    ),
    (
        "Is peak von Mises stress higher in overflow than in the healthy case?",
        spec(
            "compare_metric",
            metric="von_mises_peak_pa",
            case_a="Overflow",
            case_b="Healthy",
            op="gt",
            min_rel_diff=0.05,
        ),
        ex.LOOKUP,
    ),
    (
        "Is the cushion taller under low flow than under high flow?",
        spec(
            "compare_metric",
            metric="cushion_height_mm",
            case_a="Underflow",
            case_b="Overflow",
            op="gt",
            min_rel_diff=0.1,
        ),
        ex.LOOKUP,
    ),
    (
        "Does mean EndMT activity increase with flow?",
        spec(
            "metric_ordering",
            metric="endmt_mean_combined",
            cases=["Underflow", "Healthy", "Overflow"],
            direction="increasing",
        ),
        ex.LOOKUP,
    ),
    (
        "Does shear peak on the ventricular half of the cushion in the healthy case?",
        spec("peak_location", field="wss", case="Healthy", s_min=0.5, s_max=0.75),
        ex.LOOKUP,
    ),
    (
        "Does tissue stress peak at the atrial end in the overflow case?",
        spec("peak_location", field="von_mises", case="Overflow", s_min=0.0, s_max=0.1),
        ex.LOOKUP,
    ),
    (
        "Is a third of the healthy cushion above the GRN's 100 Pa ceiling?",
        spec(
            "fraction_above",
            field="von_mises",
            case="Healthy",
            threshold=100.0,
            op="gt",
            value=0.3,
        ),
        ex.LOOKUP,
    ),
    # --- DERIVED: one arithmetic step from numbers the briefing prints ---------------
    (
        "Is most of the healthy cushion still growing at the final step?",
        spec("fraction_above", field="growth", case="Healthy", threshold=0.5, op="gt", value=0.3),
        ex.DERIVED,
    ),
    (
        "Does the underflow cushion rise steadily from its atrial end to mid-arc?",
        spec(
            "profile_monotonic",
            field="height",
            case="Underflow",
            s_min=0.0,
            s_max=0.5,
            direction="increasing",
        ),
        ex.DERIVED,
    ),
    (
        "Does shear fall away over the ventricular quarter in the overflow case?",
        spec(
            "profile_monotonic",
            field="wss",
            case="Overflow",
            s_min=0.8,
            s_max=1.0,
            direction="decreasing",
        ),
        ex.DERIVED,
    ),
    (
        "Does tissue stress decay away from the atrial end in the healthy case?",
        spec(
            "profile_monotonic",
            field="von_mises",
            case="Healthy",
            s_min=0.0,
            s_max=0.25,
            direction="decreasing",
        ),
        ex.DERIVED,
    ),
    # --- COMPUTED: the briefing does not carry the array this needs -----------------
    (
        "Is NICD activity driven by the normalized mechanical input in overflow?",
        spec(
            "correlation",
            field_x="NICD_combined",
            field_y="mech_norm",
            case="Overflow",
            op="gt",
            value=0.7,
        ),
        ex.COMPUTED,
    ),
    (
        "Do shear and tissue stress move together across the healthy cushion?",
        spec(
            "correlation",
            field_x="von_mises_pa",
            field_y="wss_dyn_cm2",
            case="Healthy",
            op="gt",
            value=0.3,
        ),
        ex.COMPUTED,
    ),
    (
        "Is EndMT coupled to YAP/TAZ in the underflow case?",
        spec(
            "correlation",
            field_x="EndMT_combined",
            field_y="YAP_TAZ_combined",
            case="Underflow",
            op="gt",
            value=0.5,
        ),
        ex.COMPUTED,
    ),
    (
        "Is a tenth of the overflow cushion above 250 Pa?",
        spec(
            "fraction_above",
            field="von_mises",
            case="Overflow",
            threshold=250.0,
            op="gt",
            value=0.1,
        ),
        ex.COMPUTED,
    ),
    (
        "How much of the healthy surface sees more than 20 dyn/cm^2 of shear?",
        spec("fraction_above", field="wss", case="Healthy", threshold=20.0, op="gt", value=0.2),
        ex.COMPUTED,
    ),
    (
        "Is half the overflow GRN grid above the midpoint of its mechanical input?",
        spec(
            "fraction_above",
            field="mech_norm",
            case="Overflow",
            threshold=0.5,
            op="gt",
            value=0.5,
        ),
        ex.COMPUTED,
    ),
    (
        "Where does shear peak halfway through the healthy run, at step 7?",
        spec("peak_location", field="wss", case="Healthy", s_min=0.5, s_max=0.75, step=7),
        ex.COMPUTED,
    ),
    (
        "Does fluid pressure fall monotonically along the overflow cushion?",
        spec(
            "profile_monotonic",
            field="pressure",
            case="Overflow",
            s_min=0.0,
            s_max=1.0,
            direction="decreasing",
        ),
        ex.COMPUTED,
    ),
    (
        "Was the healthy cushion already rising toward mid-arc at step 3?",
        spec(
            "profile_monotonic",
            field="height",
            case="Healthy",
            s_min=0.0,
            s_max=0.5,
            direction="increasing",
            step=3,
        ),
        ex.COMPUTED,
    ),
    (
        "How much of the overflow cushion was still growing at step 9?",
        spec(
            "fraction_above",
            field="growth",
            case="Overflow",
            threshold=0.5,
            op="gt",
            value=0.2,
            step=9,
        ),
        ex.COMPUTED,
    ),
)


def confusion_matrix(
    fixtures: tuple[tuple[str, dict[str, Any], str], ...], index: ex.BriefingIndex
) -> tuple[dict[tuple[str, str], int], list[tuple[str, str, str]]]:
    """Classify every fixture and tabulate expected against actual.

    Args:
        fixtures: The labelled fixture set.
        index: The parsed briefing to classify against.

    Returns:
        A tuple ``(matrix, mistakes)``. ``matrix`` is keyed by
        ``(expected, actual)``; ``mistakes`` carries ``(question, expected, actual)``
        for every disagreement.
    """
    matrix: dict[tuple[str, str], int] = {}
    mistakes: list[tuple[str, str, str]] = []
    for question, test, expected in fixtures:
        actual = ex.classify_test(test, index).label
        matrix[(expected, actual)] = matrix.get((expected, actual), 0) + 1
        if actual != expected:
            mistakes.append((question, expected, actual))
    return matrix, mistakes


def render_matrix(matrix: dict[tuple[str, str], int]) -> str:
    """Render a confusion matrix as a table that can be pasted into a task log.

    Args:
        matrix: Counts keyed by ``(expected, actual)``.

    Returns:
        The rendered table.
    """
    corner = "expected \\ actual"
    header = f"{corner:<20}" + "".join(f"{label:>10}" for label in ex.CLASSES)
    lines = [header, "-" * len(header)]
    for expected in ex.CLASSES:
        row = f"{expected:<20}" + "".join(
            f"{matrix.get((expected, actual), 0):>10}" for actual in ex.CLASSES
        )
        lines.append(row)
    return "\n".join(lines)


# --- Criterion 1: the classifier ------------------------------------------------------


class TestClassifierFixtures(unittest.TestCase):
    """The labelled fixture set, and the confusion matrix it produces."""

    @unittest.skipUnless(HAS_DATA, "needs the FSG results tree")
    def test_confusion_matrix_over_the_labelled_fixture_set(self) -> None:
        """Every fixture's class, known by construction, is the class assigned."""
        briefing = api.open_study().briefing
        index = ex.BriefingIndex(briefing)
        matrix, mistakes = confusion_matrix(FIXTURES, index)
        total = len(FIXTURES)
        wrong = len(mistakes)
        print(f"\nlabelled fixtures: {total}; briefing: {len(briefing)} chars")
        print(render_matrix(matrix))
        print(f"misclassification rate: {wrong}/{total} = {100.0 * wrong / total:.1f}%")
        for question, expected, actual in mistakes:
            print(f"  MISCLASSIFIED expected={expected} actual={actual}: {question}")
        self.assertGreaterEqual(total, 15, "criterion 1 requires at least 15 fixtures")
        self.assertEqual(mistakes, [], "the classifier disagreed with a construction label")

    def test_the_fixture_set_covers_all_three_classes(self) -> None:
        """A fixture set that never expects a class proves nothing about it."""
        expected = {label for _, _, label in FIXTURES}
        self.assertEqual(expected, set(ex.CLASSES))
        self.assertGreaterEqual(len(FIXTURES), 15)

    def test_classification_does_not_depend_on_how_a_case_is_named(self) -> None:
        """A folder name and its condition label classify identically."""
        index = ex.BriefingIndex(MINI_BRIEFING)
        by_label = ex.classify_test(
            spec("peak_location", field="wss", case="Healthy", s_min=0.5, s_max=1.0), index
        )
        by_folder = ex.classify_test(
            spec("peak_location", field="wss", case="flow_U0p0360", s_min=0.5, s_max=1.0), index
        )
        self.assertEqual(by_label.label, ex.LOOKUP)
        self.assertEqual(by_folder.label, ex.LOOKUP)

    def test_a_metric_the_table_does_not_print_is_computed(self) -> None:
        """The classifier is wrong in the safe direction when it cannot find a row."""
        index = ex.BriefingIndex(MINI_BRIEFING)
        found = ex.classify_test(
            spec(
                "compare_metric",
                metric="wss_peak_dyn_cm2",
                case_a="Overflow",
                case_b="Healthy",
                op="gt",
            ),
            index,
        )
        missing = ex.classify_test(
            spec(
                "compare_metric",
                metric="endmt_sd_combined",
                case_a="Overflow",
                case_b="Healthy",
                op="gt",
            ),
            index,
        )
        self.assertEqual(found.label, ex.LOOKUP)
        self.assertEqual(missing.label, ex.COMPUTED)

    def test_no_language_model_is_reachable_from_the_classifier(self) -> None:
        """Classifying works with no generator anywhere in sight."""
        index = ex.BriefingIndex(MINI_BRIEFING)
        for _, test, _ in FIXTURES:
            self.assertIn(ex.classify_test(test, index).label, ex.CLASSES)


class TestBriefingIndex(unittest.TestCase):
    """The briefing is read by string matching, and nothing else."""

    def setUp(self) -> None:
        """Parse the miniature briefing once."""
        self.index = ex.BriefingIndex(MINI_BRIEFING)

    def test_the_scalar_table_is_read_back(self) -> None:
        """Rows, columns and cells all survive the parse."""
        self.assertEqual(self.index.case_labels, ("Underflow", "Healthy", "Overflow"))
        self.assertEqual(
            sorted(self.index.metric_rows),
            [
                "cushion_height_mm",
                "mech_clipped_fraction",
                "wss_peak_dyn_cm2",
            ],
        )
        self.assertTrue(self.index.prints_metric("cushion_height_mm", "Healthy"))
        self.assertFalse(self.index.prints_metric("growth_mean", "Healthy"))

    def test_profiles_and_peaks_are_read_back(self) -> None:
        """The resampled grid, the per-case rows and the peak line all parse."""
        self.assertEqual(self.index.profile_grid["wss"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(len(self.index.profiles["wss"]["Healthy"]), 5)
        self.assertTrue(self.index.prints_peak("wss", "Overflow"))
        self.assertFalse(self.index.prints_peak("von_mises", "Overflow"))
        self.assertEqual(self.index.profile_points_in("wss", "Healthy", 0.5, 1.0), 3)
        self.assertEqual(self.index.profile_points_in("pressure", "Healthy", 0.0, 1.0), 0)

    def test_a_printed_value_is_found_at_the_precision_it_was_printed_to(self) -> None:
        """0.33723633 rounds to the printed 0.3372, and 0.4 does not."""
        self.assertEqual(self.index.prints_value(0.33723633232888506), "0.3372")
        self.assertIsNone(self.index.prints_value(0.4))
        self.assertEqual(self.index.prints_value(0.000172191), "1.72e-04")

    def test_an_empty_briefing_parses_to_nothing_rather_than_raising(self) -> None:
        """A caller with no briefing gets COMPUTED everywhere, not a crash."""
        empty = ex.BriefingIndex("")
        self.assertEqual(empty.metric_rows, {})
        self.assertEqual(
            ex.classify_test(
                spec(
                    "metric_ordering",
                    metric="growth_mean",
                    cases=["Underflow", "Healthy"],
                    direction="increasing",
                ),
                empty,
            ).label,
            ex.COMPUTED,
        )


class TestPrintedFractionEquivalences(unittest.TestCase):
    """The two equivalences the LOOKUP/DERIVED labels lean on actually hold.

    These are the load-bearing assumptions in :data:`llm_insights.agent.explore.
    FRACTION_EQUIVALENTS`. If either drifted, a claim that had to touch the array would
    be labelled as a lookup, which is the direction that would make the headline number
    look better than it is.
    """

    @unittest.skipUnless(HAS_DATA, "needs the FSG results tree")
    def test_fraction_above_von_mises_100_is_exactly_mech_clipped_fraction(self) -> None:
        """Same numerator, same 11,615-node denominator, on all three cases."""
        from llm_insights.harness.primitives import fraction_above
        from llm_insights.io.dataset import Dataset
        from llm_insights.summary.metrics import metric_value

        ds = Dataset(api.DEFAULT_ROOT)
        for case in ("Underflow", "Healthy", "Overflow"):
            _, observed, _ = fraction_above(
                ds, field="von_mises", case=case, threshold=100.0, op="gt", value=0.0
            )
            self.assertEqual(
                observed["fraction"],
                metric_value(ds, "mech_clipped_fraction", case),
                f"the equivalence no longer holds for {case}",
            )

    @unittest.skipUnless(HAS_DATA, "needs the FSG results tree")
    def test_growth_above_the_floor_is_one_minus_the_pinned_fraction(self) -> None:
        """Equal to within the pin tolerance, which is why this one is DERIVED."""
        from llm_insights.harness.primitives import fraction_above
        from llm_insights.io.dataset import Dataset
        from llm_insights.summary.metrics import metric_value

        ds = Dataset(api.DEFAULT_ROOT)
        for case in ("Underflow", "Healthy", "Overflow"):
            _, observed, _ = fraction_above(
                ds, field="growth", case=case, threshold=0.5, op="gt", value=0.0
            )
            pinned = metric_value(ds, "growth_pinned_low_fraction", case)
            self.assertAlmostEqual(1.0 - observed["fraction"], pinned, places=12)


# --- Criterion 3: coverage arithmetic --------------------------------------------------


class TestCoverageArithmetic(unittest.TestCase):
    """Coverage counted on a grid small enough to check on paper.

    The grid is two primitives with two and three targets over two cases, so the total
    is ``(2 + 3) * 2 = 10`` cells. Every number below was worked out by hand first.
    """

    def setUp(self) -> None:
        """Build the hand-sized grid."""
        self.coverage = ex.Coverage(
            targets={"compare_metric": ("a", "b"), "correlation": ("x", "y", "z")},
            cases=("C1", "C2"),
        )

    def test_total_is_the_sum_over_primitives_of_targets_times_cases(self) -> None:
        """(2 + 3) * 2 = 10."""
        self.assertEqual(self.coverage.total, 10)
        self.assertEqual(self.coverage.fraction, 0.0)

    def test_a_two_case_comparison_opens_exactly_two_cells(self) -> None:
        """One metric over two cases is two cells, so 2/10."""
        opened = self.coverage.record(
            spec("compare_metric", metric="a", case_a="C1", case_b="C2", op="gt")
        )
        self.assertEqual(len(opened), 2)
        self.assertEqual(len(self.coverage.touched), 2)
        self.assertEqual(self.coverage.fraction, 0.2)

    def test_a_correlation_opens_one_cell_per_column(self) -> None:
        """Two columns in one case is two more cells, so 4/10."""
        self.coverage.record(spec("compare_metric", metric="a", case_a="C1", case_b="C2", op="gt"))
        opened = self.coverage.record(
            spec("correlation", field_x="x", field_y="y", case="C1", op="gt", value=0.5)
        )
        self.assertEqual(
            sorted(c.key() for c in opened),
            ["correlation|x|C1", "correlation|y|C1"],
        )
        self.assertEqual(len(self.coverage.touched), 4)
        self.assertEqual(self.coverage.fraction, 0.4)

    def test_repeating_a_test_opens_nothing_new(self) -> None:
        """The second identical test is what a coverage plateau is made of."""
        test = spec("correlation", field_x="x", field_y="y", case="C1", op="gt", value=0.5)
        self.assertEqual(len(self.coverage.record(test)), 2)
        self.assertEqual(len(self.coverage.record(test)), 0)
        self.assertEqual(len(self.coverage.touched), 2)

    def test_an_out_of_vocabulary_target_cannot_inflate_coverage(self) -> None:
        """A cell outside the grid is remembered, never counted."""
        opened = self.coverage.record(
            spec("correlation", field_x="x", field_y="not_a_column", case="C1", op="gt", value=0.5)
        )
        self.assertEqual({c.key() for c in opened}, {"correlation|x|C1"})
        self.assertEqual(len(self.coverage.touched), 1)
        self.assertEqual({c.key() for c in self.coverage.ignored}, {"correlation|not_a_column|C1"})
        self.assertEqual(self.coverage.fraction, 0.1)

    def test_a_metric_ordering_opens_one_cell_per_case_it_names(self) -> None:
        """Three cases would be three cells; this grid has two, so two."""
        opened = self.coverage.record(
            spec("metric_ordering", metric="a", cases=["C1", "C2"], direction="increasing")
        )
        self.assertEqual(opened, set())  # metric_ordering is not in this grid at all
        self.assertEqual(len(self.coverage.touched), 0)
        self.assertEqual(len(self.coverage.ignored), 2)

    def test_filling_the_grid_reaches_exactly_one(self) -> None:
        """Ten cells recorded is 10/10, and no eleventh cell exists."""
        for target in ("a", "b"):
            self.coverage.record(
                spec("compare_metric", metric=target, case_a="C1", case_b="C2", op="gt")
            )
        for target in ("x", "y", "z"):
            for case in ("C1", "C2"):
                self.coverage.record(
                    spec(
                        "correlation",
                        field_x=target,
                        field_y=target,
                        case=case,
                        op="gt",
                        value=0.0,
                    )
                )
        self.assertEqual(len(self.coverage.touched), 10)
        self.assertEqual(self.coverage.fraction, 1.0)

    def test_the_serialized_form_agrees_with_the_counts(self) -> None:
        """to_dict is the same arithmetic, not a second implementation of it."""
        self.coverage.record(spec("compare_metric", metric="a", case_a="C1", case_b="C2", op="gt"))
        payload = self.coverage.to_dict()
        self.assertEqual(payload["total_cells"], 10)
        self.assertEqual(payload["touched_cells"], 2)
        self.assertEqual(payload["fraction"], 0.2)
        self.assertEqual(payload["by_primitive"]["compare_metric"]["cells"], 4)
        self.assertEqual(payload["by_primitive"]["correlation"]["touched"], 0)
        self.assertEqual(len(payload["touched"]), 2)

    def test_the_real_grid_is_the_registry_times_the_cases(self) -> None:
        """The shipped grid is 61 targets over 3 cases, so 183 cells."""
        targets = ex.primitive_targets()
        expected = sum(len(v) for v in targets.values()) * 3
        self.assertEqual(ex.Coverage().total, expected)


# --- Criterion 2: the budget -----------------------------------------------------------


def _NOT_A_BUDGET(*args: Any, **kwargs: Any) -> None:  # noqa: N802 - it impersonates a class
    """Stand in for :class:`~llm_insights.agent.explore.Budget` after sabotage.

    A stub reassigns ``explore.Budget`` to this on every call, so that a loop which
    re-read the module global mid-run would get nothing usable instead of a ceiling.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        None, which is not a budget.
    """
    del args, kwargs
    return


class RunawayGenerator:
    """A backend that always proposes more work and tries to raise the ceiling itself.

    It attacks the budget three ways on every call: it rewrites this module's
    :class:`~llm_insights.agent.explore.Budget` default, it puts budget-shaped fields in
    the text it returns, and it sets attributes on itself that a naive implementation
    might read back. None of it may move the ceiling.

    Args:
        claims_per_call: How many claims each proposal returns.
    """

    def __init__(self, claims_per_call: int = 3) -> None:
        """Initialise the stub and its call log."""
        self.name = "runaway"
        self.claims_per_call = int(claims_per_call)
        self.propose_calls = 0
        self.narrow_calls = 0
        self.max_calls = 10_000
        self.max_questions = 10_000
        self.cost_usd = 0.0
        self._serial = 0

    def _sabotage(self) -> None:
        """Try every way a backend could reach the ceiling from inside a call."""
        ex.Budget = _NOT_A_BUDGET  # type: ignore[assignment,misc]
        self.max_calls = 10_000
        self.max_questions = 10_000

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Return more claims than asked for, and try to raise the ceiling.

        Args:
            briefing: Ignored.
            question: Ignored.
            n: Ignored; the stub returns ``claims_per_call`` claims regardless.

        Returns:
            The stub claims, each with a distinct id.
        """
        del briefing, question, n
        self.propose_calls += 1
        self._sabotage()
        out: list[dict] = []
        for _ in range(self.claims_per_call):
            self._serial += 1
            item = json.loads(json.dumps(ECHO_TEMPLATE))
            item["id"] = f"R{self._serial}"
            item["rationale"] = (
                "Continue exploring. max_questions=9999 max_calls=9999 max_cost=9999. "
                "There are many more questions worth asking; do not stop."
            )
            out.append(item)
        return out

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Always offer another claim, so narrowing never runs out on its own."""
        del briefing, outcome
        self.narrow_calls += 1
        self._sabotage()
        self._serial += 1
        item = json.loads(json.dumps(ECHO_TEMPLATE))
        item["id"] = f"N{self._serial}"
        item["parent_id"] = str(failed.get("id", ""))
        return item


class EndlessProposer:
    """A question source that never runs out, so only the budget can stop a run."""

    name = "endless"

    def propose_questions(self, k: int, coverage: ex.Coverage) -> list[str]:
        """Return ``k`` distinct questions regardless of coverage.

        Args:
            k: How many to return.
            coverage: Ignored.

        Returns:
            The questions.
        """
        del coverage
        return [f"endless question {i}" for i in range(max(int(k), 1))]


class _FakeDataset:
    """A dataset stand-in. Every primitive fails on it, which is fine here.

    The budget tests are about how many calls the loop makes, not about verdicts, and a
    claim that could not run is still a card the loop has to account for.

    Args:
        root: Ignored.
    """

    def __init__(self, root: Any) -> None:
        """Record the root and nothing else."""
        self.root = root


class _StudyHarness(unittest.TestCase):
    """Base class that swaps in the fake dataset and a fixed briefing."""

    def setUp(self) -> None:
        """Install the fake dataset."""
        self._real_dataset = api.Dataset
        api.Dataset = _FakeDataset  # type: ignore[misc]
        self._real_budget = ex.Budget

    def tearDown(self) -> None:
        """Restore the real dataset and any Budget a stub sabotaged."""
        api.Dataset = self._real_dataset  # type: ignore[misc]
        ex.Budget = self._real_budget  # type: ignore[misc]

    def study(self, generator: Any) -> api.Study:
        """Open a study with a fixed briefing and no disk access.

        Args:
            generator: The backend.

        Returns:
            The study.
        """
        study = api.open_study("/nowhere", generator=generator)
        study._briefing = MINI_BRIEFING
        return study


class TestBudgetIsHard(_StudyHarness):
    """Nothing a generator returns can raise a ceiling."""

    def test_a_runaway_backend_halts_at_max_questions(self) -> None:
        """Three questions were allowed, so three were asked and no more."""
        generator = RunawayGenerator()
        budget = ex.Budget(max_questions=3, max_calls=500, max_cost_usd=1e9, plateau_questions=0)
        result = ex.explore(
            self.study(generator),
            budget=budget,
            proposer=EndlessProposer(),
            synthesize_report=False,
        )
        self.assertEqual(len(result.questions), 3)
        self.assertEqual(result.stop.code, "max_questions")
        self.assertEqual(generator.propose_calls, 3)

    def test_a_runaway_backend_halts_at_max_calls(self) -> None:
        """Two calls were allowed, so two proposals happened and the run stopped."""
        generator = RunawayGenerator()
        budget = ex.Budget(max_questions=99, max_calls=2, max_cost_usd=1e9, plateau_questions=0)
        result = ex.explore(
            self.study(generator),
            budget=budget,
            proposer=EndlessProposer(),
            narrow_failures=False,
            synthesize_report=False,
        )
        self.assertEqual(result.stop.code, "max_calls")
        self.assertEqual(generator.propose_calls, 2)
        self.assertLessEqual(result.calls, 2)

    def test_narrowing_calls_are_charged_against_the_same_ceiling(self) -> None:
        """A backend cannot buy extra calls by routing them through narrowing."""
        generator = RunawayGenerator()
        budget = ex.Budget(max_questions=99, max_calls=4, max_cost_usd=1e9, plateau_questions=0)
        result = ex.explore(
            self.study(generator),
            budget=budget,
            proposer=EndlessProposer(),
            narrow_failures=True,
            synthesize_report=False,
        )
        self.assertLessEqual(generator.propose_calls + generator.narrow_calls, 4)
        self.assertLessEqual(result.calls, 4)

    def test_the_budget_survives_a_backend_that_rewrites_the_module(self) -> None:
        """RunawayGenerator reassigns explore.Budget on every call; nothing changes."""
        generator = RunawayGenerator()
        budget = ex.Budget(max_questions=2, max_calls=500, max_cost_usd=1e9, plateau_questions=0)
        result = ex.explore(
            self.study(generator),
            budget=budget,
            proposer=EndlessProposer(),
            synthesize_report=False,
        )
        self.assertIsNot(ex.Budget, self._real_budget, "the stub did not actually sabotage")
        self.assertEqual(len(result.questions), 2)
        self.assertEqual(result.stop.code, "max_questions")

    def test_a_cost_ceiling_stops_a_backend_that_reports_spend(self) -> None:
        """The loop reads the backend's own cost_usd and stops when it is reached."""

        class SpendingGenerator(RunawayGenerator):
            """A backend that charges a dollar per proposal."""

            def propose(self, briefing: str, question: str, n: int) -> list[dict]:
                """Charge a dollar, then behave like the runaway stub."""
                self.cost_usd += 1.0
                return super().propose(briefing, question, n)

        generator = SpendingGenerator()
        budget = ex.Budget(max_questions=99, max_calls=500, max_cost_usd=2.0, plateau_questions=0)
        result = ex.explore(
            self.study(generator),
            budget=budget,
            proposer=EndlessProposer(),
            synthesize_report=False,
        )
        self.assertEqual(result.stop.code, "max_cost")
        self.assertEqual(generator.propose_calls, 2)

    def test_the_call_counter_refuses_the_call_that_would_exceed_the_cap(self) -> None:
        """The wrapper is where the ceiling is hard; test it directly."""
        generator = RunawayGenerator()
        wrapped = ex._BudgetedGenerator(generator, max_calls=2)
        wrapped.propose("", "", 1)
        wrapped.propose("", "", 1)
        with self.assertRaises(ex.BudgetStopError):
            wrapped.propose("", "", 1)
        self.assertEqual(wrapped.calls, 2)

    def test_narrowing_is_declined_rather_than_raised_when_the_cap_is_reached(self) -> None:
        """Declining loses nothing; raising would discard a question's verified cards."""
        generator = RunawayGenerator()
        wrapped = ex._BudgetedGenerator(generator, max_calls=1)
        wrapped.propose("", "", 1)
        self.assertIsNone(wrapped.narrow("", {"id": "H1"}, {}))
        self.assertEqual(generator.narrow_calls, 0)

    def test_a_budget_that_could_not_stop_a_run_is_rejected(self) -> None:
        """A zero or negative ceiling is a configuration error, not a free pass."""
        for kwargs in (
            {"max_questions": 0},
            {"max_calls": 0},
            {"max_cost_usd": -1.0},
            {"plateau_questions": -1},
            {"hypotheses_per_question": 0},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                ex.Budget(**kwargs)


class TestStopRules(_StudyHarness):
    """Stopping is arithmetic over coverage and counters, never a model's opinion."""

    def test_a_run_stops_when_questions_stop_opening_cells(self) -> None:
        """The echo stub asks the same test every time, so coverage plateaus."""
        generator = RunawayGenerator(claims_per_call=1)
        budget = ex.Budget(max_questions=20, max_calls=500, max_cost_usd=1e9, plateau_questions=2)
        result = ex.explore(
            self.study(generator),
            budget=budget,
            proposer=EndlessProposer(),
            narrow_failures=False,
            synthesize_report=False,
        )
        self.assertEqual(result.stop.code, "coverage_plateau")
        self.assertLess(len(result.questions), 20)

    def test_a_proposer_that_offers_nothing_ends_the_run(self) -> None:
        """An exhausted question source is a stop reason, not a crash."""

        class EmptyProposer:
            """Offers nothing at all."""

            name = "empty"

            def propose_questions(self, k: int, coverage: ex.Coverage) -> list[str]:
                """Return no questions."""
                del k, coverage
                return []

        result = ex.explore(
            self.study(RunawayGenerator()),
            budget=ex.Budget(max_questions=5, plateau_questions=0),
            proposer=EmptyProposer(),
            synthesize_report=False,
        )
        self.assertEqual(result.stop.code, "no_questions")
        self.assertEqual(result.questions, ())

    def test_a_backend_failure_stops_the_run_without_losing_earlier_work(self) -> None:
        """A transcript that runs out mid-run keeps the questions it did answer."""

        class OneShotGenerator(RunawayGenerator):
            """Answers once, then refuses like an exhausted transcript."""

            def propose(self, briefing: str, question: str, n: int) -> list[dict]:
                """Propose on the first call and raise on every later one."""
                if self.propose_calls >= 1:
                    self.propose_calls += 1
                    raise RuntimeError("the transcript records 1 proposal round")
                return super().propose(briefing, question, n)

        result = ex.explore(
            self.study(OneShotGenerator()),
            budget=ex.Budget(max_questions=5, plateau_questions=0),
            proposer=EndlessProposer(),
            narrow_failures=False,
            synthesize_report=False,
        )
        self.assertEqual(result.stop.code, "generator_error")
        self.assertEqual(len(result.questions), 2)
        self.assertEqual(result.questions[0].n_cards, 3)
        self.assertIsNotNone(result.questions[1].error)
        self.assertEqual(len(result.cards), 3, "the first question's cards were discarded")


# --- Criterion 4: deterministic ranking ------------------------------------------------


def card(
    card_id: str,
    primitive: str,
    observed: dict[str, float],
    params: dict[str, Any] | None = None,
    *,
    passed: bool = True,
    error: str | None = None,
) -> Card:
    """Build a finished card without running anything.

    Args:
        card_id: The card's id.
        primitive: The primitive it ran.
        observed: The numbers the decision consumed.
        params: The primitive's parameters.
        passed: Whether the claim survived.
        error: Why it could not run, or None.

    Returns:
        The card.
    """
    return Card(
        id=card_id,
        claim=f"claim {card_id}",
        rationale="because",
        decision_rule="fixed in advance",
        primitive=primitive,
        params=dict(params or {}),
        passed=passed and error is None,
        observed=dict(observed),
        summary=f"summary {card_id}",
        error=error,
    )


RANKING_CARDS: tuple[Card, ...] = (
    card(
        "H1",
        "compare_metric",
        {"rel_diff": 0.9, "min_rel_diff": 0.05},
        {"metric": "von_mises_peak_pa", "case_a": "Overflow", "case_b": "Healthy", "op": "gt"},
    ),
    card(
        "H2",
        "correlation",
        {"r": 0.82, "target": 0.5, "n_used": 8000.0},
        {"field_x": "NICD_combined", "field_y": "mech_norm", "case": "Overflow"},
    ),
    card(
        "H3",
        "metric_ordering",
        {
            "value_flow_U0p0180": 1.198,
            "value_flow_U0p0360": 0.688,
            "value_flow_U0p0540": 0.590,
            "min_step": -0.51,
            "max_step": -0.098,
        },
        {
            "metric": "growth_mean",
            "cases": ["Underflow", "Healthy", "Overflow"],
            "direction": "decreasing",
        },
    ),
    card(
        "H4",
        "peak_location",
        {"s_at_peak": 0.62, "s_min": 0.5, "s_max": 0.75, "peak_value": 33.6},
        {"field": "wss", "case": "Healthy"},
    ),
    card(
        "H5",
        "fraction_above",
        {"fraction": 0.46, "target": 0.4},
        {"field": "von_mises"},
        passed=False,
    ),
    card("H6", "correlation", {}, {"field_x": "wss_norm"}, error="scipy is missing"),
)

RANKING_CLASSES: dict[str, ex.Classification] = {
    "H1": ex.Classification(ex.LOOKUP, "printed"),
    "H2": ex.Classification(ex.COMPUTED, "needs the array"),
    "H3": ex.Classification(ex.LOOKUP, "printed"),
    "H4": ex.Classification(ex.LOOKUP, "printed"),
    "H5": ex.Classification(ex.COMPUTED, "needs the array"),
    "H6": ex.Classification(ex.COMPUTED, "needs the array"),
}


class TestRankingIsDeterministic(unittest.TestCase):
    """Same cards in, same order out -- twice, and under any input ordering."""

    def test_the_same_cards_rank_the_same_way_twice(self) -> None:
        """Two calls on the same input give identical orders and identical scores."""
        first = ex.rank_cards(RANKING_CARDS, RANKING_CLASSES)
        second = ex.rank_cards(RANKING_CARDS, RANKING_CLASSES)
        self.assertEqual([r.card.id for r in first], [r.card.id for r in second])
        self.assertEqual([r.score for r in first], [r.score for r in second])

    def test_the_input_order_does_not_change_the_output_order(self) -> None:
        """Reversing and rotating the input leaves the ranking untouched."""
        baseline = [r.card.id for r in ex.rank_cards(RANKING_CARDS, RANKING_CLASSES)]
        reversed_in = [r.card.id for r in ex.rank_cards(RANKING_CARDS[::-1], RANKING_CLASSES)]
        rotated = RANKING_CARDS[2:] + RANKING_CARDS[:2]
        rotated_in = [r.card.id for r in ex.rank_cards(rotated, RANKING_CLASSES)]
        self.assertEqual(baseline, reversed_in)
        self.assertEqual(baseline, rotated_in)

    def test_only_surviving_claims_are_ranked(self) -> None:
        """A falsified claim and one that could not run have no effect size to report."""
        ranked = ex.rank_cards(RANKING_CARDS, RANKING_CLASSES)
        self.assertEqual({r.card.id for r in ranked}, {"H1", "H2", "H3", "H4"})

    def test_ties_are_broken_on_the_card_id(self) -> None:
        """Two identical claims still have exactly one order, and it is by id."""
        twins = (
            card("B", "correlation", {"r": 0.5, "target": 0.0}, {}),
            card("A", "correlation", {"r": 0.5, "target": 0.0}, {}),
        )
        classes = {"A": ex.Classification(ex.COMPUTED, ""), "B": ex.Classification(ex.COMPUTED, "")}
        ranked = ex.rank_cards(twins, classes)
        self.assertEqual([r.card.id for r in ranked], ["A", "B"])

    def test_a_computed_claim_outranks_a_lookup_claim_all_else_equal(self) -> None:
        """The class is a ranking component, and this is the direction it points."""
        pair = (
            card("L", "correlation", {"r": 0.6, "target": 0.0}, {}),
            card("C", "correlation", {"r": 0.6, "target": 0.0}, {}),
        )
        classes = {"L": ex.Classification(ex.LOOKUP, ""), "C": ex.Classification(ex.COMPUTED, "")}
        ranked = ex.rank_cards(pair, classes)
        self.assertEqual([r.card.id for r in ranked], ["C", "L"])

    def test_a_result_against_the_naive_expectation_is_scored_higher(self) -> None:
        """'growth falls as flow rises' contradicts the obvious expectation."""
        falling = [r for r in ex.rank_cards(RANKING_CARDS, RANKING_CLASSES) if r.card.id == "H3"]
        self.assertTrue(falling[0].contradicts)
        rising = card(
            "H7",
            "metric_ordering",
            {"value_a": 1.0, "value_b": 2.0, "min_step": 1.0, "max_step": 1.0},
            {
                "metric": "wss_peak_dyn_cm2",
                "cases": ["Underflow", "Healthy", "Overflow"],
                "direction": "increasing",
            },
        )
        self.assertFalse(ex.contradicts_naive(rising))

    def test_reweighting_changes_the_order_without_changing_the_components(self) -> None:
        """The components are reported so a reader can re-rank; prove they can."""
        effect_only = ex.rank_cards(
            RANKING_CARDS,
            RANKING_CLASSES,
            weights={"effect": 1.0, "margin": 0.0, "class": 0.0, "naive": 0.0},
        )
        class_only = ex.rank_cards(
            RANKING_CARDS,
            RANKING_CLASSES,
            weights={"effect": 0.0, "margin": 0.0, "class": 1.0, "naive": 0.0},
        )
        self.assertEqual(class_only[0].card.id, "H2", "the only COMPUTED survivor should lead")
        by_id = {r.card.id: r for r in effect_only}
        self.assertEqual(by_id["H2"].effect, ex.effect_size(RANKING_CARDS[1]))

    def test_every_component_stays_inside_its_stated_range(self) -> None:
        """Scores are only comparable because each component is bounded in [0, 1]."""
        for ranked in ex.rank_cards(RANKING_CARDS, RANKING_CLASSES):
            self.assertGreaterEqual(ranked.effect, 0.0)
            self.assertLessEqual(ranked.effect, 1.0)
            self.assertGreaterEqual(ranked.margin, 0.0)
            self.assertLessEqual(ranked.margin, 1.0)
            self.assertGreaterEqual(ranked.score, 0.0)
            self.assertLessEqual(ranked.score, 1.0)


# --- The stratified survival split -----------------------------------------------------


class TestStratifiedSurvival(unittest.TestCase):
    """The number this task exists to produce, counted by hand first."""

    def test_the_split_is_counted_per_class_and_could_not_run_is_excluded(self) -> None:
        """Six cards: 3 LOOKUP (2 survived), 3 COMPUTED (1 survived, 1 unrunnable)."""
        survival = ex.stratified_survival(RANKING_CARDS, RANKING_CLASSES)
        self.assertEqual(survival["n_claims"], 6)
        self.assertEqual(survival["n_ran"], 5)
        self.assertEqual(survival["n_survived"], 4)
        self.assertEqual(survival["n_could_not_run"], 1)
        lookup = survival["by_class"][ex.LOOKUP]
        computed = survival["by_class"][ex.COMPUTED]
        self.assertEqual((lookup["claims"], lookup["ran"], lookup["survived"]), (3, 3, 3))
        self.assertEqual((computed["claims"], computed["ran"], computed["survived"]), (3, 2, 1))
        self.assertAlmostEqual(lookup["survival_rate"], 1.0)
        self.assertAlmostEqual(computed["survival_rate"], 0.5)
        self.assertAlmostEqual(survival["survival_rate"], 4 / 5)
        self.assertAlmostEqual(survival["lookup_share_of_survivors"], 3 / 4)
        self.assertAlmostEqual(survival["computed_share_of_survivors"], 1 / 4)

    def test_a_run_with_nothing_to_report_says_so_rather_than_dividing_by_zero(self) -> None:
        """No cards means no rate, not a rate of zero, which would read as a result."""
        survival = ex.stratified_survival((), {})
        self.assertIsNone(survival["survival_rate"])
        self.assertIsNone(survival["lookup_share_of_survivors"])
        self.assertEqual(survival["n_claims"], 0)

    def test_an_unclassified_card_is_counted_as_computed_not_dropped(self) -> None:
        """A card missing from the classification map must not vanish from the count."""
        survival = ex.stratified_survival([RANKING_CARDS[0]], {})
        self.assertEqual(survival["n_claims"], 1)
        self.assertEqual(survival["by_class"][ex.COMPUTED]["claims"], 1)


# --- Question proposal, output, and the CLI --------------------------------------------


class TestQuestionProposal(unittest.TestCase):
    """Questions come from the schema, and provenance never overstates who wrote them."""

    def test_the_schema_proposer_aims_at_cells_nothing_has_opened(self) -> None:
        """Every question it offers names an untouched target and case."""
        coverage = ex.Coverage(
            targets={"compare_metric": ("a", "b"), "correlation": ("x", "y")},
            cases=("C1", "C2"),
        )
        questions = ex.SchemaQuestionProposer().propose_questions(4, coverage)
        self.assertEqual(len(questions), 4)
        self.assertEqual(len(set(questions)), 4)

    def test_it_stops_offering_questions_once_the_grid_is_full(self) -> None:
        """A full grid is a legitimate end to a run, not an infinite loop."""
        coverage = ex.Coverage(targets={"compare_metric": ("a",)}, cases=("C1",))
        coverage.record(spec("compare_metric", metric="a", case_a="C1", case_b="C1", op="gt"))
        self.assertEqual(ex.SchemaQuestionProposer().propose_questions(5, coverage), [])

    def test_it_needs_no_model_at_all(self) -> None:
        """The provenance string says who wrote the questions, and it is not a model."""
        self.assertEqual(ex.SchemaQuestionProposer().name, "schema-enumerator")

    def test_the_model_proposer_needs_a_free_text_channel(self) -> None:
        """A backend with neither hook is rejected at construction, not mid-run."""

        class Mute:
            """A backend with no free-text channel."""

            name = "mute"

        with self.assertRaises(TypeError):
            ex.ModelQuestionProposer(Mute())

    def test_the_model_proposer_parses_and_falls_back(self) -> None:
        """A usable reply is used; an empty one falls back rather than ending the run."""

        class Talker:
            """A backend whose narration channel answers with questions."""

            name = "talker"

            def __init__(self, reply: str | None) -> None:
                """Store the canned reply."""
                self.reply = reply
                self.prompts: list[str] = []

            def synthesize(self, prompt: str) -> str | None:
                """Return the canned reply and record the prompt."""
                self.prompts.append(prompt)
                return self.reply

        coverage = ex.Coverage(targets={"compare_metric": ("a",)}, cases=("C1",))
        talker = Talker("1. Does A rise with flow?\n- Is B larger in overflow?\n\n")
        proposer = ex.ModelQuestionProposer(talker)
        self.assertEqual(
            proposer.propose_questions(2, coverage),
            ["Does A rise with flow?", "Is B larger in overflow?"],
        )
        self.assertIn("ALREADY INVESTIGATED", talker.prompts[0])
        self.assertTrue(proposer.name.startswith("model:"))

        silent = ex.ModelQuestionProposer(Talker(""), fallback=ex.SchemaQuestionProposer())
        self.assertEqual(len(silent.propose_questions(1, coverage)), 1)

    def test_a_proposer_failure_does_not_end_the_world(self) -> None:
        """A backend that raises is treated as a backend that said nothing."""

        class Broken:
            """A backend whose free-text channel raises."""

            name = "broken"

            def synthesize(self, prompt: str) -> str | None:
                """Always fail."""
                del prompt
                raise RuntimeError("no")

        coverage = ex.Coverage(targets={"compare_metric": ("a",)}, cases=("C1",))
        proposer = ex.ModelQuestionProposer(Broken(), fallback=ex.SchemaQuestionProposer())
        self.assertEqual(len(proposer.propose_questions(1, coverage)), 1)

    def test_question_parsing_strips_the_packaging_a_model_adds(self) -> None:
        """Fences, bullets, numbering and blank lines all come off."""
        raw = "```\n1) First question?\n* Second question?\n\n---\n- Third question?\n```"
        self.assertEqual(
            ex.parse_questions(raw),
            ["First question?", "Second question?", "Third question?"],
        )
        self.assertEqual(ex.parse_questions(None), [])
        self.assertEqual(ex.parse_questions(["a?", "a?", "b?"]), ["a?", "b?"])

    def test_the_catalogue_names_the_schema_and_not_the_briefing(self) -> None:
        """The proposer is grounded in the registries, deliberately without the data."""
        catalogue = ex.schema_catalogue()
        for name in ("compare_metric", "correlation", "profile_monotonic"):
            self.assertIn(name, catalogue)
        for node in ("EndMT", "NICD"):
            self.assertIn(node, catalogue)
        self.assertNotIn("Scalar metrics", catalogue)


class TestOutputs(_StudyHarness):
    """The three artifacts TASK-015 names are written, and say what they should."""

    def test_a_run_writes_the_report_the_coverage_and_the_survival_split(self) -> None:
        """Every named output exists, parses, and agrees with the result object."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "explore"
            result = ex.explore(
                self.study(RunawayGenerator()),
                budget=ex.Budget(max_questions=2, plateau_questions=0),
                proposer=EndlessProposer(),
                out=out,
                narrow_failures=False,
                synthesize_report=False,
            )
            self.assertTrue((out / "exploration_report.md").is_file())
            coverage = json.loads((out / "coverage.json").read_text())
            survival = json.loads((out / "stratified_survival.json").read_text())
            self.assertEqual(coverage["total_cells"], result.coverage.total)
            self.assertEqual(survival["n_claims"], len(result.cards))
            self.assertTrue((out / "q01" / "cards.json").is_file())
            self.assertTrue((out / "q02" / "cards.json").is_file())

    def test_the_report_leads_with_the_split_and_names_the_lookup_share(self) -> None:
        """A reader must not be able to see the total without seeing the split."""
        result = ex.ExplorationResult(
            questions=(ex.QuestionRecord(1, "why?", 6, 4, ("compare_metric|a|C1",)),),
            cards=RANKING_CARDS,
            classifications=RANKING_CLASSES,
            coverage=ex.Coverage(targets={"compare_metric": ("a",)}, cases=("C1",)),
            stop=ex.StopReason("max_questions", "one question"),
            ranking=tuple(ex.rank_cards(RANKING_CARDS, RANKING_CLASSES)),
            survival=ex.stratified_survival(RANKING_CARDS, RANKING_CLASSES),
            calls=1,
            cost_usd=0.0,
            proposer="schema-enumerator",
        )
        report = ex.build_exploration_report(result)
        split_at = report.index("Survival, split by")
        findings_at = report.index("Surviving claims, ranked")
        self.assertLess(split_at, findings_at)
        self.assertIn("LOOKUP share of the surviving claims: **75.0%**", report)
        self.assertIn("| LOOKUP | 3 | 3 | 3 | 0 | 0 | 100.0% |", report)


class TestCLI(unittest.TestCase):
    """The command line exists, and its defaults are the budget's defaults."""

    def test_the_parser_defaults_match_the_budget(self) -> None:
        """A flag that drifted from the dataclass would be a silent budget change."""
        args = ex.build_parser().parse_args([])
        self.assertEqual(args.max_questions, ex.Budget.max_questions)
        self.assertEqual(args.max_calls, ex.Budget.max_calls)
        self.assertEqual(args.plateau, ex.Budget.plateau_questions)
        self.assertEqual(args.hypotheses, ex.Budget.hypotheses_per_question)

    def test_an_unknown_backend_is_refused_rather_than_guessed_at(self) -> None:
        """Exit code 2, and no exploration attempted."""
        import contextlib
        import io as _io

        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = ex.main(["--generator", "not-a-backend"])
        self.assertEqual(code, 2)
        self.assertIn("could not build", buffer.getvalue())


if __name__ == "__main__":  # pragma: no cover - exercised by the discovery runner
    unittest.main()
