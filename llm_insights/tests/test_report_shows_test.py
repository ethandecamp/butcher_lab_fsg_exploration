"""The top-level exploration report shows the test that ran, not only the claim.

The one-page summary a reader actually reads used to print each claim's English and
nothing else. In a real run that was actively misleading: four claims carried
byte-identical ``params`` under four different English sentences, one of them prose
about mechanical stress over a test that correlated two GRN nodes. The verdicts were
correct -- the harness ran exactly what ``params`` asked for -- but a reader of the
summary could not see that the arithmetic was not about the sentence above it.

So these tests hold the report to one property: **the claim and the mechanical
description of its own test are both on the page, together, for every card.** The
divergence fixture below is built to disagree with itself on purpose; the test asserts
that both halves of the disagreement survive into the rendered Markdown, so that a
reader can see it. Nothing here checks the two against each other -- that is a
different job, and this is display only.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_insights.agent import explore as ex
from llm_insights.cards.card import Card

#: The claim's prose is about mechanical stress; its params correlate two GRN nodes.
#: This is the real shape of the defect, copied from ``data/explore_021/``.
DIVERGENT_CLAIM = (
    "The expression of Snai1 is directly proportional to mechanical stress in the Healthy case."
)
DIVERGENT_SUMMARY = (
    "PASS: correlation YAP_TAZ_shear_only vs Snai1_shear_only in flow_U0p0360: "
    "r=0.87 >= 0.5 -> holds"
)
BROKEN_ERROR = "unknown GRN field 'YAP_TAZ'; available fields are ['mech_norm', 'wss_norm']"


def _card(
    card_id: str,
    claim: str,
    summary: str,
    params: dict[str, object],
    *,
    passed: bool = True,
    error: str | None = None,
) -> Card:
    """Build a finished card without running anything.

    Args:
        card_id: The card's id.
        claim: The plain-English claim.
        summary: The harness's own one-line description of what ran.
        params: What the primitive was actually called with.
        passed: Whether the decision rule held.
        error: Why the test could not run, or None.

    Returns:
        The card.
    """
    return Card(
        id=card_id,
        claim=claim,
        rationale="because",
        decision_rule="fixed in advance",
        primitive="correlation",
        params=dict(params),
        passed=passed and error is None,
        observed={} if error else {"r": 0.87, "target": 0.5},
        summary=summary,
        error=error,
    )


#: One survivor whose prose and test disagree, one falsified claim, one that could not
#: run. Criterion 2 wants at least one of each outcome in the same result.
CARDS: tuple[Card, ...] = (
    _card(
        "q01_H1",
        DIVERGENT_CLAIM,
        DIVERGENT_SUMMARY,
        {"field_x": "YAP_TAZ_shear_only", "field_y": "Snai1_shear_only", "case": "Healthy"},
    ),
    _card(
        "q01_H2",
        "The expression of Snai1 falls as mechanical stress rises in the Healthy case.",
        "FAIL: correlation YAP_TAZ_shear_only vs Snai1_shear_only: r=0.87 < 0.0 -> does not hold",
        {"field_x": "YAP_TAZ_shear_only", "field_y": "Snai1_shear_only", "case": "Healthy"},
        passed=False,
    ),
    _card(
        "q01_H3",
        "The expression of Snai1 tracks wall shear stress in the Overflow case.",
        f"COULD NOT RUN: {BROKEN_ERROR}",
        {"field_x": "YAP_TAZ", "field_y": "Snai1_shear_only", "case": "Overflow"},
        error=BROKEN_ERROR,
    ),
)

CLASSES: dict[str, ex.Classification] = {
    "q01_H1": ex.Classification("COMPUTED", "the briefing prints no correlation", ""),
    "q01_H2": ex.Classification("COMPUTED", "the briefing prints no correlation", ""),
    "q01_H3": ex.Classification("COMPUTED", "the briefing prints no correlation", ""),
}


def _result(
    cards: tuple[Card, ...] = CARDS,
    classifications: dict[str, ex.Classification] | None = None,
) -> ex.ExplorationResult:
    """Assemble a finished exploration around a fixed set of cards.

    Args:
        cards: The cards the run produced.
        classifications: Card id to classification. Defaults to :data:`CLASSES`.

    Returns:
        The result, with ranking and survival computed the way a real run computes them.
    """
    classes = CLASSES if classifications is None else classifications
    return ex.ExplorationResult(
        questions=(ex.QuestionRecord(1, "How does Snai1 relate to stress?", len(cards), 1, ()),),
        cards=cards,
        classifications=classes,
        coverage=ex.Coverage(targets={"correlation": ("Snai1_shear_only",)}, cases=("Healthy",)),
        stop=ex.StopReason("max_questions", "one question"),
        ranking=tuple(ex.rank_cards(cards, classes)),
        survival=ex.stratified_survival(cards, classes),
        calls=1,
        cost_usd=0.0,
        proposer="schema-enumerator",
    )


class TestTheReportShowsTheTest(unittest.TestCase):
    """A reader of the summary can see what the harness actually did."""

    def test_a_claim_and_its_test_are_both_printed_so_a_reader_sees_them_disagree(
        self,
    ) -> None:
        """Prose about mechanical stress over a test of two GRN nodes: both on the page.

        This is the whole point of the section. Asserting only that the prose appears
        would have passed before the change; asserting only the summary would not let a
        reader compare. Both must be there, and both must be about this card.
        """
        report = ex.build_exploration_report(_result())
        self.assertIn(DIVERGENT_CLAIM, report)
        self.assertIn(DIVERGENT_SUMMARY, report)
        # The disagreement is legible: the prose names a quantity the test never read,
        # and the test names two the prose never mentions.
        self.assertIn("mechanical stress", DIVERGENT_CLAIM)
        self.assertNotIn("mechanical stress", DIVERGENT_SUMMARY)
        self.assertIn("YAP_TAZ_shear_only", report)

    def test_the_mechanical_line_is_read_off_the_card_and_not_re_derived(self) -> None:
        """Whatever the card says is what the report prints, verbatim.

        A second rendering of primitive-and-params would drift from the harness's own
        wording, so the section is pinned to the card's ``summary`` field. A summary
        that shares no words with the params proves the report did not rebuild it.
        """
        odd = _card(
            "q01_H1",
            DIVERGENT_CLAIM,
            "the harness said something entirely of its own",
            {"field_x": "YAP_TAZ_shear_only", "field_y": "Snai1_shear_only"},
        )
        report = ex.build_exploration_report(_result(cards=(odd,), classifications=CLASSES))
        self.assertIn("the harness said something entirely of its own", report)

    def test_every_card_appears_including_the_ones_that_could_not_run(self) -> None:
        """Survivors, falsified claims and unrunnable ones are all listed.

        The ranked table above shows survivors only. A claim that was falsified, or one
        whose test never executed, is exactly the case a reader most needs to see the
        mechanics of, so the section is over every card in the run.
        """
        report = ex.build_exploration_report(_result())
        for card in CARDS:
            self.assertIn(card.id, report, f"{card.id} is missing from the report")
            self.assertIn(card.claim, report, f"{card.id}'s claim is missing")

    def test_a_card_that_could_not_run_shows_its_error_in_place_of_a_summary(self) -> None:
        """The error is the mechanical description when there is no test result."""
        report = ex.build_exploration_report(_result())
        self.assertIn(BROKEN_ERROR, report)
        self.assertIn("could not run", report)

    def test_each_claim_carries_its_verdict_and_its_class(self) -> None:
        """Id, verdict and class head each block, so no lookup elsewhere is needed."""
        report = ex.build_exploration_report(_result())
        self.assertIn("### q01_H1 — SURVIVED · COMPUTED", report)
        self.assertIn("### q01_H2 — FALSIFIED · COMPUTED", report)
        self.assertIn("### q01_H3 — COULD NOT RUN · COMPUTED", report)

    def test_the_classification_reason_moved_into_the_per_claim_block(self) -> None:
        """The section absorbed the old list, so a claim is described in one place."""
        report = ex.build_exploration_report(_result())
        self.assertIn("the briefing prints no correlation", report)
        self.assertNotIn("## How each claim was classified", report)

    def test_an_unclassified_card_is_still_listed_rather_than_dropped(self) -> None:
        """A missing classification hides a claim from the reader; it must not."""
        report = ex.build_exploration_report(_result(classifications={}))
        self.assertIn("q01_H1", report)
        self.assertIn("unclassified", report)

    def test_the_existing_sections_are_left_intact(self) -> None:
        """Headline, survival table, questions table and ranked table all survive."""
        report = ex.build_exploration_report(_result())
        for heading in (
            "# Autonomous exploration",
            "## Survival, split by how hard the answer was to get",
            "## Questions",
            "## Surviving claims, ranked",
            "## Every claim, beside the test that ran",
        ):
            self.assertIn(heading, report)
        self.assertLess(
            report.index("## Surviving claims, ranked"),
            report.index("## Every claim, beside the test that ran"),
            "the per-claim section belongs after the ranked table",
        )


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    unittest.main()
