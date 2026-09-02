"""Tests for the hypothesis-card model and its Markdown/HTML renderers.

Written against :mod:`unittest` rather than pytest: pytest is not installed in the
environment this project currently runs in, and pytest collects ``TestCase`` classes
unchanged, so these keep working once it is.

Run with::

    cd src && python3 -m unittest discover -s ../tests -t . -v
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from typing import Any

from llm_insights.cards.card import Card, cards_from_json, cards_from_run, cards_to_json
from llm_insights.cards.render import (
    COULD_NOT_RUN,
    FALSIFIED,
    NARROWED,
    SURVIVED,
    counts,
    format_number,
    group_into_blocks,
    render_html,
    render_markdown,
    verdict_of,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CARDS = REPO_ROOT / "data" / "example_cards.json"

QUESTION = "How does inlet flow rate reshape the AV cushion and its EndMT program?"
META: dict[str, Any] = {
    "dataset_root": "/data/one_way_fsg_model",
    "n_field_values": 348450,
    "generated_at": "2026-09-01T09:15:00Z",
    "generator": "llm_insights.agent.propose",
    "model": "test-fixture",
}


def setUpModule() -> None:
    """Keep the library's expected warnings out of the test report.

    Several tests deliberately exercise defensive paths that log a warning (a missing
    outcome, a non-numeric observation, a revision cycle). Those warnings are correct
    behaviour, not test failures, so they are swallowed rather than printed.
    """
    logger = logging.getLogger("llm_insights")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False


def make_card(**overrides: Any) -> Card:
    """Build a valid card, overriding any field.

    Args:
        **overrides: Field values replacing the defaults below.

    Returns:
        A :class:`Card` suitable for rendering.
    """
    base: dict[str, Any] = {
        "id": "H1",
        "claim": "Wall shear stress is higher in the overflow case.",
        "rationale": "Inlet velocity is 1.5x the healthy case.",
        "decision_rule": "SURVIVES if the ratio exceeds 1.1.",
        "primitive": "compare_scalar_between_cases",
        "params": {"field": "wss_dyn_cm2", "step": 14},
        "passed": True,
        "observed": {"ratio": 1.4967, "required_ratio": 1.1},
        "summary": "Ratio 1.50 clears the threshold.",
    }
    base.update(overrides)
    return Card(**base)


class _Hypothesis:
    """Duck-typed stand-in for ``llm_insights.harness.spec.Hypothesis``."""

    def __init__(self, hid: str, parent_id: str | None = None, **kwargs: Any) -> None:
        self.id = hid
        self.claim = kwargs.get("claim", f"claim {hid}")
        self.rationale = kwargs.get("rationale", f"rationale {hid}")
        self.decision_rule = kwargs.get("decision_rule", f"rule {hid}")
        self.primitive = kwargs.get("primitive", "compare_scalar_between_cases")
        self.params = kwargs.get("params", {"step": 14})
        self.parent_id = parent_id


class _Outcome:
    """Duck-typed stand-in for ``llm_insights.harness.spec.Outcome``."""

    def __init__(
        self,
        hypothesis_id: str,
        passed: bool = True,
        observed: dict[str, float] | None = None,
        summary: str = "ran",
        error: str | None = None,
    ) -> None:
        self.hypothesis_id = hypothesis_id
        self.passed = passed
        self.observed = observed if observed is not None else {"value": 1.0}
        self.summary = summary
        self.error = error


class TestRoundTrip(unittest.TestCase):
    """cards_to_json -> cards_from_json must preserve every field exactly."""

    def test_round_trip_preserves_every_field(self) -> None:
        cards = [
            make_card(),
            make_card(
                id="H2",
                claim="Variability is highest in overflow.",
                passed=False,
                observed={"overflow_iqr": 0.0412, "healthy_iqr": 0.0733},
                summary="Highest in healthy instead.",
                revised_by="H3",
            ),
            make_card(
                id="H3",
                claim="Restricted to surface nodes, variability is highest in overflow.",
                observed={"overflow_surface_iqr": 0.0688},
                revision_of="H2",
            ),
            make_card(
                id="H4",
                passed=False,
                observed={},
                summary="Never reached a verdict.",
                error="ValueError: step 22 out of range 0-14",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "cards.json"
            cards_to_json(cards, path)
            self.assertTrue(path.is_file(), "cards_to_json should create parent dirs")
            restored = cards_from_json(path)

        self.assertEqual(len(restored), len(cards))
        for original, copy in zip(cards, restored, strict=True):
            for field in fields(Card):
                self.assertEqual(
                    getattr(copy, field.name),
                    getattr(original, field.name),
                    f"field {field.name} did not survive the round trip",
                )
            self.assertEqual(copy, original)

    def test_round_trip_of_the_shipped_fixture(self) -> None:
        cards = cards_from_json(EXAMPLE_CARDS)
        self.assertEqual(len(cards), 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "again.json"
            cards_to_json(cards, path)
            self.assertEqual(cards_from_json(path), cards)

    def test_unknown_keys_are_ignored_and_defaults_apply(self) -> None:
        payload = [
            {
                "id": "H9",
                "claim": "c",
                "rationale": "r",
                "decision_rule": "d",
                "primitive": "p",
                "params": {},
                "passed": True,
                "observed": {},
                "summary": "s",
                "future_field_we_do_not_know": 42,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cards.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            (card,) = cards_from_json(path)
        self.assertIsNone(card.error)
        self.assertIsNone(card.revision_of)
        self.assertIsNone(card.revised_by)

    def test_missing_required_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cards.json"
            path.write_text(json.dumps([{"id": "H1"}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                cards_from_json(path)


class TestEscaping(unittest.TestCase):
    """User-supplied text must never reach the HTML unescaped."""

    def test_script_tag_and_ampersand_are_escaped(self) -> None:
        card = make_card(
            claim="<script>alert('xss')</script> shear & stress",
            rationale="Tom & Jerry <b>bold</b>",
            summary='He said "overflow" & left',
            decision_rule="SURVIVES if a < b & c > d",
            observed={"a<b": 1.5},
        )
        out = render_html([card], "Does <b>flow</b> & growth couple?", META)

        self.assertNotIn("<script>alert", out)
        self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", out)
        self.assertIn("shear &amp; stress", out)
        self.assertIn("Tom &amp; Jerry &lt;b&gt;bold&lt;/b&gt;", out)
        self.assertIn("&quot;overflow&quot; &amp; left", out)
        self.assertIn("a &lt; b &amp; c &gt; d", out)
        self.assertIn("a&lt;b", out)
        self.assertIn("Does &lt;b&gt;flow&lt;/b&gt; &amp; growth couple?", out)
        self.assertNotIn("<b>bold</b>", out)

    def test_params_are_escaped_in_the_monospace_block(self) -> None:
        card = make_card(params={"case": "</style><script>x</script>"})
        out = render_html([card], QUESTION, META)
        self.assertNotIn("</style><script>x", out)
        self.assertIn("&lt;/style&gt;&lt;script&gt;x&lt;/script&gt;", out)


class TestVerdicts(unittest.TestCase):
    """The four terminal states must be classified and labelled distinctly."""

    def test_error_card_is_could_not_run_not_falsified(self) -> None:
        card = make_card(passed=False, observed={}, error="ValueError: step 22 out of range")
        self.assertIs(verdict_of(card), COULD_NOT_RUN)

        out = render_html([card], QUESTION, META)
        self.assertIn("COULD NOT RUN", out)
        self.assertIn('class="card inert"', out)
        self.assertNotIn('class="card falsified"', out)
        self.assertIn("ValueError: step 22 out of range", out)

        md = render_markdown([card], QUESTION, META)
        self.assertIn("COULD NOT RUN", md)
        self.assertNotIn("FALSIFIED", md.split("## Claims")[1])

    def test_error_card_is_could_not_run_even_if_passed_is_true(self) -> None:
        card = make_card(passed=True, error="boom")
        self.assertIs(verdict_of(card), COULD_NOT_RUN)

    def test_plain_failure_is_falsified(self) -> None:
        self.assertIs(verdict_of(make_card(passed=False)), FALSIFIED)

    def test_pass_is_survived(self) -> None:
        self.assertIs(verdict_of(make_card(passed=True)), SURVIVED)

    def test_counts_include_every_state(self) -> None:
        cards = cards_from_json(EXAMPLE_CARDS)
        tally = counts(cards)
        self.assertEqual(tally[SURVIVED.label], 3)
        self.assertEqual(tally[NARROWED.label], 1)
        self.assertEqual(tally[FALSIFIED.label], 0)
        self.assertEqual(tally[COULD_NOT_RUN.label], 1)
        self.assertEqual(sum(tally.values()), len(cards))


class TestNarrowedChain(unittest.TestCase):
    """A falsified claim with a successor is the centrepiece of the report."""

    def setUp(self) -> None:
        self.parent = make_card(
            id="H3",
            claim="Variability is highest in the overflow case.",
            passed=False,
            observed={"overflow_iqr": 0.0412, "healthy_iqr": 0.0733},
            summary="Highest in the healthy case instead.",
            revised_by="H4",
        )
        self.child = make_card(
            id="H4",
            claim="Restricted to surface nodes, variability is highest in overflow.",
            passed=True,
            observed={"overflow_surface_iqr": 0.0688},
            summary="The narrowed claim survives.",
            revision_of="H3",
        )

    def test_falsified_card_with_successor_renders_as_narrowed(self) -> None:
        self.assertIs(verdict_of(self.parent), NARROWED)
        self.assertTrue(self.parent.was_narrowed)
        out = render_html([self.parent, self.child], QUESTION, META)
        self.assertIn(NARROWED.label, out)

    def test_successor_is_rendered_inside_the_same_chain_block(self) -> None:
        out = render_html([self.parent, self.child], QUESTION, META)

        self.assertEqual(out.count('<section class="block">'), 1, "chain must be one block")
        self.assertIn('<div class="chain">', out)
        chain = out.split('<div class="chain">')[1].split("</section>")[0]
        self.assertIn(self.parent.claim, chain)
        self.assertIn(self.child.claim, chain)
        self.assertIn("H3 → H4", chain)
        self.assertIn("The agent narrowed it and re-ran", chain)
        self.assertIn("↳ narrows H3", chain)
        self.assertIn("card survived revision", chain)

    def test_narrowed_card_is_stamped_falsified_as_well(self) -> None:
        """The rejection must stay visible; NARROWED alone would hide it."""
        out = render_html([self.parent, self.child], QUESTION, META)
        verdict_row = out.split('<div class="verdict">')[1].split("</div>")[0]
        self.assertIn(FALSIFIED.label, verdict_row)
        self.assertIn(NARROWED.label, verdict_row)

        md = render_markdown([self.parent, self.child], QUESTION, META)
        self.assertIn(f"{FALSIFIED.label} → ", md)

    def test_grouping_puts_parent_and_child_in_one_block(self) -> None:
        blocks = group_into_blocks([self.parent, make_card(id="H1"), self.child])
        self.assertEqual([[c.id for c in b] for b in blocks], [["H3", "H4"], ["H1"]])

    def test_grouping_survives_a_dangling_or_cyclic_link(self) -> None:
        dangling = make_card(id="H7", passed=False, revised_by="does-not-exist")
        self.assertEqual(group_into_blocks([dangling]), [[dangling]])

        a = make_card(id="A", passed=False, revised_by="B")
        b = make_card(id="B", passed=False, revised_by="A", revision_of="A")
        blocks = group_into_blocks([a, b])
        self.assertEqual([[c.id for c in block] for block in blocks], [["A", "B"]])

    def test_markdown_links_the_chain_too(self) -> None:
        md = render_markdown([self.parent, self.child], QUESTION, META)
        self.assertIn("Falsification chain: H3 → H4", md)
        self.assertIn("Narrowed by **H4** below.", md)
        self.assertIn("Narrows the falsified claim **H3**.", md)

    def test_fixture_chain_renders_as_one_block(self) -> None:
        cards = cards_from_json(EXAMPLE_CARDS)
        blocks = group_into_blocks(cards)
        self.assertEqual(
            [[c.id for c in block] for block in blocks],
            [["H1"], ["H2"], ["H3", "H4"], ["H5"]],
        )


class TestEmptyRun(unittest.TestCase):
    """Zero cards must render, not crash."""

    def test_markdown_with_no_cards(self) -> None:
        md = render_markdown([], QUESTION, META)
        self.assertIn(QUESTION, md)
        self.assertIn("No hypotheses were produced", md)

    def test_html_with_no_cards(self) -> None:
        out = render_html([], QUESTION, META)
        self.assertIn("<!doctype html>", out)
        self.assertIn("</html>", out)
        self.assertIn("nothing to falsify yet", out)
        self.assertNotIn("<article", out)

    def test_html_with_no_cards_and_empty_meta_and_question(self) -> None:
        out = render_html([], "", {})
        self.assertIn("Falsification report", out)
        self.assertNotIn("<dl", out)


class TestObservedNumbers(unittest.TestCase):
    """Every number the decision used must be visible in the rendered output."""

    def test_every_observed_number_appears_in_html(self) -> None:
        cards = cards_from_json(EXAMPLE_CARDS)
        out = render_html(cards, QUESTION, META)
        checked = 0
        for card in cards:
            for name, value in card.observed.items():
                self.assertIn(name, out, f"{card.id}: observed key {name} missing from HTML")
                self.assertIn(
                    format_number(value),
                    out,
                    f"{card.id}: observed value {value} for {name} missing from HTML",
                )
                checked += 1
        self.assertGreater(checked, 20, "fixture should exercise many observed values")

    def test_literal_values_appear_verbatim(self) -> None:
        card = make_card(observed={"ratio": 1.4967, "p95": 41.73, "n_nodes": 1462.0})
        out = render_html([card], QUESTION, META)
        self.assertIn("1.4967", out)
        self.assertIn("41.73", out)
        self.assertIn("1462", out)
        self.assertNotIn("1462.0", out)

    def test_every_observed_number_appears_in_markdown(self) -> None:
        cards = cards_from_json(EXAMPLE_CARDS)
        md = render_markdown(cards, QUESTION, META)
        for card in cards:
            for name, value in card.observed.items():
                self.assertIn(name, md)
                self.assertIn(format_number(value), md)

    def test_format_number(self) -> None:
        self.assertEqual(format_number(1462.0), "1462")
        self.assertEqual(format_number(0.0412), "0.0412")
        self.assertEqual(format_number(-3.5), "-3.5")
        self.assertEqual(format_number(0.0), "0")
        self.assertEqual(format_number(float("nan")), "nan")


class TestHtmlDocument(unittest.TestCase):
    """The HTML must be a single self-contained, projectable, printable file."""

    def setUp(self) -> None:
        self.cards = cards_from_json(EXAMPLE_CARDS)
        self.out = render_html(self.cards, QUESTION, META)

    def test_is_standalone(self) -> None:
        self.assertTrue(self.out.startswith("<!doctype html>"))
        self.assertIn("<style>", self.out)
        self.assertNotIn("<script", self.out)
        self.assertNotIn("http://", self.out)
        self.assertNotIn("https://", self.out)
        self.assertNotIn("@import", self.out)
        self.assertNotIn("cdn", self.out.lower())

    def test_has_print_and_dark_mode_blocks(self) -> None:
        self.assertIn("@media print", self.out)
        self.assertIn("@media (prefers-color-scheme: dark)", self.out)

    def test_header_states_question_counts_and_verification(self) -> None:
        self.assertIn(QUESTION, self.out)
        self.assertIn("No language model judged any result.", self.out)
        for verdict in (SURVIVED, FALSIFIED, NARROWED, COULD_NOT_RUN):
            self.assertIn(verdict.label, self.out)
        self.assertIn("/data/one_way_fsg_model", self.out)
        self.assertIn("348450", self.out)
        self.assertIn("2026-09-01T09:15:00Z", self.out)

    def test_card_sections_appear_in_the_required_order(self) -> None:
        card = self.cards[0]
        body = self.out.split('<article class="card')[1]
        positions = [
            body.index(card.claim),
            body.index("Decision rule · fixed before the"),
            body.index(card.primitive),
            body.index("Observed values the decision rule used"),
            body.index(SURVIVED.label),
        ]
        self.assertEqual(positions, sorted(positions), "card sections are out of order")

    def test_tags_are_balanced(self) -> None:
        for tag in ("article", "section", "table", "div", "dl", "ul", "footer", "header"):
            self.assertEqual(
                self.out.count(f"<{tag}"),
                self.out.count(f"</{tag}>"),
                f"unbalanced <{tag}> tags",
            )


class TestWriteReport(unittest.TestCase):
    """write_report writes both files and returns their paths."""

    def test_writes_both_files(self) -> None:
        cards = cards_from_json(EXAMPLE_CARDS)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "made" / "up"
            md_path, html_path = write_report(cards, QUESTION, META, out_dir)
            self.assertEqual(md_path, out_dir / "report.md")
            self.assertEqual(html_path, out_dir / "report.html")
            self.assertIn("Falsification report", md_path.read_text(encoding="utf-8"))
            self.assertIn("<!doctype html>", html_path.read_text(encoding="utf-8"))


class TestCardsFromRun(unittest.TestCase):
    """Joining harness objects to cards, including the revision links."""

    def test_links_revisions_in_both_directions(self) -> None:
        hyps = [
            _Hypothesis("H1"),
            _Hypothesis("H2"),
            _Hypothesis("H3", parent_id="H2"),
        ]
        outs = [
            _Outcome("H1", passed=True, observed={"ratio": 1.5}),
            _Outcome("H2", passed=False, observed={"iqr": 0.04}, summary="contradicted"),
            _Outcome("H3", passed=True, observed={"iqr": 0.07}, summary="narrowed"),
        ]
        cards = {c.id: c for c in cards_from_run(hyps, outs)}
        self.assertEqual(cards["H2"].revised_by, "H3")
        self.assertEqual(cards["H3"].revision_of, "H2")
        self.assertIsNone(cards["H1"].revised_by)
        self.assertIs(verdict_of(cards["H2"]), NARROWED)
        self.assertIs(verdict_of(cards["H3"]), SURVIVED)

    def test_error_outcome_never_counts_as_passed(self) -> None:
        (card,) = cards_from_run(
            [_Hypothesis("H1")], [_Outcome("H1", passed=True, error="ValueError: bad step")]
        )
        self.assertFalse(card.passed)
        self.assertEqual(card.error, "ValueError: bad step")
        self.assertIs(verdict_of(card), COULD_NOT_RUN)

    def test_missing_outcome_becomes_could_not_run(self) -> None:
        (card,) = cards_from_run([_Hypothesis("H1")], [_Outcome("H2")])
        self.assertIs(verdict_of(card), COULD_NOT_RUN)
        self.assertIsNotNone(card.error)

    def test_accepts_mappings_as_well_as_objects(self) -> None:
        hyps = [
            {
                "id": "H1",
                "claim": "c",
                "rationale": "r",
                "decision_rule": "d",
                "primitive": "p",
                "params": {"step": 14},
                "parent_id": None,
            }
        ]
        outs = [{"hypothesis_id": "H1", "passed": True, "observed": {"x": 2}, "summary": "s"}]
        (card,) = cards_from_run(hyps, outs)
        self.assertEqual(card.claim, "c")
        self.assertEqual(card.observed, {"x": 2.0})
        self.assertTrue(card.passed)

    def test_non_numeric_observations_are_dropped_not_fatal(self) -> None:
        (card,) = cards_from_run(
            [_Hypothesis("H1")], [_Outcome("H1", observed={"good": 1.0, "bad": "not a number"})]
        )
        self.assertEqual(card.observed, {"good": 1.0})

    def test_outcomes_without_ids_are_matched_positionally(self) -> None:
        class _Bare:
            passed = False
            observed = {"x": 1.0}
            summary = "s"
            error = None

        cards = cards_from_run([_Hypothesis("H1"), _Hypothesis("H2")], [_Bare(), _Bare()])
        self.assertEqual([c.id for c in cards], ["H1", "H2"])
        self.assertTrue(all(c.error is None for c in cards))

    def test_reads_the_real_harness_hypothesis_and_outcome(self) -> None:
        """The executable half lives on ``Hypothesis.test``; cards must reach into it."""
        try:
            from llm_insights.harness.spec import Hypothesis, Outcome, TestSpec
        except ImportError as exc:  # pragma: no cover - harness lands in parallel
            self.skipTest(f"harness.spec unavailable: {exc}")

        spec = TestSpec(
            primitive="compare_dispersion_between_cases",
            params={"node": "EndMT", "scenario": "combined"},
            decision_rule="SURVIVES if the overflow IQR is the largest of the three.",
        )
        narrower = TestSpec(
            primitive="compare_dispersion_between_cases",
            params={"node": "EndMT", "scenario": "shear_only", "region": "surface"},
            decision_rule="SURVIVES if the overflow surface IQR is the largest.",
        )
        hyps = [
            Hypothesis(id="H1", claim="Variability peaks in overflow.", rationale="r", test=spec),
            Hypothesis(
                id="H2",
                claim="On the surface, variability peaks in overflow.",
                rationale="r2",
                test=narrower,
                parent_id="H1",
            ),
        ]
        outs = [
            Outcome("H1", passed=False, observed={"overflow_iqr": 0.0412}, summary="FAIL ..."),
            Outcome("H2", passed=True, observed={"overflow_surface_iqr": 0.0688}, summary="PASS"),
        ]
        cards = {card.id: card for card in cards_from_run(hyps, outs)}

        self.assertEqual(cards["H1"].primitive, "compare_dispersion_between_cases")
        self.assertEqual(cards["H1"].params, {"node": "EndMT", "scenario": "combined"})
        self.assertIn("SURVIVES if the overflow IQR", cards["H1"].decision_rule)
        self.assertNotIn("TestSpec", cards["H1"].primitive)
        self.assertEqual(cards["H1"].revised_by, "H2")
        self.assertEqual(cards["H2"].revision_of, "H1")
        self.assertIs(verdict_of(cards["H1"]), NARROWED)

        out = render_html(list(cards.values()), QUESTION, META)
        self.assertIn('<div class="chain">', out)
        self.assertIn("0.0688", out)

    def test_renders_end_to_end_from_run(self) -> None:
        hyps = [_Hypothesis("H1"), _Hypothesis("H2", parent_id="H1")]
        outs = [_Outcome("H1", passed=False), _Outcome("H2", passed=True)]
        out = render_html(cards_from_run(hyps, outs), QUESTION, META)
        self.assertIn('<div class="chain">', out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
