"""Tests for the optional narrative summary and the numeric fence around it.

Written against :mod:`unittest` for the same reason as the rest of ``tests/``: pytest
collects ``TestCase`` classes unchanged, so these keep working either way.

No test here calls a model. Every backend is a fake, and the fakes that matter are the
ones that misbehave on purpose -- a summary quoting a number nothing measured, a
backend that raises, a backend with no ``synthesize`` method at all -- because those
are the paths that decide whether this feature can weaken the project's one claim:
that no language model judged any result.

Run with::

    cd src && python3 -m unittest discover -s ../tests -t .. -v -p "test_synthesis.py"
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import tempfile
import types
import unittest
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from llm_insights.agent.generator import TranscriptGenerator
from llm_insights.agent.loop import FLOOR_OBSERVED_KEY, investigate, write_transcript
from llm_insights.agent.subscription import (
    DEFAULT_MAX_CALLS,
    ClaudeCLIGenerator,
    supported_flags,
)
from llm_insights.agent.synthesis import (
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_prompt,
    check_numeric_containment,
    split_narration,
    supports_synthesis,
    synthesize,
    transcript_entry,
)
from llm_insights.cards.card import Card, cards_from_run, cards_to_json
from llm_insights.cards.render import (
    NARRATIVE_HEADING,
    NARRATIVE_SUBLINE,
    counts,
    render_html,
    render_markdown,
)
from llm_insights.io.dataset import NodeField

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_TRANSCRIPT = REPO_ROOT / "data" / "demo_transcript.json"
METRICS_MODULE = "llm_insights.summary.metrics"
_MISSING = object()

#: The CLI help text the fake generator advertises. Only the flags this module cares
#: about need to be here; :mod:`tests.test_subscription` pins the full set.
CLI_HELP = """Usage: claude [options] [command] [prompt]

Options:
  --disallowedTools, --disallowed-tools <tools...>  deny list
  --max-budget-usd <usd>                            hard dollar cap
  --max-turns <n>                                   cap agentic turns
  --model <model>                                   model
  --output-format <format>                          text, json, stream-json
  -p, --print                                       print mode
  --system-prompt <prompt>                          replace the system prompt
"""


def setUpModule() -> None:
    """Quieten the library's own logging; these assertions are about return values.

    The two tests that do assert on a log line use :meth:`assertLogs`, which sets the
    level on the logger it captures and is therefore unaffected by this.
    """
    logging.getLogger("llm_insights").setLevel(logging.CRITICAL)


# --- Fixtures ----------------------------------------------------------------------


def make_card(**overrides: Any) -> Card:
    """Build a finished card, overriding any field.

    Args:
        **overrides: Field values replacing the defaults below.

    Returns:
        A :class:`~llm_insights.cards.card.Card`.
    """
    base: dict[str, Any] = {
        "id": "H1",
        "claim": "Wall shear stress is higher in the overflow case.",
        "rationale": "Inlet velocity is 1.5x the healthy case.",
        "decision_rule": "SURVIVES if the ratio exceeds 1.1.",
        "primitive": "compare_metric",
        "params": {"metric": "wss_peak_dyn_cm2", "op": "gt"},
        "passed": True,
        "observed": {"ratio": 1.4967, "required_ratio": 1.1},
        "summary": "Ratio 1.50 clears the threshold.",
    }
    base.update(overrides)
    return Card(**base)


class StubDataset:
    """A stand-in for :class:`~llm_insights.io.dataset.Dataset`.

    Only the surface ``fraction_above`` touches is implemented, which is enough to run
    the loop end to end against the real primitives with no results tree present.
    """

    root = "/stub/dataset"

    def __init__(self, von_mises: np.ndarray) -> None:
        """Store the von Mises values this stub will serve."""
        self._von_mises = np.asarray(von_mises, dtype=float)

    @staticmethod
    def resolve_case(name: str) -> str:
        """Accept the real condition labels and case names."""
        from llm_insights.io.dataset import Dataset

        return Dataset.resolve_case(name)

    def nodes(self, case: str) -> NodeField:
        """Return a node field carrying the configured von Mises values."""
        n = self._von_mises.size
        return NodeField(
            case=case,
            condition="Healthy",
            step=14,
            x_mm=np.linspace(0.0, 1.0, n),
            y_mm=np.zeros(n),
            von_mises_pa=self._von_mises,
            wss_dyn_cm2=np.full(n, np.nan),
            is_surface=np.ones(n, dtype=bool),
            von_mises_raw_min_pa=float(self._von_mises.min()),
        )


class MetricDataset:
    """A dataset stub for ``compare_metric``, which only ever resolves case names."""

    root = "/stub/metrics"

    @staticmethod
    def resolve_case(name: str) -> str:
        """Accept the two synthetic cases these tests use."""
        if name in {"A", "B"}:
            return name
        raise KeyError(f"unknown case {name!r}")


@contextlib.contextmanager
def fake_metric_registry(values: dict[tuple[str, str], float]) -> Iterator[None]:
    """Install a stand-in metric registry for the duration of a block.

    The harness imports the registry lazily inside a function, so replacing the
    ``sys.modules`` entry is enough and the real module is never imported.

    Args:
        values: ``{(metric, case): value}``.

    Yields:
        None, with the fake registry installed.
    """
    module = types.ModuleType(METRICS_MODULE)
    module.METRIC_NAMES = tuple(sorted({metric for metric, _ in values}))

    def metric_value(ds: Any, name: str, case: str) -> float:
        """Look the value up in the staged mapping."""
        del ds
        return float(values[(name, case)])

    module.metric_value = metric_value
    previous = sys.modules.get(METRICS_MODULE, _MISSING)
    sys.modules[METRICS_MODULE] = module
    try:
        yield
    finally:
        if previous is _MISSING:
            del sys.modules[METRICS_MODULE]
        else:
            sys.modules[METRICS_MODULE] = previous  # type: ignore[assignment]


def fraction_hypothesis(hid: str, threshold: float, value: float, op: str = "gt") -> dict[str, Any]:
    """Build a schema-valid hypothesis over ``fraction_above``."""
    return {
        "id": hid,
        "claim": f"More than {value:.0%} of the cushion exceeds {threshold} Pa.",
        "rationale": "Fixture claim.",
        "test": {
            "primitive": "fraction_above",
            "params": {
                "field": "von_mises",
                "case": "Healthy",
                "threshold": threshold,
                "op": op,
                "value": value,
            },
            "decision_rule": f"The fraction above {threshold} Pa is {op} {value}.",
        },
    }


def metric_hypothesis(hid: str, min_rel_diff: float | None = None) -> dict[str, Any]:
    """Build a schema-valid ``compare_metric`` hypothesis, optionally with a margin."""
    params: dict[str, Any] = {
        "metric": "wss_peak_dyn_cm2",
        "case_a": "A",
        "case_b": "B",
        "op": "gt",
    }
    if min_rel_diff is not None:
        params["min_rel_diff"] = min_rel_diff
    return {
        "id": hid,
        "claim": "Peak wall shear stress is higher in A than in B.",
        "rationale": "Fixture claim.",
        "test": {
            "primitive": "compare_metric",
            "params": params,
            "decision_rule": "SURVIVES if A exceeds B by the required margin.",
        },
    }


class FakeGenerator:
    """A generator whose narration behaviour is dictated by the test.

    Args:
        proposals: Raw hypothesis mappings to return from ``propose``.
        blurbs: Summary texts returned by successive ``synthesize`` calls. A
            :class:`Exception` instance in the list is raised instead of returned,
            which is how the fault-injection tests reach the failure path.
    """

    name = "fake"

    def __init__(self, proposals: list[dict], blurbs: list[Any] | None = None) -> None:
        """Stage the proposals and the narration replies."""
        self._proposals = proposals
        self._blurbs = list(blurbs or [])
        self.synthesis_prompts: list[str] = []

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Return the staged proposals."""
        del briefing, question, n
        return json.loads(json.dumps(self._proposals))

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Decline to narrow; these tests are about the summary, not the chain."""
        del briefing, failed, outcome
        return None

    def synthesize(self, prompt: str) -> str | None:
        """Return the next staged summary, or raise a staged exception."""
        self.synthesis_prompts.append(prompt)
        if not self._blurbs:
            return None
        reply = self._blurbs.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class MuteGenerator:
    """A backend from before this feature existed: no ``synthesize`` method at all."""

    name = "mute"

    def __init__(self, proposals: list[dict]) -> None:
        """Stage the proposals."""
        self._proposals = proposals

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Return the staged proposals."""
        del briefing, question, n
        return json.loads(json.dumps(self._proposals))

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Decline to narrow."""
        del briefing, failed, outcome
        return None


# --- The containment check ---------------------------------------------------------


class TestNumericContainment(unittest.TestCase):
    """The fence: which numerals a summary is allowed to use."""

    def setUp(self) -> None:
        """Two cards with distinct observed values."""
        self.cards = [
            make_card(),
            make_card(
                id="H2",
                claim="Nearly half the domain sits at the stress ceiling.",
                decision_rule="SURVIVES if the saturated fraction exceeds 3.25 percent.",
                observed={"fraction": 0.463, "threshold_pa": 100.0},
                passed=False,
                summary="Fraction 0.463 is below the rule.",
            ),
        ]

    def test_clean_blurb_passes(self):
        """Numbers taken straight from the observed values raise no violation."""
        blurb = (
            "Peak wall shear stress came out 1.4967 times higher in the overflow case, "
            "clearing the 1.1 the rule required, while 0.463 of the domain sat at the "
            "100 Pa ceiling."
        )
        self.assertEqual(check_numeric_containment(blurb, self.cards), [])

    def test_hallucinated_figure_is_caught(self):
        """A number the cards never produced is reported, and nothing else is."""
        blurb = "Shear was 1.4967 times higher, and EndMT rose by 2.7 fold."
        self.assertEqual(check_numeric_containment(blurb, self.cards), ["2.7"])

    def test_lower_displayed_precision_is_accepted(self):
        """'about 1.5' is a rounding of 1.4967, not an invention."""
        self.assertEqual(check_numeric_containment("The ratio was about 1.5.", self.cards), [])

    def test_precision_is_the_tokens_own(self):
        """The same rounding at a precision it cannot support is still a violation.

        ``1.5`` stands for [1.45, 1.55) and contains 1.4967. ``1.50`` stands for
        [1.495, 1.505) and does not, so the check tightens as the summary claims more
        precision rather than accepting any nearby figure.
        """
        self.assertEqual(check_numeric_containment("The ratio was 1.49.", self.cards), ["1.49"])

    def test_small_integer_prose_counts_are_accepted(self):
        """Counting the claims themselves needs no measurement to back it."""
        self.assertEqual(
            check_numeric_containment("1 of the 2 claims survived falsification.", self.cards), []
        )

    def test_an_integer_above_the_card_count_is_not_a_prose_count(self):
        """The allowance stops at the number of cards, so it cannot launder a figure."""
        blurb = "9 of the claims survived."
        self.assertEqual(check_numeric_containment(blurb, self.cards), ["9"])

    def test_a_number_only_in_a_decision_rule_is_accepted(self):
        """The pre-registered rule is part of the verified record, so quoting it is fine."""
        self.assertNotIn(
            3.25, [v for card in self.cards for v in card.observed.values()], "must not be observed"
        )
        self.assertEqual(
            check_numeric_containment("The rule demanded 3.25 percent.", self.cards), []
        )

    def test_percentages_of_a_fraction_are_accepted(self):
        """46.3% and 0.463 are the same measurement in different units."""
        blurb = "46.3% of the domain saturated."
        self.assertEqual(check_numeric_containment(blurb, self.cards), [])

    def test_a_percentage_that_matches_nothing_is_still_caught(self):
        """The unit conversion is not a licence to invent a percentage."""
        blurb = "81.2% of the domain saturated."
        self.assertEqual(check_numeric_containment(blurb, self.cards), ["81.2%"])

    def test_identifiers_are_not_numerals(self):
        """Digits inside names are names: H2, wss_dyn_cm2, flow_U0p0180."""
        blurb = "Card H2 tested wss_dyn_cm2 across flow_U0p0180 and found nothing new."
        self.assertEqual(check_numeric_containment(blurb, self.cards), [])

    def test_violations_are_reported_once_and_in_order(self):
        """A repeated bad figure is named once, in the order it first appeared."""
        blurb = "It rose 2.7 fold, then 3.9 fold, then 2.7 fold again."
        self.assertEqual(check_numeric_containment(blurb, self.cards), ["2.7", "3.9"])

    def test_a_blurb_with_no_numbers_is_always_clean(self):
        """Prose alone cannot fail a numeric check."""
        self.assertEqual(
            check_numeric_containment("Higher flow made the tissue stiffer.", self.cards), []
        )


class TestSynthesisPrompt(unittest.TestCase):
    """What the model is actually shown and told."""

    def setUp(self) -> None:
        """One passing card and one falsified card that was narrowed."""
        self.cards = [
            make_card(id="H1", passed=False, revised_by="H1b"),
            make_card(id="H1b", revision_of="H1", observed={"fraction": 0.463}),
        ]

    def test_prompt_carries_every_field_a_reader_would_need(self):
        """Id, claim, verdict, rule, primitive, params and all observed values."""
        prompt = build_synthesis_prompt("Why does flow matter?", self.cards)
        self.assertIn("Why does flow matter?", prompt)
        for fragment in ("H1", "H1b", "NARROWED", "SURVIVED", "compare_metric",
                         "wss_peak_dyn_cm2", "0.463", "1.4967", "SURVIVES if the ratio"):
            self.assertIn(fragment, prompt, fragment)

    def test_prompt_names_the_chain(self):
        """A narrowed claim is linked to its successor in both directions."""
        prompt = build_synthesis_prompt("q", self.cards)
        self.assertIn("was_narrowed_into", prompt)
        self.assertIn("narrows_falsified_claim", prompt)

    def test_system_prompt_states_the_rules_that_matter(self):
        """The four instructions the containment check cannot enforce are written down."""
        lowered = SYNTHESIS_SYSTEM_PROMPT.lower()
        self.assertIn("only numbers that appear in the 'observed'", lowered)
        self.assertIn("never contradict a verdict", lowered)
        self.assertIn("chain", lowered)
        self.assertIn("saturation", lowered)
        self.assertNotIn("json array", lowered)

    def test_retry_prompt_names_the_offending_numerals(self):
        """The one retry tells the model exactly what was rejected."""
        prompt = build_synthesis_prompt("q", self.cards, ["2.7", "81%"])
        self.assertIn("2.7", prompt)
        self.assertIn("81%", prompt)
        self.assertIn("rejected", prompt)


# --- Orchestration, with fakes that misbehave on purpose ---------------------------


class TestSynthesisFlow(unittest.TestCase):
    """A finished run, narrated by a backend that behaves, and by ones that do not."""

    def setUp(self) -> None:
        """Ten nodes, all above 0 Pa, and one claim that survives against them."""
        self.ds = StubDataset(np.linspace(1.0, 10.0, 10))
        self.proposals = [fraction_hypothesis("H1", 0.0, 0.0), fraction_hypothesis("H2", 5.0, 0.2)]

    def _run(self, generator: Any) -> list[Card]:
        """Run the loop and return finished cards."""
        result = investigate(self.ds, generator, "briefing", "the question", n=2)
        self.cards = cards_from_run(result.hypotheses, result.outcomes)
        self.result = result
        return self.cards

    def test_a_compliant_blurb_renders(self):
        """A summary whose numbers check out reaches the report."""
        generator = FakeGenerator(self.proposals, ["Both of the 2 claims survived testing."])
        cards = self._run(generator)
        outcome = synthesize(generator, "the question", cards)

        self.assertTrue(outcome.rendered)
        self.assertEqual(outcome.blurb, "Both of the 2 claims survived testing.")
        self.assertEqual(outcome.status, "rendered")
        self.assertFalse(outcome.suppressed)
        self.assertEqual(outcome.provenance(), {"synthesis": "rendered"})
        self.assertIn(outcome.blurb, render_markdown(cards, "q", {}, outcome.blurb))

    def test_an_invented_number_is_suppressed_and_leaves_the_cards_untouched(self):
        """The headline fault-injection case: a hallucinated figure loses the whole blurb.

        The retry is offered once and fails too, so nothing is rendered; the warning
        names the numeral; the provenance records the suppression; and the cards and
        the verdict tally are byte-identical to the same run with no summary at all.
        """
        control = self._run(MuteGenerator(self.proposals))
        control_json = json.dumps([asdict(card) for card in control], sort_keys=True)
        control_tally = counts(control)

        generator = FakeGenerator(
            self.proposals,
            ["EndMT rose 2.7 fold across the cushion.", "It still rose 2.7 fold."],
        )
        cards = self._run(generator)
        with self.assertLogs("llm_insights.agent.synthesis", level="WARNING") as logs:
            outcome = synthesize(generator, "the question", cards)

        self.assertFalse(outcome.rendered)
        self.assertIsNone(outcome.blurb)
        self.assertTrue(outcome.suppressed)
        self.assertEqual(outcome.violations, ("2.7",))
        self.assertTrue(outcome.retried)
        self.assertIn("2.7", "\n".join(logs.output))
        self.assertIn("suppressed", outcome.status)
        self.assertIn("2.7", outcome.status)
        self.assertEqual(outcome.provenance()["synthesis_violations"], "2.7")

        self.assertEqual(json.dumps([asdict(c) for c in cards], sort_keys=True), control_json)
        self.assertEqual(counts(cards), control_tally)
        self.assertNotIn("2.7", render_markdown(cards, "q", {}, outcome.blurb))
        self.assertNotIn("2.7", render_html(cards, "q", {}, outcome.blurb))

    def test_exactly_one_retry_is_ever_spent(self):
        """A second failure suppresses permanently rather than trying again."""
        generator = FakeGenerator(
            self.proposals, ["Rose 2.7 fold.", "Rose 3.9 fold.", "Rose 4.1 fold."]
        )
        cards = self._run(generator)
        outcome = synthesize(generator, "the question", cards)
        self.assertEqual(len(generator.synthesis_prompts), 2, "one attempt plus one retry")
        self.assertEqual(outcome.violations, ("3.9",), "the retry's own numerals are reported")
        self.assertFalse(outcome.rendered)

    def test_a_retry_that_complies_is_rendered_and_says_so(self):
        """The one retry is worth having: a corrected summary still reaches the report."""
        generator = FakeGenerator(
            self.proposals, ["Rose 2.7 fold.", "Both of the 2 claims survived."]
        )
        cards = self._run(generator)
        outcome = synthesize(generator, "the question", cards)
        self.assertTrue(outcome.rendered)
        self.assertTrue(outcome.retried)
        self.assertEqual(outcome.status, "rendered after one retry")
        self.assertIn("2.7", generator.synthesis_prompts[1], "the retry names what was rejected")

    def test_retry_can_be_switched_off(self):
        """With ``allow_retry`` False the first failure is final and costs one call."""
        generator = FakeGenerator(self.proposals, ["Rose 2.7 fold.", "Both of the 2 survived."])
        cards = self._run(generator)
        outcome = synthesize(generator, "the question", cards, allow_retry=False)
        self.assertEqual(len(generator.synthesis_prompts), 1)
        self.assertFalse(outcome.rendered)
        self.assertFalse(outcome.retried)

    def test_a_backend_that_raises_never_costs_a_verdict(self):
        """A generator failure is caught, recorded, and leaves every result standing."""
        generator = FakeGenerator(self.proposals, [RuntimeError("the CLI exited 1")])
        cards = self._run(generator)
        with self.assertLogs("llm_insights.agent.synthesis", level="WARNING"):
            outcome = synthesize(generator, "the question", cards)

        self.assertIsNone(outcome.blurb)
        self.assertIn("RuntimeError", outcome.status)
        self.assertIn("the CLI exited 1", outcome.status)
        self.assertEqual([c.id for c in cards], ["H1", "H2"])
        self.assertTrue(all(c.passed for c in cards))
        self.assertEqual(self.result.meta["n_survived"], 2)

    def test_a_backend_without_the_method_is_simply_skipped(self):
        """An older backend needs no changes; the section is omitted and said so."""
        generator = MuteGenerator(self.proposals)
        self.assertFalse(supports_synthesis(generator))
        cards = self._run(generator)
        outcome = synthesize(generator, "the question", cards)
        self.assertIsNone(outcome.blurb)
        self.assertFalse(outcome.attempted)
        self.assertIn("mute", outcome.status)
        self.assertNotIn(NARRATIVE_HEADING, render_markdown(cards, "q", {}, outcome.blurb))

    def test_an_empty_reply_renders_nothing(self):
        """A backend that returns nothing is not an error and is not a blurb either."""
        generator = FakeGenerator(self.proposals, [""])
        cards = self._run(generator)
        outcome = synthesize(generator, "the question", cards)
        self.assertIsNone(outcome.blurb)
        self.assertTrue(outcome.attempted)

    def test_fenced_prose_is_unwrapped(self):
        """A model that wraps its prose in a fence still gets rendered."""
        generator = FakeGenerator(self.proposals, ["```\nBoth of the 2 claims survived.\n```"])
        cards = self._run(generator)
        outcome = synthesize(generator, "the question", cards)
        self.assertEqual(outcome.blurb, "Both of the 2 claims survived.")

    def test_no_cards_means_no_call(self):
        """An empty run has nothing to narrate, and no allowance is spent finding out."""
        generator = FakeGenerator([], ["should never be asked for"])
        outcome = synthesize(generator, "q", [])
        self.assertEqual(generator.synthesis_prompts, [])
        self.assertIsNone(outcome.blurb)


# --- Rendering ---------------------------------------------------------------------


class TestRendering(unittest.TestCase):
    """The blurb must be visible, and unmistakably not a verdict."""

    def setUp(self) -> None:
        """One card and one compliant summary."""
        self.cards = [make_card()]
        self.blurb = "The single claim survived, with a ratio of 1.4967."

    def test_markdown_fences_the_blurb(self):
        """Heading, subline and the text itself, below the claims it narrates."""
        text = render_markdown(self.cards, "Q", {}, self.blurb)
        self.assertIn(f"## {NARRATIVE_HEADING}", text)
        self.assertIn(NARRATIVE_SUBLINE, text)
        self.assertIn(f"> {self.blurb}", text)
        self.assertLess(
            text.index("| Verdict | Count |"),
            text.index(NARRATIVE_HEADING),
            "the tally is above the narration",
        )
        self.assertLess(
            text.index("## Claims"),
            text.index(NARRATIVE_HEADING),
            "the narration sits below the claims it describes",
        )

    def test_markdown_without_a_blurb_is_unchanged(self):
        """Omitting the summary must not perturb the report that existed before it."""
        self.assertEqual(
            render_markdown(self.cards, "Q", {}), render_markdown(self.cards, "Q", {}, None)
        )
        self.assertNotIn(NARRATIVE_HEADING, render_markdown(self.cards, "Q", {}))

    def test_html_marks_it_as_narration(self):
        """A tinted, dashed callout with a label saying what it is."""
        doc = render_html(self.cards, "Q", {}, self.blurb)
        self.assertIn('class="narration"', doc)
        self.assertIn("Model-generated narration", doc)
        self.assertIn(self.blurb, doc)
        self.assertIn(".narration {", doc, "its styles ship with it")
        self.assertLess(doc.index('class="tally"'), doc.index('class="narration"'))
        self.assertLess(
            doc.index("</main>"),
            doc.index('class="narration"'),
            "the narration follows the cards, it does not head the report",
        )

    def test_html_without_a_blurb_carries_neither_block_nor_styles(self):
        """With no summary the document is byte-for-byte the one it was before."""
        doc = render_html(self.cards, "Q", {})
        self.assertNotIn("narration", doc)
        self.assertEqual(doc, render_html(self.cards, "Q", {}, None))

    def test_html_stays_self_contained(self):
        """No external asset may sneak in with the callout."""
        doc = render_html(self.cards, "Q", {}, self.blurb)
        for forbidden in ("http://", "https://", "<script", "<link", "<img"):
            self.assertNotIn(forbidden, doc, forbidden)

    def test_the_blurb_is_escaped(self):
        """Model text is untrusted input to the renderer like any other."""
        doc = render_html(self.cards, "Q", {}, "<script>alert(1)</script>")
        self.assertNotIn("<script>alert", doc)
        self.assertIn("&lt;script&gt;", doc)

    def test_the_verdict_line_and_tally_keep_their_place(self):
        """The summary is additive: nothing existing moves relative to anything else."""
        with_blurb = render_markdown(self.cards, "Q", {}, self.blurb)
        without = render_markdown(self.cards, "Q", {})
        for anchor in ("# Falsification report", "**Question.** Q", "| Verdict | Count |"):
            self.assertIn(anchor, with_blurb)
            self.assertIn(anchor, without)
        self.assertLess(with_blurb.index("**Question.**"), with_blurb.index(NARRATIVE_HEADING))

    def test_provenance_labels_the_synthesis_keys(self):
        """A suppressed summary is visible in the provenance block, not silent."""
        meta = {
            "synthesis": "suppressed: unsupported numerals (2.7)",
            "synthesis_violations": "2.7",
        }
        text = render_markdown(self.cards, "Q", meta)
        self.assertIn("Narrative summary", text)
        self.assertIn("suppressed", text)
        self.assertIn("2.7", text)


# --- Transcript round trip ---------------------------------------------------------


class TestTranscriptRoundTrip(unittest.TestCase):
    """A recorded summary must replay, and an old transcript must still load."""

    def setUp(self) -> None:
        """A dataset, a temporary directory, and one surviving claim."""
        self.ds = StubDataset(np.linspace(1.0, 10.0, 10))
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.proposals = [fraction_hypothesis("H1", 0.0, 0.0)]
        self.blurb = "The 1 claim proposed survived testing."

    def _record(self) -> Path:
        """Run once with a narrating fake and write the transcript."""
        generator = FakeGenerator(self.proposals, [self.blurb])
        result = investigate(self.ds, generator, "briefing", "the question", n=1)
        cards = cards_from_run(result.hypotheses, result.outcomes)
        outcome = synthesize(generator, result.question, cards)
        self.assertTrue(outcome.rendered, "the fixture blurb must pass the check")
        entry = transcript_entry(outcome, result.question, len(cards))
        return write_transcript(result, self.tmp / "transcript.json", entry)

    def test_a_recorded_blurb_replays_byte_for_byte(self):
        """The replay reproduces the recorded text, and re-checks it on the way."""
        path = self._record()
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "2")
        self.assertEqual([e["kind"] for e in payload["entries"]][-1], "synthesize")

        replayed = TranscriptGenerator(path)
        self.assertEqual(replayed.recorded_synthesis, self.blurb)
        result = investigate(self.ds, replayed, "briefing", "the question", n=1)
        cards = cards_from_run(result.hypotheses, result.outcomes)
        outcome = synthesize(replayed, result.question, cards)
        self.assertEqual(outcome.blurb, self.blurb)
        self.assertEqual(outcome.status, "rendered")

    def test_a_run_without_a_summary_writes_a_version_1_transcript(self):
        """The file a run has always written is unchanged when nothing was narrated."""
        result = investigate(self.ds, MuteGenerator(self.proposals), "b", "q", n=1)
        path = write_transcript(result, self.tmp / "plain.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "1")
        self.assertNotIn("synthesize", [e["kind"] for e in payload["entries"]])

    def test_a_suppressed_blurb_is_recorded_and_suppressed_again_on_replay(self):
        """A replay re-runs the check; being written down does not make it acceptable."""
        generator = FakeGenerator(self.proposals, ["Rose 2.7 fold.", "Still 2.7 fold."])
        result = investigate(self.ds, generator, "b", "the question", n=1)
        cards = cards_from_run(result.hypotheses, result.outcomes)
        outcome = synthesize(generator, result.question, cards)
        self.assertTrue(outcome.suppressed)
        path = write_transcript(
            result, self.tmp / "bad.json", transcript_entry(outcome, result.question, len(cards))
        )

        replayed = TranscriptGenerator(path)
        self.assertIn("2.7", str(replayed.recorded_synthesis))
        replay_outcome = synthesize(replayed, "the question", cards)
        self.assertIsNone(replay_outcome.blurb)
        self.assertTrue(replay_outcome.suppressed)

    def test_the_shipped_demo_transcript_still_replays(self):
        """The old-format file the demo runs on has no summary and loads unchanged."""
        self.assertTrue(DEMO_TRANSCRIPT.is_file(), DEMO_TRANSCRIPT)
        replayed = TranscriptGenerator(DEMO_TRANSCRIPT)
        self.assertIsNone(replayed.recorded_synthesis)
        proposals = replayed.propose("b", "q", 6)
        self.assertEqual([p["id"] for p in proposals], ["H1", "H2", "H3", "H4", "H5", "H6"])
        self.assertEqual(replayed.narrow("b", {"id": "H3"}, {})["id"], "H3b")
        self.assertIsNone(replayed.synthesize("ignored"))
        self.assertEqual(
            replayed.recorded_question,
            "What differs between the healthy and overflow cases, and why?",
        )

    def test_asking_an_old_transcript_to_narrate_is_not_an_error(self):
        """It renders nothing and says so, rather than failing the run."""
        replayed = TranscriptGenerator(DEMO_TRANSCRIPT)
        outcome = synthesize(replayed, "q", [make_card()])
        self.assertIsNone(outcome.blurb)
        self.assertTrue(outcome.attempted)
        self.assertIsNone(outcome.error)


# --- The CLI backend's guardrails --------------------------------------------------


def envelope(text: str, cost: float = 0.0161) -> str:
    """Build a CLI response envelope matching the real one's field names."""
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "num_turns": 1,
            "total_cost_usd": cost,
            "usage": {"input_tokens": 10, "output_tokens": 87},
            "result": text,
        }
    )


class TestCliSynthesis(unittest.TestCase):
    """One extra call, under exactly the guardrails every other call runs under."""

    def setUp(self) -> None:
        """Build a CLI generator whose binary resolves but is never executed."""
        supported_flags.cache_clear()
        self.addCleanup(supported_flags.cache_clear)
        which = mock.patch(
            "llm_insights.agent.subscription.shutil.which", return_value="/usr/bin/claude"
        )
        which.start()
        self.addCleanup(which.stop)
        help_run = mock.patch(
            "llm_insights.agent.subscription.subprocess.run",
            return_value=mock.Mock(stdout=CLI_HELP, stderr="", returncode=0),
        )
        help_run.start()
        self.addCleanup(help_run.stop)

        self.gen = ClaudeCLIGenerator(model="sonnet")
        self.addCleanup(self.gen.close)
        self.argvs: list[list[str]] = []
        self.ds = StubDataset(np.linspace(1.0, 10.0, 10))
        self.cards = [make_card()]

    def _serve(self, replies: dict[str, str]) -> None:
        """Install a transport that answers by inspecting the prompt it is given.

        Args:
            replies: Maps a marker found in the prompt to the ``result`` text.
        """

        def transport(argv: list[str], cwd: str) -> str:
            del cwd
            self.argvs.append(list(argv))
            prompt = argv[2]
            for marker, reply in replies.items():
                if marker in prompt:
                    return envelope(reply)
            raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

        self.gen.transport = transport

    def test_the_summary_call_replaces_the_json_system_prompt(self):
        """The hypothesis system prompt orders strict JSON, which would fight this call."""
        self._serve({"RESULTS (": "The 1 claim survived."})
        outcome = synthesize(self.gen, "q", self.cards)
        self.assertEqual(outcome.blurb, "The 1 claim survived.")
        argv = self.argvs[0]
        self.assertEqual(argv[argv.index("--system-prompt") + 1], SYNTHESIS_SYSTEM_PROMPT)

    def test_the_summary_call_keeps_every_guardrail(self):
        """One turn, no tools, the budget cap, and the empty scratch directory."""
        self._serve({"RESULTS (": "The 1 claim survived."})
        synthesize(self.gen, "q", self.cards)
        argv = self.argvs[0]
        self.assertEqual(argv[argv.index("--max-turns") + 1], "1")
        self.assertIn("--disallowedTools", argv)
        self.assertEqual(argv[argv.index("--max-budget-usd") + 1], "0.50")
        self.assertEqual(argv[argv.index("--model") + 1], "sonnet")

    def test_it_is_counted_against_the_call_ceiling(self):
        """The summary is a call like any other and is accounted like any other."""
        self._serve({"RESULTS (": "The 1 claim survived."})
        before = self.gen.calls
        synthesize(self.gen, "q", self.cards)
        self.assertEqual(self.gen.calls, before + 1)
        self.assertGreater(self.gen.cost_usd, 0.0)

    def test_it_cannot_breach_the_ceiling(self):
        """At the limit the call is refused before the subprocess, not after it.

        This is the property that makes the feature safe to add to a metered backend:
        the ceiling is checked first, so the worst case is a missing paragraph, never
        an extra call. The failure is caught and the verified results still stand.
        """
        self._serve({"RESULTS (": "The 1 claim survived."})
        self.gen.calls = self.gen.max_calls
        outcome = synthesize(self.gen, "q", self.cards)
        self.assertEqual(self.argvs, [], "no subprocess may be launched at the ceiling")
        self.assertEqual(self.gen.calls, self.gen.max_calls, "and no call may be counted")
        self.assertIn("BudgetExceededError", outcome.status)
        self.assertIsNone(outcome.blurb)

    def test_a_full_run_plus_a_summary_stays_inside_the_default_ceiling(self):
        """Worst case is one proposal, one narrowing per claim, and one summary.

        With the default ``--n 5`` that is seven calls against a ceiling of twelve, and
        the arithmetic is pinned here so raising ``--n`` past the ceiling is caught by a
        test rather than by a run stopping in front of an audience.
        """
        failing = [fraction_hypothesis(f"H{i}", 5.0, 0.9) for i in range(1, 6)]
        successor = fraction_hypothesis("N1", 5.0, 0.4)
        self._serve(
            {
                "Propose exactly": json.dumps(failing),
                "narrower claim": json.dumps(successor),
                "RESULTS (": "All 5 claims were narrowed.",
            }
        )
        result = investigate(self.ds, self.gen, "briefing", "q", n=5)
        cards = cards_from_run(result.hypotheses, result.outcomes)
        synthesize(self.gen, "q", cards)

        self.assertEqual(self.gen.calls, 7, "1 proposal + 5 narrowings + 1 summary")
        self.assertLessEqual(self.gen.calls, DEFAULT_MAX_CALLS)
        self.assertLessEqual(1 + 5 + 1, DEFAULT_MAX_CALLS)


# --- The --min-rel-diff floor ------------------------------------------------------


class TestMinRelDiffFloor(unittest.TestCase):
    """The operator may tighten a claim's noise guard, and may never loosen one."""

    def setUp(self) -> None:
        """Two cases whose peak differs by 1%."""
        self.ds = MetricDataset()
        self.values = {("wss_peak_dyn_cm2", "A"): 10.1, ("wss_peak_dyn_cm2", "B"): 10.0}

    def _run(self, proposal: dict[str, Any], floor: float) -> Any:
        """Run one hypothesis against the fake registry with the given floor."""
        with fake_metric_registry(self.values):
            return investigate(
                self.ds,
                MuteGenerator([proposal]),
                "briefing",
                "q",
                n=1,
                min_rel_diff_floor=floor,
            )

    def test_the_default_floor_changes_nothing(self):
        """A 1% difference passes when nothing is required of it."""
        result = self._run(metric_hypothesis("H1"), 0.0)
        self.assertTrue(result.outcomes[0].passed)
        self.assertAlmostEqual(result.outcomes[0].observed["rel_diff"], 0.01)
        self.assertEqual(result.outcomes[0].observed["min_rel_diff"], 0.0)
        self.assertNotIn(FLOOR_OBSERVED_KEY, result.outcomes[0].observed)
        self.assertNotIn("min_rel_diff_floor", result.meta)

    def test_a_floor_above_the_observed_difference_refutes_the_claim(self):
        """The same claim, the same data, and a margin the operator considers real."""
        result = self._run(metric_hypothesis("H1"), 0.05)
        outcome = result.outcomes[0]
        self.assertFalse(outcome.passed)
        self.assertIsNone(outcome.error, "a refutation, not a failure to run")
        self.assertEqual(outcome.observed["ordering_ok"], 1.0, "the ordering itself holds")
        self.assertEqual(outcome.observed["margin_ok"], 0.0)
        self.assertEqual(outcome.observed["min_rel_diff"], 0.05)

    def test_the_floor_is_visible_on_the_card_it_affected(self):
        """A reader must see that the stricter margin came from the operator."""
        result = self._run(metric_hypothesis("H1"), 0.05)
        cards = cards_from_run(result.hypotheses, result.outcomes)
        self.assertEqual(cards[0].observed[FLOOR_OBSERVED_KEY], 0.05)
        self.assertEqual(cards[0].params["min_rel_diff"], 0.05)
        self.assertIn(FLOOR_OBSERVED_KEY, render_markdown(cards, "q", result.meta))

    def test_a_stricter_model_margin_is_preserved(self):
        """It is a floor, not an override: the model may be stricter than the operator."""
        result = self._run(metric_hypothesis("H1", min_rel_diff=0.5), 0.05)
        outcome = result.outcomes[0]
        self.assertEqual(outcome.observed["min_rel_diff"], 0.5, "the model's own margin stands")
        self.assertNotIn(FLOOR_OBSERVED_KEY, outcome.observed)
        self.assertEqual(result.hypotheses[0].test.params["min_rel_diff"], 0.5)
        self.assertNotIn("min_rel_diff_floor_raised", result.meta)

    def test_the_floor_is_recorded_in_the_provenance(self):
        """Both the floor and the claims it bound are named in the report."""
        result = self._run(metric_hypothesis("H1"), 0.05)
        self.assertEqual(result.meta["min_rel_diff_floor"], 0.05)
        self.assertEqual(result.meta["min_rel_diff_floor_raised"], "H1")
        text = render_markdown([], "q", result.meta)
        self.assertIn("Operator min_rel_diff floor", text)
        self.assertIn("0.05", text)

    def test_the_floor_leaves_other_primitives_alone(self):
        """Only ``compare_metric`` has a noise guard, so only it is touched."""
        ds = StubDataset(np.linspace(1.0, 10.0, 10))
        result = investigate(
            ds,
            MuteGenerator([fraction_hypothesis("H1", 0.0, 0.0)]),
            "briefing",
            "q",
            n=1,
            min_rel_diff_floor=0.5,
        )
        self.assertTrue(result.outcomes[0].passed)
        self.assertNotIn("min_rel_diff", result.hypotheses[0].test.params)
        self.assertNotIn(FLOOR_OBSERVED_KEY, result.outcomes[0].observed)
        self.assertNotIn("min_rel_diff_floor_raised", result.meta)


if __name__ == "__main__":  # pragma: no cover - direct invocation
    unittest.main()


class TestSplitNarration(unittest.TestCase):
    """Parsing the two labelled parts out of one reply."""

    def test_both_labels_split_cleanly(self):
        """The ordinary case."""
        headline, summary = split_narration(
            "HEADLINE: Flow suppresses growth.\nSUMMARY: The longer version. And more."
        )
        self.assertEqual(headline, "Flow suppresses growth.")
        self.assertEqual(summary, "The longer version. And more.")

    def test_an_unlabelled_reply_becomes_the_summary(self):
        """A model that ignores the labels must not lose its work."""
        headline, summary = split_narration("just some prose with no labels")
        self.assertIsNone(headline)
        self.assertEqual(summary, "just some prose with no labels")

    def test_a_summary_label_alone_still_parses(self):
        """Half-compliance keeps the half that matters."""
        headline, summary = split_narration("SUMMARY: only a body here.")
        self.assertIsNone(headline)
        self.assertEqual(summary, "only a body here.")

    def test_a_headline_label_alone_is_treated_as_the_body(self):
        """With nothing to expand on, one paragraph is the summary, not a lede."""
        headline, summary = split_narration("HEADLINE: a lede with no body after it")
        self.assertIsNone(headline)
        self.assertEqual(summary, "a lede with no body after it")

    def test_labels_out_of_order_keep_the_summary(self):
        """A reply that leads with SUMMARY is not silently mangled."""
        headline, summary = split_narration("SUMMARY: the body.\nHEADLINE: the lede.")
        self.assertIsNone(headline)
        self.assertIn("the body.", summary)

    def test_multi_line_parts_are_collapsed(self):
        """Each part renders as one paragraph in both output formats."""
        headline, summary = split_narration(
            "HEADLINE: one\nline lede\nSUMMARY: a body\nover lines"
        )
        self.assertEqual(headline, "one line lede")
        self.assertEqual(summary, "a body over lines")


class TestHeadlineEndToEnd(unittest.TestCase):
    """The headline through synthesize() and out into both renderers."""

    def setUp(self) -> None:
        """One card, and a backend that returns a labelled reply."""
        self.cards = [make_card()]
        self.labelled = (
            "HEADLINE: The single claim held up.\n"
            "SUMMARY: The claim survived its pre-registered test at a ratio of 1.4967. "
            "Nothing was refuted. There is not much more to say."
        )

    def _generator(self, reply):
        """Return a stub backend whose synthesize() returns ``reply``."""

        class _Stub:
            name = "stub"

            def propose(self, briefing, question, n):
                return []

            def narrow(self, briefing, failed, outcome):
                return None

            def synthesize(self, prompt):
                del prompt
                return reply

        return _Stub()

    def test_synthesize_splits_the_reply(self):
        """Both parts come back on the result."""
        outcome = synthesize(self._generator(self.labelled), "Q", self.cards)
        self.assertEqual(outcome.headline, "The single claim held up.")
        self.assertIn("survived its pre-registered test", outcome.blurb)
        self.assertNotIn("HEADLINE", outcome.blurb)

    def test_an_unlabelled_reply_still_renders_a_summary(self):
        """Backwards compatible: the old shape keeps working, headline just absent."""
        outcome = synthesize(self._generator("A plain paragraph with 1.4967 in it."), "Q", self.cards)
        self.assertIsNone(outcome.headline)
        self.assertEqual(outcome.blurb, "A plain paragraph with 1.4967 in it.")

    def test_a_bad_number_anywhere_suppresses_both_parts(self):
        """The check runs over the whole reply; half a rejected reply is not shown."""
        outcome = synthesize(
            self._generator("HEADLINE: It held.\nSUMMARY: It held at p=0.003."),
            "Q",
            self.cards,
            allow_retry=False,
        )
        self.assertIsNone(outcome.blurb)
        self.assertIsNone(outcome.headline)
        self.assertTrue(outcome.suppressed)

    def test_a_number_in_the_headline_is_checked_too(self):
        """The lede is model prose like any other and gets no exemption."""
        outcome = synthesize(
            self._generator("HEADLINE: It held at 99.9 percent.\nSUMMARY: It held."),
            "Q",
            self.cards,
            allow_retry=False,
        )
        self.assertTrue(outcome.suppressed)
        self.assertIn("99.9", outcome.violations)

    def test_markdown_puts_the_headline_above_the_summary(self):
        """Order on the page: heading, lede, disclaimer, body."""
        text = render_markdown(self.cards, "Q", {}, "the body", "the lede")
        self.assertIn("**the lede**", text)
        self.assertLess(text.index("**the lede**"), text.index("the body"))
        self.assertLess(text.index(NARRATIVE_HEADING), text.index("**the lede**"))

    def test_markdown_without_a_headline_is_the_previous_output(self):
        """Omitting it must not perturb a report produced before the lede existed."""
        self.assertEqual(
            render_markdown(self.cards, "Q", {}, "the body"),
            render_markdown(self.cards, "Q", {}, "the body", None),
        )

    def test_html_renders_the_headline_in_its_own_element(self):
        """It is styled, escaped, and inside the narration callout."""
        doc = render_html(self.cards, "Q", {}, "the body", "the lede")
        self.assertIn('<p class="lede">the lede</p>', doc)
        self.assertIn(".narration .lede {", doc)
        self.assertLess(doc.index('class="lede"'), doc.index('class="blurb"'))

    def test_the_html_headline_is_escaped(self):
        """Model text is untrusted input to the renderer like any other."""
        doc = render_html(self.cards, "Q", {}, "the body", "<script>alert(1)</script>")
        self.assertNotIn("<script>alert", doc)

    def test_html_without_a_headline_is_the_previous_output(self):
        """Same additive guarantee as the markdown renderer."""
        self.assertEqual(
            render_html(self.cards, "Q", {}, "the body"),
            render_html(self.cards, "Q", {}, "the body", None),
        )

    def test_the_transcript_records_the_whole_reply_so_a_replay_resplits(self):
        """Storing the raw labelled text is what makes a replay reproduce both parts."""
        outcome = synthesize(self._generator(self.labelled), "Q", self.cards)
        entry = transcript_entry(outcome, "Q", len(self.cards))
        self.assertIn("HEADLINE:", entry["response"])
        self.assertEqual(split_narration(entry["response"])[0], outcome.headline)


class TestToleranceFloor(unittest.TestCase):
    """A verbatim quote of an observed value must survive re-computation.

    Regression for a real suppression: a summary quoted r = 0.6369388762466359 from a
    card computed with scipy.stats, and re-checking it against the same correlation
    computed in numpy gave 0.6369388762466373. Both correct, 2e-15 apart, and the
    displayed-precision tolerance of 5e-17 called it invented.
    """

    def setUp(self) -> None:
        """A card whose observed r is the numpy value."""
        self.cards = [make_card(observed={"r": 0.6369388762466373, "n_used": 8873.0})]

    def test_a_last_ulp_difference_is_tolerated(self):
        """The exact case that suppressed a real run."""
        blurb = "Mechanical input tracked EndMT (r = 0.6369388762466359 across 8873 points)."
        self.assertEqual(check_numeric_containment(blurb, self.cards), [])

    def test_a_rounded_quote_still_passes(self):
        """The original behaviour is unchanged: rounding is allowed."""
        self.assertEqual(check_numeric_containment("r was about 0.637.", self.cards), [])

    def test_an_invented_number_is_still_caught(self):
        """The floor must not turn the check off."""
        self.assertEqual(
            check_numeric_containment("Significant at p=0.003.", self.cards), ["0.003"]
        )

    def test_a_number_outside_the_floor_is_still_caught(self):
        """Nine significant figures of agreement is required, not four."""
        self.assertEqual(
            check_numeric_containment("r was 0.63693880 exactly.", self.cards),
            ["0.63693880"],
        )

    def test_the_floor_scales_with_magnitude(self):
        """A large value gets a proportionally larger absolute tolerance."""
        cards = [make_card(observed={"peak": 858.9761234567890})]
        self.assertEqual(
            check_numeric_containment("Peak stress reached 858.9761234567123 Pa.", cards), []
        )
        self.assertEqual(
            check_numeric_containment("Peak stress reached 858.9761000000000 Pa.", cards),
            ["858.9761000000000"],
        )


class TestTheHeadlineAnswersTheQuestion(unittest.TestCase):
    """The lede is the answer to what was asked, not a digest of the report.

    A run is started with a question on the command line. The report's own verdict
    table says what happened to each claim; what it never says in words is what the
    answer to that question turned out to be. That is the headline's job, and it is
    the only place in the whole report where the question gets answered in prose.
    """

    def test_the_system_prompt_frames_the_task_as_answering(self):
        """Framing first: the model is told a question was asked and is to answer it."""
        self.assertIn("answer that question", SYNTHESIS_SYSTEM_PROMPT)

    def test_the_headline_rule_demands_a_direct_answer(self):
        """Not 'the most important finding' -- the answer, in the question's terms."""
        self.assertIn("ANSWER THE QUESTION", SYNTHESIS_SYSTEM_PROMPT)
        self.assertIn("in its own terms", SYNTHESIS_SYSTEM_PROMPT)

    def test_the_headline_rule_rejects_a_report_digest(self):
        """The failure mode worth naming explicitly in the prompt."""
        self.assertIn("Five claims were tested and four survived", SYNTHESIS_SYSTEM_PROMPT)
        self.assertIn("Lead with the answer", SYNTHESIS_SYSTEM_PROMPT)

    def test_the_summary_carries_the_evidence_for_that_answer(self):
        """The long part supports the headline rather than restating it."""
        self.assertIn("evidence for that answer", SYNTHESIS_SYSTEM_PROMPT)

    def test_an_unanswerable_question_gets_an_honest_partial_answer(self):
        """Neither a confident answer the cards do not support, nor a refusal."""
        self.assertIn("not settled", SYNTHESIS_SYSTEM_PROMPT)
        self.assertIn("never a refusal to answer at all", SYNTHESIS_SYSTEM_PROMPT)

    def test_the_user_turn_leads_with_the_question(self):
        """The question must reach the model before the results it is answered from."""
        prompt = build_synthesis_prompt("Does flow suppress growth?", [make_card()])
        self.assertLess(prompt.index("QUESTION"), prompt.index("RESULTS ("))
        self.assertIn("Does flow suppress growth?", prompt)

    def test_the_user_turn_asks_for_the_answer_explicitly(self):
        """Restated at the point of asking, not only in the system prompt."""
        prompt = build_synthesis_prompt("Does flow suppress growth?", [make_card()])
        self.assertIn("Answer the question at the top of this message", prompt)
        self.assertIn("which answers it directly", prompt)

    def test_the_retry_prompt_still_asks_for_the_answer(self):
        """A retry must not quietly become a different task."""
        prompt = build_synthesis_prompt("Does flow suppress growth?", [make_card()], ["2.7"])
        self.assertIn("Answer the question at the top of this message", prompt)
        self.assertIn("2.7", prompt)

