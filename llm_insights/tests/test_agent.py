"""Tests for the agent layer: prompts, response parsing, backends, and the loop.

Written against :mod:`unittest` rather than pytest, because pytest is not installed
in the environment this project currently runs in. pytest collects ``TestCase``
classes unchanged, so these keep working once it is.

No test here makes a network call. The Anthropic backend is exercised through an
injected transport, and the assertions on it are about the shape of the request and
about the API key never escaping into a representation, a log, or run metadata.

Run with::

    cd src && python3 -m unittest discover -s ../tests -t .. -v
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from llm_insights.agent.generator import (
    ANTHROPIC_URL,
    ANTHROPIC_VERSION,
    API_KEY_ENV,
    AnthropicGenerator,
    EchoGenerator,
    TranscriptGenerator,
    build_system_prompt,
    build_user_prompt,
    make_generator,
    parse_response,
    parse_single,
)
from llm_insights.agent.loop import investigate, write_transcript
from llm_insights.cards.card import cards_from_run
from llm_insights.harness.primitives import PRIMITIVES
from llm_insights.io.dataset import NodeField

FAKE_KEY = "sk-ant-not-a-real-key-0123456789"


def setUpModule() -> None:
    """Silence the modules under test; these assertions are about return values."""
    logging.disable(logging.CRITICAL)


def tearDownModule() -> None:
    """Restore logging for anything that runs afterwards."""
    logging.disable(logging.NOTSET)


# --- Test doubles ------------------------------------------------------------------


class StubDataset:
    """A minimal stand-in for :class:`~llm_insights.io.dataset.Dataset`.

    Only the surface the ``fraction_above`` primitive touches is implemented, so the
    loop can be exercised end to end against real primitives without the multi-gigabyte
    results tree being present.
    """

    root = "/stub/dataset"

    def __init__(self, von_mises: np.ndarray) -> None:
        """Store the von Mises values this stub will serve."""
        self._von_mises = np.asarray(von_mises, dtype=float)

    @staticmethod
    def resolve_case(name: str) -> str:
        """Accept the three real condition labels and case names."""
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


def _hypothesis(hid: str, threshold: float, value: float, op: str = "gt") -> dict[str, Any]:
    """Build a schema-valid hypothesis mapping over ``fraction_above``."""
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


# --- parse_response ----------------------------------------------------------------


class ParseResponseTest(unittest.TestCase):
    """Every packaging a model plausibly wraps its JSON in."""

    def test_bare_json_array(self) -> None:
        """A plain array comes back as a list of mappings."""
        text = json.dumps([_hypothesis("H1", 0.0, 0.5), _hypothesis("H2", 1.0, 0.1)])
        items = parse_response(text)
        self.assertEqual([item["id"] for item in items], ["H1", "H2"])

    def test_json_inside_fences(self) -> None:
        """A ```json fenced block is unwrapped."""
        text = "```json\n" + json.dumps([_hypothesis("H1", 0.0, 0.5)]) + "\n```"
        self.assertEqual([item["id"] for item in parse_response(text)], ["H1"])

    def test_json_with_prose_around_it(self) -> None:
        """Prose before and after the array is ignored."""
        text = (
            "Here are the hypotheses I propose, based on the briefing:\n\n"
            + json.dumps([_hypothesis("H7", 2.0, 0.25)])
            + "\n\nI hope these are useful. Let me know if you want more."
        )
        items = parse_response(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "H7")

    def test_single_object_instead_of_array(self) -> None:
        """A lone object is returned as a one-element list."""
        items = parse_response(json.dumps(_hypothesis("H1", 0.0, 0.5)))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "H1")

    def test_object_wrapping_the_array(self) -> None:
        """An array nested under a 'hypotheses' key is unwrapped."""
        text = json.dumps({"hypotheses": [_hypothesis("H1", 0.0, 0.5)]})
        self.assertEqual([item["id"] for item in parse_response(text)], ["H1"])

    def test_garbage_raises_clearly(self) -> None:
        """Text with no JSON in it raises, and the message says why."""
        with self.assertRaises(ValueError) as caught:
            parse_response("I am afraid I cannot help with that request.")
        message = str(caught.exception)
        self.assertIn("no JSON", message)
        self.assertIn("I am afraid", message)

    def test_empty_response_raises(self) -> None:
        """An empty response raises rather than yielding an empty list."""
        with self.assertRaises(ValueError):
            parse_response("   ")

    def test_parse_single_accepts_null(self) -> None:
        """An explicit null from a narrowing request means 'no successor'."""
        self.assertIsNone(parse_single("null"))


# --- Prompts -----------------------------------------------------------------------


class PromptTest(unittest.TestCase):
    """The prompt must describe the code, not a memory of the code."""

    def test_system_prompt_names_every_primitive(self) -> None:
        """All six registered primitives appear by name. This is the drift guard."""
        prompt = build_system_prompt()
        self.assertEqual(len(PRIMITIVES), 6, "the prompt guard assumes six primitives")
        for name in PRIMITIVES:
            self.assertIn(name, prompt, f"{name} is registered but absent from the prompt")

    def test_system_prompt_lists_real_parameter_names(self) -> None:
        """Parameter names are introspected, so a signature change moves the prompt."""
        prompt = build_system_prompt()
        for param in ("min_rel_diff", "s_min", "s_max", "min_fraction", "field_x", "case_a"):
            self.assertIn(param, prompt)

    def test_system_prompt_states_the_rules(self) -> None:
        """The four rules the loop depends on are stated."""
        prompt = build_system_prompt().lower()
        self.assertIn("decision rule before", prompt)
        self.assertIn("qualitative", prompt)
        self.assertIn("strict json", prompt)

    def test_system_prompt_is_identical_across_calls(self) -> None:
        """Caching only hits if the system prompt is byte-identical every time."""
        self.assertEqual(build_system_prompt(), build_system_prompt())

    def test_user_prompt_carries_the_varying_parts(self) -> None:
        """Briefing, question and count go in the user turn, not the system turn."""
        text = build_user_prompt("BRIEF TEXT", "Why does the cushion grow?", 3)
        self.assertIn("BRIEF TEXT", text)
        self.assertIn("Why does the cushion grow?", text)
        self.assertIn("exactly 3", text)
        self.assertNotIn("BRIEF TEXT", build_system_prompt())


# --- AnthropicGenerator ------------------------------------------------------------


class AnthropicGeneratorTest(unittest.TestCase):
    """The HTTP backend, exercised without touching the network."""

    def test_missing_key_raises_naming_the_variable(self) -> None:
        """With the key unset, construction fails and names the variable."""
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaises(RuntimeError) as caught,
        ):
            AnthropicGenerator()
        self.assertIn(API_KEY_ENV, str(caught.exception))

    def test_request_shape(self) -> None:
        """The request carries the documented URL, headers and body."""
        captured: dict[str, Any] = {}

        def transport(url: str, payload: Any, headers: Any, timeout: float) -> dict:
            captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
            return {"content": [{"type": "text", "text": json.dumps([_hypothesis("H1", 0, 0.5)])}]}

        with mock.patch.dict(os.environ, {API_KEY_ENV: FAKE_KEY}, clear=True):
            generator = AnthropicGenerator(model="claude-test-1", max_tokens=555, temperature=0.25)
            generator.transport = transport
            items = generator.propose("BRIEFING", "QUESTION", 2)

        self.assertEqual([item["id"] for item in items], ["H1"])
        self.assertEqual(captured["url"], ANTHROPIC_URL)
        self.assertEqual(captured["headers"]["x-api-key"], FAKE_KEY)
        self.assertEqual(captured["headers"]["anthropic-version"], ANTHROPIC_VERSION)
        self.assertEqual(captured["headers"]["content-type"], "application/json")

        payload = captured["payload"]
        self.assertEqual(payload["model"], "claude-test-1")
        self.assertEqual(payload["max_tokens"], 555)
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertIn("BRIEFING", payload["messages"][0]["content"])
        self.assertIn("QUESTION", payload["messages"][0]["content"])
        self.assertEqual(payload["system"][0]["text"], build_system_prompt())
        self.assertEqual(payload["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertTrue(json.dumps(payload))  # the body must be JSON-serialisable

    def test_system_prompt_identical_across_two_calls(self) -> None:
        """Both calls send the same system block, which is what caching requires."""
        systems: list[str] = []

        def transport(url: str, payload: Any, headers: Any, timeout: float) -> dict:
            systems.append(payload["system"][0]["text"])
            return {"content": [{"type": "text", "text": json.dumps(_hypothesis("H1", 0, 0.5))}]}

        with mock.patch.dict(os.environ, {API_KEY_ENV: FAKE_KEY}, clear=True):
            generator = AnthropicGenerator()
            generator.transport = transport
            generator.propose("brief one", "question one", 1)
            generator.narrow("brief two", _hypothesis("H1", 0, 0.5), {"passed": False})

        self.assertEqual(len(systems), 2)
        self.assertEqual(systems[0], systems[1])

    def test_key_never_appears_in_repr_or_metadata(self) -> None:
        """The key is not in repr(), not on the instance, and not in run metadata."""
        with mock.patch.dict(os.environ, {API_KEY_ENV: FAKE_KEY}, clear=True):
            generator = AnthropicGenerator(model="claude-test-1")
            generator.transport = lambda url, payload, headers, timeout: {
                "content": [{"type": "text", "text": json.dumps([_hypothesis("H1", 0.0, 0.0)])}]
            }
            self.assertNotIn(FAKE_KEY, repr(generator))
            self.assertNotIn(FAKE_KEY, str(vars(generator)))
            result = investigate(
                StubDataset(np.array([1.0, 2.0, 3.0])),
                generator,
                "briefing",
                "question",
                n=1,
            )
        self.assertNotIn(FAKE_KEY, json.dumps(result.meta, default=str))
        self.assertNotIn(FAKE_KEY, json.dumps(result.transcript, default=str))
        self.assertEqual(result.meta["generator"], "anthropic:claude-test-1")
        self.assertEqual(result.meta["model"], "claude-test-1")

    def test_key_never_reaches_the_logs(self) -> None:
        """Nothing the backend logs while calling the API contains the key."""
        with mock.patch.dict(os.environ, {API_KEY_ENV: FAKE_KEY}, clear=True):
            generator = AnthropicGenerator()
            generator.transport = lambda url, payload, headers, timeout: {
                "content": [{"type": "text", "text": json.dumps([_hypothesis("H1", 0.0, 0.0)])}],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
            logging.disable(logging.NOTSET)
            try:
                with self.assertLogs("llm_insights.agent.generator", level="DEBUG") as logs:
                    generator.propose("briefing", "question", 1)
            finally:
                logging.disable(logging.CRITICAL)
        self.assertNotIn(FAKE_KEY, "\n".join(logs.output))

    def test_http_error_message_excludes_the_key(self) -> None:
        """A transport failure surfaces as RuntimeError without leaking the key."""

        def transport(url: str, payload: Any, headers: Any, timeout: float) -> dict:
            raise RuntimeError("the Anthropic API returned HTTP 401: invalid x-api-key")

        with mock.patch.dict(os.environ, {API_KEY_ENV: FAKE_KEY}, clear=True):
            generator = AnthropicGenerator()
            generator.transport = transport
            with self.assertRaises(RuntimeError) as caught:
                generator.propose("briefing", "question", 1)
        self.assertNotIn(FAKE_KEY, str(caught.exception))


# --- The loop ----------------------------------------------------------------------


class LoopTest(unittest.TestCase):
    """End-to-end behaviour of :func:`investigate`."""

    def setUp(self) -> None:
        """Ten nodes, all above 0 Pa and all below 100 Pa."""
        self.ds = StubDataset(np.linspace(1.0, 10.0, 10))

    def test_echo_run_produces_cards(self) -> None:
        """A full run with the offline stub yields one card per admissible claim."""
        result = investigate(self.ds, EchoGenerator(), "briefing", "the question", n=3)
        self.assertEqual(len(result.hypotheses), 3)
        self.assertEqual(len(result.outcomes), 3)
        self.assertEqual(result.meta["n_proposed"], 3)
        self.assertEqual(result.meta["n_admissible"], 3)
        self.assertEqual(result.meta["n_rejected"], 0)
        self.assertEqual(result.meta["n_survived"], 3)
        self.assertEqual(result.meta["generator"], "echo")
        self.assertEqual(result.meta["dataset_root"], "/stub/dataset")
        self.assertEqual(result.meta["schema_version"], "1")
        self.assertTrue(result.meta["generated_at"].endswith("Z"))

        cards = cards_from_run(result.hypotheses, result.outcomes)
        self.assertEqual(len(cards), 3)
        self.assertTrue(all(card.passed for card in cards))
        self.assertTrue(all(card.observed for card in cards))

    def test_invalid_proposal_is_recorded_not_dropped(self) -> None:
        """A malformed claim is counted as rejected and its reason is kept."""
        good = _hypothesis("H1", 0.0, 0.0)
        bad = {"id": "H2", "claim": "no rationale, no test"}

        class TwoItemGenerator:
            name = "fixture"

            def propose(self, briefing: str, question: str, n: int) -> list[dict]:
                return [good, bad]

            def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
                return None

        result = investigate(self.ds, TwoItemGenerator(), "briefing", "q", n=2)
        self.assertEqual(result.meta["n_proposed"], 2)
        self.assertEqual(result.meta["n_admissible"], 1)
        self.assertEqual(result.meta["n_rejected"], 1)
        self.assertIn("rationale", result.meta["rejection_reasons"])
        rejections = [e for e in result.transcript if e["kind"] == "rejected"]
        self.assertEqual(len(rejections), 1)
        self.assertEqual(rejections[0]["raw"], bad)
        self.assertEqual([h.id for h in result.hypotheses], ["H1"])

    def test_failure_triggers_exactly_one_narrowing(self) -> None:
        """A refuted claim gets one successor, linked by parent_id, and stops there."""
        failing = _hypothesis("H1", 5.0, 0.9)  # the fraction above 5 Pa is 0.5, not > 0.9
        successor = _hypothesis("H1b", 5.0, 0.4)
        successor["claim"] = "Most of the cushion is stressed."
        calls: list[str] = []

        class NarrowingGenerator:
            name = "fixture"

            def propose(self, briefing: str, question: str, n: int) -> list[dict]:
                return [failing]

            def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
                calls.append(str(failed["id"]))
                # The observed numbers must reach the generator, or narrowing is blind.
                assert "fraction" in outcome["observed"]
                return json.loads(json.dumps(successor))

        result = investigate(self.ds, NarrowingGenerator(), "briefing", "q", n=1)

        self.assertEqual(calls, ["H1"], "exactly one narrowing round should have run")
        self.assertEqual([h.id for h in result.hypotheses], ["H1", "H1b"])
        self.assertIsNone(result.hypotheses[0].parent_id)
        self.assertEqual(result.hypotheses[1].parent_id, "H1")
        self.assertFalse(result.outcomes[0].passed)
        self.assertTrue(result.outcomes[1].passed)
        self.assertEqual(result.meta["n_narrowed"], 1)
        self.assertEqual(result.meta["n_falsified"], 0)
        self.assertEqual(result.meta["n_survived"], 1)
        self.assertEqual(result.meta["n_proposed"], 2)

        cards = {card.id: card for card in cards_from_run(result.hypotheses, result.outcomes)}
        self.assertEqual(cards["H1"].revised_by, "H1b")
        self.assertEqual(cards["H1b"].revision_of, "H1")

    def test_could_not_run_is_not_narrowed(self) -> None:
        """A claim that could not be evaluated is never treated as a refutation."""
        broken = _hypothesis("H1", 0.0, 0.5)
        broken["test"]["params"]["case"] = "Nonexistent"
        narrow_calls: list[str] = []

        class Generator:
            name = "fixture"

            def propose(self, briefing: str, question: str, n: int) -> list[dict]:
                return [broken]

            def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
                narrow_calls.append(str(failed["id"]))
                return None

        result = investigate(self.ds, Generator(), "briefing", "q", n=1)
        self.assertEqual(narrow_calls, [])
        self.assertEqual(result.meta["n_could_not_run"], 1)
        self.assertEqual(result.meta["n_falsified"], 0)
        self.assertIsNotNone(result.outcomes[0].error)

    def test_narrowing_can_be_switched_off(self) -> None:
        """With narrow_failures False, a refuted claim stands alone."""
        failing = _hypothesis("H1", 5.0, 0.9)

        class Generator:
            name = "fixture"

            def propose(self, briefing: str, question: str, n: int) -> list[dict]:
                return [failing]

            def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
                raise AssertionError("narrow must not be called")

        result = investigate(self.ds, Generator(), "briefing", "q", n=1, narrow_failures=False)
        self.assertEqual(len(result.hypotheses), 1)
        self.assertEqual(result.meta["n_falsified"], 1)
        self.assertEqual(result.meta["n_narrowed"], 0)

    def test_counts_partition_the_admissible_claims(self) -> None:
        """Survived + falsified + narrowed + could-not-run equals admissible."""
        result = investigate(self.ds, EchoGenerator(), "briefing", "q", n=4)
        meta = result.meta
        total = (
            meta["n_survived"] + meta["n_falsified"] + meta["n_narrowed"] + meta["n_could_not_run"]
        )
        self.assertEqual(total, meta["n_admissible"])


# --- TranscriptGenerator -----------------------------------------------------------


class TranscriptTest(unittest.TestCase):
    """A recorded run must replay to the same hypotheses and the same verdicts."""

    def setUp(self) -> None:
        """A dataset and a temporary directory for the transcript."""
        self.ds = StubDataset(np.linspace(1.0, 10.0, 10))
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _record(self) -> Path:
        """Run once with a fixture generator that both fails and narrows, and save it."""
        failing = _hypothesis("H1", 5.0, 0.9)
        surviving = _hypothesis("H2", 0.0, 0.0)
        invalid = {"id": "H3", "claim": "malformed", "rationale": "x"}
        successor = _hypothesis("H1b", 5.0, 0.4)

        class Fixture:
            name = "fixture:v1"
            model = "fixture-model"

            def propose(self, briefing: str, question: str, n: int) -> list[dict]:
                return [failing, surviving, invalid]

            def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
                return json.loads(json.dumps(successor))

        self.original = investigate(self.ds, Fixture(), "briefing", "the question", n=3)
        return write_transcript(self.original, self.tmp / "transcript.json")

    def test_replay_reproduces_the_run(self) -> None:
        """Replaying the transcript gives identical claims, outcomes and counts."""
        path = self._record()
        replay = investigate(self.ds, TranscriptGenerator(path), "briefing", "the question", n=3)

        self.assertEqual(
            [h.id for h in replay.hypotheses], [h.id for h in self.original.hypotheses]
        )
        self.assertEqual(
            [h.claim for h in replay.hypotheses], [h.claim for h in self.original.hypotheses]
        )
        self.assertEqual(
            [h.parent_id for h in replay.hypotheses],
            [h.parent_id for h in self.original.hypotheses],
        )
        self.assertEqual(
            [(o.passed, o.summary) for o in replay.outcomes],
            [(o.passed, o.summary) for o in self.original.outcomes],
        )
        for key in ("n_proposed", "n_admissible", "n_rejected", "n_survived", "n_narrowed"):
            self.assertEqual(replay.meta[key], self.original.meta[key], key)

    def test_replay_names_its_source(self) -> None:
        """The replay is labelled as a replay and records the original backend."""
        path = self._record()
        generator = TranscriptGenerator(path)
        self.assertEqual(generator.name, f"transcript:{path}")
        self.assertEqual(generator.source_model, "fixture-model")

        replay = investigate(self.ds, generator, "briefing", "the question", n=3)
        self.assertTrue(replay.meta["generator"].startswith("transcript:"))
        self.assertIn("fixture:v1", replay.meta["replayed_from"])
        self.assertEqual(replay.meta["model"], "fixture-model")

    def test_bare_entry_list_is_accepted(self) -> None:
        """A transcript stored as a plain list of entries still loads."""
        path = self.tmp / "bare.json"
        entries = [{"kind": "propose", "response": [_hypothesis("H1", 0.0, 0.0)]}]
        path.write_text(json.dumps(entries), encoding="utf-8")
        result = investigate(self.ds, TranscriptGenerator(path), "b", "q", n=1)
        self.assertEqual([h.id for h in result.hypotheses], ["H1"])

    def test_exhausted_transcript_raises_clearly(self) -> None:
        """Asking for a round the transcript does not have is an error, not silence."""
        path = self.tmp / "empty.json"
        path.write_text(json.dumps([]), encoding="utf-8")
        generator = TranscriptGenerator(path)
        with self.assertRaises(RuntimeError) as caught:
            generator.propose("b", "q", 1)
        self.assertIn(str(path), str(caught.exception))

    def test_non_transcript_file_raises(self) -> None:
        """A JSON file that is not a transcript is rejected by name."""
        path = self.tmp / "not_a_transcript.json"
        path.write_text(json.dumps({"something": "else"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            TranscriptGenerator(path)


# --- Factory -----------------------------------------------------------------------


class FactoryTest(unittest.TestCase):
    """The CLI's generator factory."""

    def test_echo(self) -> None:
        """The echo backend needs no arguments."""
        self.assertIsInstance(make_generator("echo"), EchoGenerator)

    def test_transcript_requires_a_path(self) -> None:
        """Selecting transcript without a file is a usage error."""
        with self.assertRaises(ValueError) as caught:
            make_generator("transcript")
        self.assertIn("--transcript", str(caught.exception))

    def test_unknown_kind(self) -> None:
        """An unknown backend name is rejected."""
        with self.assertRaises(ValueError):
            make_generator("openai")


if __name__ == "__main__":  # pragma: no cover - direct invocation
    unittest.main()
