"""The callable API: build-once, remember-across-questions, and CLI parity.

Four properties are asserted here, and each one is a thing that would otherwise fail
silently:

* the dataset and the briefing are built **once** per study, not once per question;
* a follow-up is told what is already settled, and a fresh ask is not;
* two questions in one study cannot hand two claims the same id;
* the cards the API produces are the cards the CLI produces, field for field.

The last of those is the one that matters most. :mod:`llm_insights.agent.run` is now a
wrapper around :class:`llm_insights.api.Study`, and the whole justification for that
refactor is that observable behaviour did not change. So it is tested against the
shipped demo transcript rather than asserted in a docstring.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import llm_insights.summary.briefing as briefing_module
from llm_insights import api
from llm_insights.agent import run as run_cli
from llm_insights.agent.generator import (
    ECHO_TEMPLATE,
    TranscriptGenerator,
    build_followup_prompt,
)
from llm_insights.cards.card import cards_from_json

REPO = Path(__file__).resolve().parents[1]
DEMO_TRANSCRIPT = REPO / "data" / "demo_transcript.json"
HAS_DATA = (api.DEFAULT_ROOT / "FSG Results").is_dir()
DEMO_QUESTION = "What differs between the healthy and overflow cases, and why?"


class RecordingGenerator:
    """A stub that records every prompt it was given and returns fixed claims.

    Args:
        n_claims: How many distinct claims each ``propose`` returns.
    """

    def __init__(self, n_claims: int = 2) -> None:
        """Initialise the stub with empty prompt logs."""
        self.name = "recording"
        self.n_claims = int(n_claims)
        self.questions: list[str] = []
        self.briefings: list[str] = []
        self.closed = 0
        self._round = 0

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Record the prompt and return ``n_claims`` claims with round-scoped ids.

        Args:
            briefing: The briefing text, recorded.
            question: The question text, recorded. This is what a follow-up decorates.
            n: Ignored; the stub returns a fixed number of claims.

        Returns:
            The stub claims.
        """
        del n
        self.briefings.append(briefing)
        self.questions.append(question)
        self._round += 1
        out: list[dict] = []
        for index in range(self.n_claims):
            item = json.loads(json.dumps(ECHO_TEMPLATE))
            # Every round proposes the SAME ids on purpose: the study is what has to
            # keep them apart, and a stub that helpfully varied them would hide the bug.
            item["id"] = f"H{index + 1}"
            item["claim"] = f"round {self._round} claim {index + 1}"
            out.append(item)
        return out

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Decline to narrow; narrowing is not what these tests are about.

        Args:
            briefing: Ignored.
            failed: Ignored.
            outcome: Ignored.

        Returns:
            None, always.
        """
        del briefing, failed, outcome
        return None

    def close(self) -> None:
        """Count the close, so propagation through the study can be asserted."""
        self.closed += 1


class CountingDataset:
    """A dataset stand-in that counts how many times it was constructed.

    Args:
        root: Whatever root the study was given. Not read.
    """

    constructed = 0

    def __init__(self, root: Any) -> None:
        """Record the root and count this construction."""
        self.root = root
        CountingDataset.constructed += 1


class _FakeDatasetMixin:
    """Swaps :class:`llm_insights.api.Dataset` for a counting stand-in."""

    def install_fake_dataset(self) -> None:
        """Replace the real Dataset and reset the construction counter."""
        CountingDataset.constructed = 0
        self._real_dataset = api.Dataset
        api.Dataset = CountingDataset  # type: ignore[misc]

    def restore_dataset(self) -> None:
        """Put the real Dataset back."""
        api.Dataset = self._real_dataset  # type: ignore[misc]


class TestBuiltOnce(unittest.TestCase, _FakeDatasetMixin):
    """The expensive halves of a run are paid for once per study."""

    def setUp(self) -> None:
        """Install the counting dataset."""
        self.install_fake_dataset()

    def tearDown(self) -> None:
        """Restore the real dataset."""
        self.restore_dataset()

    def test_dataset_and_briefing_are_built_once_across_three_asks(self) -> None:
        """Three questions in one study construct one dataset and one briefing."""
        generator = RecordingGenerator()
        study = api.open_study("/nowhere", generator=generator)
        builds: list[str] = []
        real_build = briefing_module.build_briefing

        def counting_build(ds: Any, *args: Any, **kwargs: Any) -> str:
            """Count a briefing build and return a fixed string."""
            del args, kwargs
            builds.append(str(ds.root))
            return "BRIEFING TEXT"

        briefing_module.build_briefing = counting_build  # type: ignore[assignment]
        try:
            for index in range(3):
                study.ask(f"question {index}", n=2)
        finally:
            briefing_module.build_briefing = real_build  # type: ignore[assignment]

        self.assertEqual(CountingDataset.constructed, 1, "the dataset was rebuilt")
        self.assertEqual(len(builds), 1, f"the briefing was built {len(builds)} times")
        self.assertEqual(len(study.history), 3)
        self.assertEqual(set(generator.briefings), {"BRIEFING TEXT"})

    def test_a_study_that_is_never_asked_builds_nothing(self) -> None:
        """Constructing a study is free; nothing is read until it is used."""
        api.open_study("/nowhere", generator=RecordingGenerator())
        self.assertEqual(CountingDataset.constructed, 0)


class TestFollowUpMemory(unittest.TestCase, _FakeDatasetMixin):
    """A follow-up carries prior work; a fresh ask does not."""

    def setUp(self) -> None:
        """Install the counting dataset and a study with a canned briefing."""
        self.install_fake_dataset()
        self.study = api.open_study("/nowhere", generator=RecordingGenerator())
        # Pre-seed the cache so these tests never reach the summary package.
        self.study._briefing = "BRIEFING"

    def tearDown(self) -> None:
        """Restore the real dataset."""
        self.restore_dataset()

    def test_prior_ids_appear_in_a_follow_up_and_not_in_a_fresh_ask(self) -> None:
        """The settled claims reach a follow-up's prompt, and only a follow-up's."""
        generator: RecordingGenerator = self.study.generator
        first = self.study.ask("what happens?", n=2)
        prior_ids = [card.id for card in first.cards]
        # The stub numbers its claims H1, H2 on every call, as a real backend does; the
        # study stamps the question number on so the second question is not lost to a
        # duplicate-id rejection. See tests/test_explore_ids.py.
        self.assertEqual(prior_ids, ["q01_H1", "q01_H2"])

        self.study.follow_up("why did that happen?", n=2)
        follow_up_prompt = generator.questions[-1]
        for hypothesis_id in prior_ids:
            self.assertIn(hypothesis_id, follow_up_prompt)
        self.assertIn("round 1 claim 1", follow_up_prompt)
        self.assertIn("why did that happen?", follow_up_prompt)

        self.study.ask("a completely fresh question", n=2)
        fresh_prompt = generator.questions[-1]
        self.assertEqual(fresh_prompt, "a completely fresh question")
        for hypothesis_id in prior_ids:
            self.assertNotIn(f"- {hypothesis_id} [", fresh_prompt)

    def test_the_question_recorded_in_meta_is_the_one_the_human_asked(self) -> None:
        """Decorating the backend's prompt must not change the report's header."""
        self.study.ask("first", n=1)
        answer = self.study.follow_up("second", n=1)
        self.assertEqual(answer.question, "second")
        self.assertEqual(answer.meta["question"], "second")

    def test_ids_are_unique_across_two_questions_in_one_study(self) -> None:
        """A second question cannot reuse a spent id, and is not thrown away for trying.

        The stub reproposes H1 and H2 on the second question, which is what a real
        backend does: each proposal call is independent and has no memory of the ids the
        last one spent. Both claims survive, under ids stamped with the question that
        asked for them, so the study still holds one card per id and a card still names
        where it came from.
        """
        self.study.ask("first", n=2)
        self.study.follow_up("second", n=2)
        ids = [card.id for card in self.study.cards]
        self.assertEqual(len(ids), len(set(ids)), f"duplicate ids: {ids}")
        self.assertEqual(ids, ["q01_H1", "q01_H2", "q02_H1", "q02_H2"])
        self.assertEqual(self.study.history[1].meta["n_rejected"], 0)
        self.assertEqual(self.study.history[1].meta["n_admissible"], 2)

    def test_history_and_cards_accumulate(self) -> None:
        """Both accessors report what the study has done so far."""
        self.study.ask("first", n=2)
        self.assertEqual(len(self.study.history), 1)
        self.assertEqual(len(self.study.cards), 2)

    def test_close_propagates_to_the_generator(self) -> None:
        """Closing a study closes its backend exactly once."""
        generator: RecordingGenerator = self.study.generator
        self.study.close()
        self.assertEqual(generator.closed, 1)

    def test_context_manager_closes_the_generator(self) -> None:
        """Leaving a ``with`` block closes the backend."""
        generator = RecordingGenerator()
        with api.open_study("/nowhere", generator=generator) as study:
            self.assertIsInstance(study, api.Study)
        self.assertEqual(generator.closed, 1)

    def test_a_study_without_a_generator_refuses_to_ask(self) -> None:
        """A missing backend is an error at ask time, naming the fix."""
        study = api.open_study("/nowhere")
        with self.assertRaises(RuntimeError) as ctx:
            study.ask("anything")
        self.assertIn("no generator", str(ctx.exception))


class TestFollowUpPrompt(unittest.TestCase):
    """The prompt builder itself, independent of any study."""

    def test_no_prior_work_leaves_the_question_untouched(self) -> None:
        """The first question of a session is never decorated."""
        self.assertEqual(build_followup_prompt("plain question", []), "plain question")

    def test_prior_work_names_ids_claims_and_verdicts(self) -> None:
        """A follow-up prompt carries what is settled, and the question itself."""
        text = build_followup_prompt(
            "why?",
            [{"id": "H7", "claim": "stress rises downstream", "verdict": "FALSIFIED"}],
        )
        self.assertIn("H7", text)
        self.assertIn("stress rises downstream", text)
        self.assertIn("FALSIFIED", text)
        self.assertIn("why?", text)

    def test_observed_numbers_are_not_leaked_into_the_prompt(self) -> None:
        """Telling the model an answer would turn its next test into a lookup."""
        text = build_followup_prompt(
            "why?",
            [{"id": "H7", "claim": "a claim", "verdict": "SURVIVED", "observed": {"x": 12345.0}}],
        )
        self.assertNotIn("12345", text)


class TestBriefingFile(unittest.TestCase):
    """``briefing_file`` short-circuits the summary package entirely."""

    def test_briefing_is_read_from_the_file_and_only_once(self) -> None:
        """The file is read on first access and cached thereafter."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "briefing.md"
            path.write_text("FROM FILE", encoding="utf-8")
            study = api.open_study("/nowhere", briefing_file=path)
            self.assertEqual(study.briefing, "FROM FILE")
            path.write_text("CHANGED ON DISK", encoding="utf-8")
            self.assertEqual(study.briefing, "FROM FILE", "the briefing was re-read")


@unittest.skipUnless(HAS_DATA, "the FSG dataset is not present")
@unittest.skipUnless(DEMO_TRANSCRIPT.is_file(), "the demo transcript is not present")
class TestCLIParity(unittest.TestCase):
    """The API and the CLI produce the same cards, field for field.

    This is the acceptance gate for making :mod:`llm_insights.agent.run` a wrapper
    around :class:`llm_insights.api.Study`.
    """

    def test_api_cards_equal_cli_cards(self) -> None:
        """Every field of every card matches, through both entry points."""
        study = api.open_study(api.DEFAULT_ROOT, generator=TranscriptGenerator(DEMO_TRANSCRIPT))
        answer = study.ask(DEMO_QUESTION, n=5)
        study.close()

        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                status = run_cli.main(
                    [
                        "--generator",
                        "transcript",
                        "--transcript",
                        str(DEMO_TRANSCRIPT),
                        "--question",
                        DEMO_QUESTION,
                        "--n",
                        "5",
                        "--out",
                        tmp,
                        "--root",
                        str(api.DEFAULT_ROOT),
                        "--log-level",
                        "ERROR",
                    ]
                )
            self.assertEqual(status, 0, buffer.getvalue())
            cli_cards = cards_from_json(Path(tmp) / "cards.json")

        self.assertEqual(len(cli_cards), len(answer.cards))
        self.assertGreater(len(cli_cards), 0, "the replay produced no cards to compare")
        for from_cli, from_api in zip(cli_cards, answer.cards, strict=True):
            self.assertEqual(asdict(from_cli), asdict(from_api))

    def test_the_cli_still_writes_all_four_artifacts(self) -> None:
        """The four files and the four stdout labels are unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                status = run_cli.main(
                    [
                        "--generator",
                        "transcript",
                        "--transcript",
                        str(DEMO_TRANSCRIPT),
                        "--out",
                        tmp,
                        "--root",
                        str(api.DEFAULT_ROOT),
                        "--log-level",
                        "ERROR",
                    ]
                )
            self.assertEqual(status, 0, buffer.getvalue())
            for name in ("cards.json", "report.md", "report.html", "transcript.json"):
                self.assertTrue((Path(tmp) / name).is_file(), f"{name} was not written")
            stdout = buffer.getvalue()
            for label in ("cards:", "markdown:", "html:", "transcript:"):
                self.assertIn(label, stdout)

    def test_a_bad_root_still_exits_two_before_any_backend_is_built(self) -> None:
        """A bad ``--root`` fails before anything that could cost money."""
        buffer = io.StringIO()
        with redirect_stdout(buffer), self.assertRaises(SystemExit) as ctx:
            run_cli.main(
                ["--root", "/definitely/not/here", "--skip-preflight", "--log-level", "ERROR"]
            )
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":  # pragma: no cover - test entry point
    unittest.main()
