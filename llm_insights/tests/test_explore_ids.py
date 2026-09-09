"""Hypothesis ids must survive a multi-question run.

``api.Study`` keeps one ``_seen_ids`` set for a whole study, and ``loop._validate``
rejects a repeated id at the *propose* stage. Every proposal call is independent, so a
model has no way to know which ids a previous question spent and reliably restarts at
``H1``. The result, measured on two live runs on 2026-09-09, is that question 1 produces
cards and every question after it produces none.

The stub here reproduces exactly that: it always returns ``H1``..``H5``, whatever it is
asked. What is under test is not the stub's behaviour but the loop's — a run that asks
three questions must end with cards for all three, with every id still unique across the
run and every card still traceable to the claim that produced it.

The dataset is a stand-in on which every primitive fails. That is deliberate: this is a
test about id bookkeeping at the propose stage, and a claim that could not run is still a
card the loop has to account for, so the assertion holds without the 2.1 GB results tree.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_insights import api

#: How many questions the regression asks. Three, not two, so a fix that merely made the
#: *second* question work would still be caught.
N_QUESTIONS = 3

#: How many hypotheses the stub returns per question.
N_PER_QUESTION = 5


def _hypothesis(hid: str, threshold: float) -> dict[str, Any]:
    """Build a schema-valid hypothesis mapping over ``fraction_above``.

    Args:
        hid: The id the stub hands back. Always one of ``H1``..``H5``.
        threshold: The stress threshold, varied only so the five claims differ in text.

    Returns:
        The raw mapping a generator would return.
    """
    return {
        "id": hid,
        "claim": f"More than a tenth of the cushion exceeds {threshold:g} Pa.",
        "rationale": "Fixture claim; its verdict is irrelevant to this test.",
        "test": {
            "primitive": "fraction_above",
            "params": {
                "field": "von_mises",
                "case": "Healthy",
                "threshold": threshold,
                "op": "gt",
                "value": 0.1,
            },
            "decision_rule": "Refuted if the fraction is at most 0.1.",
        },
    }


class RestartingGenerator:
    """A backend that always numbers its claims from ``H1``, as a real model does.

    Each ``propose`` call is a fresh request with no memory of the last one, so the ids
    it returns are only ever unique *within* one question. Nothing about this is
    misbehaviour, which is the point: the loop must cope with it.

    Attributes:
        name: Provenance string, as the :class:`~llm_insights.agent.generator.Generator`
            protocol requires.
        propose_calls: How many times :meth:`propose` was called.
    """

    name = "stub:restarting-ids"

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.propose_calls = 0

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Return ``H1``..``H5``, ignoring the briefing and the question.

        Args:
            briefing: Unused.
            question: Unused.
            n: How many claims were asked for; honoured up to five.

        Returns:
            The raw hypothesis mappings.
        """
        del briefing, question
        self.propose_calls += 1
        return [_hypothesis(f"H{i}", 100.0 * i) for i in range(1, min(int(n), 5) + 1)]

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Decline to narrow, so the test measures the proposal path alone.

        Args:
            briefing: Unused.
            failed: Unused.
            outcome: Unused.

        Returns:
            None, always.
        """
        del briefing, failed, outcome
        return None


class EchoingNarrower(RestartingGenerator):
    """A backend whose successor copies the id spelling it was shown.

    ``build_narrow_prompt`` serializes the refuted claim, prefix and all, so a model
    that has just been shown ``q05_H5`` writes ``q05_H6`` back. Stamping the prefix on
    a second time produced ``q05_q05_H6`` in the live run of 2026-09-09: still unique,
    still traceable, and still wrong -- an id is provenance, and provenance that repeats
    itself is a bug the next reader has to explain away.
    """

    name = "stub:echoing-narrower"

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Return a successor whose id already carries the question prefix.

        Args:
            briefing: Unused.
            failed: The refuted claim, whose id is read for its prefix.
            outcome: Unused.

        Returns:
            The raw successor mapping.
        """
        del briefing, outcome
        parent_id = str(failed["id"])
        prefix = parent_id.rsplit("_", 1)[0] + "_" if "_" in parent_id else ""
        successor = _hypothesis(f"{prefix}H9", 999.0)
        successor["claim"] = "A narrower claim, echoing the id spelling it was shown."
        return successor


class _FakeDataset:
    """A dataset stand-in. Every primitive fails on it, which is fine here.

    Args:
        root: Ignored.
    """

    def __init__(self, root: Any) -> None:
        """Record the root and nothing else."""
        self.root = root


class _RefutingDataset(_FakeDataset):
    """A dataset on which ``fraction_above`` runs and refutes, so narrowing is reached.

    Args:
        root: Ignored.
    """

    root = "/stub/refuting"

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

    def nodes(self, case: str) -> Any:
        """Return a node field whose von Mises values are all below any threshold used.

        Args:
            case: The resolved case name.

        Returns:
            A :class:`~llm_insights.io.dataset.NodeField`.
        """
        import numpy as np

        from llm_insights.io.dataset import NodeField

        values = np.zeros(8, dtype=float)
        return NodeField(
            case=case,
            condition="Healthy",
            step=14,
            x_mm=np.linspace(0.0, 1.0, 8),
            y_mm=np.zeros(8),
            von_mises_pa=values,
            wss_dyn_cm2=np.full(8, np.nan),
            is_surface=np.ones(8, dtype=bool),
            von_mises_raw_min_pa=0.0,
        )


class TestIdsSurviveMultipleQuestions(unittest.TestCase):
    """Every question in a study must produce cards, not only the first."""

    def setUp(self) -> None:
        """Install the fake dataset so no results tree is needed."""
        self._real_dataset = api.Dataset
        api.Dataset = _FakeDataset  # type: ignore[misc]

    def tearDown(self) -> None:
        """Restore the real dataset class."""
        api.Dataset = self._real_dataset  # type: ignore[misc]

    def _study(self, generator: Any) -> api.Study:
        """Open a study with a fixed briefing and no disk access.

        Args:
            generator: The backend to use.

        Returns:
            The study.
        """
        study = api.open_study("/nowhere", generator=generator)
        study._briefing = "a fixed briefing; nothing here reads it"
        return study

    def test_every_question_produces_cards(self) -> None:
        """Three questions, five claims each, fifteen cards -- none lost to an id."""
        generator = RestartingGenerator()
        study = self._study(generator)

        answers = [
            study.ask(f"question {index}", n=N_PER_QUESTION) for index in range(1, N_QUESTIONS + 1)
        ]

        self.assertEqual(generator.propose_calls, N_QUESTIONS)
        for index, answer in enumerate(answers, start=1):
            with self.subTest(question=index):
                self.assertEqual(
                    len(answer.cards),
                    N_PER_QUESTION,
                    f"question {index} produced {len(answer.cards)} card(s); "
                    f"{N_PER_QUESTION} were proposed and none should be lost to an id",
                )
                self.assertEqual(answer.meta["n_rejected"], 0)

    def test_ids_stay_unique_across_the_whole_run(self) -> None:
        """An id may appear exactly once in a study, however many questions it spans."""
        study = self._study(RestartingGenerator())
        answers = [
            study.ask(f"question {index}", n=N_PER_QUESTION) for index in range(1, N_QUESTIONS + 1)
        ]

        ids = [card.id for answer in answers for card in answer.cards]
        self.assertEqual(len(ids), N_QUESTIONS * N_PER_QUESTION)
        self.assertEqual(len(set(ids)), len(ids), f"ids repeat across questions: {sorted(ids)}")

    def test_a_card_names_the_question_that_produced_it(self) -> None:
        """Provenance survives: an id says which question the claim came from."""
        study = self._study(RestartingGenerator())
        answers = [
            study.ask(f"question {index}", n=N_PER_QUESTION) for index in range(1, N_QUESTIONS + 1)
        ]

        for index, answer in enumerate(answers, start=1):
            with self.subTest(question=index):
                for card in answer.cards:
                    self.assertTrue(
                        card.id.startswith(f"q{index:02d}_"),
                        f"card id {card.id!r} does not name question {index}",
                    )


class TestNarrowingDoesNotRepeatThePrefix(unittest.TestCase):
    """A successor that echoes the prefix it was shown must not be given a second one."""

    def setUp(self) -> None:
        """Install a dataset on which claims are refuted rather than erroring."""
        self._real_dataset = api.Dataset
        api.Dataset = _RefutingDataset  # type: ignore[misc]

    def tearDown(self) -> None:
        """Restore the real dataset class."""
        api.Dataset = self._real_dataset  # type: ignore[misc]

    def test_a_successor_id_carries_the_prefix_exactly_once(self) -> None:
        """``q02_q02_H9`` is not an acceptable id for anything."""
        study = api.open_study("/nowhere", generator=EchoingNarrower())
        study._briefing = "a fixed briefing; nothing here reads it"

        study.ask("question one", n=1)
        answer = study.ask("question two", n=1)

        successors = [card.id for card in answer.cards if card.revision_of]
        self.assertTrue(successors, "the fixture did not produce a successor to check")
        for card_id in successors:
            with self.subTest(card_id=card_id):
                self.assertEqual(
                    card_id.count("q02_"),
                    1,
                    f"successor id {card_id!r} repeats its question prefix",
                )


if __name__ == "__main__":
    unittest.main()
