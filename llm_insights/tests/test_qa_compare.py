"""The Q/A comparison runner: four arms, resumable, and a table nobody graded.

No model is called here. A stub stands in for both backends, and the harness arm runs
the real pipeline -- real briefing, real primitives, real dataset -- underneath it, so
what is exercised is everything except the model's text.

Properties asserted, each one a thing that would otherwise fail silently:

* every arm writes one ``answer.json`` per question, and a rerun skips finished ones;
* a failing backend stops the arm after two consecutive failures, writes nothing for
  the failed questions, and the next run picks them up;
* the harness arm and the baseline arm read byte-identical briefings, and the report
  says loudly when they did not;
* the Sonnet backend is built with no fallback model, so a "Sonnet" column cannot
  quietly contain another model's answers;
* claim ids name the question they belong to even on a resumed run;
* the table has Dan's answer and all four arms, and a withheld harness summary still
  shows what survived.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_insights import api
from llm_insights.agent import qa_compare as qc
from llm_insights.agent.generator import ECHO_TEMPLATE

HAS_DATA = (api.DEFAULT_ROOT / "FSG Results").is_dir()

ITEMS = [
    qc.QAItem(1, "Where does compaction emerge?", "atrial - more compaction"),
    qc.QAItem(2, "When do the cases diverge?", "a few iterations in"),
    qc.QAItem(3, "Minimal signature?", None),
]


class StubGenerator:
    """Stands in for both backends: proposes the echo claim, answers in labels.

    Args:
        fail_on: Question substrings whose calls raise, to simulate an outage.
    """

    def __init__(self, fail_on: tuple[str, ...] = (), fail_summary: bool = False) -> None:
        """Initialise counters."""
        self.name = "stub:test"
        self.calls = 0
        self.fail_on = fail_on
        self.fail_summary = fail_summary
        self.baseline_prompts: list[tuple[str, str | None]] = []
        self.proposed_briefings: list[str] = []

    def _maybe_fail(self, text: str) -> None:
        for marker in self.fail_on:
            if marker in text:
                raise RuntimeError(f"simulated outage on {marker!r}")

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Return one valid claim, the package's own echo template."""
        del n
        self.calls += 1
        self._maybe_fail(question)
        self.proposed_briefings.append(briefing)
        return [json.loads(json.dumps(ECHO_TEMPLATE))]

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Decline to narrow."""
        del briefing, failed, outcome
        self.calls += 1
        return None

    def synthesize(self, prompt: str) -> str | None:
        """Write a labelled summary with no numerals, so it always passes the check."""
        del prompt
        self.calls += 1
        if self.fail_summary:
            raise RuntimeError("Claude AI usage limit reached")
        return "HEADLINE: The cases differ.\nSUMMARY: The tested claim held. It is one claim."

    def complete(self, user_text: str, system: str | None = None) -> str:
        """Answer a baseline question; the reply includes one invented number."""
        self.calls += 1
        self._maybe_fail(user_text)
        self.baseline_prompts.append((user_text, system))
        return (
            "HEADLINE: Atrial side compacts.\nSUMMARY: The briefing suggests 123456.789 as a value."
        )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _run(
    out: Path, arm: str, gen: StubGenerator, items: list[qc.QAItem] = ITEMS, **kw: Any
) -> dict:
    with redirect_stdout(io.StringIO()):
        briefing = qc.prepare_briefing(out, None)
        return qc.run_arm("llama", arm, items, generator=gen, briefing=briefing, out=out, **kw)


class TestLoadQA(unittest.TestCase):
    """The shipped question file is Dan's, intact."""

    def test_the_shipped_file_has_all_ten_questions(self) -> None:
        """The shipped file has all ten questions."""
        items = qc.load_qa(qc.DEFAULT_QA_FILE)
        self.assertEqual([i.number for i in items], list(range(1, 11)))
        self.assertIsNone(items[9].dan, "Dan left question 10 blank")
        self.assertTrue(all(i.dan for i in items[:9]))
        self.assertIn("compaction and expansion/elongation", items[0].question)

    def test_duplicate_numbers_are_rejected(self) -> None:
        """Duplicate numbers are rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qa.json"
            path.write_text(
                json.dumps(
                    {"items": [{"number": 1, "question": "a"}, {"number": 1, "question": "b"}]}
                )
            )
            with self.assertRaises(ValueError):
                qc.load_qa(path)

    def test_selecting_a_missing_question_is_an_error(self) -> None:
        """Selecting a missing question is an error."""
        with self.assertRaises(ValueError):
            qc.select(ITEMS, [7])
        self.assertEqual([i.number for i in qc.select(ITEMS, [3, 1])], [1, 3])


class TestBaselinePrompt(unittest.TestCase):
    """The baseline asks for the same shape of answer the harness summary gives."""

    def test_same_labels_as_the_harness_summary(self) -> None:
        """Same labels as the harness summary."""
        from llm_insights.agent.synthesis import HEADLINE_LABEL, SUMMARY_LABEL

        system = qc.baseline_system_prompt()
        self.assertIn(HEADLINE_LABEL, system)
        self.assertIn(SUMMARY_LABEL, system)

    def test_prompt_carries_the_briefing_and_the_question_verbatim(self) -> None:
        """Prompt carries the briefing and the question verbatim."""
        prompt = qc.build_baseline_prompt("BRIEF-TEXT", "Why?")
        self.assertIn("BRIEF-TEXT", prompt)
        self.assertTrue(prompt.rstrip().endswith("Why?"))


class TestSonnetHasNoFallback(unittest.TestCase):
    """A column headed Sonnet must not contain Haiku's answers."""

    def test_fallback_model_is_disabled(self) -> None:
        """Fallback model is disabled."""
        from llm_insights.agent import subscription

        captured: dict[str, Any] = {}

        class Fake:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        original = subscription.ClaudeCLIGenerator
        subscription.ClaudeCLIGenerator = Fake  # type: ignore[misc]
        try:
            qc.make_backend("sonnet", max_calls=99, max_budget_usd=3.0)
        finally:
            subscription.ClaudeCLIGenerator = original  # type: ignore[misc]
        self.assertIsNone(captured["fallback_model"])
        self.assertEqual(captured["model"], "sonnet")
        self.assertEqual(captured["max_calls"], 99)

    def test_default_ceilings_cover_ten_questions(self) -> None:
        """Default ceilings cover ten questions."""
        # propose + up to five narrows + two summary calls, plus one baseline call
        self.assertGreaterEqual(qc.DEFAULT_MAX_CALLS, 10 * (1 + 5 + 2 + 1))


@unittest.skipUnless(HAS_DATA, "FSG results not present")
class TestArms(unittest.TestCase):
    """Both arms end to end against the real pipeline, with a stub model."""

    def test_harness_arm_writes_cards_and_a_summary(self) -> None:
        """Harness arm writes cards and a summary."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            gen = StubGenerator()
            counts = _run(out, "harness", gen)
            self.assertEqual(counts, {"done": 3, "skipped": 0, "failed": 0})
            rec = json.loads((out / "llama/harness/q02/answer.json").read_text())
            self.assertEqual(rec["headline"], "The cases differ.")
            self.assertEqual(rec["summary_status"], "rendered")
            self.assertEqual(len(rec["cards"]), 1)
            self.assertTrue(rec["cards"][0]["id"].startswith("q02_"))
            self.assertIn(rec["cards"][0]["class"], {"LOOKUP", "DERIVED", "COMPUTED"})
            self.assertEqual(sum(rec["tally"].values()), 1)
            self.assertTrue((out / "llama/harness/q02/cards.json").is_file())
            # the harness read the exact briefing written for the run
            self.assertEqual(gen.proposed_briefings[0], (out / "briefing.md").read_text())

    def test_baseline_arm_reads_the_same_briefing_and_flags_invented_numbers(self) -> None:
        """Baseline arm reads the same briefing and flags invented numbers."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            gen = StubGenerator()
            _run(out, "baseline", gen)
            briefing = (out / "briefing.md").read_text()
            user, system = gen.baseline_prompts[0]
            self.assertIn(briefing, user)
            self.assertEqual(system, qc.baseline_system_prompt())
            rec = json.loads((out / "llama/baseline/q01/answer.json").read_text())
            self.assertEqual(rec["headline"], "Atrial side compacts.")
            self.assertEqual(rec["numbers_not_in_briefing"], ["123456.789"])
            self.assertEqual(rec["briefing_sha256"], qc.sha256(briefing))

    def test_rerun_skips_finished_questions_and_force_redoes_them(self) -> None:
        """Rerun skips finished questions and force redoes them."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _run(out, "baseline", StubGenerator())
            gen = StubGenerator()
            self.assertEqual(_run(out, "baseline", gen), {"done": 0, "skipped": 3, "failed": 0})
            self.assertEqual(gen.calls, 0)
            self.assertEqual(_run(out, "baseline", gen, force=True)["done"], 3)

    def test_outage_stops_the_arm_and_the_next_run_resumes(self) -> None:
        """Outage stops the arm and the next run resumes."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            # questions 2 and 3 fail: two in a row, so the arm stops
            gen = StubGenerator(fail_on=("When do", "Minimal"))
            self.assertEqual(_run(out, "harness", gen), {"done": 1, "skipped": 0, "failed": 2})
            self.assertFalse((out / "llama/harness/q02/answer.json").exists())
            counts = _run(out, "harness", StubGenerator())
            self.assertEqual(counts, {"done": 2, "skipped": 1, "failed": 0})
            # resumed question 3 is still labelled as question 3
            rec = json.loads((out / "llama/harness/q03/answer.json").read_text())
            self.assertTrue(rec["cards"][0]["id"].startswith("q03_"))

    def test_a_failure_mid_question_is_not_saved_as_finished(self) -> None:
        """A failure mid question is not saved as finished."""
        # The loop and synthesis both swallow backend errors by design. Found by fault
        # injection: a usage limit hitting the summary call saved the question as done
        # with its summary missing, and a rerun never redid it.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            counts = _run(out, "harness", StubGenerator(fail_summary=True))
            self.assertEqual(counts, {"done": 0, "skipped": 0, "failed": 2})
            self.assertFalse((out / "llama/harness/q01/answer.json").exists())
            self.assertEqual(_run(out, "harness", StubGenerator())["done"], 3)

    def test_first_outage_alone_does_not_stop_the_arm(self) -> None:
        """First outage alone does not stop the arm."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            counts = _run(out, "baseline", StubGenerator(fail_on=("Where does",)))
            self.assertEqual(counts, {"done": 2, "skipped": 0, "failed": 1})


@unittest.skipUnless(HAS_DATA, "FSG results not present")
class TestReport(unittest.TestCase):
    """The table: Dan's column, four arms, and honest cells."""

    def _both_arms(self, out: Path) -> None:
        _run(out, "harness", StubGenerator())
        _run(out, "baseline", StubGenerator())

    def test_table_has_dan_and_all_four_arms(self) -> None:
        """Table has dan and all four arms."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._both_arms(out)
            paths = qc.write_report(out, ITEMS)
            rows = _rows(paths["table_csv"])
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                list(rows[0]),
                [
                    "#",
                    "Question",
                    "Dan",
                    "Llama (harness)",
                    "Llama (no harness)",
                    "Sonnet (harness)",
                    "Sonnet (no harness)",
                ],
            )
            self.assertEqual(rows[2]["Dan"], "(no answer given)")
            self.assertTrue(rows[0]["Llama (harness)"].startswith("The cases differ."))
            self.assertIn("[Tests: 1 ", rows[0]["Llama (harness)"])
            self.assertEqual(rows[0]["Sonnet (harness)"], qc.NOT_RUN)
            md = paths["table_md"].read_text()
            self.assertIn("Every arm read the same briefing", md)
            self.assertIn("| Sonnet (no harness) |", md)
            stats = {r["Arm"]: r for r in _rows(paths["stats_csv"])}
            self.assertEqual(stats["Llama (harness)"]["Questions answered"], "3/3")
            self.assertEqual(
                stats["Llama (no harness)"]["Numbers not in briefing"], "3 (unchecked)"
            )
            self.assertEqual(len(_rows(paths["cards_csv"])), 3)

    def test_mismatched_briefings_are_flagged(self) -> None:
        """Mismatched briefings are flagged."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._both_arms(out)
            path = out / "llama/baseline/q01/answer.json"
            rec = json.loads(path.read_text())
            rec["briefing_sha256"] = "0" * 64
            path.write_text(json.dumps(rec))
            md = qc.write_report(out, ITEMS)["table_md"].read_text()
            self.assertIn("did not all read the same briefing", md)

    def test_withheld_summary_still_shows_survivors(self) -> None:
        """Withheld summary still shows survivors."""
        cell = qc.answer_cell(
            {
                "arm": "harness",
                "headline": None,
                "summary": None,
                "summary_status": "suppressed: 1 unsupported numeral",
                "cards": [{"claim": "WSS rises with flow", "verdict": "SURVIVED"}],
                "tally": {"SURVIVED": 1, "FALSIFIED": 2},
            }
        )
        self.assertIn("Summary withheld", cell)
        self.assertIn("WSS rises with flow", cell)
        self.assertIn("1 survived, 2 falsified", cell)
        self.assertIn("falsified then narrowed", qc._tally_text({"NARROWED": 1}))

    def test_pipes_do_not_break_the_markdown_table(self) -> None:
        """Pipes do not break the markdown table."""
        table = qc.markdown_table([{"a": "x | y", "b": "line\nbreak"}])
        self.assertIn("x \\| y", table)
        self.assertEqual(len(table.strip().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
