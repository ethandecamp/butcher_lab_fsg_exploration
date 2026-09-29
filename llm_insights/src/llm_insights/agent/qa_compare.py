"""Compare how two models answer Dan's questions, with and without the harness.

Four arms run over the same questions and the same briefing:

* ``llama/harness`` -- the local model proposes claims, the harness tests them, and the
  same model writes a checked summary of the verdicts. This is the normal pipeline.
* ``llama/baseline`` -- the local model reads the same briefing and just answers. No
  claims, no tests, no numeric check.
* ``sonnet/harness`` and ``sonnet/baseline`` -- the same two, through the Claude Code CLI
  on the subscription (no API key).

Both arms are asked for the same shape of answer -- a one-to-two sentence headline and a
three-to-six sentence summary -- so a reader comparing them is comparing content rather
than length or format.

Nothing here grades an answer. The output is a table that puts Dan's answer beside each
arm's, for a human to judge, plus the counts a reader needs to weigh the harness arms
(how many claims ran, survived, were falsified, or could not run, and how many of the
survivors were lookups of numbers already printed in the briefing).

Every question's result is written as soon as it finishes, and a question that already
has a result is skipped, so a run that stops part way -- a usage limit, a closed laptop,
Ollama not running -- resumes where it stopped when the same command is run again.

Typical use, from ``llm_insights/``::

    PYTHONPATH=src python3 -m llm_insights.agent.qa_compare run      # all four arms
    PYTHONPATH=src python3 -m llm_insights.agent.qa_compare report   # table only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

LOG = logging.getLogger(__name__)

#: ``llm_insights/``, the project directory. Paths default to absolute locations under
#: it, so the command behaves the same from any working directory.
PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[3]

#: Dan's question/answer pairs.
DEFAULT_QA_FILE: Final[Path] = PROJECT_DIR / "qa" / "dan_qa.json"

#: Where every arm's per-question results and the final table are written.
DEFAULT_OUT: Final[Path] = PROJECT_DIR / "data" / "qa_compare"

MODELS: Final[tuple[str, ...]] = ("llama", "sonnet")
ARMS: Final[tuple[str, ...]] = ("harness", "baseline")

#: Column headings for the comparison table, in column order.
ARM_HEADINGS: Final[dict[tuple[str, str], str]] = {
    ("llama", "harness"): "Llama (harness)",
    ("llama", "baseline"): "Llama (no harness)",
    ("sonnet", "harness"): "Sonnet (harness)",
    ("sonnet", "baseline"): "Sonnet (no harness)",
}

#: Claims requested per question in the harness arms. Matches the Sep 9 llama run.
DEFAULT_N: Final[int] = 5

#: The Claude model alias the ``sonnet`` arms use.
DEFAULT_CLI_MODEL: Final[str] = "sonnet"

#: Call ceiling for the one CLI generator both Sonnet arms share. A harness question
#: costs one proposal, up to one narrowing per falsified claim, and one or two summary
#: calls -- at most about eight -- and a baseline question costs one, so ten questions
#: need at most about ninety. The package default of twelve would stop at question two.
DEFAULT_MAX_CALLS: Final[int] = 120

#: Cost ceiling, in USD, for that generator. On a subscription this is the CLI's
#: equivalent-spend estimate, not an amount billed; it exists to stop a runaway.
DEFAULT_MAX_BUDGET_USD: Final[float] = 15.0

#: An arm stops after this many questions fail in a row. Two consecutive failures almost
#: always mean the backend is down or the usage limit was hit, and continuing would
#: only burn through the remaining questions producing nothing.
MAX_CONSECUTIVE_FAILURES: Final[int] = 2

#: Integers at or below this are treated as prose counts ("three of the five claims")
#: when counting a baseline answer's numbers that the briefing does not print.
SMALL_COUNT: Final[int] = 10

#: The file each finished question writes. Its presence is what "already done" means.
ANSWER_FILE: Final[str] = "answer.json"

#: What an empty cell says.
NOT_RUN: Final[str] = "(not run)"

_NUMERAL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


# --- the baseline prompt -------------------------------------------------------------


def baseline_system_prompt() -> str:
    """Build the system prompt for the no-harness arm.

    Deliberately mirrors :data:`llm_insights.agent.synthesis.SYNTHESIS_SYSTEM_PROMPT` --
    the same two labelled parts, the same sentence budgets, the same "answer the
    question, not the report" instruction, the same honest-partial-answer rule -- so
    that the only difference between the arms a reader sees is where the evidence came
    from: tested claims, or the briefing alone.

    Returns:
        The system prompt.
    """
    from llm_insights.agent.synthesis import (
        HEADLINE_LABEL,
        HEADLINE_MAX_SENTENCES,
        HEADLINE_MIN_SENTENCES,
        MAX_SENTENCES,
        MIN_SENTENCES,
        SUMMARY_LABEL,
    )

    return (
        "You answer a question about a fluid-solid-growth simulation of a developing "
        "heart valve cushion coupled to a gene-regulatory network model.\n"
        "\n"
        "You are given a briefing that summarises the simulation's outputs for three "
        "inlet flow conditions. Answer the question from the briefing, for a biologist "
        "who built the model.\n"
        "\n"
        "RULES\n"
        "1. Write exactly two labelled parts, in this order and nothing else:\n"
        f"   {HEADLINE_LABEL} {HEADLINE_MIN_SENTENCES} to {HEADLINE_MAX_SENTENCES} "
        "sentences that ANSWER THE QUESTION, directly, in its own terms. A reader who "
        "stops here must come away with the answer. Lead with the answer, not with the "
        "evidence for it.\n"
        f"   {SUMMARY_LABEL} {MIN_SENTENCES} to {MAX_SENTENCES} sentences giving the "
        "evidence for that answer and any qualification it needs. The summary must "
        "stand on its own.\n"
        "   Plain English prose in both. No markdown, no further headings, no bullet "
        "points, no JSON, no code fences.\n"
        "2. Use only numbers that appear in the briefing. Do not state a number you were "
        "not given.\n"
        "3. If the numbers implicate saturation -- a large fraction of the domain at a "
        "normalization ceiling, a clipped or pinned quantity -- say plainly that the "
        "effect may be partly an artefact of that ceiling rather than biology.\n"
        "4. Answer only as far as the briefing allows. If it does not settle the "
        "question, the headline says what it does establish and that the question is "
        "not settled -- an honest partial answer, never a confident one the briefing "
        "does not support, and never a refusal to answer at all.\n"
        "5. Do not recommend further work and do not describe your own reasoning. "
        "Return the two labelled parts and nothing else."
    )


def build_baseline_prompt(briefing: str, question: str) -> str:
    """Build the user turn for the no-harness arm.

    Args:
        briefing: The briefing, byte-identical to the one the harness arm reads.
        question: The question, exactly as Dan wrote it.

    Returns:
        The user turn.
    """
    return f"BRIEFING\n\n{briefing}\n\n---\n\nQUESTION\n\n{question}\n"


# --- inputs ------------------------------------------------------------------------


@dataclass(frozen=True)
class QAItem:
    """One of Dan's questions and his answer to it.

    Attributes:
        number: 1-based question number; also the directory name (``q01``) and the
            hypothesis id prefix (``q01_H1``).
        question: The question, verbatim.
        dan: Dan's answer, verbatim, or None where he left it blank.
    """

    number: int
    question: str
    dan: str | None

    @property
    def slug(self) -> str:
        """The directory name for this question, e.g. ``q03``."""
        return f"q{self.number:02d}"


def load_qa(path: str | Path) -> list[QAItem]:
    """Read the question/answer file.

    Args:
        path: A JSON file with an ``items`` list of ``{number, question, dan}``.

    Returns:
        The items, in file order.

    Raises:
        ValueError: If the file is malformed or two items share a number.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} has no 'items' list")
    items: list[QAItem] = []
    for entry in entries:
        try:
            number = int(entry["number"])
            question = str(entry["question"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: malformed item {entry!r}") from exc
        if not question:
            raise ValueError(f"{path}: item {number} has an empty question")
        dan = entry.get("dan")
        items.append(QAItem(number, question, str(dan).strip() if dan else None))
    numbers = [item.number for item in items]
    duplicates = sorted(n for n, count in Counter(numbers).items() if count > 1)
    if duplicates:
        raise ValueError(f"{path}: question numbers repeat: {duplicates}")
    return items


def select(items: Sequence[QAItem], numbers: Iterable[int] | None) -> list[QAItem]:
    """Keep only the requested question numbers.

    Args:
        items: Every item.
        numbers: The numbers to keep, or None for all.

    Returns:
        The kept items, in file order.

    Raises:
        ValueError: If a requested number does not exist.
    """
    if not numbers:
        return list(items)
    wanted = set(numbers)
    missing = wanted - {item.number for item in items}
    if missing:
        raise ValueError(f"no such question number(s): {sorted(missing)}")
    return [item for item in items if item.number in wanted]


def sha256(text: str) -> str:
    """Return the hex SHA-256 of ``text``, the same digest the loop records."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- backends ----------------------------------------------------------------------


def make_backend(
    model: str,
    *,
    cli_model: str = DEFAULT_CLI_MODEL,
    max_calls: int = DEFAULT_MAX_CALLS,
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
) -> Any:
    """Build the generator for one model.

    The Sonnet arm is built with **no fallback model**. The CLI backend normally retries
    on Haiku when the account cannot use the requested model, which is the right
    default for a demo and the wrong one here: a column headed "Sonnet" must not
    silently contain Haiku's answers. It fails loudly instead.

    Args:
        model: ``"llama"`` or ``"sonnet"``.
        cli_model: The Claude model alias for the Sonnet arm.
        max_calls: Call ceiling for the CLI generator.
        max_budget_usd: Cost ceiling for the CLI generator.

    Returns:
        The generator.

    Raises:
        ValueError: On an unknown model.
        RuntimeError: If the backend cannot be reached (no ``claude`` on PATH).
    """
    if model == "llama":
        from llm_insights.agent.local import LocalGenerator

        return LocalGenerator()
    if model == "sonnet":
        from llm_insights.agent.subscription import ClaudeCLIGenerator

        return ClaudeCLIGenerator(
            model=cli_model,
            max_calls=max_calls,
            max_budget_usd=max_budget_usd,
            fallback_model=None,
        )
    raise ValueError(f"unknown model {model!r}; expected one of {', '.join(MODELS)}")


class _ErrorRecorder:
    """Forward everything to a backend, and remember every exception it raises.

    The loop deliberately swallows a failed narrowing call, and synthesis turns a failed
    summary call into a status string, because in a normal run the verified cards are
    still worth keeping. Here that would be wrong: if the usage limit lands halfway
    through a question, the question would be saved as finished with a claim never
    narrowed or its summary missing, and a rerun would never redo it -- so one column
    would be quietly degraded against the other. Recording at the backend boundary
    catches every such failure however the caller handles it.

    Args:
        inner: The real backend.
    """

    _WRAPPED: Final[frozenset[str]] = frozenset({"propose", "narrow", "synthesize", "complete"})

    def __init__(self, inner: Any) -> None:
        """Wrap ``inner`` with an empty error list."""
        self._inner = inner
        self.errors: list[str] = []

    def __getattr__(self, item: str) -> Any:
        """Return the backend's attribute, wrapping the model-calling methods."""
        attr = getattr(self._inner, item)
        if item not in self._WRAPPED or not callable(attr):
            return attr

        def recorded(*args: Any, **kwargs: Any) -> Any:
            try:
                return attr(*args, **kwargs)
            except Exception as exc:
                self.errors.append(f"{item}: {type(exc).__name__}: {exc}")
                raise

        return recorded


class IncompleteQuestionError(RuntimeError):
    """A backend call failed part way through a question, so its result is not kept."""


def _usage(generator: Any) -> dict[str, Any]:
    """Snapshot the counters a generator keeps, for per-question accounting."""
    return {
        "calls": getattr(generator, "calls", None),
        "cost_usd": getattr(generator, "cost_usd", None),
    }


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Difference two :func:`_usage` snapshots, leaving None where unknown."""
    out: dict[str, Any] = {}
    for key in before:
        a, b = before[key], after[key]
        out[key] = None if a is None or b is None else round(b - a, 6)
    return out


# --- one question ------------------------------------------------------------------


def run_harness_question(study: Any, item: QAItem, qdir: Path, n: int) -> dict[str, Any]:
    """Run one question through the full pipeline and describe the result.

    Args:
        study: A :class:`llm_insights.api.Study` holding this arm's generator.
        item: The question.
        qdir: Where the cards, report and transcript go.
        n: Claims to request.

    Returns:
        The fields this arm contributes to ``answer.json``.
    """
    from llm_insights.agent.explore import BriefingIndex, classify_card
    from llm_insights.cards.render import verdict_of

    # The Study numbers its questions itself, starting from one, so a resumed run
    # that skips q01-q03 would otherwise label question 4's claims q01_*. Setting the
    # counter keeps each claim's id naming the question it belongs to. Ids stay unique
    # within the study either way; this is only about what the id says.
    study._questions_started = item.number - 1
    answer = study.ask(item.question, n=n, out=qdir, synthesize_summary=True)
    index = BriefingIndex(study.briefing)

    cards: list[dict[str, Any]] = []
    for card in answer.cards:
        cards.append(
            {
                "id": card.id,
                "claim": card.claim,
                "primitive": card.primitive,
                "params": card.params,
                "verdict": verdict_of(card).label,
                "what_ran": card.error or card.summary,
                "observed": card.observed,
                "class": classify_card(card, index).label,
                "revision_of": card.revision_of,
            }
        )
    return {
        "headline": answer.headline,
        "summary": answer.blurb,
        "summary_status": str(answer.meta.get("synthesis", "")),
        "cards": cards,
        "tally": dict(Counter(card["verdict"] for card in cards)),
    }


def unsupported_numbers(text: str, briefing_index: Any) -> list[str]:
    """List the numerals in ``text`` that the briefing does not print.

    Small integers are skipped as likely prose counts. A reading aid for the baseline
    arm, whose answer -- unlike the harness summary -- is not checked before it is shown.

    Args:
        text: The answer.
        briefing_index: A parsed :class:`~llm_insights.agent.explore.BriefingIndex`.

    Returns:
        The unsupported numerals, as written, in order.
    """
    out: list[str] = []
    for token in _NUMERAL_RE.findall(text or ""):
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        if value.is_integer() and abs(value) <= SMALL_COUNT:
            continue
        if briefing_index.prints_value(value) is None:
            out.append(token)
    return out


def run_baseline_question(generator: Any, briefing: str, item: QAItem) -> dict[str, Any]:
    """Ask one question with the briefing and nothing else.

    Args:
        generator: A backend with ``complete(user_text, system)``.
        briefing: The briefing.
        item: The question.

    Returns:
        The fields this arm contributes to ``answer.json``.
    """
    from llm_insights.agent.explore import BriefingIndex
    from llm_insights.agent.synthesis import split_narration

    raw = generator.complete(
        build_baseline_prompt(briefing, item.question), baseline_system_prompt()
    )
    text = "\n".join(" ".join(line.split()) for line in str(raw or "").splitlines()).strip()
    headline, summary = split_narration(text)
    return {
        "headline": headline,
        "summary": summary or None,
        "raw": raw,
        "numbers_not_in_briefing": unsupported_numbers(text, BriefingIndex(briefing)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary file so a killed run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# --- one arm -----------------------------------------------------------------------


def run_arm(
    model: str,
    arm: str,
    items: Sequence[QAItem],
    *,
    generator: Any,
    briefing: str,
    out: Path,
    n: int = DEFAULT_N,
    force: bool = False,
    dataset_root: Path | None = None,
) -> dict[str, int]:
    """Run every question for one model and one arm, skipping finished ones.

    Args:
        model: ``"llama"`` or ``"sonnet"``, used for the output path and labels.
        arm: ``"harness"`` or ``"baseline"``.
        items: The questions to run.
        generator: The model's backend.
        briefing: The briefing every arm reads.
        out: The output root.
        n: Claims per question, harness arm only.
        force: Re-run questions that already have a result.
        dataset_root: The FSG results root, or None for the package default.

    Returns:
        Counts of ``done`` (run now), ``skipped`` (already had a result) and
        ``failed`` questions.
    """
    from llm_insights.api import open_study

    arm_dir = out / model / arm
    counts = {"done": 0, "skipped": 0, "failed": 0}
    recorder = _ErrorRecorder(generator)
    study = None
    if arm == "harness":
        # One study per arm, reading the briefing that was written to disk for this
        # run, so the harness sees exactly the bytes the baseline sees. A fresh study
        # per arm also keeps one model's claim ids from colliding with another's.
        study = open_study(dataset_root, generator=recorder, briefing_file=out / "briefing.md")
    brief_sha = sha256(briefing)
    consecutive = 0

    for item in items:
        qdir = arm_dir / item.slug
        answer_path = qdir / ANSWER_FILE
        label = f"[{model}/{arm} {item.slug}]"
        if answer_path.is_file() and not force:
            counts["skipped"] += 1
            print(f"{label} already done, skipping (use --force to redo)", flush=True)
            continue

        print(f"{label} {item.question[:70]}...", flush=True)
        before = _usage(generator)
        started = time.monotonic()
        recorder.errors.clear()
        try:
            if arm == "harness":
                fields = run_harness_question(study, item, qdir, n)
            else:
                fields = run_baseline_question(recorder, briefing, item)
            if recorder.errors:
                raise IncompleteQuestionError(
                    f"{len(recorder.errors)} model call(s) failed mid-question, so this "
                    f"result is not kept; first was {recorder.errors[0]}"
                )
        except Exception as exc:  # recorded and reported; the next run retries it
            counts["failed"] += 1
            consecutive += 1
            message = f"{type(exc).__name__}: {exc}"
            print(
                f"{label} FAILED after {time.monotonic() - started:.0f}s -- {message}", flush=True
            )
            if "BudgetExceededError" in f"{type(exc).__name__} {exc}":
                print(
                    f"{label} the call or cost ceiling was reached; stopping this arm", flush=True
                )
                break
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"{label} {consecutive} failures in a row -- the backend is probably "
                    "down or out of usage; stopping this arm. Run the same command again "
                    "to resume from here.",
                    flush=True,
                )
                break
            continue

        consecutive = 0
        seconds = round(time.monotonic() - started, 1)
        record = {
            "number": item.number,
            "question": item.question,
            "dan": item.dan,
            "model": model,
            "arm": arm,
            "generator": str(getattr(generator, "name", type(generator).__name__)),
            "briefing_sha256": brief_sha,
            "seconds": seconds,
            **_delta(before, _usage(generator)),
            **fields,
        }
        _write_json(answer_path, record)
        counts["done"] += 1
        tail = ""
        if arm == "harness":
            tail = "  " + _tally_text(record.get("tally", {}))
        print(f"{label} done in {seconds:.0f}s{tail}", flush=True)

    if study is not None:
        study.generator = None  # the caller owns the generator and closes it
    return counts


# --- the table ---------------------------------------------------------------------


def load_answers(out: Path, items: Sequence[QAItem]) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Read every finished ``answer.json`` under ``out``.

    Args:
        out: The output root.
        items: The questions.

    Returns:
        Answers keyed by ``(model, arm, number)``.
    """
    found: dict[tuple[str, str, int], dict[str, Any]] = {}
    for model in MODELS:
        for arm in ARMS:
            for item in items:
                path = out / model / arm / item.slug / ANSWER_FILE
                if path.is_file():
                    found[(model, arm, item.number)] = json.loads(path.read_text(encoding="utf-8"))
    return found


def _tally_text(tally: dict[str, int]) -> str:
    """Render a verdict tally as ``2 survived, 1 falsified, ...`` in a fixed order."""
    order = ("SURVIVED", "FALSIFIED", "NARROWED", "COULD NOT RUN", "ENVIRONMENT FAULT")
    words = {"NARROWED": "falsified then narrowed"}
    parts = [f"{tally[k]} {words.get(k, k.lower())}" for k in order if tally.get(k)]
    parts += [f"{v} {k.lower()}" for k, v in tally.items() if k not in order and v]
    return ", ".join(parts) or "no claims"


def answer_cell(record: dict[str, Any] | None) -> str:
    """The text of one arm's answer to one question, for the table.

    A harness answer whose summary was withheld by the numeric check is not left
    blank: the cell says why, and lists the claims that survived, so the reader still
    sees what the run established.

    Args:
        record: The ``answer.json`` contents, or None if not run.

    Returns:
        The cell text.
    """
    if record is None:
        return NOT_RUN
    headline = (record.get("headline") or "").strip()
    summary = (record.get("summary") or "").strip()
    body = " ".join(part for part in (headline, summary) if part)
    if record.get("arm") == "harness":
        if not summary:
            status = record.get("summary_status") or "no summary"
            survivors = [
                c["claim"] for c in record.get("cards", []) if c.get("verdict") == "SURVIVED"
            ]
            body = f"(Summary withheld -- {status}.)"
            if survivors:
                body += " Claims that survived testing: " + " / ".join(survivors)
        body += f" [Tests: {_tally_text(record.get('tally', {}))}]"
    return body or "(empty answer)"


def comparison_rows(
    items: Sequence[QAItem], answers: dict[tuple[str, str, int], dict[str, Any]]
) -> list[dict[str, str]]:
    """Build the comparison table: one row per question, one column per source.

    Args:
        items: The questions.
        answers: As :func:`load_answers` returns.

    Returns:
        The rows, as ordered mappings.
    """
    rows: list[dict[str, str]] = []
    for item in items:
        row = {
            "#": str(item.number),
            "Question": item.question,
            "Dan": item.dan or "(no answer given)",
        }
        for model in MODELS:
            for arm in ARMS:
                row[ARM_HEADINGS[(model, arm)]] = answer_cell(
                    answers.get((model, arm, item.number))
                )
        rows.append(row)
    return rows


def stats_rows(
    items: Sequence[QAItem], answers: dict[tuple[str, str, int], dict[str, Any]]
) -> list[dict[str, str]]:
    """Build the per-arm run statistics table.

    Args:
        items: The questions.
        answers: As :func:`load_answers` returns.

    Returns:
        One row per arm.
    """
    rows: list[dict[str, str]] = []
    for model in MODELS:
        for arm in ARMS:
            recs = [
                answers[(model, arm, i.number)] for i in items if (model, arm, i.number) in answers
            ]
            row = {
                "Arm": ARM_HEADINGS[(model, arm)],
                "Model actually used": ", ".join(sorted({r.get("generator", "?") for r in recs}))
                or "-",
                "Questions answered": f"{len(recs)}/{len(items)}",
                "Total time (min)": f"{sum(r.get('seconds') or 0 for r in recs) / 60:.1f}"
                if recs
                else "-",
                "Model calls": str(sum(r.get("calls") or 0 for r in recs)) if recs else "-",
            }
            if arm == "harness":
                tally: Counter[str] = Counter()
                survivors_by_class: Counter[str] = Counter()
                rendered = 0
                for r in recs:
                    tally.update(r.get("tally", {}))
                    rendered += 1 if r.get("summary") else 0
                    for c in r.get("cards", []):
                        if c.get("verdict") == "SURVIVED":
                            survivors_by_class[c.get("class", "?")] += 1
                n_surv = tally.get("SURVIVED", 0)
                row["Claims tested"] = str(sum(tally.values())) if recs else "-"
                row["Survived / falsified / could not run"] = (
                    f"{n_surv} / {tally.get('FALSIFIED', 0) + tally.get('NARROWED', 0)} / "
                    f"{tally.get('COULD NOT RUN', 0) + tally.get('ENVIRONMENT FAULT', 0)}"
                    if recs
                    else "-"
                )
                row["Survivors that were LOOKUPs"] = (
                    f"{survivors_by_class.get('LOOKUP', 0)}/{n_surv}" if n_surv else "-"
                )
                row["Summaries shown / withheld"] = (
                    f"{rendered} / {len(recs) - rendered}" if recs else "-"
                )
                row["Numbers not in briefing"] = "0 (checked)" if recs else "-"
            else:
                row["Claims tested"] = "n/a"
                row["Survived / falsified / could not run"] = "n/a"
                row["Survivors that were LOOKUPs"] = "n/a"
                row["Summaries shown / withheld"] = "n/a"
                row["Numbers not in briefing"] = (
                    str(sum(len(r.get("numbers_not_in_briefing", [])) for r in recs))
                    + " (unchecked)"
                    if recs
                    else "-"
                )
            rows.append(row)
    return rows


def card_rows(
    items: Sequence[QAItem], answers: dict[tuple[str, str, int], dict[str, Any]]
) -> list[dict[str, str]]:
    """Every harness claim, with the test that actually ran beside it.

    Args:
        items: The questions.
        answers: As :func:`load_answers` returns.

    Returns:
        One row per card.
    """
    rows: list[dict[str, str]] = []
    for item in items:
        for model in MODELS:
            record = answers.get((model, "harness", item.number))
            for c in (record or {}).get("cards", []):
                params = json.dumps(c.get("params", {}), sort_keys=True)
                rows.append(
                    {
                        "#": str(item.number),
                        "Arm": ARM_HEADINGS[(model, "harness")],
                        "Claim id": str(c.get("id", "")),
                        "Claim": str(c.get("claim", "")),
                        "Test": f"{c.get('primitive', '')} {params}",
                        "What ran": str(c.get("what_ran") or ""),
                        "Verdict": str(c.get("verdict", "")),
                        "Class": str(c.get("class", "")),
                    }
                )
    return rows


def _md_cell(text: str) -> str:
    """Make text safe inside a markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def markdown_table(rows: Sequence[dict[str, str]]) -> str:
    """Render rows as a GitHub-flavoured markdown table.

    Args:
        rows: Rows sharing one set of keys.

    Returns:
        The table, or a one-line note if there are no rows.
    """
    if not rows:
        return "_(nothing yet)_\n"
    heads = list(rows[0])
    lines = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in heads) + "|"]
    lines += ["| " + " | ".join(_md_cell(r.get(h, "")) for h in heads) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write rows to CSV with a header, or a header-less empty file if none."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_report(out: Path, items: Sequence[QAItem]) -> dict[str, Path]:
    """Write the comparison table and its companions.

    Files written under ``out``:

    * ``qa_comparison.csv`` -- the table: one row per question; Dan, then four arms.
    * ``qa_comparison.md`` -- the same table, then the run statistics, then notes.
    * ``qa_run_stats.csv`` -- the run statistics alone.
    * ``qa_cards.csv`` -- every harness claim beside the test that ran.

    Args:
        out: The output root.
        items: The questions.

    Returns:
        The written paths, keyed by short name.
    """
    out.mkdir(parents=True, exist_ok=True)
    answers = load_answers(out, items)
    comp = comparison_rows(items, answers)
    stats = stats_rows(items, answers)
    cards = card_rows(items, answers)

    shas = sorted({r.get("briefing_sha256", "?") for r in answers.values()})
    if len(shas) > 1:
        brief_note = (
            f"**WARNING: the arms did not all read the same briefing** ({len(shas)} different "
            "SHA-256s). Delete the stale arm's folder and rerun it before comparing."
        )
    elif shas:
        brief_note = f"Every arm read the same briefing (SHA-256 `{shas[0][:12]}`)."
    else:
        brief_note = "No arm has run yet."

    paths = {
        "table_csv": out / "qa_comparison.csv",
        "table_md": out / "qa_comparison.md",
        "stats_csv": out / "qa_run_stats.csv",
        "cards_csv": out / "qa_cards.csv",
    }
    _write_csv(paths["table_csv"], comp)
    _write_csv(paths["stats_csv"], stats)
    _write_csv(paths["cards_csv"], cards)

    md = [
        "# Dan's questions: Llama vs Sonnet, with and without the harness\n",
        markdown_table(comp),
        "\n## Run statistics\n",
        markdown_table(stats),
        "\n## Notes\n",
        f"- {brief_note}",
        "- **Harness** = the model proposes claims, executable tests decide each one, then the "
        "model summarises the verdicts. A summary containing any number not on a card is "
        "withheld, so harness answers never state an unverified number.",
        "- **No harness** = the model reads the same briefing and answers directly. Nothing is "
        "tested and nothing is checked.",
        "- **LOOKUP** = a claim whose answer is already printed in the briefing. A high LOOKUP "
        "share means survivors mostly confirm numbers the model was shown.",
        "- Sonnet runs through the Claude Code CLI on a subscription, with no fallback model.",
        "- Dan's answers are his expectations as the modeller, several explicitly hedged. "
        "They are the reference column, not an answer key; nothing here was graded automatically.",
        "- Per-claim detail for the harness arms: `qa_cards.csv`.",
        "",
    ]
    paths["table_md"].write_text("\n".join(md), encoding="utf-8")
    return paths


# --- the command -------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python3 -m llm_insights.agent.qa_compare",
        description="Run Dan's questions through Llama and Sonnet, with and without the "
        "harness, and write a comparison table.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--qa", type=Path, default=DEFAULT_QA_FILE, help="question/answer JSON")
    common.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")

    run = sub.add_parser("run", parents=[common], help="run arms, then write the table")
    run.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    run.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    run.add_argument("--questions", nargs="+", type=int, help="question numbers (default: all)")
    run.add_argument("--n", type=int, default=DEFAULT_N, help="claims per question (harness)")
    run.add_argument(
        "--force", action="store_true", help="redo questions that already have results"
    )
    run.add_argument(
        "--cli-model", default=DEFAULT_CLI_MODEL, help="Claude model alias for the sonnet arms"
    )
    run.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS, help="CLI call ceiling")
    run.add_argument(
        "--max-budget-usd", type=float, default=DEFAULT_MAX_BUDGET_USD, help="CLI cost ceiling"
    )
    run.add_argument(
        "--dataset-root", type=Path, default=None, help="FSG results root (default: in-repo)"
    )
    run.add_argument("--skip-preflight", action="store_true", help="skip the venv/dependency check")

    sub.add_parser("report", parents=[common], help="rebuild the table from finished results")
    return parser


def prepare_briefing(out: Path, dataset_root: Path | None) -> str:
    """Build the briefing once and write it where every arm will read it.

    If a previous run wrote a different briefing, the new one replaces it and a warning
    is printed: the arms that already ran read the old one, and the report will say so.

    Args:
        out: The output root.
        dataset_root: The FSG results root, or None for the default.

    Returns:
        The briefing text.
    """
    from llm_insights.api import open_study

    briefing = open_study(dataset_root).briefing
    path = out / "briefing.md"
    out.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") != briefing:
        print(
            "WARNING: the briefing has changed since results in this folder were produced. "
            "Arms that already ran used the old one; the report will flag the mismatch.",
            flush=True,
        )
    path.write_text(briefing, encoding="utf-8")
    print(f"briefing: {len(briefing):,} chars, sha256 {sha256(briefing)[:12]}", flush=True)
    return briefing


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command.

    Args:
        argv: Arguments, or None for ``sys.argv``.

    Returns:
        The exit status: 0 when every requested question has a result, 1 otherwise.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    items = load_qa(args.qa)

    if args.command == "report":
        paths = write_report(args.out, items)
        print(f"wrote {paths['table_md']}\n      {paths['table_csv']}")
        return 0

    if not args.skip_preflight:
        from llm_insights import preflight

        if not preflight.report():
            return 2

    chosen = select(items, args.questions)
    briefing = prepare_briefing(args.out, args.dataset_root)
    incomplete = False
    for model in MODELS:
        if model not in args.models:
            continue
        try:
            generator = make_backend(
                model,
                cli_model=args.cli_model,
                max_calls=args.max_calls,
                max_budget_usd=args.max_budget_usd,
            )
        except Exception as exc:
            print(f"[{model}] cannot start this backend -- {type(exc).__name__}: {exc}", flush=True)
            incomplete = True
            continue
        try:
            for arm in ARMS:
                if arm not in args.arms:
                    continue
                counts = run_arm(
                    model,
                    arm,
                    chosen,
                    generator=generator,
                    briefing=briefing,
                    out=args.out,
                    n=args.n,
                    force=args.force,
                    dataset_root=args.dataset_root,
                )
                print(
                    f"[{model}/{arm}] {counts['done']} run, {counts['skipped']} skipped, "
                    f"{counts['failed']} failed",
                    flush=True,
                )
                if counts["done"] + counts["skipped"] < len(chosen):
                    incomplete = True
            summary = getattr(generator, "usage_summary", None)
            if callable(summary):
                print(f"[{model}] usage: {summary()}", flush=True)
        finally:
            close = getattr(generator, "close", None)
            if callable(close):
                close()

    paths = write_report(args.out, items)
    print(f"\nwrote {paths['table_md']}\n      {paths['table_csv']}", flush=True)
    if incomplete:
        print(
            "some questions have no result yet -- run the same command again to resume", flush=True
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
