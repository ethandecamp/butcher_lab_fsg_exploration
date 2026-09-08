"""The package's callable surface: a study you can ask more than one question.

:func:`llm_insights.agent.loop.investigate` has always been the real entry point, but
it is stateless and reachable only through ``argparse``. Both cost something:

* **Stateless.** Every run starts blank. Ask a follow-up and the model happily
  re-proposes a hypothesis the simulator already refuted, because nothing tells it
  otherwise; and two questions in one session can hand two different claims the same
  id, which makes the accumulated cards ambiguous.
* **CLI-only.** The expensive half of a run is opening the dataset and building the
  briefing, and a caller that wanted to ask three questions paid for that three times.

A :class:`Study` fixes both. It builds the :class:`~llm_insights.io.dataset.Dataset`
and the briefing **once**, lazily, and keeps the ids and the verdicts of everything
asked so far. :meth:`Study.follow_up` differs from :meth:`Study.ask` in exactly one
way: the ids, claims and verdicts already settled are put in front of the model so it
does not re-propose them. Nothing else changes -- no verdict, no number, no primitive.

Nothing here judges anything. The generator writes claims; the harness runs them;
arithmetic decides. This module only arranges for that to happen more than once
without paying setup costs twice.

Typical use::

    from llm_insights.api import open_study

    with open_study() as study:
        first = study.ask("How much does inlet flow rate matter?", n=6)
        second = study.follow_up("Why did H4 fail?", n=3)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from llm_insights.agent.generator import build_followup_prompt
from llm_insights.agent.loop import investigate, write_transcript
from llm_insights.agent.synthesis import synthesize, transcript_entry
from llm_insights.cards.card import Card, cards_from_run, cards_to_json
from llm_insights.cards.render import verdict_of, write_report
from llm_insights.io.dataset import Dataset

LOG = logging.getLogger(__name__)

#: How many hypotheses a question asks for when the caller does not say.
DEFAULT_N: Final[int] = 5


def _default_root() -> Path:
    """Locate the FSG results tree, preferring the checkout this package lives in.

    In a normal checkout the model sits at ``<repo>/one_way_fsg_model``, two levels
    above ``src/llm_insights``. A staged copy is used as a fallback so the tool still
    runs in sandboxes where only the extracted artifacts are present.

    Returns:
        The first candidate path that exists, or the in-repo path if none do, so the
        error message names the location a user would actually expect.
    """
    in_repo = Path(__file__).resolve().parents[2].parent / "one_way_fsg_model"
    staged = Path("/mnt/user-data/uploads/butcher_lab_fsg_exploration/one_way_fsg_model")
    for candidate in (in_repo, staged):
        if (candidate / "FSG Results").is_dir():
            return candidate
    return in_repo


#: Where the dataset is looked for when no root is given.
DEFAULT_ROOT: Path = _default_root()


@dataclass(frozen=True)
class Answer:
    """Everything one question produced.

    Attributes:
        question: The question as it was asked, not the text the backend was given.
            A follow-up decorates the latter with prior work (see
            :func:`~llm_insights.agent.generator.build_followup_prompt`); the report
            header must still say what the human asked.
        cards: The finished :class:`~llm_insights.cards.card.Card` objects, in report
            order.
        meta: The provenance and count block :func:`investigate` built.
        blurb: The checked plain-English summary, or None when none was requested or
            it did not survive the numeric containment check.
        headline: The one-line lede above ``blurb``, or None.
        paths: Files written for this question, keyed by a short label, empty when no
            output directory was given.
    """

    question: str
    cards: tuple[Card, ...]
    meta: dict[str, Any]
    blurb: str | None = None
    headline: str | None = None
    paths: dict[str, Path] = field(default_factory=dict)

    def prior_entries(self) -> list[dict[str, str]]:
        """Describe this answer's claims for a later follow-up prompt.

        Returns:
            One mapping per card with ``id``, ``claim`` and ``verdict``. Deliberately
            no observed numbers: a follow-up is told what is settled so it does not
            repeat it, not told the answers, which would let it write a claim it has
            already been given the result of.
        """
        return [
            {"id": card.id, "claim": card.claim, "verdict": verdict_of(card).label}
            for card in self.cards
        ]


class _FollowUpGenerator:
    """A generator that folds prior work into the question, and delegates the rest.

    Every backend builds its own user turn from
    :func:`~llm_insights.agent.generator.build_user_prompt`, so the question is the
    one place session context can be injected without reaching into a backend or
    widening the :class:`~llm_insights.agent.generator.Generator` protocol. Everything
    other than :meth:`propose` -- ``narrow``, ``synthesize``, ``close``,
    ``usage_summary``, ``name``, ``model``, ``recorded_from`` -- is forwarded to the
    wrapped backend by :meth:`__getattr__`, so provenance still names the real
    backend rather than this wrapper.

    Args:
        inner: The real generator.
        prior: The claims already settled, as
            :meth:`Answer.prior_entries` returns them.
    """

    def __init__(self, inner: Any, prior: Sequence[Mapping[str, Any]]) -> None:
        self._inner = inner
        self._prior = [dict(item) for item in prior]

    def __repr__(self) -> str:
        """Return a representation naming the wrapped backend."""
        return f"_FollowUpGenerator({self._inner!r}, {len(self._prior)} prior claim(s))"

    def __getattr__(self, item: str) -> Any:
        """Forward every attribute this wrapper does not define to the backend.

        Args:
            item: The attribute name.

        Returns:
            The backend's attribute.

        Raises:
            AttributeError: If the backend does not have it either.
        """
        return getattr(self._inner, item)

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Ask the backend for hypotheses, with the settled claims in front.

        Args:
            briefing: The factual summary, passed through unchanged.
            question: The follow-up question, as the human asked it.
            n: How many hypotheses to request.

        Returns:
            Raw mappings, unvalidated, exactly as the backend returned them.
        """
        return self._inner.propose(briefing, build_followup_prompt(question, self._prior), n)

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Narrow a refuted claim through the backend, unmodified.

        Narrowing already carries the falsified claim and every observed number, so
        there is nothing a session context would add.

        Args:
            briefing: The factual summary.
            failed: The falsified hypothesis, serialized.
            outcome: Its outcome.

        Returns:
            The successor mapping, or None.
        """
        return self._inner.narrow(briefing, failed, outcome)


class Study:
    """One dataset, one briefing, and every question asked against them.

    The dataset and the briefing are built at most once, on first use, and reused by
    every question. Hypothesis ids are unique across the whole study, not merely
    within one question.

    Args:
        root: The dataset root. Defaults to :data:`DEFAULT_ROOT`.
        generator: The backend that writes claims. Required before asking anything;
            it is a constructor argument rather than a per-question one because a
            backend holds a session and a budget across questions.
        briefing_file: Read the briefing from this file instead of building it from
            the dataset. The same escape hatch the CLI exposes.
    """

    def __init__(  # noqa: D107 - documented on the class
        self,
        root: str | Path | None = None,
        *,
        generator: Any = None,
        briefing_file: str | Path | None = None,
    ) -> None:
        self.root: Path = Path(root) if root is not None else DEFAULT_ROOT
        self.generator = generator
        self.briefing_file: Path | None = Path(briefing_file) if briefing_file else None
        self._dataset: Dataset | None = None
        self._briefing: str | None = None
        self._history: list[Answer] = []
        self._seen_ids: set[str] = set()

    def __repr__(self) -> str:
        """Return a short representation naming the root and how much has been asked."""
        return f"Study(root={str(self.root)!r}, questions={len(self._history)})"

    def __enter__(self) -> Study:
        """Return this study, so it can be used as a context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the generator's resources on the way out.

        Args:
            *exc: The exception triple, unused; a failure still closes the backend.
        """
        del exc
        self.close()

    # --- the built-once halves ------------------------------------------------------

    @property
    def dataset(self) -> Dataset:
        """The dataset, opened on first access and reused afterwards."""
        if self._dataset is None:
            LOG.info("opening dataset at %s", self.root)
            self._dataset = Dataset(self.root)
        return self._dataset

    @property
    def briefing(self) -> str:
        """The briefing, built on first access and reused afterwards.

        The summary package is imported lazily for the same reason
        :mod:`llm_insights.agent.run` imports it lazily: a caller that supplies
        ``briefing_file`` can work without it.

        Returns:
            The briefing text.
        """
        if self._briefing is None:
            if self.briefing_file is not None:
                LOG.info("reading briefing from %s", self.briefing_file)
                self._briefing = self.briefing_file.read_text(encoding="utf-8")
            else:
                from llm_insights.summary.briefing import build_briefing

                LOG.info("building briefing from %s", self.root)
                self._briefing = str(build_briefing(self.dataset))
        return self._briefing

    # --- accumulated state ----------------------------------------------------------

    @property
    def history(self) -> list[Answer]:
        """Every :class:`Answer` so far, in the order the questions were asked."""
        return list(self._history)

    @property
    def cards(self) -> list[Card]:
        """Every card from every question so far, in order."""
        return [card for answer in self._history for card in answer.cards]

    @property
    def seen_ids(self) -> set[str]:
        """Every hypothesis id spent so far, which no later question may reuse."""
        return set(self._seen_ids)

    # --- asking ---------------------------------------------------------------------

    def ask(
        self,
        question: str,
        n: int = DEFAULT_N,
        *,
        out: str | Path | None = None,
        narrow_failures: bool = True,
        max_narrow_rounds: int = 1,
        min_rel_diff_floor: float = 0.0,
        synthesize_summary: bool = False,
    ) -> Answer:
        """Investigate one question, with no reference to anything asked before.

        Args:
            question: The question to investigate.
            n: How many hypotheses to ask for.
            out: Directory for ``cards.json``, the report and the transcript. None
                writes nothing.
            narrow_failures: When False, refuted claims are left standing.
            max_narrow_rounds: How many successive narrowing rounds to allow.
            min_rel_diff_floor: The operator's floor on ``compare_metric`` margins.
            synthesize_summary: Ask the backend for a checked plain-English summary
                after every verdict is fixed. Off by default, as on the CLI.

        Returns:
            The :class:`Answer`, which is also appended to :attr:`history`.

        Raises:
            RuntimeError: If no generator was given to the study.
        """
        return self._run(
            question,
            n,
            prior=(),
            **self._options(
                out, narrow_failures, max_narrow_rounds, min_rel_diff_floor, synthesize_summary
            ),
        )

    def follow_up(
        self,
        question: str,
        n: int = DEFAULT_N,
        *,
        out: str | Path | None = None,
        narrow_failures: bool = True,
        max_narrow_rounds: int = 1,
        min_rel_diff_floor: float = 0.0,
        synthesize_summary: bool = False,
    ) -> Answer:
        """Investigate one question, telling the model what is already settled.

        The only difference from :meth:`ask` is the proposal prompt: the ids, claims
        and verdicts of every card so far are put in front of the question so the
        model does not re-propose settled ground. Verification is untouched.

        Args:
            question: The follow-up question.
            n: How many hypotheses to ask for.
            out: Directory for the written artifacts, or None.
            narrow_failures: When False, refuted claims are left standing.
            max_narrow_rounds: How many successive narrowing rounds to allow.
            min_rel_diff_floor: The operator's floor on ``compare_metric`` margins.
            synthesize_summary: Ask for a checked plain-English summary.

        Returns:
            The :class:`Answer`.

        Raises:
            RuntimeError: If no generator was given to the study.
        """
        prior = [entry for answer in self._history for entry in answer.prior_entries()]
        return self._run(
            question,
            n,
            prior=prior,
            **self._options(
                out, narrow_failures, max_narrow_rounds, min_rel_diff_floor, synthesize_summary
            ),
        )

    @staticmethod
    def _options(
        out: str | Path | None,
        narrow_failures: bool,
        max_narrow_rounds: int,
        min_rel_diff_floor: float,
        synthesize_summary: bool,
    ) -> dict[str, Any]:
        """Bundle the options :meth:`ask` and :meth:`follow_up` share.

        Args:
            out: Output directory, or None.
            narrow_failures: Whether to narrow refuted claims.
            max_narrow_rounds: Narrowing round ceiling.
            min_rel_diff_floor: The operator's margin floor.
            synthesize_summary: Whether to ask for a narrative summary.

        Returns:
            The keyword arguments for :meth:`_run`.
        """
        return {
            "out": out,
            "narrow_failures": narrow_failures,
            "max_narrow_rounds": max_narrow_rounds,
            "min_rel_diff_floor": min_rel_diff_floor,
            "synthesize_summary": synthesize_summary,
        }

    def _run(
        self,
        question: str,
        n: int,
        *,
        prior: Sequence[Mapping[str, Any]],
        out: str | Path | None,
        narrow_failures: bool,
        max_narrow_rounds: int,
        min_rel_diff_floor: float,
        synthesize_summary: bool,
    ) -> Answer:
        """Run one question. The single path :meth:`ask` and :meth:`follow_up` share.

        Args:
            question: The question, as the human asked it.
            n: How many hypotheses to request.
            prior: Settled claims to put in front of the model, empty for :meth:`ask`.
            out: Output directory, or None.
            narrow_failures: Whether to narrow refuted claims.
            max_narrow_rounds: Narrowing round ceiling.
            min_rel_diff_floor: The operator's margin floor.
            synthesize_summary: Whether to ask for a narrative summary.

        Returns:
            The :class:`Answer`.

        Raises:
            RuntimeError: If the study has no generator.
        """
        if self.generator is None:
            raise RuntimeError(
                "this Study has no generator, so it cannot ask anything; pass one to "
                "open_study(generator=...)"
            )
        backend = _FollowUpGenerator(self.generator, prior) if prior else self.generator

        result = investigate(
            self.dataset,
            backend,
            self.briefing,
            question,
            n=n,
            narrow_failures=narrow_failures,
            max_narrow_rounds=max_narrow_rounds,
            min_rel_diff_floor=min_rel_diff_floor,
            seen_ids=self._seen_ids,
        )
        cards = cards_from_run(result.hypotheses, result.outcomes)

        # Synthesis runs after every verdict is fixed and the cards are frozen. It can
        # only add a paragraph; nothing below re-reads a verdict.
        narration = None
        narration_headline = None
        synthesis_record = None
        if synthesize_summary:
            outcome = synthesize(self.generator, result.question, cards)
            result.meta.update(outcome.provenance())
            narration = outcome.blurb
            narration_headline = outcome.headline
            synthesis_record = transcript_entry(outcome, result.question, len(cards))

        paths: dict[str, Path] = {}
        if out is not None:
            out_dir = Path(out)
            paths["cards:"] = out_dir / "cards.json"
            cards_to_json(cards, paths["cards:"])
            md_path, html_path = write_report(
                cards, result.question, result.meta, out_dir, narration, narration_headline
            )
            paths["markdown:"] = md_path
            paths["html:"] = html_path
            paths["transcript:"] = write_transcript(
                result, out_dir / "transcript.json", synthesis_record
            )

        answer = Answer(
            question=question,
            cards=tuple(cards),
            meta=result.meta,
            blurb=narration,
            headline=narration_headline,
            paths=paths,
        )
        self._history.append(answer)
        LOG.info(
            "study now holds %d question(s) and %d card(s)", len(self._history), len(self.cards)
        )
        return answer

    def close(self) -> None:
        """Release the generator's resources, if it holds any."""
        close = getattr(self.generator, "close", None)
        if callable(close):
            close()


def open_study(
    root: str | Path | None = None,
    *,
    generator: Any = None,
    briefing_file: str | Path | None = None,
) -> Study:
    """Open a study over one dataset.

    Nothing is read from disk here. The dataset and the briefing are built on first
    use, so constructing a study is free and a study that is never asked anything
    costs nothing.

    Args:
        root: The dataset root. Defaults to :data:`DEFAULT_ROOT`.
        generator: The backend that writes claims.
        briefing_file: Read the briefing from this file instead of building it.

    Returns:
        The :class:`Study`.
    """
    return Study(root, generator=generator, briefing_file=briefing_file)


__all__ = [
    "DEFAULT_N",
    "DEFAULT_ROOT",
    "Answer",
    "Study",
    "open_study",
]
