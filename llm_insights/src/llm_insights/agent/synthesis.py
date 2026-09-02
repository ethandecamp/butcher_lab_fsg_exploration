"""Optional plain-English narration of a finished run, and the check that fences it.

Everything else in this package keeps the language model away from the verdict. It
proposes claims; :mod:`llm_insights.harness.runner` executes them; arithmetic in
:mod:`llm_insights.harness.primitives` decides what passed. This module is the one
place where model-written prose sits next to verified numbers, so it is built to make
that prose *checkable* rather than to make it persuasive:

* It runs **after** verification and receives finished
  :class:`~llm_insights.cards.card.Card` objects. There is no path from here back into
  a verdict: cards are frozen, nothing is re-run, and the blurb is passed to the
  renderers as a separate argument that never touches the tally or the card bodies.
* Every numeral the model writes is checked against the cards by
  :func:`check_numeric_containment`. A number that is not in some card's ``observed``
  mapping, not written in a card's own claim / rationale / decision rule, and not a
  small prose count is a **violation**, and one violation suppresses the whole blurb.
  At most one retry is spent naming the offending numerals back to the model.
* A backend that cannot narrate simply has no ``synthesize`` method, and the report
  omits the section. That is checked with :func:`getattr`, not with a protocol method,
  so no existing backend has to change to keep working.

The feature is off unless ``--synthesize`` is passed. With it off, no call is made and
nothing about a run changes.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from llm_insights.cards.card import Card
from llm_insights.cards.render import verdict_of

LOG = logging.getLogger(__name__)

#: How many sentences the model is asked for. Not enforced after the fact: suppressing
#: an otherwise-supported blurb over a sentence count would trade a real result for a
#: cosmetic one.
MIN_SENTENCES: Final[int] = 3
MAX_SENTENCES: Final[int] = 6

#: Sentence budget for the one-line lede that sits above the summary. A reader who
#: stops after this should still have the run's actual finding, not a description of
#: the report's structure.
HEADLINE_MIN_SENTENCES: Final[int] = 1
HEADLINE_MAX_SENTENCES: Final[int] = 2

#: Labels the model is asked to put in front of each part. Parsing on labels rather
#: than on paragraph breaks keeps a model that writes one long paragraph from silently
#: losing its headline into the body.
HEADLINE_LABEL: Final[str] = "HEADLINE:"
SUMMARY_LABEL: Final[str] = "SUMMARY:"

#: System prompt for the narration call. Deliberately *not*
#: :func:`llm_insights.agent.generator.build_system_prompt`, which orders strict JSON
#: and no prose -- exactly the opposite of what this call wants.
SYNTHESIS_SYSTEM_PROMPT: Final[str] = (
    "You write a short plain-English summary of a hypothesis-testing run over a "
    "fluid-solid-growth simulation of a developing heart valve cushion coupled to a "
    "gene-regulatory network model.\n"
    "\n"
    "A question was asked. Claims were written to probe it and every one of them was "
    "then tested by executable code. You are not judging anything and you cannot "
    "change any verdict. Your job is to answer that question from the verified "
    "results, for a biologist who will not read the table underneath.\n"
    "\n"
    "RULES\n"
    "1. Write exactly two labelled parts, in this order and nothing else:\n"
    f"   {HEADLINE_LABEL} {HEADLINE_MIN_SENTENCES} to {HEADLINE_MAX_SENTENCES} "
    "sentences that ANSWER THE QUESTION, directly, in its own terms. This is the "
    "single most important line in the report: a reader who stops here must come away "
    "with the answer, not with a description of the report. If the question is "
    "'How does mechanical stress relate to growth?', the headline says how it relates "
    "-- 'Higher mechanical loading went with less growth, not more, across all three "
    "flow conditions' -- and never 'Five claims were tested and four survived'. Lead "
    "with the answer, not with the evidence for it.\n"
    f"   {SUMMARY_LABEL} {MIN_SENTENCES} to {MAX_SENTENCES} sentences giving the "
    "evidence for that answer and any qualification it needs. The summary must stand "
    "on its own; do not open it with 'additionally' or otherwise write it as a "
    "continuation of the headline.\n"
    "   Plain English prose in both. No markdown, no further headings, no bullet "
    "points, no JSON, no code fences.\n"
    "2. Use only numbers that appear in the 'observed' block of some card. Do not "
    "round to a figure that is not there, do not add up numbers to make a new one, "
    "and do not state a number you were not given. Every numeral you write is checked "
    "against these cards by a program, and a single unsupported numeral suppresses "
    "your entire summary. Prose counts of the claims themselves (for example 'four of "
    "the five claims') are allowed.\n"
    "3. Never contradict a verdict. A claim marked FALSIFIED was refuted; do not "
    "describe it as supported, likely, or nearly true. A claim marked COULD NOT RUN "
    "produced no evidence in either direction; say so rather than guessing.\n"
    "4. Where a falsified claim was followed by a narrower one that survived, describe "
    "that as a chain: what was rejected, and what smaller claim replaced it. That "
    "sequence is the most informative thing in the run.\n"
    "5. If the numbers implicate saturation -- a large fraction of the domain sitting "
    "at a normalization ceiling, a clipped or pinned quantity, a variance that shrinks "
    "where the raw spread grows -- say plainly that the effect may be partly an "
    "artefact of that ceiling rather than biology.\n"
    "6. Answer only as far as the verified results allow. If they do not settle the "
    "question, the headline says what they do establish and that the question is not "
    "settled -- an honest partial answer, never a confident one the cards do not "
    "support, and never a refusal to answer at all.\n"
    "7. Do not speculate beyond the cards, do not recommend further work, and do not "
    "describe your own reasoning. Return the two labelled parts and nothing else."
)

#: Numeric literals: an optional sign, digits with optional thousands separators, an
#: optional fractional part, an optional exponent, and an optional percent sign.
_NUMERAL_RE: Final[re.Pattern[str]] = re.compile(
    r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
)

#: Characters that, immediately before a match, mean the digits are part of an
#: identifier rather than a number: ``H3``, ``wss_dyn_cm2``, ``flow_U0p0180``.
_IDENTIFIER_PREFIX: Final[str] = "_."

#: Floor on the match tolerance, as a fraction of the value's magnitude. Absorbs the
#: last-few-ulp differences between two correct implementations of the same statistic,
#: or the same one on two machines, without letting an invented figure through: nine
#: significant figures of agreement between unrelated quantities does not occur here.
_RELATIVE_FLOOR: Final[float] = 1e-9

#: Fields whose text counts as a place a numeral may legitimately come from. These are
#: the parts of a card the model itself wrote before any test ran, plus the rule the
#: run was held to; a number quoted out of one of them is quoting the record.
_TEXT_FIELDS: Final[tuple[str, ...]] = ("claim", "rationale", "decision_rule")


@dataclass(frozen=True)
class SynthesisResult:
    """What one narration attempt produced, and why it did or did not render.

    Attributes:
        blurb: The summary text to render, or None when nothing may be rendered.
        headline: The one-line lede that sits above ``blurb``, or None when the model
            did not label one. It is suppressed together with ``blurb`` and never on
            its own: the containment check runs over the whole reply, so a headline
            surviving a rejected summary would mean showing the unchecked half of a
            reply whose other half was rejected.
        status: One line for the report's provenance block, e.g. ``"rendered"`` or
            ``"suppressed: ..."``. Always set.
        attempted: True if a model call was actually made.
        suppressed: True if a blurb came back and the containment check rejected it.
        violations: The unsupported numerals, in the order they appeared.
        retried: True if the one permitted retry was spent.
        error: The backend failure that stopped the attempt, or None.
        raw: The last text the backend returned, kept for the transcript so a replay
            re-runs the same check over the same text. None when no text arrived.
    """

    blurb: str | None
    status: str
    headline: str | None = None
    attempted: bool = False
    suppressed: bool = False
    violations: tuple[str, ...] = ()
    retried: bool = False
    error: str | None = None
    raw: str | None = None

    @property
    def rendered(self) -> bool:
        """True when a blurb survived the containment check and will be shown."""
        return self.blurb is not None

    def provenance(self) -> dict[str, str]:
        """Return the provenance entries describing this attempt.

        Returns:
            A mapping to merge into a run's ``meta``. ``synthesis`` is always present;
            ``synthesis_violations`` appears only when a blurb was suppressed, so a
            suppressed summary is visible in the report rather than silent.
        """
        entries = {"synthesis": self.status}
        if self.suppressed and self.violations:
            entries["synthesis_violations"] = ", ".join(self.violations)
        return entries


def supports_synthesis(generator: Any) -> bool:
    """Report whether a backend can write a narrative summary.

    Duck-typed on purpose. The capability is optional, so a backend advertises it by
    having the method rather than by implementing a wider protocol, and every backend
    written before this feature keeps working untouched.

    Args:
        generator: Any generator.

    Returns:
        True if the generator exposes a callable ``synthesize``.
    """
    return callable(getattr(generator, "synthesize", None))


# --- Prompt ------------------------------------------------------------------------


def card_payload(card: Card) -> dict[str, Any]:
    """Describe one finished card for the narration prompt.

    Every field the summary is allowed to lean on is here and nothing else is: the
    verdict as the harness classified it, the rule that was fixed before the test ran,
    the primitive and its parameters, and the complete ``observed`` mapping. Sending
    the whole of ``observed`` matters -- a model given a rounded subset would have to
    invent the rest, and :func:`check_numeric_containment` would then suppress it.

    Args:
        card: The card to describe.

    Returns:
        A JSON-serializable mapping.
    """
    payload: dict[str, Any] = {
        "id": card.id,
        "claim": card.claim,
        "verdict": verdict_of(card).label,
        "decision_rule": card.decision_rule,
        "primitive": card.primitive,
        "params": dict(card.params),
        "observed": dict(card.observed),
    }
    if card.error:
        payload["could_not_run_because"] = card.error
    if card.revision_of:
        payload["narrows_falsified_claim"] = card.revision_of
    if card.revised_by:
        payload["was_narrowed_into"] = card.revised_by
    return payload


def build_synthesis_prompt(
    question: str, cards: Sequence[Card], violations: Sequence[str] = ()
) -> str:
    """Build the user turn asking for the narrative summary.

    Args:
        question: The question the run set out to answer.
        cards: Every finished card, in report order.
        violations: Numerals a previous attempt used that the cards do not support.
            When present the prompt names them back and asks for one corrected
            summary; this is the single retry described in the module docstring.

    Returns:
        The user message text.
    """
    payload = [card_payload(card) for card in cards]
    lines = [
        "QUESTION",
        question.strip() or "(not recorded)",
        "",
        f"RESULTS ({len(cards)} claim{'' if len(cards) == 1 else 's'}, already verified "
        "in code)",
        json.dumps(payload, indent=2, sort_keys=False, default=str),
        "",
    ]
    if violations:
        lines += [
            "Your previous summary was rejected because it contained "
            f"{len(violations)} number(s) that appear nowhere in the results above: "
            + ", ".join(str(v) for v in violations)
            + ".",
            "",
            "Write the summary again using only numbers that are present in an "
            "'observed' block, or no numbers at all. Do not restate the rejected "
            "figures in any form.",
            "",
        ]
    lines.append(
        f"Answer the question at the top of this message. Write the two labelled parts "
        f"now -- {HEADLINE_LABEL}, which answers it directly, then {SUMMARY_LABEL}, "
        "which gives the evidence. Follow the rules in your instructions and return "
        "only those two parts."
    )
    return "\n".join(lines)


# --- The containment check ---------------------------------------------------------


def _numerals(text: str) -> list[tuple[str, float]]:
    """Extract numeric literals from text, ignoring digits inside identifiers.

    ``H3``, ``wss_dyn_cm2`` and ``flow_U0p0180`` are names, not measurements, so a
    match whose first character is preceded by a letter, a digit, an underscore or a
    dot is skipped. A number followed by letters (``1.5x``, ``100Pa``) is still a
    number and is still checked.

    Args:
        text: Any text.

    Returns:
        ``(token, value)`` pairs in order of appearance. ``value`` drops thousands
        separators and any trailing percent sign, so ``"46.3%"`` yields ``46.3``.
    """
    found: list[tuple[str, float]] = []
    for match in _NUMERAL_RE.finditer(text):
        start = match.start()
        if start > 0 and (text[start - 1].isalnum() or text[start - 1] in _IDENTIFIER_PREFIX):
            continue
        token = match.group(0)
        try:
            value = float(token.rstrip("%").replace(",", ""))
        except ValueError:  # pragma: no cover - the pattern cannot produce one
            continue
        if math.isfinite(value):
            found.append((token, value))
    return found


def _tolerance(token: str, value: float = 0.0) -> float:
    """Return the tolerance a numeral must be matched within.

    Normally this is the precision the numeral itself displays: ``"1.5"`` is written to
    one decimal, so it stands for anything in ``[1.45, 1.55)`` and matches an observed
    ``1.4967``; ``"5"`` stands for ``[4.5, 5.5)``. That is what lets a summary round
    without being accused of inventing, while still catching a figure that is simply
    not in the data.

    Displayed precision alone is too strict at the top end. A model that copies an
    observed value verbatim quotes all seventeen significant figures, which asks for
    agreement to about 5e-17 -- tighter than floating point is reproducible. This bit
    a real run: a summary quoted ``r = 0.6369388762466359`` from a card computed with
    ``scipy.stats``, and re-checking it against the same correlation computed in numpy
    gave ``0.6369388762466373``. Both are correct, they differ by 2e-15 relative, and
    the summary was suppressed for inventing a number that was in fact its own. So the
    tolerance is floored at :data:`_RELATIVE_FLOOR` of the value's magnitude. Two
    genuinely different quantities in this dataset agreeing to nine significant figures
    does not happen, so the check loses nothing it was catching.

    Args:
        token: The numeral as it was written.
        value: The numeral's value, used for the relative floor.

    Returns:
        The larger of half a unit in the token's last displayed place and
        :data:`_RELATIVE_FLOOR` times the magnitude of ``value``.
    """
    body = token.rstrip("%").replace(",", "").lstrip("+-")
    exponent = 0
    if "e" in body.lower():
        mantissa, _, exp = body.lower().partition("e")
        try:
            exponent = int(exp)
        except ValueError:  # pragma: no cover - the pattern cannot produce one
            exponent = 0
        body = mantissa
    decimals = len(body.split(".")[1]) if "." in body else 0
    displayed = 0.5 * (10.0**-decimals) * (10.0**exponent)
    return max(displayed, abs(value) * _RELATIVE_FLOOR)


def _supported_values(cards: Sequence[Card]) -> tuple[set[float], set[str]]:
    """Collect every number the cards support, and every numeral they spell.

    Args:
        cards: The finished cards.

    Returns:
        A tuple ``(values, tokens)``: the finite numbers from every ``observed``
        mapping and from the text of every card's claim, rationale and decision rule,
        and the literal numeral strings those texts contain.
    """
    values: set[float] = set()
    tokens: set[str] = set()
    for card in cards:
        for value in card.observed.values():
            if math.isfinite(value):
                values.add(float(value))
        for field_name in _TEXT_FIELDS:
            for token, value in _numerals(str(getattr(card, field_name, "") or "")):
                tokens.add(token)
                values.add(value)
    return values, tokens


def check_numeric_containment(blurb: str, cards: Sequence[Card]) -> list[str]:
    """Return the numerals in ``blurb`` that the cards do not support.

    A numeral is supported when any of these holds:

    1. It matches a value in some card's ``observed`` mapping, within the tolerance
       implied by its own displayed precision (see :func:`_tolerance`). A token
       written as a percentage is also compared against the same values scaled by 100,
       because ``"46.3%"`` and ``0.463`` are the same measurement in different units.
    2. It appears in some card's claim, rationale or decision rule -- verbatim, or as
       the same value within that same tolerance.
    3. It is a whole number no greater than the number of cards, which is what makes
       "four of the five claims" sayable without quoting a measurement.

    Anything else is a violation, and one violation suppresses the whole blurb. The
    check is deliberately about numerals only: it cannot tell whether prose contradicts
    a verdict, which is why the prompt forbids that separately and why the verdict
    table, not the summary, remains the authoritative statement of what happened.

    Args:
        blurb: The model's summary text.
        cards: The finished cards the summary describes.

    Returns:
        The unsupported numerals as written, in order of first appearance, without
        duplicates. Empty when every numeral is supported.
    """
    cards = list(cards)
    values, tokens = _supported_values(cards)
    n_cards = len(cards)

    violations: list[str] = []
    for token, value in _numerals(blurb):
        if token in violations:
            continue
        if token in tokens:
            continue
        is_whole = "." not in token and "%" not in token and "e" not in token.lower()
        if is_whole and 0 <= value <= n_cards:
            continue
        tol = _tolerance(token, value)
        if any(abs(candidate - value) <= tol for candidate in values):
            continue
        if token.endswith("%") and any(
            abs(candidate * 100.0 - value) <= tol for candidate in values
        ):
            continue
        violations.append(token)
    return violations


# --- Orchestration -----------------------------------------------------------------


def _clean(raw: Any) -> str:
    """Normalize a backend's reply into one plain-text paragraph.

    Strips markdown fences the model may have wrapped the prose in, then collapses all
    whitespace so the same text renders identically in Markdown and in HTML.

    Args:
        raw: Whatever the backend returned.

    Returns:
        The cleaned text, possibly empty.
    """
    text = str(raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text.strip("`")
        newline = text.find("\n")
        if newline != -1 and text[:newline].strip().lower() in {"", "text", "markdown"}:
            text = text[newline + 1 :]
    # Collapse horizontal whitespace but keep line breaks: the labels the reply is
    # split on live at the start of a line, and flattening the reply to one line first
    # would make an unlabelled body indistinguishable from a labelled one.
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


def split_narration(text: str) -> tuple[str | None, str]:
    """Split a labelled reply into its headline and its summary.

    A model that ignores the labels must not lose its work, so an unlabelled reply
    becomes the summary with no headline rather than an error: the long summary is the
    part that existed first and the part a reader most needs.

    Args:
        text: The cleaned reply, with line structure preserved.

    Returns:
        A tuple ``(headline_or_None, summary)``. The summary is empty only when the
        reply was empty.
    """
    head_at = text.find(HEADLINE_LABEL)
    body_at = text.find(SUMMARY_LABEL)
    if head_at == -1 and body_at == -1:
        return None, " ".join(text.split())
    if body_at == -1:  # headline labelled, body not: everything after the label is one
        tail = text[head_at + len(HEADLINE_LABEL) :]
        return None, " ".join(tail.split())
    if head_at == -1 or head_at > body_at:
        return None, " ".join(text[body_at + len(SUMMARY_LABEL) :].split())
    headline = " ".join(text[head_at + len(HEADLINE_LABEL) : body_at].split())
    summary = " ".join(text[body_at + len(SUMMARY_LABEL) :].split())
    return (headline or None), summary


def _ask(generator: Any, prompt: str) -> tuple[str, str | None]:
    """Make one narration call, converting any backend failure into a message.

    A generator error must never discard verified results, so this mirrors
    :func:`llm_insights.agent.loop._narrow_one`: the exception is caught, named, and
    returned rather than propagated.

    Args:
        generator: The backend to ask.
        prompt: The complete user turn.

    Returns:
        A tuple ``(text, error)``. Exactly one of them is meaningful: on failure the
        text is empty and the error names the exception type.
    """
    try:
        raw = generator.synthesize(prompt)
    except Exception as exc:  # a backend failure must not discard verified results
        message = f"{type(exc).__name__}: {exc}"
        LOG.warning("the narrative summary could not be produced: %s", message)
        return "", message
    return _clean(raw), None


def synthesize(
    generator: Any,
    question: str,
    cards: Sequence[Card],
    *,
    allow_retry: bool = True,
) -> SynthesisResult:
    """Ask for a narrative summary of a finished run, and check it before allowing it.

    Nothing here can change a verdict: the cards are already frozen, no test is re-run,
    and a failure at any point returns a result whose ``blurb`` is None, which the
    renderers treat as "omit the section".

    Args:
        generator: The generator the run used. One without a ``synthesize`` method is
            reported as unavailable and no call is made.
        question: The question the run set out to answer.
        cards: The finished cards, in report order.
        allow_retry: Whether one -- and only one -- retry may be spent when the first
            summary fails :func:`check_numeric_containment`.

    Returns:
        The :class:`SynthesisResult`. Its ``status`` is written into the report's
        provenance block whether or not a blurb survived.
    """
    cards = list(cards)
    if not cards:
        LOG.info("no cards to summarise, so no narrative summary was requested")
        return SynthesisResult(blurb=None, status="not attempted: the run produced no claims")

    name = str(getattr(generator, "name", type(generator).__name__))
    if not supports_synthesis(generator):
        LOG.info("the %s backend cannot write a narrative summary", name)
        return SynthesisResult(
            blurb=None, status=f"not available: the {name} backend cannot write one"
        )

    LOG.info("asking %s for a narrative summary of %d card(s)", name, len(cards))
    text, error = _ask(generator, build_synthesis_prompt(question, cards))
    if error is not None:
        return SynthesisResult(blurb=None, status=f"failed: {error}", attempted=True, error=error)
    if not text:
        LOG.warning("%s returned an empty narrative summary", name)
        return SynthesisResult(
            blurb=None, status="not rendered: the backend returned no text", attempted=True
        )

    violations = check_numeric_containment(text, cards)
    if not violations:
        LOG.info("the narrative summary passed the numeric containment check")
        headline, summary = split_narration(text)
        if headline is None:
            LOG.warning("the reply carried no %s label; rendering the summary alone", HEADLINE_LABEL)
        return SynthesisResult(
            blurb=summary, headline=headline, status="rendered", attempted=True, raw=text
        )

    LOG.warning(
        "the narrative summary used %d numeral(s) the cards do not support: %s",
        len(violations),
        ", ".join(violations),
    )
    if not allow_retry:
        return _suppressed(text, violations, retried=False)

    LOG.info("retrying the narrative summary once, naming the unsupported numerals")
    retry_text, retry_error = _ask(generator, build_synthesis_prompt(question, cards, violations))
    if retry_error is not None or not retry_text:
        return _suppressed(text, violations, retried=True)
    retry_violations = check_numeric_containment(retry_text, cards)
    if not retry_violations:
        LOG.info("the retried narrative summary passed the numeric containment check")
        headline, summary = split_narration(retry_text)
        return SynthesisResult(
            blurb=summary,
            headline=headline,
            status="rendered after one retry",
            attempted=True,
            retried=True,
            raw=retry_text,
        )
    LOG.warning(
        "the retried narrative summary still used unsupported numerals (%s); suppressed",
        ", ".join(retry_violations),
    )
    return _suppressed(retry_text, retry_violations, retried=True)


def _suppressed(text: str, violations: Sequence[str], *, retried: bool) -> SynthesisResult:
    """Build the result for a blurb the containment check rejected.

    The text is still carried on the result, so a transcript records what the model
    actually wrote and a replay re-runs the same check over the same words. It is not
    carried in ``blurb``, which is the only field a renderer reads.

    Args:
        text: The rejected summary.
        violations: The unsupported numerals.
        retried: Whether the one permitted retry had been spent.

    Returns:
        A suppressed :class:`SynthesisResult`.
    """
    joined = ", ".join(violations)
    suffix = " after one retry" if retried else ""
    return SynthesisResult(
        blurb=None,
        status=f"suppressed{suffix}: unsupported numerals ({joined})",
        attempted=True,
        suppressed=True,
        violations=tuple(violations),
        retried=retried,
        raw=text,
    )


def transcript_entry(result: SynthesisResult, question: str, n_cards: int) -> dict[str, Any]:
    """Describe one narration attempt for the run transcript.

    Args:
        result: The attempt to record.
        question: The question the run answered.
        n_cards: How many cards the summary described.

    Returns:
        A transcript entry. ``kind`` is ``"synthesize"``, which every reader written
        before this feature ignores, so an older replay of a newer transcript still
        works.
    """
    entry: dict[str, Any] = {
        "kind": "synthesize",
        "request": {"question": question, "n_cards": int(n_cards)},
        "response": result.raw,
        "status": result.status,
    }
    if result.violations:
        entry["violations"] = list(result.violations)
    if result.error:
        entry["error"] = result.error
    return entry


def recorded_blurb(entries: Sequence[Mapping[str, Any]]) -> str | None:
    """Find the summary text recorded in a transcript, if it has one.

    Args:
        entries: The transcript's entries.

    Returns:
        The recorded text, or None when the transcript records no narration -- which
        is the case for every transcript written before this feature existed.
    """
    for entry in entries:
        if entry.get("kind") == "synthesize":
            response = entry.get("response")
            if isinstance(response, str) and response.strip():
                return response
    return None


__all__ = [
    "HEADLINE_LABEL",
    "HEADLINE_MAX_SENTENCES",
    "HEADLINE_MIN_SENTENCES",
    "SUMMARY_LABEL",
    "split_narration",
    "MAX_SENTENCES",
    "MIN_SENTENCES",
    "SYNTHESIS_SYSTEM_PROMPT",
    "SynthesisResult",
    "build_synthesis_prompt",
    "card_payload",
    "check_numeric_containment",
    "recorded_blurb",
    "supports_synthesis",
    "synthesize",
    "transcript_entry",
]
