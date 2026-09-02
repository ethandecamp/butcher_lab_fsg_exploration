"""The investigation loop: propose, verify, narrow what the simulator refuted.

One pass through :func:`investigate` is: ask a generator for N claims, validate each
one strictly, run the survivors through the harness, and for every claim the
simulator *refuted* ask the generator for a narrower successor and run that too.

Three commitments shape the code:

* **Nothing is dropped silently.** A proposal that fails validation is recorded as a
  rejection with the reason, counted in ``meta``, and written into the transcript. A
  run that got four usable claims out of five must say so.
* **A refutation and a failure to run are different events.** Only outcomes that
  actually ran and did not pass are narrowed. A claim that could not be evaluated is
  evidence about the hypothesis or the harness, never about the model, and asking for
  a "narrower" version of it would manufacture a result out of a bug.
* **Provenance is recorded, not inferred.** ``meta`` names the generator, the model
  where there is one, and — for a replay — the backend that originally produced the
  text. The transcript holds every raw generator response, so any run can be replayed
  byte-for-byte by :class:`llm_insights.agent.generator.TranscriptGenerator`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_insights.agent.generator import TRANSCRIPT_SCHEMA_VERSION, generator_model
from llm_insights.harness.runner import run, run_all
from llm_insights.harness.spec import (
    Hypothesis,
    Outcome,
    hypothesis_from_dict,
    hypothesis_to_dict,
    outcome_to_dict,
)

LOG = logging.getLogger(__name__)

#: Version of the ``meta`` payload written by :func:`investigate`.
SCHEMA_VERSION: str = "1"

#: Longest rejection summary written into ``meta``; the full list lives in the
#: transcript, which is written to disk beside the report.
_SUMMARY_CHARS: int = 400


@dataclass(frozen=True)
class RunResult:
    """Everything one investigation produced.

    Attributes:
        question: The question the run set out to answer.
        hypotheses: Every admissible :class:`~llm_insights.harness.spec.Hypothesis`,
            in the order it was proposed, successors immediately after their parents.
        outcomes: One :class:`~llm_insights.harness.spec.Outcome` per hypothesis, in
            the same order.
        transcript: Every raw generator exchange plus every rejection, in order. This
            is the replay format; see :func:`write_transcript`.
        meta: Provenance and counts. See :func:`investigate` for each key.
    """

    question: str
    hypotheses: list[Hypothesis]
    outcomes: list[Outcome]
    transcript: list[dict]
    meta: dict[str, Any]


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _digest(text: str) -> str:
    """Return a short sha256 of the briefing, so a replay can be checked against it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _validate(
    raw_items: Sequence[Any], seen_ids: set[str], transcript: list[dict], stage: str
) -> tuple[list[Hypothesis], int]:
    """Validate raw generator output, recording every rejection.

    Args:
        raw_items: The mappings the generator returned.
        seen_ids: Ids already used in this run. Mutated with each accepted id.
        transcript: Appended to with one ``kind="rejected"`` entry per rejection.
        stage: ``"propose"`` or ``"narrow"``, recorded on rejections.

    Returns:
        A tuple ``(admissible, n_rejected)``.
    """
    admissible: list[Hypothesis] = []
    rejected = 0
    for index, raw in enumerate(raw_items):
        try:
            hypothesis = hypothesis_from_dict(raw)
            if hypothesis.id in seen_ids:
                raise ValueError(
                    f"hypothesis: id {hypothesis.id!r} was already used in this run; ids "
                    "must be unique so outcomes can be matched to claims"
                )
        except ValueError as exc:
            reason = str(exc)
            LOG.warning("rejected proposal %d at stage %s: %s", index, stage, reason)
            transcript.append(
                {"kind": "rejected", "stage": stage, "index": index, "reason": reason, "raw": raw}
            )
            rejected += 1
            continue
        seen_ids.add(hypothesis.id)
        admissible.append(hypothesis)
    return admissible, rejected


def _unique_id(candidate: str, seen_ids: set[str]) -> str:
    """Return an id not already used, suffixing the candidate if it collides."""
    if candidate not in seen_ids:
        return candidate
    for suffix in range(2, 100):
        alternative = f"{candidate}_r{suffix}"
        if alternative not in seen_ids:
            LOG.warning("successor id %s was already taken; using %s", candidate, alternative)
            return alternative
    raise ValueError(f"could not find a free id based on {candidate!r}")


def _refuted(pairs: Sequence[tuple[Hypothesis, Outcome]]) -> list[tuple[Hypothesis, Outcome]]:
    """Select the pairs the simulator actually refuted.

    A refutation means the test ran and the decision rule did not hold. An outcome
    carrying an ``error`` is excluded: it produced no evidence, so there is nothing to
    narrow.
    """
    return [(h, o) for h, o in pairs if o.error is None and not o.passed]


def investigate(
    ds: Any,
    generator: Any,
    briefing: str,
    question: str,
    n: int = 5,
    narrow_failures: bool = True,
    max_narrow_rounds: int = 1,
) -> RunResult:
    """Run one investigation end to end.

    The sequence is: propose ``n`` claims, validate each strictly, run all admissible
    claims against ``ds``, then — for each claim the simulator refuted — ask the
    generator for a narrower successor, link it to its parent through ``parent_id``,
    and run it too. Successors may themselves be narrowed, up to ``max_narrow_rounds``.

    Args:
        ds: The :class:`~llm_insights.io.dataset.Dataset` to test against. Passed
            straight to the harness; nothing here reads it.
        generator: Any :class:`~llm_insights.agent.generator.Generator`.
        briefing: The factual summary of the dataset given to the generator.
        question: The question the run is answering.
        n: How many claims to ask for.
        narrow_failures: When False, refuted claims are left standing and no
            narrowing is attempted.
        max_narrow_rounds: How many successive narrowing rounds to allow. One round
            means a refuted claim may get a successor, but that successor may not.

    Returns:
        The :class:`RunResult`. Its ``meta`` carries:

        ``generator``, ``model``, ``replayed_from`` (present only for a replay),
        ``generated_at`` (UTC ISO-8601), ``dataset_root``, ``question``,
        ``briefing_sha256``, ``schema_version``, and the counts ``n_proposed``,
        ``n_admissible``, ``n_rejected``, ``n_survived``, ``n_falsified``,
        ``n_narrowed``, ``n_could_not_run``.

        ``n_proposed`` counts every raw mapping the generator returned, from the
        proposal round and every narrowing round together; ``n_rejected`` counts those
        that failed validation. The four verdict counts partition the admissible
        claims exactly as the report's tally does: ``n_narrowed`` counts refuted
        claims that got a successor, ``n_falsified`` counts refuted claims that did
        not, and a successor is itself counted as survived or falsified.
    """
    transcript: list[dict] = []
    seen_ids: set[str] = set()
    briefing = briefing or ""
    n_proposed = 0
    n_rejected = 0

    LOG.info("proposing %d hypotheses via %s", n, getattr(generator, "name", generator))
    raw_items = generator.propose(briefing, question, n)
    raw_list = list(raw_items or [])
    n_proposed += len(raw_list)
    transcript.append(
        {
            "kind": "propose",
            "request": {"question": question, "n": int(n), "briefing_sha256": _digest(briefing)},
            "response": raw_list,
        }
    )

    hypotheses, rejected = _validate(raw_list, seen_ids, transcript, "propose")
    n_rejected += rejected
    outcomes = run_all(ds, hypotheses)
    narrowed_parent_ids: set[str] = set()

    pending = list(zip(hypotheses, outcomes, strict=True))
    rounds = int(max_narrow_rounds) if narrow_failures else 0
    for round_index in range(max(rounds, 0)):
        targets = _refuted(pending)
        if not targets:
            break
        LOG.info("narrowing round %d: %d refuted claim(s)", round_index + 1, len(targets))
        pending = []
        for parent, outcome in targets:
            successor, received, rejected = _narrow_one(
                generator, briefing, parent, outcome, seen_ids, transcript
            )
            n_proposed += received
            n_rejected += rejected
            if successor is None:
                continue
            successor_outcome = run(ds, successor)
            hypotheses.append(successor)
            outcomes.append(successor_outcome)
            narrowed_parent_ids.add(parent.id)
            pending.append((successor, successor_outcome))

    meta = _build_meta(
        generator=generator,
        ds=ds,
        question=question,
        briefing=briefing,
        hypotheses=hypotheses,
        outcomes=outcomes,
        narrowed_parent_ids=narrowed_parent_ids,
        n_proposed=n_proposed,
        n_rejected=n_rejected,
        transcript=transcript,
    )
    LOG.info(
        "run complete: %d admissible, %d survived, %d falsified, %d narrowed, %d could not run",
        meta["n_admissible"],
        meta["n_survived"],
        meta["n_falsified"],
        meta["n_narrowed"],
        meta["n_could_not_run"],
    )
    return RunResult(
        question=question,
        hypotheses=hypotheses,
        outcomes=outcomes,
        transcript=transcript,
        meta=meta,
    )


def _narrow_one(
    generator: Any,
    briefing: str,
    parent: Hypothesis,
    outcome: Outcome,
    seen_ids: set[str],
    transcript: list[dict],
) -> tuple[Hypothesis | None, int, int]:
    """Ask for one successor to a refuted claim and validate it.

    The successor's ``parent_id`` is set here rather than trusted from the generator,
    so the chain in the report reflects what the loop actually did. A successor that
    reuses its parent's id is renamed rather than rejected: the id is bookkeeping, and
    losing a legitimate narrowing over it would be the wrong trade.

    Args:
        generator: The generator to ask.
        briefing: The factual summary, passed through unchanged.
        parent: The refuted hypothesis.
        outcome: Its outcome, including every observed number.
        seen_ids: Ids already used; mutated when a successor is accepted.
        transcript: Appended to with the raw exchange and any rejection.

    Returns:
        A tuple ``(successor_or_None, n_received, n_rejected)``, where ``n_received``
        counts raw mappings the generator returned. A generator error is caught and
        recorded rather than aborting the run, because the claims already verified are
        still valid results.
    """
    failed_payload = hypothesis_to_dict(parent)
    outcome_payload = outcome_to_dict(outcome)
    try:
        raw = generator.narrow(briefing, failed_payload, outcome_payload)
    except Exception as exc:  # a backend failure must not discard verified results
        message = f"{type(exc).__name__}: {exc}"
        LOG.warning("narrowing %s failed: %s", parent.id, message)
        transcript.append(
            {"kind": "narrow", "failed_id": parent.id, "response": None, "error": message}
        )
        return None, 0, 0

    transcript.append({"kind": "narrow", "failed_id": parent.id, "response": raw})
    if raw is None:
        LOG.info("generator declined to narrow %s", parent.id)
        return None, 0, 0

    # Validate against a set that excludes the parent's own id, so reusing it is a
    # rename below rather than a duplicate-id rejection here.
    scratch = set(seen_ids)
    scratch.discard(parent.id)
    candidates, rejected = _validate([raw], scratch, transcript, "narrow")
    if not candidates:
        return None, 1, rejected

    successor = candidates[0]
    if successor.parent_id != parent.id:
        successor = replace(successor, parent_id=parent.id)
    final_id = _unique_id(successor.id, seen_ids)
    if final_id != successor.id:
        successor = replace(successor, id=final_id)
    seen_ids.add(final_id)
    LOG.info("narrowed %s into %s", parent.id, successor.id)
    return successor, 1, rejected


def _build_meta(
    *,
    generator: Any,
    ds: Any,
    question: str,
    briefing: str,
    hypotheses: Sequence[Hypothesis],
    outcomes: Sequence[Outcome],
    narrowed_parent_ids: set[str],
    n_proposed: int,
    n_rejected: int,
    transcript: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the provenance and count block. See :func:`investigate` for the keys."""
    n_survived = sum(1 for o in outcomes if o.error is None and o.passed)
    n_could_not_run = sum(1 for o in outcomes if o.error is not None)
    refuted = [o for o in outcomes if o.error is None and not o.passed]
    n_narrowed = sum(1 for o in refuted if o.hypothesis_id in narrowed_parent_ids)
    n_falsified = len(refuted) - n_narrowed

    reasons = [str(e.get("reason", "")) for e in transcript if e.get("kind") == "rejected"]
    summary = "; ".join(reasons)
    if len(summary) > _SUMMARY_CHARS:
        summary = summary[: _SUMMARY_CHARS - 3] + "..."

    root = getattr(ds, "root", None)
    meta: dict[str, Any] = {
        "generator": str(getattr(generator, "name", type(generator).__name__)),
        "model": generator_model(generator),
        "generated_at": _now(),
        "dataset_root": str(root) if root is not None else None,
        "question": question,
        "briefing_sha256": _digest(briefing),
        "schema_version": SCHEMA_VERSION,
        "n_proposed": int(n_proposed),
        "n_admissible": len(hypotheses),
        "n_rejected": int(n_rejected),
        "n_survived": int(n_survived),
        "n_falsified": int(n_falsified),
        "n_narrowed": int(n_narrowed),
        "n_could_not_run": int(n_could_not_run),
    }
    if summary:
        meta["rejection_reasons"] = summary
    replayed = getattr(generator, "recorded_from", None)
    if replayed:
        meta["replayed_from"] = json.dumps(dict(replayed), sort_keys=True, default=str)
    return meta


def write_transcript(result: RunResult, path: str | Path) -> Path:
    """Write a run's transcript in the format :class:`TranscriptGenerator` replays.

    The file records which backend originally produced the text, so a replay of it
    can report the true origin of the claims rather than claiming to be a live call.

    Args:
        result: The run to record.
        path: Destination file. Parent directories are created.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "recorded_from": {
            "generator": result.meta.get("generator"),
            "model": result.meta.get("model"),
            "generated_at": result.meta.get("generated_at"),
            "question": result.question,
        },
        "entries": list(result.transcript),
    }
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    LOG.info("wrote transcript with %d entries to %s", len(result.transcript), target)
    return target


__all__ = ["SCHEMA_VERSION", "RunResult", "investigate", "write_transcript"]
