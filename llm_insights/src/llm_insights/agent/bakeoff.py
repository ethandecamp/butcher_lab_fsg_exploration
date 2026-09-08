r"""Measure candidate local models against the strict validator, and print the table.

Choosing a local model by reading benchmark cards is guessing. The only question that
matters for this package is narrower and answerable: *given this exact system prompt
and this exact briefing, what fraction of what the model returns survives
:func:`llm_insights.harness.spec.hypothesis_from_dict`?* A model that writes elegant
prose and cannot hold a JSON schema is useless here, and a 4B model that holds it is
worth more than an 8B model that does not.

So this module runs N proposal calls against each candidate through the real
:class:`~llm_insights.agent.local.LocalGenerator` -- same prompt, same parser, same
validator the loop uses -- and reports, per model:

* the **strict-validator rejection rate**: of the hypothesis objects that came back,
  how many the validator threw out;
* the **usable yield**: how many valid hypotheses were obtained against how many were
  asked for, which is the number that survives a call returning no JSON at all;
* mean latency and mean prompt/completion tokens per call;
* the distribution of rejection reasons, bucketed by what the validator complained
  about.

Nothing here judges a *claim*. It counts schema failures, which is arithmetic. The
bake-off is measurement infrastructure, not a verification path, and no result it
produces reaches a card.

``--canary`` is the other half of TASK-016 and answers a different question: did the
model actually receive the whole prompt? It appends a unique sentence to the very end
of the real briefing and asks for it back verbatim. A model whose context silently
truncated the prompt cannot answer, and that is the only way to tell truncation from a
model that simply read carelessly.

Run it::

    PYTHONPATH=src python3 -m llm_insights.agent.bakeoff \\
        --models llama3.1:8b qwen3:8b --rounds 5 --n 5

    PYTHONPATH=src python3 -m llm_insights.agent.bakeoff --canary --models llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import secrets
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from llm_insights.agent.local import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    LocalBackendError,
    LocalGenerator,
)
from llm_insights.harness.spec import hypothesis_from_dict

LOG = logging.getLogger(__name__)

#: The question the bake-off asks. Fixed so two runs of this module are comparable;
#: it is the demo question, so the measurement reflects the workload that matters.
DEFAULT_QUESTION: Final[str] = "What differs between the healthy and overflow cases, and why?"

#: How many proposal calls each model gets, and how many hypotheses each asks for.
DEFAULT_ROUNDS: Final[int] = 5
DEFAULT_N: Final[int] = 5

#: Patterns that bucket a validator message into a short reason. Ordered: the first
#: match wins. Anything unmatched is bucketed by its own leading words, so a new
#: failure mode shows up as itself rather than being folded into "other".
_REASON_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    (r"unexpected field\(s\) (\[[^\]]*\])", r"unexpected field(s) \1"),
    (r"missing required field '([^']+)'", r"missing field '\1'"),
    (r"field '([^']+)' must not be empty", r"empty field '\1'"),
    (r"field 'primitive' names '([^']+)', which is not a known test", r"unknown primitive"),
    (r"field '([^']+)' must be (\w+)", r"field '\1' wrong type"),
    (r"must be an object/dict", "not an object"),
)

#: The sentence appended to the briefing by ``--canary``. ``{token}`` is filled with a
#: fresh random word so no model can answer it from memory of an earlier run.
CANARY_TEMPLATE: Final[str] = (
    "END-OF-BRIEFING MARKER. The verification passphrase for this briefing is "
    "{token}. Nothing follows this line."
)

#: What the canary asks for. Deliberately trivial: the test is whether the end of the
#: prompt arrived, not whether the model can reason.
CANARY_INSTRUCTION: Final[str] = (
    "The text above is a briefing. Its final line states a verification passphrase. "
    "Reply with that passphrase and nothing else: no explanation, no punctuation, no "
    "quotation marks."
)

#: Completion budget for a canary call. Generous on purpose. The canary must measure
#: one thing -- did the end of the prompt arrive -- and at 32 tokens it did not: a
#: reasoning model such as ``qwen3:8b`` spent the whole budget inside ``<think>``,
#: Ollama routed that to its ``reasoning`` field, and the reply came back with
#: ``finish_reason: length`` and an empty ``content``. That is indistinguishable from
#: a truncated prompt, and confounding the instrument is worse than paying for a few
#: hundred more tokens.
CANARY_MAX_TOKENS: Final[int] = 512

#: System prompt for the canary call. The hypothesis system prompt orders strict JSON,
#: which is the wrong instruction for a one-word answer.
CANARY_SYSTEM: Final[str] = (
    "You answer questions about a document you have been given. You quote from it "
    "exactly and you never invent text that is not in it."
)


def classify_rejection(message: str) -> str:
    """Bucket a validator error message into a short, countable reason.

    Args:
        message: The :class:`ValueError` message from
            :func:`~llm_insights.harness.spec.hypothesis_from_dict`.

    Returns:
        A short reason string suitable as a histogram key.
    """
    for pattern, replacement in _REASON_PATTERNS:
        match = re.search(pattern, message)
        if match:
            return match.expand(replacement) if "\\" in replacement else replacement
    return " ".join(message.split()[:6])


@dataclass
class ModelReport:
    """Everything one candidate model's run measured.

    Attributes:
        model: The model tag as the server names it.
        json_mode: Whether ``response_format: json_object`` was requested.
        rounds: Proposal calls attempted.
        asked: Hypotheses requested across all calls.
        call_failures: Calls that produced no parseable hypothesis list at all --
            a transport error, or a reply with no JSON in it. Every hypothesis that
            call would have carried is lost, which is why ``rejection_rate`` alone is
            not the whole story.
        proposed: Hypothesis objects that came back and reached the validator.
        accepted: Of those, how many the strict validator accepted.
        rejected: Of those, how many it threw out.
        latencies: Wall-clock seconds per call that reached the server.
        prompt_tokens: Prompt tokens the server reported, per call.
        completion_tokens: Completion tokens the server reported, per call.
        reasons: Histogram of rejection reasons.
        errors: One line per failed call, kept so a bad run is diagnosable.
    """

    model: str
    json_mode: bool = False
    rounds: int = 0
    asked: int = 0
    call_failures: int = 0
    proposed: int = 0
    accepted: int = 0
    rejected: int = 0
    latencies: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def rejection_rate(self) -> float | None:
        """Fraction of returned hypothesis objects the validator threw out.

        Returns:
            The fraction in [0, 1], or None when nothing reached the validator, in
            which case reporting 0.0 would read as a perfect score for a model that
            produced nothing.
        """
        return self.rejected / self.proposed if self.proposed else None

    @property
    def usable_yield(self) -> float | None:
        """Valid hypotheses obtained, over hypotheses asked for.

        This is the number that does not flatter a model whose calls fail outright:
        a call returning no JSON contributes zero valid hypotheses here and is
        invisible to :attr:`rejection_rate`.

        Returns:
            The fraction, or None when nothing was asked for.
        """
        return self.accepted / self.asked if self.asked else None

    @property
    def mean_latency(self) -> float | None:
        """Mean wall-clock seconds per call that reached the server."""
        return sum(self.latencies) / len(self.latencies) if self.latencies else None

    @property
    def mean_prompt_tokens(self) -> float | None:
        """Mean prompt tokens per call, as the server reported them."""
        return sum(self.prompt_tokens) / len(self.prompt_tokens) if self.prompt_tokens else None

    @property
    def mean_completion_tokens(self) -> float | None:
        """Mean completion tokens per call, as the server reported them."""
        if not self.completion_tokens:
            return None
        return sum(self.completion_tokens) / len(self.completion_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report, derived numbers included.

        Returns:
            A JSON-ready mapping.
        """
        data = asdict(self)
        data["rejection_rate"] = self.rejection_rate
        data["usable_yield"] = self.usable_yield
        data["mean_latency_s"] = self.mean_latency
        data["mean_prompt_tokens"] = self.mean_prompt_tokens
        data["mean_completion_tokens"] = self.mean_completion_tokens
        return data


def score_batch(items: Sequence[Any], report: ModelReport) -> None:
    """Run one call's returned objects past the strict validator and count the result.

    Args:
        items: Whatever the generator's ``propose`` returned.
        report: The report to accumulate into, mutated in place.
    """
    counter = Counter(report.reasons)
    for item in items:
        report.proposed += 1
        try:
            hypothesis_from_dict(item)
        except (ValueError, TypeError) as exc:
            report.rejected += 1
            counter[classify_rejection(str(exc))] += 1
        else:
            report.accepted += 1
    report.reasons = dict(counter)


def run_model(
    model: str,
    briefing: str,
    question: str,
    *,
    rounds: int = DEFAULT_ROUNDS,
    n: int = DEFAULT_N,
    url: str | None = None,
    json_mode: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    make_generator: Any = LocalGenerator,
) -> ModelReport:
    """Run the bake-off for one candidate model.

    Args:
        model: Model tag to ask the server for.
        briefing: The factual summary the proposals must be about.
        question: The question to ask.
        rounds: How many proposal calls to make.
        n: How many hypotheses each call asks for.
        url: Base URL of the server, or None for the environment default.
        json_mode: Ask for ``response_format: json_object``.
        max_tokens: Sampling ceiling per call.
        temperature: Sampling temperature.
        make_generator: Factory for the backend. Injectable so a test can run this
            function end to end without a server.

    Returns:
        The finished :class:`ModelReport`.
    """
    generator = make_generator(
        model=model,
        url=url,
        json_mode=json_mode,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    report = ModelReport(model=model, json_mode=json_mode)

    for index in range(max(int(rounds), 0)):
        report.rounds += 1
        report.asked += n
        before_in = generator.tokens["input"]
        before_out = generator.tokens["output"]
        started = time.monotonic()
        try:
            items = generator.propose(briefing, question, n)
        except (LocalBackendError, ValueError) as exc:
            report.call_failures += 1
            report.errors.append(f"round {index + 1}: {type(exc).__name__}: {exc}")
            LOG.warning("%s round %d failed: %s", model, index + 1, exc)
            continue
        finally:
            report.latencies.append(time.monotonic() - started)
            report.prompt_tokens.append(generator.tokens["input"] - before_in)
            report.completion_tokens.append(generator.tokens["output"] - before_out)
        LOG.info("%s round %d returned %d object(s)", model, index + 1, len(items))
        score_batch(items, report)

    close = getattr(generator, "close", None)
    if callable(close):
        close()
    return report


def _cell(value: float | None, spec: str) -> str:
    """Format an optional number for the table, or ``n/a``.

    Args:
        value: The number, or None.
        spec: A format spec such as ``".0%"``.

    Returns:
        The formatted cell.
    """
    return "n/a" if value is None else format(value, spec)


def format_table(reports: Sequence[ModelReport]) -> str:
    """Render the bake-off result as a markdown table plus reason histograms.

    Args:
        reports: One report per candidate, in the order they were run.

    Returns:
        The rendered text.
    """
    header = (
        "| model | json | calls | failed | proposed | rejected | rejection rate | "
        "usable yield | mean latency | mean tok in | mean tok out |"
    )
    rule = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, rule]
    for report in reports:
        lines.append(
            "| {model} | {json} | {rounds} | {failed} | {proposed} | {rejected} | "
            "{rate} | {yield_} | {latency} | {tok_in} | {tok_out} |".format(
                model=report.model,
                json="yes" if report.json_mode else "no",
                rounds=report.rounds,
                failed=report.call_failures,
                proposed=report.proposed,
                rejected=report.rejected,
                rate=_cell(report.rejection_rate, ".1%"),
                yield_=_cell(report.usable_yield, ".1%"),
                latency=_cell(report.mean_latency, ".1f") + " s",
                tok_in=_cell(report.mean_prompt_tokens, ".0f"),
                tok_out=_cell(report.mean_completion_tokens, ".0f"),
            )
        )

    for report in reports:
        label = f"{report.model}{' (json mode)' if report.json_mode else ''}"
        lines.append("")
        lines.append(f"### {label} — why claims were rejected")
        if not report.reasons:
            lines.append("")
            lines.append(
                "(nothing was rejected)" if report.proposed else "(nothing reached the validator)"
            )
        else:
            lines.append("")
            for reason, count in sorted(report.reasons.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {count} x {reason}")
        for error in report.errors:
            lines.append(f"- CALL FAILED — {error}")
    return "\n".join(lines)


# --- the canary --------------------------------------------------------------------


@dataclass(frozen=True)
class CanaryResult:
    """What one canary check found.

    Attributes:
        model: The model that was asked.
        passphrase: The random passphrase placed at the very end of the briefing.
        reply: The model's reply, stripped.
        passed: True only if the passphrase appears in the reply.
        prompt_chars: Length of the user turn actually sent.
        context_length: The context length the server reported, or None.
        prompt_tokens: Prompt tokens the server reported for the call, or None.
    """

    model: str
    passphrase: str
    reply: str
    passed: bool
    prompt_chars: int
    context_length: int | None
    prompt_tokens: int | None

    def describe(self) -> str:
        """Render the result as the lines to paste into a task log.

        Returns:
            The rendered text.
        """
        return "\n".join(
            [
                f"model:            {self.model}",
                f"briefing + canary: {self.prompt_chars} characters sent",
                f"server context:   {self.context_length}",
                f"prompt tokens:    {self.prompt_tokens}",
                f"passphrase:       {self.passphrase}",
                f"model replied:    {self.reply!r}",
                f"CANARY:           {'PASS' if self.passed else 'FAIL'}",
            ]
        )


def run_canary(
    model: str,
    briefing: str,
    *,
    url: str | None = None,
    make_generator: Any = LocalGenerator,
    passphrase: str | None = None,
) -> CanaryResult:
    """Prove the model received the end of the briefing, not a truncated prefix.

    A passphrase that exists nowhere else is appended to the very last line of the
    real briefing. If the model can quote it, every byte before it arrived too. If it
    cannot, the prompt was truncated -- silently, because that is what an
    over-long prompt does -- and every claim measured through that server is about a
    briefing the model only partly read.

    Args:
        model: Model tag to ask.
        briefing: The full briefing, exactly as the loop would send it.
        url: Base URL of the server, or None for the environment default.
        make_generator: Factory for the backend, injectable for tests.
        passphrase: Fixed passphrase, for a deterministic test. Defaults to a fresh
            random one, so no model can answer from memory of an earlier run.

    Returns:
        The :class:`CanaryResult`.
    """
    token = passphrase or f"ELDERBERRY-{secrets.randbelow(9000) + 1000}"
    user = "\n".join(
        [briefing.rstrip(), "", CANARY_TEMPLATE.format(token=token), "", CANARY_INSTRUCTION]
    )
    generator = make_generator(model=model, url=url, max_tokens=CANARY_MAX_TOKENS, temperature=0.0)
    reply = generator.complete(user, CANARY_SYSTEM).strip()
    prompt_tokens = generator.tokens["input"] or None
    return CanaryResult(
        model=model,
        passphrase=token,
        reply=reply,
        passed=token in reply,
        prompt_chars=len(user),
        context_length=generator.context_length,
        prompt_tokens=prompt_tokens,
    )


# --- CLI ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="llm_insights.agent.bakeoff",
        description=(
            "Run N proposal calls against each candidate local model and report the "
            "strict-validator rejection rate, latency and token cost of each."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="model tags to compare, as the local server names them",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="calls per model")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="hypotheses requested per call")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="the question to ask")
    parser.add_argument("--url", default=None, help="base URL of the local server")
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help=(
            "also run each model with response_format=json_object, which constrains "
            "the reply to a JSON object while the schema is an array"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--briefing-file",
        type=Path,
        default=None,
        help="read the briefing from this file instead of building it from the dataset",
    )
    parser.add_argument("--root", type=Path, default=None, help="dataset root")
    parser.add_argument(
        "--canary",
        action="store_true",
        help=(
            "instead of the bake-off, append a unique passphrase to the end of the "
            "briefing and check that each model can quote it back, proving the prompt "
            "was not silently truncated"
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="write the raw report as JSON here")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def load_briefing(briefing_file: Path | None, root: Path | None) -> str:
    """Load the briefing the bake-off measures against.

    Args:
        briefing_file: A saved briefing to read, or None to build one.
        root: Dataset root, or None for the package default.

    Returns:
        The briefing text.
    """
    if briefing_file is not None:
        return briefing_file.read_text(encoding="utf-8")
    from llm_insights.api import DEFAULT_ROOT
    from llm_insights.io.dataset import Dataset
    from llm_insights.summary.briefing import build_briefing

    return str(build_briefing(Dataset(root if root is not None else DEFAULT_ROOT)))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bake-off, or the canary, and print the result.

    Args:
        argv: Command-line arguments, or None to read :data:`sys.argv`.

    Returns:
        Process exit status: 0 on success, 1 if a canary failed.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    briefing = load_briefing(args.briefing_file, args.root)
    print(f"briefing: {len(briefing)} characters\n")

    if args.canary:
        results = [run_canary(model, briefing, url=args.url) for model in args.models]
        for result in results:
            print(result.describe())
            print()
        if args.out is not None:
            args.out.write_text(
                json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
            )
        return 0 if all(r.passed for r in results) else 1

    reports: list[ModelReport] = []
    modes = [False, True] if args.json_mode else [False]
    for model in args.models:
        for mode in modes:
            LOG.info("bake-off: %s (json_mode=%s)", model, mode)
            reports.append(
                run_model(
                    model,
                    briefing,
                    args.question,
                    rounds=args.rounds,
                    n=args.n,
                    url=args.url,
                    json_mode=mode,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            )
    print(format_table(reports))
    if args.out is not None:
        args.out.write_text(json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CANARY_INSTRUCTION",
    "CANARY_MAX_TOKENS",
    "CANARY_SYSTEM",
    "CANARY_TEMPLATE",
    "DEFAULT_N",
    "DEFAULT_QUESTION",
    "DEFAULT_ROUNDS",
    "CanaryResult",
    "ModelReport",
    "build_parser",
    "classify_rejection",
    "format_table",
    "load_briefing",
    "main",
    "run_canary",
    "run_model",
    "score_batch",
]
