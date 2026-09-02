"""Command-line entry point: run one investigation and write the report.

Usage::

    python -m llm_insights.agent.run --generator transcript --transcript data/run.json

The default backend is ``claude-cli``, which reaches the model through the Claude Code
CLI and therefore needs no API key. It is the default rather than ``transcript``
because a replay ignores the question it is given: defaulting to a replay would answer
a new question with old cards and say nothing about it, which is a worse failure in
front of an audience than a backend that refuses to start. A transcript replay is one
flag away, and the CLI backend refuses to run at all if the binary is missing.

Whichever backend runs, its identity is written into the report's provenance block.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from llm_insights.agent.generator import (
    GENERATOR_KINDS,
    build_system_prompt,
    build_user_prompt,
    make_generator,
)
from llm_insights.agent.loop import investigate, write_transcript
from llm_insights.cards.card import cards_from_run, cards_to_json
from llm_insights.cards.render import write_report
from llm_insights.io.dataset import Dataset

LOG = logging.getLogger(__name__)


def _default_root() -> Path:
    """Locate the FSG results tree, preferring the checkout this package lives in.

    In a normal checkout the model sits at ``<repo>/one_way_fsg_model``, two levels
    above ``src/llm_insights``. A staged copy is used as a fallback so the tool still
    runs in sandboxes where only the extracted artifacts are present.

    Returns:
        The first candidate path that exists, or the in-repo path if none do, so the
        error message names the location a user would actually expect.
    """
    in_repo = Path(__file__).resolve().parents[3].parent / "one_way_fsg_model"
    staged = Path("/mnt/user-data/uploads/butcher_lab_fsg_exploration/one_way_fsg_model")
    for candidate in (in_repo, staged):
        if (candidate / "FSG Results").is_dir():
            return candidate
    return in_repo


DEFAULT_ROOT: Path = _default_root()

#: The question a run answers when none is given.
DEFAULT_QUESTION: str = "How does inlet flow rate reshape the AV cushion and its EndMT program?"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="llm_insights.agent.run",
        description=(
            "Propose hypotheses about the FSG + GRN simulation, verify each one with "
            "executable code, and write a falsification report."
        ),
    )
    parser.add_argument(
        "--generator",
        choices=GENERATOR_KINDS,
        default="claude-cli",
        help=(
            "which backend writes the hypotheses (default: claude-cli, which needs no "
            "API key); paste is a manual fallback and is never chosen automatically"
        ),
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="recorded run to replay, required by --generator transcript",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="the question to investigate")
    parser.add_argument("--n", type=int, default=5, help="how many hypotheses to ask for")
    parser.add_argument(
        "--model",
        default=None,
        help="model id, used by --generator anthropic and --generator claude-cli",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help=(
            "ceiling on model calls for --generator claude-cli; a full run makes six "
            "to ten, so the default of 12 catches a runaway without firing in normal use"
        ),
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=None,
        help="cumulative cost ceiling for --generator claude-cli (default: 0.50)",
    )
    parser.add_argument(
        "--claude-binary",
        default=None,
        help="path to the claude executable, if it is not on PATH",
    )
    parser.add_argument(
        "--paste-dir",
        type=Path,
        default=None,
        help="where --generator paste writes its prompt and reply files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print the exact prompt the generator would send and exit without calling "
            "a model; use it to inspect cost before spending any allowance"
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data"), help="directory for the cards and report"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="dataset root")
    parser.add_argument(
        "--briefing-file",
        type=Path,
        default=None,
        help=(
            "read the briefing from this file instead of building it from the dataset; "
            "an escape hatch for running before the summary module lands"
        ),
    )
    parser.add_argument(
        "--no-narrow",
        action="store_true",
        help="do not ask the generator to narrow refuted claims",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="logging verbosity (default: INFO)",
    )
    return parser


def _load_briefing(ds: Dataset, briefing_file: Path | None) -> str:
    """Produce the briefing text the generator is given.

    The summary package is imported here rather than at module scope because it is
    being written in parallel with this one; importing it lazily means this CLI stays
    usable, and says exactly what is missing when it is not.

    Args:
        ds: The dataset to summarise.
        briefing_file: An explicit briefing to read instead, or None.

    Returns:
        The briefing text.

    Raises:
        SystemExit: With status 2 if neither a briefing file nor the summary module is
            available.
    """
    if briefing_file is not None:
        LOG.info("reading briefing from %s", briefing_file)
        return briefing_file.read_text(encoding="utf-8")
    try:
        from llm_insights.summary.briefing import build_briefing
    except ImportError as exc:
        print(
            "error: llm_insights.summary.briefing could not be imported, so no briefing "
            f"can be built ({exc}). That module is written separately; either install "
            "it, or pass --briefing-file PATH with a briefing to use instead.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    LOG.info("building briefing from %s", ds.root)
    return str(build_briefing(ds))


def _summarise(meta: dict[str, Any], paths: dict[str, Path]) -> str:
    """Format the closing human summary printed to stdout.

    Args:
        meta: The run metadata.
        paths: Written files, keyed by a short label.

    Returns:
        The text to print.
    """
    lines = [
        "",
        f"Generator:   {meta.get('generator')}",
        f"Model:       {meta.get('model') or '(none)'}",
    ]
    if meta.get("replayed_from"):
        lines.append(f"Replay of:   {meta['replayed_from']}")
    lines += [
        f"Dataset:     {meta.get('dataset_root')}",
        "",
        f"  proposed        {meta.get('n_proposed')}",
        f"  admissible      {meta.get('n_admissible')}",
        f"  rejected        {meta.get('n_rejected')}",
        f"  survived        {meta.get('n_survived')}",
        f"  falsified       {meta.get('n_falsified')}",
        f"  narrowed        {meta.get('n_narrowed')}",
        f"  could not run   {meta.get('n_could_not_run')}",
        "",
    ]
    lines += [f"{label:<12} {path}" for label, path in paths.items()]
    return "\n".join(lines)


def _open_dataset(root: Path) -> Dataset:
    """Open the dataset, exiting with a usable message if it cannot be read.

    Args:
        root: The dataset root.

    Returns:
        The opened dataset.

    Raises:
        SystemExit: With status 2 if the root cannot be opened.
    """
    try:
        return Dataset(root)
    except (FileNotFoundError, OSError) as exc:
        print(f"error: could not open the dataset at {root}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def _warn_on_stale_replay(generator: Any, question: str) -> None:
    """Warn when a replay is being asked a question it did not answer.

    A :class:`~llm_insights.agent.generator.TranscriptGenerator` ignores the question
    it is given and returns recorded claims. That is correct behaviour for a replay,
    and dangerous in a live demo: without this warning the report would carry the new
    question in its header and the old run's answers underneath it.

    Args:
        generator: The generator about to run.
        question: The question this run was given.
    """
    recorded = getattr(generator, "recorded_question", None)
    if recorded and str(recorded).strip() != question.strip():
        print(
            "warning: this transcript recorded the question\n"
            f"           {recorded!r}\n"
            f"         but this run was given\n"
            f"           {question!r}\n"
            "         A replay returns its recorded claims regardless of the question, "
            "so the report\n         will pair your question with the old run's answers. "
            "Use --generator claude-cli\n         to actually answer this question.",
            file=sys.stderr,
        )


def _close(generator: Any) -> None:
    """Release a backend's resources if it holds any.

    Args:
        generator: The generator that has finished, whatever its backend.
    """
    close = getattr(generator, "close", None)
    if callable(close):
        close()


def _dry_run(ds: Dataset, args: argparse.Namespace) -> int:
    """Print the prompts a live run would send, and make no model call.

    Args:
        ds: The dataset to summarise.
        args: Parsed arguments.

    Returns:
        Process exit status 0.
    """
    briefing = _load_briefing(ds, args.briefing_file)
    system = build_system_prompt()
    user = build_user_prompt(briefing, args.question, args.n)
    print("=" * 78)
    print("SYSTEM PROMPT")
    print("=" * 78)
    print(system)
    print()
    print("=" * 78)
    print("USER PROMPT (proposal round)")
    print("=" * 78)
    print(user)
    print()
    sys.stdout.flush()
    total = len(system) + len(user)
    print(
        f"dry run: {len(system)} + {len(user)} = {total} characters, roughly "
        f"{total // 4} tokens for the first call. No model was called.",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one investigation from the command line.

    Args:
        argv: Argument list, defaulting to :data:`sys.argv`.

    Returns:
        Process exit status: 0 on success, 2 on a usage or environment error.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    ds = _open_dataset(args.root)
    if args.dry_run:
        return _dry_run(ds, args)

    try:
        generator = make_generator(
            args.generator,
            transcript=args.transcript,
            model=args.model,
            max_calls=args.max_calls,
            max_budget_usd=args.max_budget_usd,
            binary=args.claude_binary,
            paste_dir=args.paste_dir,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _warn_on_stale_replay(generator, args.question)

    briefing = _load_briefing(ds, args.briefing_file)
    try:
        result = investigate(
            ds,
            generator,
            briefing,
            args.question,
            n=args.n,
            narrow_failures=not args.no_narrow,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        _close(generator)
        # The proposal round is the one generator call the loop does not guard, because
        # a run with no claims has nothing to report. Fail with the backend's own
        # message rather than a traceback, and say what to try instead.
        print(f"error: the {args.generator} backend failed: {exc}", file=sys.stderr)
        if args.generator == "claude-cli":
            print(
                "hint: check `claude -p hello` works in this terminal. If the model "
                "was refused, try --model sonnet. If your allowance is exhausted, run "
                "--generator transcript --transcript data/demo_transcript.json to "
                "replay the recorded demo.",
                file=sys.stderr,
            )
        return 2

    cards = cards_from_run(result.hypotheses, result.outcomes)
    out_dir = Path(args.out)
    cards_path = out_dir / "cards.json"
    cards_to_json(cards, cards_path)
    md_path, html_path = write_report(cards, result.question, result.meta, out_dir)
    transcript_path = write_transcript(result, out_dir / "transcript.json")

    print(
        _summarise(
            result.meta,
            {
                "cards:": cards_path,
                "markdown:": md_path,
                "html:": html_path,
                "transcript:": transcript_path,
            },
        )
    )
    usage = getattr(generator, "usage_summary", None)
    if callable(usage):
        print(f"Usage:       {usage()}\n")
    _close(generator)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
