"""Environment checks that run before a single model call is spent.

This module exists because of one incident. On 2026-09-02 a live run produced a card
reading ``COULD NOT RUN: ModuleNotFoundError: No module named 'scipy'``. Nothing was
wrong with the project -- scipy was declared and installed -- the run simply used an
interpreter that was not the virtualenv's, because every documented command sets
``PYTHONPATH=src`` and therefore works whether or not the environment is activated.
That convenience removed a loud failure and installed a quiet one in its place.

Two things made it worse than a missing package normally is. The import sat deep inside
a primitive, so it failed only when a hypothesis happened to use that primitive; and the
failure arrived through the same channel a genuinely unevaluable claim uses, so an
environment fault was displayed as though it were a fact about the hypothesis.

The check therefore runs up front and says so plainly. It is deliberately cheap and
deliberately non-fatal for anything that is merely suspicious: a warning that stops a
demo is worse than the problem it warns about.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)

#: Modules the harness imports at run time. Every one of these is needed to evaluate
#: some primitive, so a missing one is fatal rather than advisory.
REQUIRED_MODULES: tuple[str, ...] = ("numpy", "pandas")


@dataclass(frozen=True)
class Problem:
    """One thing wrong with the environment.

    Attributes:
        fatal: True when a run cannot produce trustworthy results at all.
        message: What is wrong, in one line.
        fix: The command or action that resolves it.
    """

    fatal: bool
    message: str
    fix: str


def _package_root() -> Path:
    """Return the directory holding ``pyproject.toml``, i.e. the project root."""
    return Path(__file__).resolve().parents[2]


def in_virtualenv() -> bool:
    """Report whether the running interpreter is inside a virtual environment."""
    return sys.prefix != sys.base_prefix


def check(required: tuple[str, ...] = REQUIRED_MODULES) -> list[Problem]:
    """Inspect the environment and return everything worth telling the user.

    Args:
        required: Module names that must import for the harness to evaluate every
            primitive.

    Returns:
        The problems found, fatal ones first. An empty list means the environment is
        fit to run.
    """
    problems: list[Problem] = []
    root = _package_root()
    venv = root / ".venv"
    activate = f"source {venv / 'bin' / 'activate'}"

    missing = []
    for name in required:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)

    if missing:
        hint = (
            f"{activate} && pip install -e \".[dev]\""
            if venv.is_dir()
            else 'pip install -e ".[dev]"'
        )
        problems.append(
            Problem(
                fatal=True,
                message=(
                    f"cannot import {', '.join(missing)}, which the harness needs to "
                    f"evaluate its tests (interpreter: {sys.executable})"
                ),
                fix=hint,
            )
        )
    elif venv.is_dir() and not in_virtualenv():
        # Not fatal: the imports above already succeeded, so this run can proceed. It is
        # still worth saying, because it is the exact shape of the 2026-09-02 incident
        # and the next dependency to go missing will go missing the same way.
        problems.append(
            Problem(
                fatal=False,
                message=(
                    f"running outside the project virtualenv ({sys.executable}); the "
                    f"required modules resolved anyway, but any dependency this "
                    f"interpreter lacks will surface mid-run as a failed test rather "
                    f"than as a setup error"
                ),
                fix=activate,
            )
        )

    problems.sort(key=lambda p: not p.fatal)
    return problems


def format_problems(problems: list[Problem]) -> str:
    """Render problems as a block suitable for stderr.

    Args:
        problems: What :func:`check` returned.

    Returns:
        The rendered text, or an empty string when there is nothing to report.
    """
    if not problems:
        return ""
    lines = ["Environment check:"]
    for problem in problems:
        lines.append(f"  {'ERROR  ' if problem.fatal else 'WARNING'}  {problem.message}")
        lines.append(f"           fix: {problem.fix}")
    return "\n".join(lines)


def report(problems: list[Problem] | None = None) -> bool:
    """Print any problems and report whether it is safe to continue.

    Args:
        problems: Problems to report. Runs :func:`check` when omitted.

    Returns:
        False if a fatal problem was found, True otherwise.
    """
    found = check() if problems is None else problems
    text = format_problems(found)
    if text:
        print(text, file=sys.stderr)
    fatal = any(p.fatal for p in found)
    if fatal:
        print(
            "Refusing to start: fix the errors above first. No model calls were made.",
            file=sys.stderr,
        )
    return not fatal


__all__ = [
    "REQUIRED_MODULES",
    "Problem",
    "check",
    "format_problems",
    "in_virtualenv",
    "report",
]
