"""Generators that authenticate by subscription login instead of an API key.

:mod:`llm_insights.agent.generator` reaches the model over the Anthropic Messages API,
which needs ``ANTHROPIC_API_KEY``. The two backends here need no key at all:

* :class:`ClaudeCLIGenerator` shells out to the Claude Code CLI (``claude -p``), which
  authenticates against a Claude subscription. This is the live backend the demo uses.
* :class:`PasteGenerator` prints the prompt, waits for a human to run it in a Claude
  chat window, and reads the reply back. It is a *fallback*, kept so a demo can still
  be given when the CLI is unavailable, and it is never selected automatically.

Both produce raw mappings in the same schema as every other generator, and both reuse
:mod:`llm_insights.agent.generator`'s prompt construction and response parsing
verbatim. Nothing downstream -- the harness, the cards, the report -- can tell which
backend wrote a claim, which is the property that makes swapping them safe.

Three cost guardrails apply to :class:`ClaudeCLIGenerator`, because a subscription
meters usage against a rolling allowance rather than a bill:

1. Each call runs in an empty temporary directory, so ``CLAUDE.md`` auto-discovery
   finds nothing and the repository's instructions are not read into the context of
   every request. This is also a correctness property: a claim must be written from
   the briefing alone. Note that the CLI's ``--bare`` flag would suppress discovery
   more thoroughly and **must not be used here** -- it also restricts authentication
   to ``ANTHROPIC_API_KEY``, and this module deliberately runs without one, so a
   ``--bare`` call would fail to authenticate at all.
2. Tools are forbidden and the turn limit is one, so a call is a single round trip
   rather than an agent session.
3. A call ceiling and a cumulative cost ceiling are enforced here, and the CLI's own
   ``--max-budget-usd`` is passed as well when the installed version supports it.

``ANTHROPIC_API_KEY`` is stripped from the subprocess environment on purpose: its
presence would silently route the call through metered API billing, which is the one
thing a subscription-backed run exists to avoid.

Logging uses the standard library rather than loguru, matching the rest of this
package. See TASK-006 for the reconciliation note.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, TextIO

from llm_insights.agent.generator import (
    build_narrow_prompt,
    build_system_prompt,
    build_user_prompt,
    parse_response,
    parse_single,
)

LOG = logging.getLogger(__name__)

#: Executable name looked up on PATH.
CLAUDE_BINARY: Final[str] = "claude"

#: Default model. The cheapest tier that can hold the output schema, chosen so a demo
#: run costs a negligible slice of a subscription's rolling allowance.
DEFAULT_CLI_MODEL: Final[str] = "haiku"

#: Model switched to when the account cannot use :data:`DEFAULT_CLI_MODEL`. A run is
#: roughly 40k input and 16k output tokens, so this is still a small slice of a
#: subscription allowance; set it to None to fail loudly instead of switching.
FALLBACK_CLI_MODEL: Final[str] = "sonnet"

#: Substrings that mark a CLI failure as "this account cannot use that model" rather
#: than a transport problem. Matched case-insensitively against the CLI's own stderr.
MODEL_UNAVAILABLE_MARKERS: Final[tuple[str, ...]] = (
    "model not found",
    "unknown model",
    "invalid model",
    "model is not available",
    "not available on your plan",
    "does not have access to",
    "unsupported model",
)

#: Hard ceiling on subprocess calls per generator instance. A full investigation makes
#: roughly six to ten, so this catches a runaway without firing in normal use.
DEFAULT_MAX_CALLS: Final[int] = 12

#: Cumulative cost ceiling in USD for one generator instance. The CLI reports a
#: per-call cost even on a subscription, where it is an equivalent-spend estimate
#: rather than an amount billed.
DEFAULT_MAX_BUDGET_USD: Final[float] = 0.50

#: Per-call subprocess timeout in seconds.
DEFAULT_TIMEOUT: Final[float] = 300.0

#: Environment variable removed from the subprocess environment. See the module
#: docstring: leaving it set would route a "no key needed" run through API billing.
API_KEY_ENV: Final[str] = "ANTHROPIC_API_KEY"

#: Marks a CLI failure as "you passed an option I do not have". Matched
#: case-insensitively; see :meth:`ClaudeCLIGenerator.call_once`, which retries once
#: with the minimal invocation rather than losing the run to a flag disagreement.
#: Every marker must name an *option*. An earlier draft included the substring
#: ``"error: unknown"``, which also matched ``"API Error: unknown model 'haiku'"`` --
#: so a model-availability failure spent an extra call on a pointless retry and then
#: permanently cleared the guardrail flags. Keep these narrow.
UNKNOWN_OPTION_MARKERS: Final[tuple[str, ...]] = (
    "unknown option",
    "unknown argument",
    "unknown flag",
    "unrecognized option",
    "unrecognised option",
    "invalid option",
    "bad option",
)

#: Flags this module uses when ``claude --help`` advertises them. Every one is
#: optional: an older CLI that lacks a flag still runs, with a warning naming the
#: guardrail that could not be applied.
OPTIONAL_FLAGS: Final[tuple[str, ...]] = (
    "--max-turns",
    "--max-budget-usd",
    "--system-prompt",
    "--allowedTools",
    "--allowed-tools",
    "--disallowedTools",
    "--disallowed-tools",
    "--strict-mcp-config",
)

#: Tools forbidden by name when the CLI offers ``--disallowedTools``. Preferred over
#: passing an empty ``--allowedTools`` list, because that flag is variadic and an
#: empty argument is a fragile way to say "nothing".
FORBIDDEN_TOOLS: Final[str] = ",".join(
    (
        "Bash",
        "BashOutput",
        "Edit",
        "Glob",
        "Grep",
        "KillShell",
        "NotebookEdit",
        "Read",
        "Skill",
        "SlashCommand",
        "Task",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
        "Write",
    )
)

#: What to tell a user whose CLI is missing.
INSTALL_HINT: Final[str] = (
    "install it with `npm install -g @anthropic-ai/claude-code`, run `claude` once to "
    "log in, then retry; or run with --generator transcript to replay a recorded run"
)


class BudgetExceededError(RuntimeError):
    """Raised when a run reaches its call ceiling or its cost ceiling.

    Subclasses :class:`RuntimeError` so existing error handling catches it unchanged,
    while a caller that wants to tell a budget stop from a transport failure still can.
    """


def _declared_flags(help_text: str) -> frozenset[str]:
    """Extract the option names an ``--help`` listing actually declares.

    A plain substring search over the whole help text is wrong, and dangerously so:
    the real CLI describes ``--bare`` with prose that names ``--system-prompt[-file]``,
    ``--add-dir``, ``--mcp-config`` and others. Treating a flag mentioned in someone
    else's description as supported would put an unknown option on the command line,
    and the CLI aborts on those -- losing the whole run.

    An option is only counted when it is declared, which is identified structurally:
    ``--help`` lists options at one fixed indent and wraps their descriptions at a
    deeper one, so the shallowest indent at which a line begins with ``-`` is the
    option column, and only lines at exactly that indent are declarations.

    Args:
        help_text: The combined stdout and stderr of ``claude --help``.

    Returns:
        Every long option the listing declares, including comma-separated aliases.
    """
    candidates: list[tuple[int, str]] = []
    for raw in help_text.splitlines():
        line = raw.strip()
        if not line.startswith("-") or not raw[:1].isspace():
            continue
        candidates.append((len(raw) - len(raw.lstrip()), line))
    if not candidates:
        return frozenset()

    option_column = min(indent for indent, _ in candidates)
    declared: set[str] = set()
    for indent, line in candidates:
        if indent != option_column:
            continue
        # The option column ends at the first two-space gap, where the description
        # starts. Aliases before that gap are comma-separated.
        for token in re.split(r"\s{2,}", line)[0].split(","):
            name = token.strip().split(" ")[0].split("=")[0].split("[")[0]
            if name.startswith("--"):
                declared.add(name)
    return frozenset(declared)


@lru_cache(maxsize=4)
def supported_flags(binary: str) -> frozenset[str]:
    """Report which optional flags the installed CLI advertises.

    The CLI's flag set changes between versions, and passing an unrecognised flag
    aborts the run. Rather than pin a version, this probes ``--help`` once per
    executable and uses only what is actually there.

    Args:
        binary: Path to the ``claude`` executable.

    Returns:
        The subset of :data:`OPTIONAL_FLAGS` the help text mentions, or an empty set
        if help cannot be read, which degrades to the minimal invocation.
    """
    try:
        completed = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.warning("could not read `%s --help` (%s); using the minimal invocation", binary, exc)
        return frozenset()
    declared = _declared_flags(f"{completed.stdout}\n{completed.stderr}")
    found = frozenset(flag for flag in OPTIONAL_FLAGS if flag in declared)
    LOG.debug("%s advertises %d of %d optional flags", binary, len(found), len(OPTIONAL_FLAGS))
    return found


def child_environment() -> dict[str, str]:
    """Build the subprocess environment with the API key removed.

    Returns:
        A copy of the current environment without :data:`API_KEY_ENV`.
    """
    env = dict(os.environ)
    if env.pop(API_KEY_ENV, None) is not None:
        LOG.info(
            "removed %s from the child environment; this call uses the "
            "subscription, not metered API billing",
            API_KEY_ENV,
        )
    return env


class ClaudeCLIGenerator:
    """Propose and narrow hypotheses by shelling out to the Claude Code CLI.

    Each call is one ``claude -p`` subprocess, run in an empty temporary directory
    with tools disabled and a one-turn limit, so it behaves as a single completion
    request rather than an agent session.

    The CLI wraps the prompt in its own agent instructions unless ``--system-prompt``
    is supported, in which case this module replaces them. :attr:`prompt_is_exact`
    records which happened, so a report can state whether the model saw exactly the
    prompt this package built.

    Args:
        model: Model alias or id passed to ``--model``.
        binary: Executable name or path, resolved on PATH at construction.
        max_calls: Hard ceiling on subprocess calls for this instance.
        max_budget_usd: Cumulative cost ceiling in USD for this instance.
        timeout: Per-call subprocess timeout in seconds.
        fallback_model: Model retried once if the account cannot use ``model``, or
            None to fail loudly instead of switching.

    Raises:
        RuntimeError: At construction, if the executable is not on PATH. Failing here
            rather than mid-run means a missing CLI costs nothing.
    """

    def __init__(  # noqa: D107 - documented on the class
        self,
        model: str = DEFAULT_CLI_MODEL,
        binary: str = CLAUDE_BINARY,
        max_calls: int = DEFAULT_MAX_CALLS,
        max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
        timeout: float = DEFAULT_TIMEOUT,
        fallback_model: str | None = FALLBACK_CLI_MODEL,
    ) -> None:
        resolved = shutil.which(binary)
        if resolved is None:
            raise RuntimeError(
                f"the Claude Code CLI ({binary!r}) is not on PATH, so no model can be "
                f"called without an API key; {INSTALL_HINT}"
            )
        self.binary = resolved
        self.model = str(model)
        self.max_calls = int(max_calls)
        self.max_budget_usd = float(max_budget_usd)
        self.timeout = float(timeout)
        self.fallback_model = str(fallback_model) if fallback_model else None
        #: Empty scratch directory every call runs in, so ``CLAUDE.md`` auto-discovery
        #: finds nothing. One per instance, not one per call: a directory the CLI has
        #: not seen before can trip its folder-trust check, and a single directory
        #: makes that a once-per-run problem instead of a once-per-call one.
        self.workdir = tempfile.mkdtemp(prefix="llm-insights-cli-")
        #: True once :attr:`model` has been switched to :attr:`fallback_model`.
        self.model_fallback_used = False
        self.name = f"claude-cli:{self.model}"

        self.flags = supported_flags(self.binary)
        self.prompt_is_exact = "--system-prompt" in self.flags

        #: Calls made so far, counted against :attr:`max_calls`.
        self.calls: int = 0
        #: Cumulative reported cost in USD, counted against :attr:`max_budget_usd`.
        self.cost_usd: float = 0.0
        #: Token totals reported by the CLI, when it reports them.
        self.tokens: dict[str, int] = {"input": 0, "output": 0}
        #: Injection point for tests. Signature matches :meth:`run_cli`.
        self.transport = self.run_cli

        if not self.prompt_is_exact:
            LOG.warning(
                "this CLI has no --system-prompt, so its own agent instructions wrap "
                "the prompt; the run is still valid but is not a clean instrument"
            )
        for flag in ("--max-turns", "--max-budget-usd"):
            if flag not in self.flags:
                LOG.warning("this CLI has no %s, so that guardrail is not applied", flag)
        LOG.info(
            "Claude Code generator ready: model %s, at most %d calls and $%.2f",
            self.model,
            self.max_calls,
            self.max_budget_usd,
        )

    def __repr__(self) -> str:
        """Return a short representation of the backend and its budget."""
        return (
            f"ClaudeCLIGenerator(model={self.model!r}, max_calls={self.max_calls}, "
            f"max_budget_usd={self.max_budget_usd})"
        )

    @property
    def tool_restriction(self) -> tuple[str, str] | None:
        """Return the flag and value that stop this CLI from using tools.

        A tool call turns one completion into an agent turn, which costs a second
        full-context request. Forbidding tools by name is preferred; an empty
        ``--allowedTools`` list is the fallback for a CLI that has no deny flag.

        Returns:
            The ``(flag, value)`` pair to append, or None when neither spelling is
            advertised and tools cannot be restricted.
        """
        for flag in ("--disallowedTools", "--disallowed-tools"):
            if flag in self.flags:
                return flag, FORBIDDEN_TOOLS
        for flag in ("--allowedTools", "--allowed-tools"):
            if flag in self.flags:
                return flag, ""
        return None

    def build_argv(self, prompt: str) -> list[str]:
        """Build the full command line for one call.

        Only flags the installed CLI advertises are included, so an older version runs
        with fewer guardrails rather than failing on an unrecognised option.

        Args:
            prompt: The complete prompt text, already including the system prompt when
                this CLI cannot take one separately.

        Returns:
            The argv list, ready for :func:`subprocess.run`.
        """
        argv = [self.binary, "-p", prompt, "--output-format", "json", "--model", self.model]
        if "--system-prompt" in self.flags:
            argv += ["--system-prompt", build_system_prompt()]
        if "--max-turns" in self.flags:
            argv += ["--max-turns", "1"]
        tools = self.tool_restriction
        if tools is not None:
            argv += [tools[0], tools[1]]
        if "--strict-mcp-config" in self.flags:
            argv.append("--strict-mcp-config")
        if "--max-budget-usd" in self.flags:
            # The full ceiling, not what is left of it. Each call is its own CLI
            # session, so this caps a single runaway call; the cumulative total is
            # this module's job, in check_budget. Passing the remainder would hand
            # the last permitted call a cap smaller than one call costs, turning a
            # clean stop into a confusing mid-call abort.
            argv += ["--max-budget-usd", f"{self.max_budget_usd:.2f}"]
        return argv

    def run_cli(self, argv: Sequence[str], cwd: str) -> str:
        """Run one CLI invocation and return its stdout.

        Args:
            argv: The command line.
            cwd: Working directory: an empty temporary directory, so the CLI cannot
                discover a project's ``CLAUDE.md`` or settings.

        Returns:
            The process's stdout.

        Raises:
            RuntimeError: On a non-zero exit, a timeout, or a failure to launch. The
                message carries the CLI's stderr, which is where an expired login or
                an exhausted allowance is reported.
        """
        try:
            completed = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd,
                env=child_environment(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"the Claude Code CLI did not respond within {self.timeout:.0f}s"
            ) from None
        except OSError as exc:
            raise RuntimeError(f"could not run the Claude Code CLI: {exc}") from None
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[:800]
            raise RuntimeError(f"the Claude Code CLI exited {completed.returncode}: {detail}")
        return completed.stdout

    def account(self, envelope: Mapping[str, Any]) -> None:
        """Record cost and token usage from one response envelope.

        Args:
            envelope: The decoded ``--output-format json`` object.
        """
        cost = envelope.get("total_cost_usd")
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            self.cost_usd += float(cost)
        usage = envelope.get("usage")
        if isinstance(usage, Mapping):
            for key, field in (("input", "input_tokens"), ("output", "output_tokens")):
                value = usage.get(field)
                if isinstance(value, int) and not isinstance(value, bool):
                    self.tokens[key] += value
        LOG.info(
            "call %d/%d: $%.4f cumulative, %d in / %d out tokens",
            self.calls,
            self.max_calls,
            self.cost_usd,
            self.tokens["input"],
            self.tokens["output"],
        )

    def extract(self, stdout: str) -> str:
        """Pull the assistant text out of the CLI's JSON envelope.

        Args:
            stdout: The process's stdout.

        Returns:
            The assistant's text.

        Raises:
            RuntimeError: If the envelope is not a JSON object, reports an error, or
                carries no result text.
        """
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            excerpt = stdout.strip().replace("\n", " ")[:200]
            raise RuntimeError(
                f"the Claude Code CLI did not return JSON; it began: {excerpt!r}"
            ) from None
        if not isinstance(envelope, Mapping):
            raise RuntimeError("the Claude Code CLI returned JSON that is not an object")
        self.account(envelope)
        if envelope.get("is_error"):
            subtype = envelope.get("subtype") or "unknown"
            detail = envelope.get("result")
            suffix = f": {str(detail)[:400]}" if isinstance(detail, str) and detail.strip() else ""
            raise RuntimeError(f"the Claude Code CLI reported an error ({subtype}){suffix}")
        text = envelope.get("result")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("the Claude Code CLI returned an empty result")
        return text

    def check_budget(self) -> None:
        """Stop the run before a call that would breach a ceiling.

        Raises:
            BudgetExceededError: If the call ceiling or the cost ceiling is reached.
        """
        if self.calls >= self.max_calls:
            raise BudgetExceededError(
                f"stopping: this run has already made {self.calls} model calls, which "
                "is its limit. Raise it with --max-calls if that is genuinely expected."
            )
        if self.cost_usd >= self.max_budget_usd:
            raise BudgetExceededError(
                f"stopping: this run has reached its ${self.max_budget_usd:.2f} ceiling "
                f"(${self.cost_usd:.4f} used). Raise it with --max-budget-usd."
            )

    def full_prompt(self, user_text: str) -> str:
        """Return the prompt string passed to the CLI as the ``-p`` argument.

        When the CLI accepts ``--system-prompt`` the system half is sent separately
        and this is the user half alone; otherwise the two are concatenated, because
        a prompt the model never sees is not a guardrail.

        Args:
            user_text: The varying half of the request.

        Returns:
            The text for ``-p``.
        """
        if "--system-prompt" in self.flags:
            return user_text
        return f"{build_system_prompt()}\n\n---\n\n{user_text}"

    def can_fall_back(self, message: str) -> bool:
        """Report whether a failure is "this account cannot use that model".

        Only that specific failure justifies silently spending a different model's
        allowance. A transport error, an exhausted quota or a bad prompt must not
        trigger a retry, because retrying those spends twice for the same failure.

        Args:
            message: The error message from the CLI.

        Returns:
            True if a one-time switch to :attr:`fallback_model` is warranted.
        """
        if self.fallback_model is None or self.model_fallback_used:
            return False
        if self.fallback_model == self.model:
            return False
        lowered = message.lower()
        return any(marker in lowered for marker in MODEL_UNAVAILABLE_MARKERS)

    def call_once(self, user_text: str) -> str:
        """Run one subprocess call, retrying once without optional flags if needed.

        The flag probe reads ``--help``, which is a good signal and not a guarantee.
        If the CLI rejects the command line as containing an unknown option, the
        guardrail flags are dropped and the call is retried once, because a demo that
        runs with fewer guardrails beats a demo that does not run. The reduced flag
        set is kept for the rest of the instance, so the fallback costs one call, not
        one per call.

        A failure :meth:`can_fall_back` recognises is re-raised untouched, so that
        one is handled by :meth:`complete` as a model switch. Only one retry may fire
        for one failure; otherwise a wrong model name would cost two calls and cost
        the run its guardrails on the way through.

        Args:
            user_text: The varying half of the request.

        Returns:
            The assistant's text.

        Raises:
            RuntimeError: On any CLI or parsing failure.
        """
        prompt = self.full_prompt(user_text)
        self.calls += 1
        try:
            return self.extract(self.transport(self.build_argv(prompt), self.workdir))
        except RuntimeError as exc:
            # can_fall_back takes precedence: a model the account cannot use is not a
            # flag problem, and retrying it here would spend a call to learn nothing
            # and then strip the guardrails for the rest of the run.
            if not self.flags or self.can_fall_back(str(exc)):
                raise
            if not self._is_unknown_option(str(exc)):
                raise
            LOG.warning(
                "this CLI rejected an option it advertises (%s); retrying once with "
                "the minimal invocation, so this run has fewer guardrails",
                exc,
            )
        self.flags = frozenset()
        self.prompt_is_exact = False
        self.check_budget()
        self.calls += 1
        return self.extract(
            self.transport(self.build_argv(self.full_prompt(user_text)), self.workdir)
        )

    @staticmethod
    def _is_unknown_option(message: str) -> bool:
        """Report whether a CLI failure was caused by an option it does not know.

        Args:
            message: The error message from the CLI.

        Returns:
            True if the message names an unknown or invalid option.
        """
        lowered = message.lower()
        return any(marker in lowered for marker in UNKNOWN_OPTION_MARKERS)

    def complete(self, user_text: str) -> str:
        """Make one call and return the assistant's text.

        If the account cannot use :attr:`model`, the call is retried once on
        :attr:`fallback_model` and :attr:`name` is updated, so the report names the
        model that actually answered rather than the one that was requested.

        Args:
            user_text: The varying half of the request.

        Returns:
            The assistant's text.

        Raises:
            BudgetExceededError: If a ceiling is reached before the call.
            RuntimeError: On any CLI or parsing failure.
        """
        self.check_budget()
        try:
            return self.call_once(user_text)
        except BudgetExceededError:
            raise
        except RuntimeError as exc:
            if not self.can_fall_back(str(exc)):
                raise
            LOG.warning(
                "this account cannot use %s (%s); retrying once on %s",
                self.model,
                exc,
                self.fallback_model,
            )
        self.model = str(self.fallback_model)
        self.model_fallback_used = True
        self.name = f"claude-cli:{self.model}"
        self.check_budget()
        return self.call_once(user_text)

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Ask the CLI for ``n`` hypotheses.

        Args:
            briefing: The factual summary of the dataset.
            question: The question the run is answering.
            n: How many hypotheses to request.

        Returns:
            Raw mappings, unvalidated.
        """
        LOG.info("asking the Claude Code CLI (%s) for %d hypotheses", self.model, n)
        return parse_response(self.complete(build_user_prompt(briefing, question, n)))

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Ask the CLI to narrow one falsified claim.

        Args:
            briefing: The factual summary of the dataset.
            failed: The falsified hypothesis, serialized.
            outcome: Its outcome, with every observed number.

        Returns:
            The successor mapping, or None if the model declined.
        """
        LOG.info("asking the Claude Code CLI to narrow %s", failed.get("id"))
        return parse_single(self.complete(build_narrow_prompt(briefing, failed, outcome)))

    def close(self) -> None:
        """Remove the scratch directory. Safe to call more than once."""
        shutil.rmtree(self.workdir, ignore_errors=True)

    def __enter__(self) -> ClaudeCLIGenerator:
        """Return self, so the generator can be used as a context manager.

        Returns:
            This generator.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Remove the scratch directory on exit.

        Args:
            *exc: Exception information, unused.
        """
        del exc
        self.close()

    def usage_summary(self) -> str:
        """Format what this run consumed, for printing after an investigation.

        Returns:
            A one-line summary of calls, cost and tokens.
        """
        return (
            f"{self.calls} call(s), ${self.cost_usd:.4f}, "
            f"{self.tokens['input']} in / {self.tokens['output']} out tokens"
        )


# --- Manual fallback ---------------------------------------------------------------

#: Clipboard commands by platform: (copy argv, paste argv).
CLIPBOARD: Final[dict[str, tuple[list[str], list[str]]]] = {
    "Darwin": (["pbcopy"], ["pbpaste"]),
    "Linux": (["xclip", "-selection", "clipboard"], ["xclip", "-selection", "clipboard", "-o"]),
}


def clipboard_commands() -> tuple[list[str], list[str]] | None:
    """Return the copy and paste commands for this platform, if both are installed.

    Returns:
        The two argv lists, or None when the clipboard cannot be used.
    """
    pair = CLIPBOARD.get(platform.system())
    if pair is None:
        return None
    copy_argv, paste_argv = pair
    if shutil.which(copy_argv[0]) is None or shutil.which(paste_argv[0]) is None:
        return None
    return copy_argv, paste_argv


class PasteGenerator:
    """Route prompts through a human and a Claude chat window.

    This is the fallback for a machine with no CLI and no API key. It writes each
    prompt to a file, copies it to the clipboard when it can, and waits for the
    operator to paste the model's reply back. Everything downstream is unchanged: the
    harness still executes every test in code, so a run driven this way is exactly as
    verifiable as any other.

    It is deliberately never chosen automatically. A run reaches this class only when
    someone asked for it by name.

    Args:
        workdir: Directory for the prompt and reply files.
        stdin: Stream the operator's acknowledgement is read from.
        stdout: Stream the instructions are printed to.
    """

    def __init__(  # noqa: D107 - documented on the class
        self,
        workdir: str | Path = "data/paste",
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self.model: str | None = None
        self.name = "paste:human-in-the-loop"
        self.calls = 0
        self.clipboard = clipboard_commands()
        LOG.info("paste generator ready; prompts and replies go to %s", self.workdir)

    def __repr__(self) -> str:
        """Return a short representation naming the working directory."""
        return f"PasteGenerator(workdir={str(self.workdir)!r})"

    def copy_to_clipboard(self, text: str) -> bool:
        """Put ``text`` on the clipboard.

        Args:
            text: The text to copy.

        Returns:
            True if the clipboard was written.
        """
        if self.clipboard is None:
            return False
        try:
            subprocess.run(self.clipboard[0], input=text, text=True, timeout=30.0, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            LOG.warning("could not write the clipboard (%s)", exc)
            return False
        return True

    def read_clipboard(self) -> str:
        """Read the clipboard.

        Returns:
            The clipboard contents, or an empty string if it cannot be read.
        """
        if self.clipboard is None:
            return ""
        try:
            completed = subprocess.run(
                self.clipboard[1], capture_output=True, text=True, timeout=30.0, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            LOG.warning("could not read the clipboard (%s)", exc)
            return ""
        return completed.stdout

    def exchange(self, label: str, prompt: str) -> str:
        """Run one prompt past the operator and collect the reply.

        Args:
            label: Short name for this exchange, used in the filenames.
            prompt: The complete prompt, system instructions included.

        Returns:
            The operator-supplied reply text.

        Raises:
            RuntimeError: If no reply is found in the clipboard or the reply file.
        """
        self.calls += 1
        stem = f"{self.calls:02d}_{label}"
        prompt_path = self.workdir / f"{stem}_prompt.txt"
        reply_path = self.workdir / f"{stem}_reply.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        copied = self.copy_to_clipboard(prompt)

        where = (
            "  the prompt is on your clipboard; paste it into a Claude chat"
            if copied
            else f"  open {prompt_path} and paste its contents into a Claude chat"
        )
        print(
            "\n".join(
                [
                    "",
                    f"  === paste exchange {self.calls}: {label} ===",
                    f"  prompt written to {prompt_path}",
                    where,
                    "  copy Claude's whole reply, then press Enter here",
                    f"  (or save the reply to {reply_path} and press Enter)",
                    "",
                ]
            ),
            file=self.stdout,
        )
        self.stdin.readline()

        reply = self.read_clipboard().strip()
        if not reply and reply_path.exists():
            reply = reply_path.read_text(encoding="utf-8").strip()
        if not reply:
            raise RuntimeError(
                f"no reply found: the clipboard was empty and {reply_path} does not "
                "exist. Copy Claude's reply, or save it to that file, then rerun."
            )
        reply_path.write_text(reply, encoding="utf-8")
        return reply

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Ask the operator to run the proposal prompt.

        Args:
            briefing: The factual summary of the dataset.
            question: The question the run is answering.
            n: How many hypotheses to request.

        Returns:
            Raw mappings, unvalidated.
        """
        body = build_user_prompt(briefing, question, n)
        prompt = f"{build_system_prompt()}\n\n---\n\n{body}"
        return parse_response(self.exchange("propose", prompt))

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Ask the operator to run the narrowing prompt.

        Args:
            briefing: The factual summary of the dataset.
            failed: The falsified hypothesis, serialized.
            outcome: Its outcome, with every observed number.

        Returns:
            The successor mapping, or None if the reply declined.
        """
        body = build_narrow_prompt(briefing, failed, outcome)
        prompt = f"{build_system_prompt()}\n\n---\n\n{body}"
        label = f"narrow_{failed.get('id', 'unknown')}"
        return parse_single(self.exchange(label, prompt))


__all__ = [
    "API_KEY_ENV",
    "CLAUDE_BINARY",
    "CLIPBOARD",
    "DEFAULT_CLI_MODEL",
    "DEFAULT_MAX_BUDGET_USD",
    "DEFAULT_MAX_CALLS",
    "DEFAULT_TIMEOUT",
    "FALLBACK_CLI_MODEL",
    "FORBIDDEN_TOOLS",
    "INSTALL_HINT",
    "MODEL_UNAVAILABLE_MARKERS",
    "OPTIONAL_FLAGS",
    "UNKNOWN_OPTION_MARKERS",
    "BudgetExceededError",
    "ClaudeCLIGenerator",
    "PasteGenerator",
    "child_environment",
    "clipboard_commands",
    "supported_flags",
]
