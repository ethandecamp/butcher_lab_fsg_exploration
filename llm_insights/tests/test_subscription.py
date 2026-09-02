"""Tests for the subscription-backed generators: the Claude Code CLI and the paste fallback.

Written against :mod:`unittest` for the same reason as :mod:`tests.test_agent`, and
following the same rule: no test here runs the real CLI or makes a network call. The
subprocess boundary is crossed only through an injected transport, or through a stub
executable created in a temporary directory.

The envelope fixtures mirror a real ``claude -p --output-format json`` response
captured from the CLI, fenced ``result`` string included, so a change in how this
package reads that envelope fails here rather than in front of an audience.

Run with::

    cd src && python3 -m unittest discover -s ../tests -t .. -v
"""

from __future__ import annotations

import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from llm_insights.agent.generator import (
    GENERATOR_KINDS,
    TranscriptGenerator,
    build_system_prompt,
    make_generator,
)
from llm_insights.agent.subscription import (
    API_KEY_ENV,
    FALLBACK_CLI_MODEL,
    FORBIDDEN_TOOLS,
    BudgetExceededError,
    ClaudeCLIGenerator,
    PasteGenerator,
    _declared_flags,
    child_environment,
    supported_flags,
)

#: Every flag the real CLI advertises that this package looks for, plus ``--bare``,
#: which it must never use. Kept in one place so a test can drop individual lines.
FULL_HELP = """Usage: claude [options] [command] [prompt]

Options:
  --allowedTools, --allowed-tools <tools...>        allow list
  --append-system-prompt <prompt>                   append
  --bare                                            minimal mode
  --disallowedTools, --disallowed-tools <tools...>  deny list
  --max-budget-usd <usd>                            hard dollar cap
  --max-turns <n>                                   cap agentic turns
  --model <model>                                   model
  --output-format <format>                          text, json, stream-json
  -p, --print                                       print mode
  --strict-mcp-config                               only use --mcp-config servers
  --system-prompt <prompt>                          replace the system prompt
"""

#: A hypothesis the harness would accept, used as the CLI's answer.
VALID_CLAIM: dict[str, Any] = {
    "id": "H1",
    "claim": "Peak wall shear stress rises with inlet velocity.",
    "rationale": "Higher inlet velocity should raise near-wall gradients.",
    "test": {
        "primitive": "fraction_above",
        "params": {
            "field": "von_mises",
            "case": "Healthy",
            "threshold": 0.0,
            "op": "gt",
            "value": 0.0,
        },
        "decision_rule": "More than 0% of solid nodes exceed 0 Pa.",
    },
}


def envelope(
    text: str, *, error: bool = False, cost: float = 0.0161, tokens: tuple[int, int] = (10, 87)
) -> str:
    """Build a CLI response envelope matching the real one's field names.

    Args:
        text: The ``result`` string.
        error: Whether to set ``is_error``.
        cost: Reported cost in USD.
        tokens: ``(input, output)`` token counts.

    Returns:
        The envelope, serialized as the CLI would print it.
    """
    return json.dumps(
        {
            "type": "result",
            "subtype": "error_during_execution" if error else "success",
            "is_error": error,
            "num_turns": 1,
            "session_id": "fake",
            "total_cost_usd": cost,
            "usage": {
                "input_tokens": tokens[0],
                "output_tokens": tokens[1],
                "cache_read_input_tokens": 7548,
                "cache_creation_input_tokens": 7435,
            },
            "result": text,
        }
    )


def fenced(payload: Any) -> str:
    """Wrap ``payload`` in a markdown JSON fence, as the real CLI does.

    Args:
        payload: Any JSON-serializable value.

    Returns:
        The fenced text.
    """
    return "```json\n" + json.dumps(payload) + "\n```"


class GeneratorFixture(unittest.TestCase):
    """Base class providing a :class:`ClaudeCLIGenerator` with no real CLI behind it."""

    help_text = FULL_HELP

    def setUp(self) -> None:
        """Build a generator whose binary resolves but is never executed."""
        supported_flags.cache_clear()
        self.addCleanup(supported_flags.cache_clear)
        patcher = mock.patch(
            "llm_insights.agent.subscription.shutil.which", return_value="/usr/bin/claude"
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        run_patcher = mock.patch(
            "llm_insights.agent.subscription.subprocess.run",
            return_value=mock.Mock(stdout=self.help_text, stderr="", returncode=0),
        )
        run_patcher.start()
        self.addCleanup(run_patcher.stop)

        self.gen = ClaudeCLIGenerator(model="sonnet")
        self.addCleanup(self.gen.close)
        self.calls: list[list[str]] = []

    def answer(self, *responses: str) -> None:
        """Install a transport that returns ``responses`` in order.

        Args:
            *responses: Envelope strings, one per expected call.
        """
        queue = list(responses)

        def transport(argv: list[str], cwd: str) -> str:
            self.calls.append(list(argv))
            self.assertTrue(Path(cwd).is_dir(), "the CLI must run in a real directory")
            self.assertEqual(list(Path(cwd).iterdir()), [], "the scratch dir must be empty")
            return queue.pop(0)

        self.gen.transport = transport


class TestConstruction(unittest.TestCase):
    """Construction fails fast and helpfully when the CLI is absent."""

    def test_missing_binary_raises_with_an_actionable_message(self):
        """A missing CLI must be reported before any work is done."""
        supported_flags.cache_clear()
        with (
            mock.patch("llm_insights.agent.subscription.shutil.which", return_value=None),
            self.assertRaises(RuntimeError) as caught,
        ):
            ClaudeCLIGenerator()
        message = str(caught.exception)
        self.assertIn("not on PATH", message)
        self.assertIn("npm install -g @anthropic-ai/claude-code", message)
        self.assertIn("--generator transcript", message)


class TestArgv(GeneratorFixture):
    """The command line carries every guardrail the installed CLI supports."""

    def test_argv_carries_the_guardrails(self):
        """Model, turn limit, tool ban and budget are all present."""
        argv = self.gen.build_argv("user text")
        self.assertEqual(argv[1], "-p")
        self.assertIn("--output-format", argv)
        self.assertEqual(argv[argv.index("--model") + 1], "sonnet")
        self.assertEqual(argv[argv.index("--max-turns") + 1], "1")
        self.assertEqual(argv[argv.index("--disallowedTools") + 1], FORBIDDEN_TOOLS)
        self.assertIn("--strict-mcp-config", argv)
        self.assertEqual(argv[argv.index("--max-budget-usd") + 1], "0.50")

    def test_bare_is_never_passed(self):
        """``--bare`` restricts auth to an API key, which a keyless run does not have.

        This is a regression guard, not a style preference: passing ``--bare`` here
        would stop the CLI reading the subscription login and every call would fail
        to authenticate.
        """
        self.assertIn("--bare", self.help_text, "the fixture must advertise the flag")
        self.assertNotIn("--bare", self.gen.build_argv("user text"))

    def test_system_prompt_is_sent_separately_when_supported(self):
        """With ``--system-prompt`` available, the user turn carries only the user half."""
        self.assertTrue(self.gen.prompt_is_exact)
        argv = self.gen.build_argv(self.gen.full_prompt("USER-HALF"))
        self.assertEqual(argv[argv.index("--system-prompt") + 1], build_system_prompt())
        self.assertEqual(argv[2], "USER-HALF")

    def test_budget_flag_is_the_whole_ceiling_not_the_remainder(self):
        """Each call is its own CLI session, so the flag caps one call, not the run.

        Passing what is left would hand the last permitted call a cap smaller than a
        call costs, turning a clean stop into a mid-call abort. The cumulative total
        is this module's job, in :meth:`check_budget`.
        """
        self.gen.cost_usd = 0.49
        argv = self.gen.build_argv("user text")
        self.assertEqual(argv[argv.index("--max-budget-usd") + 1], "0.50")


class TestDeclaredFlags(unittest.TestCase):
    """Only options a help listing *declares* count as supported."""

    def test_a_flag_named_in_another_flags_prose_is_not_support(self):
        """This is the bug that would have broken the demo, so it is pinned here.

        The real CLI documents ``--bare`` with prose naming several other options. A
        substring search over the whole help text would report those as supported,
        put an unknown option on the command line, and lose the run.
        """
        help_text = (
            "Options:\n"
            "  --bare      Minimal mode. Explicitly provide context via:\n"
            "              --system-prompt[-file], --add-dir, --mcp-config.\n"
            "  --model <model>   the model\n"
        )
        declared = _declared_flags(help_text)
        self.assertIn("--bare", declared)
        self.assertIn("--model", declared)
        self.assertNotIn("--system-prompt", declared)
        self.assertNotIn("--add-dir", declared)
        self.assertNotIn("--mcp-config", declared)

    def test_comma_separated_aliases_are_both_declared(self):
        """``--allowedTools, --allowed-tools`` declares two spellings."""
        declared = _declared_flags("  --allowedTools, --allowed-tools <tools...>  allow\n")
        self.assertIn("--allowedTools", declared)
        self.assertIn("--allowed-tools", declared)

    def test_prose_at_the_left_margin_is_ignored(self):
        """A usage line or free text is not an option declaration."""
        self.assertEqual(_declared_flags("Pass --model to choose a model.\n"), frozenset())


class TestUnknownOptionRetry(GeneratorFixture):
    """A flag the CLI rejects degrades the run rather than ending it."""

    def test_unknown_option_retries_with_the_minimal_invocation(self):
        """The second attempt carries none of the optional flags."""
        queue = ["error: unknown option '--strict-mcp-config'", envelope(fenced([VALID_CLAIM]))]

        def transport(argv: list[str], cwd: str) -> str:  # noqa: ARG001 - protocol shape
            self.calls.append(list(argv))
            item = queue.pop(0)
            if not item.startswith("{"):
                raise RuntimeError(f"the Claude Code CLI exited 1: {item}")
            return item

        self.gen.transport = transport
        items = self.gen.propose("briefing", "question", 1)

        self.assertEqual(items, [VALID_CLAIM])
        self.assertEqual(len(self.calls), 2)
        for flag in ("--strict-mcp-config", "--max-turns", "--system-prompt"):
            self.assertNotIn(flag, self.calls[1])
        self.assertIn("--model", self.calls[1])

    def test_the_reduced_flag_set_is_kept(self):
        """The fallback must cost one call, not one per call."""
        self.gen.transport = lambda argv, cwd: envelope(fenced([VALID_CLAIM]))  # noqa: ARG005
        self.gen.flags = frozenset()
        self.gen.prompt_is_exact = False
        argv = self.gen.build_argv("user text")
        self.assertNotIn("--max-turns", argv)

    def test_a_model_failure_does_not_trigger_the_flag_retry(self):
        """Only one retry may fire for one failure.

        Regression: ``"API Error: unknown model 'sonnet'"`` matched an over-broad
        ``"error: unknown"`` marker, so a model-availability failure spent a call on a
        pointless flag retry *and* permanently cleared the guardrails before the model
        switch ran. Caught by an end-to-end run against a stub, not by a unit test.
        """
        queue = [
            "API Error: unknown model 'sonnet' for this account",
            envelope(fenced([VALID_CLAIM])),
        ]

        def transport(argv: list[str], cwd: str) -> str:  # noqa: ARG001 - protocol shape
            self.calls.append(list(argv))
            item = queue.pop(0)
            if not item.startswith("{"):
                raise RuntimeError(f"the Claude Code CLI exited 1: {item}")
            return item

        self.gen.transport = transport
        self.gen.propose("briefing", "question", 1)

        self.assertEqual(len(self.calls), 2, "exactly one retry, not two")
        self.assertEqual(self.gen.model, FALLBACK_CLI_MODEL)
        self.assertTrue(self.gen.flags, "the guardrail flags must survive a model switch")
        self.assertIn("--max-turns", self.calls[1])

    def test_a_second_unknown_option_failure_is_not_retried(self):
        """With no flags left to drop, the failure is real and must surface."""
        self.gen.flags = frozenset()

        def transport(argv: list[str], cwd: str) -> str:  # noqa: ARG001 - protocol shape
            self.calls.append(list(argv))
            raise RuntimeError("error: unknown option '--model'")

        self.gen.transport = transport
        with self.assertRaises(RuntimeError):
            self.gen.complete("anything")
        self.assertEqual(len(self.calls), 1)


class TestOlderCLI(GeneratorFixture):
    """A CLI advertising none of the optional flags still runs."""

    help_text = "Usage: claude [options] [prompt]\n  -p, --print\n  --output-format <format>\n"

    def test_unsupported_flags_are_omitted(self):
        """Only the always-present flags are used."""
        argv = self.gen.build_argv("user text")
        for flag in ("--max-turns", "--disallowedTools", "--max-budget-usd", "--system-prompt"):
            self.assertNotIn(flag, argv)

    def test_system_prompt_is_folded_into_the_user_turn(self):
        """A prompt the model never sees is not a guardrail, so it is concatenated."""
        self.assertFalse(self.gen.prompt_is_exact)
        prompt = self.gen.full_prompt("USER-HALF")
        self.assertTrue(prompt.startswith(build_system_prompt()))
        self.assertTrue(prompt.endswith("USER-HALF"))


class TestAllowToolsFallback(GeneratorFixture):
    """A CLI with only an allow-list flag is restricted with an empty allow list."""

    help_text = "Usage: claude\n  --allowedTools <tools...>\n  -p\n  --output-format <format>\n"

    def test_empty_allow_list_is_used(self):
        """``--allowedTools ''`` is the fallback when there is no deny flag."""
        argv = self.gen.build_argv("user text")
        self.assertEqual(argv[argv.index("--allowedTools") + 1], "")


class TestEnvelope(GeneratorFixture):
    """Reading the CLI's JSON envelope."""

    def test_result_is_extracted_and_usage_accounted(self):
        """Cost and tokens accumulate from the envelope."""
        text = self.gen.extract(envelope("hello", cost=0.02, tokens=(11, 22)))
        self.assertEqual(text, "hello")
        self.assertAlmostEqual(self.gen.cost_usd, 0.02)
        self.assertEqual(self.gen.tokens, {"input": 11, "output": 22})

    def test_non_json_output_is_reported_with_an_excerpt(self):
        """A CLI that printed a login prompt must not look like a parse bug."""
        with self.assertRaises(RuntimeError) as caught:
            self.gen.extract("Please run `claude` and log in first.")
        self.assertIn("did not return JSON", str(caught.exception))
        self.assertIn("log in", str(caught.exception))

    def test_error_envelope_is_raised_with_the_clis_own_text(self):
        """``is_error`` is honoured, and the CLI's message survives into the error."""
        with self.assertRaises(RuntimeError) as caught:
            self.gen.extract(envelope("Credit balance is too low", error=True))
        self.assertIn("error_during_execution", str(caught.exception))
        self.assertIn("Credit balance is too low", str(caught.exception))

    def test_empty_result_is_raised(self):
        """An empty result is a failure, not an empty hypothesis list."""
        with self.assertRaises(RuntimeError):
            self.gen.extract(envelope("   "))

    def test_cost_is_counted_before_an_error_is_raised(self):
        """A failed call still spent allowance, so it must still be accounted."""
        with self.assertRaises(RuntimeError):
            self.gen.extract(envelope("partial", error=True, cost=0.03))
        self.assertAlmostEqual(self.gen.cost_usd, 0.03)


class TestBudget(GeneratorFixture):
    """The call and cost ceilings stop a run before it spends more."""

    def test_call_ceiling_stops_the_run(self):
        """Reaching the call limit raises rather than making another call."""
        self.gen.max_calls = 2
        self.gen.calls = 2
        with self.assertRaises(BudgetExceededError) as caught:
            self.gen.check_budget()
        self.assertIn("--max-calls", str(caught.exception))

    def test_cost_ceiling_stops_the_run(self):
        """Reaching the cost limit raises rather than making another call."""
        self.gen.cost_usd = 0.75
        with self.assertRaises(BudgetExceededError) as caught:
            self.gen.check_budget()
        self.assertIn("--max-budget-usd", str(caught.exception))

    def test_ceiling_is_checked_before_the_subprocess_runs(self):
        """No subprocess may be spawned once the ceiling is reached."""
        self.answer(envelope(fenced([VALID_CLAIM])))
        self.gen.max_calls = 0
        with self.assertRaises(BudgetExceededError):
            self.gen.complete("anything")
        self.assertEqual(self.calls, [])

    def test_budget_error_is_a_runtime_error(self):
        """Existing handlers catch it without change."""
        self.assertTrue(issubclass(BudgetExceededError, RuntimeError))


class TestModelFallback(GeneratorFixture):
    """Switching models is allowed only for "this account cannot use that model"."""

    def test_unavailable_model_falls_back_once(self):
        """A model the account cannot use is retried on the fallback, and named as such."""
        queue = ["unknown model: sonnet", envelope(fenced([VALID_CLAIM]))]

        def transport(argv: list[str], cwd: str) -> str:  # noqa: ARG001 - protocol shape
            self.calls.append(list(argv))
            item = queue.pop(0)
            if not item.startswith("{"):
                raise RuntimeError(f"the Claude Code CLI exited 1: {item}")
            return item

        self.gen.transport = transport
        items = self.gen.propose("briefing", "question", 1)

        self.assertEqual(len(items), 1)
        self.assertEqual(self.gen.model, FALLBACK_CLI_MODEL)
        self.assertEqual(self.gen.name, f"claude-cli:{FALLBACK_CLI_MODEL}")
        self.assertEqual(self.calls[1][self.calls[1].index("--model") + 1], FALLBACK_CLI_MODEL)

    def test_other_failures_do_not_retry(self):
        """A transport failure must not be paid for twice."""

        def transport(argv: list[str], cwd: str) -> str:  # noqa: ARG001 - protocol shape
            self.calls.append(list(argv))
            raise RuntimeError("could not reach the service")

        self.gen.transport = transport
        with self.assertRaises(RuntimeError):
            self.gen.complete("anything")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.gen.model, "sonnet")

    def test_fallback_happens_at_most_once(self):
        """A second unavailable-model error is not retried again."""
        self.gen.model_fallback_used = True
        self.assertFalse(self.gen.can_fall_back("unknown model: sonnet"))

    def test_fallback_disabled_when_none(self):
        """``fallback_model=None`` means fail loudly."""
        self.gen.fallback_model = None
        self.assertFalse(self.gen.can_fall_back("unknown model: sonnet"))

    def test_budget_error_is_not_a_fallback_trigger(self):
        """A ceiling must never be escaped by switching model."""
        self.gen.max_calls = 0
        with self.assertRaises(BudgetExceededError):
            self.gen.complete("anything")
        self.assertEqual(self.gen.model, "sonnet")


class TestProposeAndNarrow(GeneratorFixture):
    """The generator returns raw mappings and never judges them."""

    def test_fenced_json_is_parsed(self):
        """The real CLI fences its JSON; the parser must see through that."""
        self.answer(envelope(fenced([VALID_CLAIM])))
        items = self.gen.propose("briefing", "question", 1)
        self.assertEqual(items, [VALID_CLAIM])

    def test_narrow_returns_none_on_null(self):
        """An explicit refusal to narrow is not an error."""
        self.answer(envelope("null"))
        self.assertIsNone(self.gen.narrow("briefing", {"id": "H1"}, {"passed": False}))

    def test_narrow_returns_the_successor(self):
        """A narrowed claim comes back as a single mapping."""
        self.answer(envelope(fenced(VALID_CLAIM)))
        result = self.gen.narrow("briefing", {"id": "H1"}, {"passed": False})
        self.assertEqual(result, VALID_CLAIM)

    def test_name_records_the_backend_and_model(self):
        """Provenance must not be mistakable for an API call or a replay."""
        self.assertEqual(self.gen.name, "claude-cli:sonnet")

    def test_usage_summary_reports_what_was_spent(self):
        """The closing line names calls, cost and tokens."""
        self.answer(envelope(fenced([VALID_CLAIM]), cost=0.02, tokens=(5, 6)))
        self.gen.propose("briefing", "question", 1)
        summary = self.gen.usage_summary()
        self.assertIn("1 call(s)", summary)
        self.assertIn("$0.0200", summary)
        self.assertIn("5 in / 6 out", summary)


class TestEnvironment(unittest.TestCase):
    """The child process must not inherit an API key."""

    def test_api_key_is_stripped(self):
        """A key in the parent environment would silently route through API billing."""
        with mock.patch.dict(os.environ, {API_KEY_ENV: "sk-ant-should-not-propagate"}):
            env = child_environment()
        self.assertNotIn(API_KEY_ENV, env)

    def test_other_variables_survive(self):
        """Stripping the key must not strip PATH along with it."""
        with mock.patch.dict(os.environ, {"LLM_INSIGHTS_MARKER": "kept"}):
            env = child_environment()
        self.assertEqual(env.get("LLM_INSIGHTS_MARKER"), "kept")


class TestFlagProbe(unittest.TestCase):
    """Flag support is read from the installed CLI, not assumed."""

    def setUp(self) -> None:
        """Clear the probe cache so each test sees its own help text."""
        supported_flags.cache_clear()
        self.addCleanup(supported_flags.cache_clear)

    def test_flags_are_read_from_help(self):
        """Only advertised flags are reported."""
        with mock.patch(
            "llm_insights.agent.subscription.subprocess.run",
            return_value=mock.Mock(stdout=FULL_HELP, stderr="", returncode=0),
        ):
            flags = supported_flags("/usr/bin/claude")
        self.assertIn("--max-turns", flags)
        self.assertIn("--system-prompt", flags)

    def test_unreadable_help_degrades_to_the_minimal_invocation(self):
        """A CLI whose help cannot be read still runs, with fewer guardrails."""
        with mock.patch(
            "llm_insights.agent.subscription.subprocess.run", side_effect=OSError("boom")
        ):
            self.assertEqual(supported_flags("/usr/bin/claude"), frozenset())


class TestRealSubprocess(unittest.TestCase):
    """One end-to-end pass over a stub executable, so the subprocess call itself is covered."""

    def setUp(self) -> None:
        """Write a stub `claude` that echoes an envelope and records its environment."""
        supported_flags.cache_clear()
        self.addCleanup(supported_flags.cache_clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.record = Path(self.tmp.name) / "record.json"
        self.stub = Path(self.tmp.name) / "claude"
        self.stub.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "argv = sys.argv[1:]\n"
            "if '--help' in argv:\n"
            f"    print({FULL_HELP!r})\n"
            "    sys.exit(0)\n"
            f"open({str(self.record)!r}, 'w').write(json.dumps("
            "{'argv': argv, 'cwd': os.getcwd(), "
            f"'key': {API_KEY_ENV!r} in os.environ}}))\n"
            "print(json.dumps({'type': 'result', 'is_error': False, "
            "'total_cost_usd': 0.01, 'usage': {'input_tokens': 1, 'output_tokens': 2}, "
            f"'result': {fenced([VALID_CLAIM])!r}}}))\n",
            encoding="utf-8",
        )
        self.stub.chmod(self.stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    def test_a_real_subprocess_round_trip(self):
        """The generator runs the executable, reads its envelope, and cleans up."""
        with mock.patch.dict(os.environ, {API_KEY_ENV: "sk-ant-should-not-propagate"}):
            gen = ClaudeCLIGenerator(model="sonnet", binary=str(self.stub))
            workdir = gen.workdir
            try:
                items = gen.propose("briefing", "question", 1)
            finally:
                gen.close()

        self.assertEqual(items, [VALID_CLAIM])
        recorded = json.loads(self.record.read_text())
        self.assertFalse(recorded["key"], "the API key must not reach the child")
        self.assertEqual(recorded["cwd"], os.path.realpath(workdir))
        self.assertNotIn("--bare", recorded["argv"])
        self.assertFalse(Path(workdir).exists(), "close() must remove the scratch directory")

    def test_a_failing_executable_is_reported_with_its_stderr(self):
        """The CLI's own message is what a user needs to see, so it must survive."""
        failing = Path(self.tmp.name) / "failing"
        failing.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "if '--help' in sys.argv[1:]:\n"
            f"    print({FULL_HELP!r})\n"
            "    sys.exit(0)\n"
            "print('Invalid API key - please run /login', file=sys.stderr)\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        failing.chmod(failing.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        gen = ClaudeCLIGenerator(model="sonnet", binary=str(failing))
        self.addCleanup(gen.close)
        with self.assertRaises(RuntimeError) as caught:
            gen.propose("briefing", "question", 1)
        self.assertIn("exited 1", str(caught.exception))
        self.assertIn("/login", str(caught.exception))


class TestPasteGenerator(unittest.TestCase):
    """The manual fallback reads a reply from the clipboard or from a file."""

    def setUp(self) -> None:
        """Build a paste generator with no clipboard and scripted stdin."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = io.StringIO()
        self.gen = PasteGenerator(
            workdir=Path(self.tmp.name) / "paste",
            stdin=io.StringIO("\n\n\n"),
            stdout=self.out,
        )
        self.gen.clipboard = None

    def test_prompt_is_written_and_reply_read_from_file(self):
        """With no clipboard, the reply file is the channel."""
        reply = self.gen.workdir / "01_propose_reply.txt"
        reply.parent.mkdir(parents=True, exist_ok=True)
        reply.write_text(fenced([VALID_CLAIM]), encoding="utf-8")

        items = self.gen.propose("briefing", "question", 1)

        self.assertEqual(items, [VALID_CLAIM])
        prompt = (self.gen.workdir / "01_propose_prompt.txt").read_text(encoding="utf-8")
        self.assertIn(build_system_prompt(), prompt)
        self.assertIn("briefing", prompt)

    def test_missing_reply_is_an_actionable_error(self):
        """Silence must not be read as an empty hypothesis list."""
        with self.assertRaises(RuntimeError) as caught:
            self.gen.propose("briefing", "question", 1)
        self.assertIn("no reply found", str(caught.exception))

    def test_clipboard_reply_wins_when_present(self):
        """The clipboard is the fast path, checked before the file."""
        self.gen.clipboard = (["true"], ["true"])
        with mock.patch.object(self.gen, "read_clipboard", return_value=fenced([VALID_CLAIM])):
            items = self.gen.propose("briefing", "question", 1)
        self.assertEqual(items, [VALID_CLAIM])

    def test_name_marks_the_run_as_human_routed(self):
        """A report must be able to say a human carried the prompt."""
        self.assertEqual(self.gen.name, "paste:human-in-the-loop")


class TestFactory(unittest.TestCase):
    """``make_generator`` reaches the new backends and reports unknown ones."""

    def test_claude_cli_kind(self):
        """The CLI backend is constructed with the requested budget."""
        supported_flags.cache_clear()
        self.addCleanup(supported_flags.cache_clear)
        with (
            mock.patch(
                "llm_insights.agent.subscription.shutil.which", return_value="/usr/bin/claude"
            ),
            mock.patch(
                "llm_insights.agent.subscription.subprocess.run",
                return_value=mock.Mock(stdout=FULL_HELP, stderr="", returncode=0),
            ),
        ):
            gen = make_generator("claude-cli", model="sonnet", max_calls=3, max_budget_usd=0.25)
        self.addCleanup(gen.close)
        self.assertIsInstance(gen, ClaudeCLIGenerator)
        self.assertEqual(gen.max_calls, 3)
        self.assertEqual(gen.max_budget_usd, 0.25)

    def test_paste_kind(self):
        """The paste backend is reachable only by name."""
        with tempfile.TemporaryDirectory() as tmp:
            gen = make_generator("paste", paste_dir=Path(tmp) / "paste")
        self.assertIsInstance(gen, PasteGenerator)

    def test_unknown_kind_lists_every_option(self):
        """The error must name what the user could have typed instead."""
        with self.assertRaises(ValueError) as caught:
            make_generator("gpt")
        for kind in GENERATOR_KINDS:
            self.assertIn(kind, str(caught.exception))

    def test_claude_cli_is_a_documented_kind(self):
        """The CLI backend must be selectable from the command line."""
        self.assertIn("claude-cli", GENERATOR_KINDS)
        self.assertIn("paste", GENERATOR_KINDS)


class TestRecordedQuestion(unittest.TestCase):
    """A transcript knows which question it answered, so a stale replay can be caught."""

    def write(self, payload: Any) -> Path:
        """Write a transcript file.

        Args:
            payload: The transcript object.

        Returns:
            The path written.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "transcript.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_question_is_recovered(self):
        """The recorded question comes back from the proposal entry."""
        path = self.write(
            {
                "entries": [
                    {
                        "kind": "propose",
                        "request": {"question": "the old question", "n": 5},
                        "response": [VALID_CLAIM],
                    }
                ]
            }
        )
        self.assertEqual(TranscriptGenerator(path).recorded_question, "the old question")

    def test_absent_question_is_none(self):
        """A transcript that records the question nowhere reports None rather than failing."""
        path = self.write({"entries": [{"kind": "propose", "response": [VALID_CLAIM]}]})
        self.assertIsNone(TranscriptGenerator(path).recorded_question)

    def test_question_is_recovered_from_recorded_from(self):
        """The shipped demo transcript records it only at file level, so that counts too.

        This was a real miss: the warning silently never fired for the one transcript
        the demo actually ships with.
        """
        path = self.write(
            {
                "recorded_from": {"question": "the old question", "model": "claude-opus-5"},
                "entries": [{"kind": "propose", "response": [VALID_CLAIM]}],
            }
        )
        self.assertEqual(TranscriptGenerator(path).recorded_question, "the old question")


if __name__ == "__main__":  # pragma: no cover - test entry point
    unittest.main()
