"""Tests for the local open-weights backend and the bake-off that chose its default.

Two rules govern this file, and both come straight from TASK-016.

**No network.** Nothing here reaches a model, and nothing leaves this machine. Two
mechanisms are used and no third one is: an injected ``transport`` callable, and a
``http.server`` stub bound to ``127.0.0.1`` on an ephemeral port. The suite passes with
no model server running, which is the point -- a test that needs Ollama up is a test
that fails on Dan's machine.

**Fault injection, not just unit tests.** TASK-007 shipped a live bug that all 49 of
its unit tests passed, and what found it was standing up a server that misbehaved on
purpose. So :class:`FaultInjectionTests` runs the *real* :func:`llm_insights.agent.local.http_json`
-- real sockets, real ``urllib``, real timeouts -- against six servers that each break
in a different way: malformed JSON, a body truncated mid-object, valid JSON of the
wrong schema, HTTP 500, a refused connection, and a hang past the timeout. Each one
must surface as a :class:`~llm_insights.agent.local.LocalBackendError` whose message
says what happened, never as a raw ``URLError`` or ``JSONDecodeError`` from three
frames down.

Run with::

    cd src && python3 -m unittest discover -s ../tests -t .. -v
"""

from __future__ import annotations

import json
import threading
import time
import unittest
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar
from unittest import mock

from llm_insights.agent import run as run_module
from llm_insights.agent.bakeoff import (
    CANARY_TEMPLATE,
    ModelReport,
    classify_rejection,
    format_table,
    run_canary,
    run_model,
    score_batch,
)
from llm_insights.agent.generator import (
    GENERATOR_KINDS,
    build_system_prompt,
    make_generator,
)
from llm_insights.agent.local import (
    LOCAL_CONTEXT_ENV,
    LOCAL_MODEL_ENV,
    LOCAL_URL_ENV,
    ContextTooSmallError,
    LocalBackendError,
    LocalGenerator,
    estimate_tokens,
    http_json,
    request_headers,
)
from llm_insights.agent.subscription import API_KEY_ENV, child_environment

#: A hypothesis the strict validator accepts. Uses ``fraction_above``, which needs no
#: metric registry, so this fixture stays valid in an environment without one.
VALID_HYPOTHESIS: dict[str, Any] = {
    "id": "H1",
    "claim": "Some of the cushion carries non-zero von Mises stress at the final step.",
    "rationale": "A fixture claim, valid by construction.",
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

#: A hypothesis the strict validator rejects: an extra top-level field.
EXTRA_FIELD_HYPOTHESIS: dict[str, Any] = {**VALID_HYPOTHESIS, "id": "H2", "confidence": 0.9}

#: A hypothesis the strict validator rejects: it names a test that does not exist.
UNKNOWN_PRIMITIVE_HYPOTHESIS: dict[str, Any] = {
    **VALID_HYPOTHESIS,
    "id": "H3",
    "test": {**VALID_HYPOTHESIS["test"], "primitive": "vibes_check"},
}


def envelope(text: str, prompt_tokens: int = 3200, completion_tokens: int = 400) -> dict[str, Any]:
    """Build a chat-completions response envelope carrying ``text``.

    Mirrors the shape Ollama's ``/v1/chat/completions`` really returns, usage block
    included, so a change in how this package reads that envelope fails here.

    Args:
        text: The assistant content.
        prompt_tokens: Reported prompt tokens.
        completion_tokens: Reported completion tokens.

    Returns:
        The envelope.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "stub",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class RecordingTransport:
    """A transport that records every request and replays scripted responses.

    Args:
        responses: One entry per ``/v1/chat/completions`` call, in order. An entry
            that is an exception is raised instead of returned. The last entry repeats
            once exhausted, so a test does not have to count calls it does not care
            about.
        context_length: Context length to report from ``/api/ps``, or None to report
            nothing, which is what a server that has not loaded the model does.
    """

    def __init__(  # noqa: D107 - documented on the class
        self, responses: Sequence[Any] | None = None, context_length: int | None = 32768
    ) -> None:
        self.responses = list(responses) if responses else []
        self.context_length = context_length
        #: Every call, as ``(url, payload, headers, timeout)``.
        self.calls: list[tuple[str, Any, dict[str, str], float]] = []
        self._index = 0

    def __call__(self, url: str, payload: Any, headers: dict[str, str], timeout: float) -> Any:
        """Record the request and return the scripted response.

        Args:
            url: Endpoint.
            payload: Request body, or None for a GET.
            headers: Request headers.
            timeout: Socket timeout.

        Returns:
            The scripted response body.

        Raises:
            LocalBackendError: If the endpoint is one this stub does not serve, or the
                script says this call fails.
        """
        self.calls.append((url, payload, dict(headers), timeout))
        if url.endswith("/api/ps"):
            if self.context_length is None:
                return {"models": []}
            return {"models": [{"model": "stub-model", "context_length": self.context_length}]}
        if url.endswith("/props"):
            raise LocalBackendError("no /props on this stub")
        if not self.responses:
            return envelope("[]")
        item = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def chat_calls(self) -> list[tuple[str, Any, dict[str, str], float]]:
        """Only the ``/v1/chat/completions`` calls, in order."""
        return [call for call in self.calls if call[0].endswith("/v1/chat/completions")]


def stub_generator(
    responses: Sequence[Any] | None = None, *, context_length: int | None = 32768, **kwargs: Any
) -> LocalGenerator:
    """Build a :class:`LocalGenerator` wired to a :class:`RecordingTransport`.

    Args:
        responses: The scripted chat responses.
        context_length: Context length the stub server reports.
        **kwargs: Passed through to :class:`LocalGenerator`.

    Returns:
        The generator, with ``.transport`` already installed.
    """
    kwargs.setdefault("model", "stub-model")
    kwargs.setdefault("url", "http://127.0.0.1:1")
    generator = LocalGenerator(**kwargs)
    generator.transport = RecordingTransport(responses, context_length)
    return generator


class ConfigurationTests(unittest.TestCase):
    """The backend is configured by environment variable and holds no credential."""

    def test_environment_variables_configure_url_and_model(self):
        """Environment variables configure url and model."""
        with mock.patch.dict(
            "os.environ",
            {LOCAL_URL_ENV: "http://127.0.0.1:9999/", LOCAL_MODEL_ENV: "qwen3:8b"},
            clear=False,
        ):
            generator = LocalGenerator()
        self.assertEqual(generator.url, "http://127.0.0.1:9999")
        self.assertEqual(generator.model, "qwen3:8b")
        self.assertEqual(generator.name, "local:qwen3:8b")

    def test_explicit_arguments_beat_the_environment(self):
        """Explicit arguments beat the environment."""
        with mock.patch.dict(
            "os.environ",
            {LOCAL_URL_ENV: "http://127.0.0.1:9999", LOCAL_MODEL_ENV: "qwen3:8b"},
            clear=False,
        ):
            generator = LocalGenerator(model="llama3.1:8b", url="http://127.0.0.1:11434")
        self.assertEqual(generator.model, "llama3.1:8b")
        self.assertEqual(generator.url, "http://127.0.0.1:11434")

    def test_headers_carry_no_credential(self):
        """Headers carry no credential."""
        headers = request_headers()
        self.assertEqual(set(headers), {"content-type", "accept"})
        lowered = {key.lower() for key in headers}
        self.assertNotIn("authorization", lowered)
        self.assertNotIn("x-api-key", lowered)

    def test_repr_names_the_model_not_a_secret(self):
        """Repr names the model not a secret."""
        generator = stub_generator()
        self.assertIn("stub-model", repr(generator))
        self.assertNotIn("key", repr(generator).lower())


class ApiKeyIsolationTests(unittest.TestCase):
    """TASK-016 criterion 4: no Anthropic key reaches a local call, ever.

    ``LocalGenerator`` starts no subprocess, so there is no child environment of its
    own to scrub. The property that matters is the one below: with the key exported,
    the value appears in nothing this backend sends. The shared scrubber the
    subprocess-backed generators use is pinned here too, so a regression in either
    half fails a test in this file.
    """

    SECRET = "sk-ant-do-not-leak-0123456789"

    def test_no_request_contains_the_api_key(self):
        """No request contains the api key."""
        generator = stub_generator([envelope(json.dumps([VALID_HYPOTHESIS]))])
        with mock.patch.dict("os.environ", {API_KEY_ENV: self.SECRET}, clear=False):
            generator.propose("briefing", "question", 1)
        transport = generator.transport
        for url, payload, headers, _timeout in transport.calls:
            blob = json.dumps({"url": url, "payload": payload, "headers": headers})
            self.assertNotIn(self.SECRET, blob)
            self.assertNotIn(API_KEY_ENV, blob)

    def test_shared_child_environment_still_strips_the_key(self):
        """Shared child environment still strips the key."""
        with mock.patch.dict("os.environ", {API_KEY_ENV: self.SECRET}, clear=False):
            self.assertNotIn(API_KEY_ENV, child_environment())

    def test_the_key_is_not_needed_to_construct_a_local_generator(self):
        """The key is not needed to construct a local generator."""
        env = {key: value for key, value in __import__("os").environ.items() if key != API_KEY_ENV}
        with mock.patch.dict("os.environ", env, clear=True):
            generator = LocalGenerator(model="m", url="http://127.0.0.1:1")
        self.assertEqual(generator.model, "m")


class ContextWindowTests(unittest.TestCase):
    """The silent-truncation hazard: refuse rather than send a prompt that will not fit."""

    def test_estimate_errs_high(self):
        """Estimate errs high."""
        # Measured on 2026-09-08: the assembled proposal prompt is 18,980 characters
        # and Ollama reported 6,248 prompt tokens for llama3.1:8b and 7,277 for
        # qwen3:8b. The estimate must sit above the worse of those, never below, or
        # the context check waves through prompts the server truncates in silence.
        self.assertGreater(estimate_tokens("x" * 18980), 7277)
        self.assertGreater(estimate_tokens("x" * 12662), 5672)

    def test_refuses_when_the_server_reports_4096(self):
        """Refuses when the server reports 4096."""
        generator = stub_generator(context_length=4096)
        with self.assertRaises(ContextTooSmallError) as caught:
            generator.check_context(build_system_prompt(), "x" * 12662)
        message = str(caught.exception)
        self.assertIn("4096", message)
        self.assertIn("truncated silently", message)
        self.assertIn("OLLAMA_CONTEXT_LENGTH", message)

    def test_refusal_happens_before_any_chat_request(self):
        """Refusal happens before any chat request."""
        generator = stub_generator([envelope(json.dumps([VALID_HYPOTHESIS]))], context_length=4096)
        with self.assertRaises(ContextTooSmallError):
            generator.propose("x" * 12662, "question", 5)
        self.assertEqual(generator.transport.chat_calls, [])
        self.assertEqual(generator.calls, 0)

    def test_accepts_when_the_server_reports_32768(self):
        """Accepts when the server reports 32768."""
        generator = stub_generator(context_length=32768)
        generator.check_context(build_system_prompt(), "x" * 12662)
        self.assertEqual(generator.context_length, 32768)

    def test_reads_llama_cpp_props_when_ollama_says_nothing(self):
        """Reads llama cpp props when ollama says nothing."""

        def transport(url: str, payload: Any, headers: Any, timeout: float) -> Any:
            del payload, headers, timeout
            if url.endswith("/api/ps"):
                return {"models": []}
            if url.endswith("/props"):
                return {"default_generation_settings": {"n_ctx": 16384}}
            return envelope("[]")

        generator = LocalGenerator(model="m", url="http://127.0.0.1:1")
        generator.transport = transport
        self.assertEqual(generator.server_context_length(), 16384)

    def test_environment_override_wins(self):
        """Environment override wins."""
        generator = stub_generator(context_length=4096)
        with mock.patch.dict("os.environ", {LOCAL_CONTEXT_ENV: "65536"}, clear=False):
            self.assertEqual(generator.server_context_length(), 65536)

    def test_a_nonsense_override_is_ignored_not_obeyed(self):
        """A nonsense override is ignored not obeyed."""
        generator = stub_generator(context_length=8192)
        with mock.patch.dict("os.environ", {LOCAL_CONTEXT_ENV: "banana"}, clear=False):
            self.assertEqual(generator.server_context_length(), 8192)

    def test_warm_up_runs_once_when_the_model_is_not_loaded(self):
        """Warm up runs once when the model is not loaded."""
        generator = stub_generator(context_length=None)
        self.assertIsNone(generator.server_context_length())
        self.assertIsNone(generator.server_context_length())
        warmups = [call for call in generator.transport.chat_calls if call[1]["max_tokens"] == 1]
        self.assertEqual(len(warmups), 1)

    def test_unknown_context_warns_but_does_not_refuse(self):
        """Unknown context warns but does not refuse."""
        generator = stub_generator(context_length=None)
        with self.assertLogs("llm_insights.agent.local", level="WARNING") as logged:
            generator.check_context("system", "user")
        self.assertTrue(any(LOCAL_CONTEXT_ENV in line for line in logged.output))


class EnvelopeTests(unittest.TestCase):
    """Every shape a misbehaving server can return must fail loudly."""

    def test_missing_choices_is_named(self):
        """Missing choices is named."""
        with self.assertRaises(LocalBackendError) as caught:
            LocalGenerator.extract({"result": "hello"})
        self.assertIn("no 'choices' list", str(caught.exception))

    def test_non_object_body_is_named(self):
        """Non object body is named."""
        with self.assertRaises(LocalBackendError) as caught:
            LocalGenerator.extract(["not", "an", "object"])
        self.assertIn("not an object", str(caught.exception))

    def test_server_reported_error_is_surfaced(self):
        """Server reported error is surfaced."""
        with self.assertRaises(LocalBackendError) as caught:
            LocalGenerator.extract({"error": {"message": "model not found"}})
        self.assertIn("model not found", str(caught.exception))

    def test_blank_content_is_rejected(self):
        """Blank content is rejected."""
        with self.assertRaises(LocalBackendError) as caught:
            LocalGenerator.extract(envelope("   "))
        self.assertIn("no message content", str(caught.exception))

    def test_empty_choices_list_is_rejected(self):
        """Empty choices list is rejected."""
        with self.assertRaises(LocalBackendError):
            LocalGenerator.extract({"choices": []})


class ProtocolTests(unittest.TestCase):
    """The backend satisfies the Generator protocol the loop requires."""

    def test_propose_returns_raw_mappings(self):
        """Propose returns raw mappings."""
        generator = stub_generator(
            [envelope(json.dumps([VALID_HYPOTHESIS, EXTRA_FIELD_HYPOTHESIS]))]
        )
        items = generator.propose("briefing", "question", 2)
        self.assertEqual([item["id"] for item in items], ["H1", "H2"])
        # Unvalidated on purpose: the loop must be able to count rejections.
        self.assertIn("confidence", items[1])

    def test_propose_survives_prose_around_the_json(self):
        """Propose survives prose around the json."""
        text = (
            "Sure! Here you go:\n```json\n"
            + json.dumps([VALID_HYPOTHESIS])
            + "\n```\nHope that helps."
        )
        generator = stub_generator([envelope(text)])
        self.assertEqual(len(generator.propose("b", "q", 1)), 1)

    def test_narrow_accepts_an_explicit_null(self):
        """Narrow accepts an explicit null."""
        generator = stub_generator([envelope("null")])
        self.assertIsNone(generator.narrow("b", {"id": "H1"}, {"passed": False}))

    def test_narrow_returns_the_successor(self):
        """Narrow returns the successor."""
        generator = stub_generator([envelope(json.dumps(VALID_HYPOTHESIS))])
        successor = generator.narrow("b", {"id": "H1"}, {"passed": False})
        self.assertEqual(successor["id"], "H1")

    def test_synthesize_sends_the_synthesis_system_prompt(self):
        """Synthesize sends the synthesis system prompt."""
        from llm_insights.agent.synthesis import SYNTHESIS_SYSTEM_PROMPT

        generator = stub_generator([envelope("HEADLINE: x\nSUMMARY: y")])
        generator.synthesize("the run went like this")
        system = generator.transport.chat_calls[-1][1]["messages"][0]["content"]
        self.assertEqual(system, SYNTHESIS_SYSTEM_PROMPT)
        self.assertNotEqual(system, build_system_prompt())

    def test_the_two_messages_are_exactly_what_this_package_built(self):
        """The two messages are exactly what this package built."""
        generator = stub_generator([envelope(json.dumps([VALID_HYPOTHESIS]))])
        generator.propose("THE BRIEFING", "THE QUESTION", 3)
        messages = generator.transport.chat_calls[-1][1]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], build_system_prompt())
        self.assertIn("THE BRIEFING", messages[1]["content"])
        self.assertIn("THE QUESTION", messages[1]["content"])

    def test_json_mode_is_off_unless_asked_for(self):
        """Json mode is off unless asked for."""
        payload = envelope(json.dumps([VALID_HYPOTHESIS]))
        plain = stub_generator([payload])
        plain.propose("b", "q", 1)
        self.assertNotIn("response_format", plain.transport.chat_calls[-1][1])

        forced = stub_generator([payload], json_mode=True)
        forced.propose("b", "q", 1)
        self.assertEqual(
            forced.transport.chat_calls[-1][1]["response_format"], {"type": "json_object"}
        )

    def test_usage_is_accumulated_and_summarized(self):
        """Usage is accumulated and summarized."""
        generator = stub_generator([envelope(json.dumps([VALID_HYPOTHESIS]), 3210, 415)])
        generator.propose("b", "q", 1)
        self.assertEqual(generator.tokens, {"input": 3210, "output": 415})
        summary = generator.usage_summary()
        self.assertIn("3210 in / 415 out", summary)
        self.assertIn("$0.00", summary)

    def test_close_is_safe_to_call_twice(self):
        """Close is safe to call twice."""
        generator = stub_generator()
        generator.close()
        generator.close()


class FactoryTests(unittest.TestCase):
    """``--generator local`` is registered, and nothing else moved."""

    def test_local_is_a_generator_kind(self):
        """Local is a generator kind."""
        self.assertIn("local", GENERATOR_KINDS)

    def test_the_existing_kinds_are_all_still_there(self):
        """The existing kinds are all still there."""
        for kind in ("claude-cli", "anthropic", "transcript", "paste", "echo"):
            self.assertIn(kind, GENERATOR_KINDS)

    def test_make_generator_builds_a_local_backend(self):
        """Make generator builds a local backend."""
        generator = make_generator("local", model="llama3.1:8b")
        self.assertIsInstance(generator, LocalGenerator)
        self.assertEqual(generator.name, "local:llama3.1:8b")

    def test_the_cli_accepts_generator_local(self):
        """The cli accepts generator local."""
        args = run_module.build_parser().parse_args(["--generator", "local"])
        self.assertEqual(args.generator, "local")


# --- fault injection ----------------------------------------------------------------


class _FaultHandler(BaseHTTPRequestHandler):
    """A request handler whose misbehaviour is set by the server that owns it."""

    protocol_version = "HTTP/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server
        """Silence the handler's stderr logging so the suite output stays readable.

        Args:
            format: Ignored.
            *args: Ignored.
        """
        del format, args

    def _respond(self) -> None:
        """Write whatever misbehaviour this server was configured with."""
        mode = self.server.mode  # type: ignore[attr-defined]
        if mode == "hang":
            time.sleep(5.0)
        if mode == "http_500":
            body = b'{"error":"internal server error, model failed to load"}'
            self.send_response(500)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        bodies = {
            "malformed": b"not json at all, just an apology from the proxy",
            "truncated": b'{"choices": [{"message": {"content": "[{\\"id\\": \\"H1\\"',
            "wrong_schema": b'{"result": "hello", "done": true}',
            "hang": b'{"choices": []}',
        }
        body = bodies.get(mode, b"{}")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Serve a GET, misbehaving as configured."""
        self._respond()

    def do_POST(self) -> None:
        """Serve a POST, misbehaving as configured."""
        length = int(self.headers.get("content-length") or 0)
        if length:
            self.rfile.read(length)
        self._respond()


class _QuietHTTPServer(HTTPServer):
    """An :class:`HTTPServer` that does not print a traceback when a client vanishes.

    The timeout test deliberately abandons a request mid-flight, and the handler
    thread then hits ``BrokenPipeError`` writing to a socket nobody is reading. That
    is the test working; printing its traceback into the suite output is not.
    """

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Swallow a handler exception instead of printing it.

        Args:
            request: The socket, unused.
            client_address: The peer address, unused.
        """
        del request, client_address


class _FaultServer:
    """A loopback HTTP server that breaks on purpose.

    Args:
        mode: Which misbehaviour to serve.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.httpd = _QuietHTTPServer(("127.0.0.1", 0), _FaultHandler)
        self.httpd.mode = mode  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        """The base URL this server is listening on."""
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self) -> _FaultServer:
        """Start serving and return self."""
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop serving.

        Args:
            *exc: Exception information, unused.
        """
        del exc
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5.0)


def _dead_port() -> int:
    """Return a loopback port with nothing listening on it.

    A server is bound and immediately closed, so the port was real and is now free.
    That is a far more reliable way to get a refused connection than picking a number.

    Returns:
        The port number.
    """
    server = _QuietHTTPServer(("127.0.0.1", 0), _FaultHandler)
    port = server.server_address[1]
    server.server_close()
    return port


class FaultInjectionTests(unittest.TestCase):
    """Six servers that misbehave on purpose, run against the real transport.

    Nothing is mocked below the socket: :func:`http_json` opens a real connection to
    a real ``http.server`` on loopback. This is the technique TASK-007 credits with
    finding a bug forty-nine unit tests had passed.
    """

    #: A generous context, set so the context probe never fires and each test
    #: exercises exactly the fault it is named for.
    ENV: ClassVar[dict[str, str]] = {LOCAL_CONTEXT_ENV: "131072"}

    def _generator(self, url: str, timeout: float = 10.0) -> LocalGenerator:
        """Build a real (unstubbed) generator pointed at ``url``.

        Args:
            url: Base URL of the misbehaving server.
            timeout: Socket timeout in seconds.

        Returns:
            The generator.
        """
        return LocalGenerator(model="stub-model", url=url, timeout=timeout, max_tokens=16)

    def test_malformed_json_body(self):
        """Malformed json body."""
        with _FaultServer("malformed") as server, mock.patch.dict("os.environ", self.ENV):
            generator = self._generator(server.url)
            with self.assertRaises(LocalBackendError) as caught:
                generator.propose("briefing", "question", 3)
        message = str(caught.exception)
        self.assertIn("not JSON", message)
        self.assertIn("apology from the proxy", message)

    def test_body_truncated_mid_object(self):
        """Body truncated mid object."""
        with _FaultServer("truncated") as server, mock.patch.dict("os.environ", self.ENV):
            generator = self._generator(server.url)
            with self.assertRaises(LocalBackendError) as caught:
                generator.propose("briefing", "question", 3)
        self.assertIn("not JSON", str(caught.exception))

    def test_valid_json_of_the_wrong_schema(self):
        """Valid json of the wrong schema."""
        with _FaultServer("wrong_schema") as server, mock.patch.dict("os.environ", self.ENV):
            generator = self._generator(server.url)
            with self.assertRaises(LocalBackendError) as caught:
                generator.propose("briefing", "question", 3)
        self.assertIn("no 'choices' list", str(caught.exception))

    def test_http_500(self):
        """Http 500."""
        with _FaultServer("http_500") as server, mock.patch.dict("os.environ", self.ENV):
            generator = self._generator(server.url)
            with self.assertRaises(LocalBackendError) as caught:
                generator.propose("briefing", "question", 3)
        message = str(caught.exception)
        self.assertIn("HTTP 500", message)
        self.assertIn("model failed to load", message)

    def test_connection_refused(self):
        """Connection refused."""
        url = f"http://127.0.0.1:{_dead_port()}"
        with mock.patch.dict("os.environ", self.ENV):
            generator = self._generator(url)
            with self.assertRaises(LocalBackendError) as caught:
                generator.propose("briefing", "question", 3)
        message = str(caught.exception)
        self.assertIn("could not reach the local model server", message)
        self.assertIn(LOCAL_URL_ENV, message)

    def test_hang_past_the_timeout(self):
        """Hang past the timeout."""
        with _FaultServer("hang") as server, mock.patch.dict("os.environ", self.ENV):
            generator = self._generator(server.url, timeout=0.5)
            started = time.monotonic()
            with self.assertRaises(LocalBackendError):
                generator.propose("briefing", "question", 3)
            elapsed = time.monotonic() - started
        # The whole point of a timeout is that it fires well before the 5s hang.
        self.assertLess(elapsed, 4.0)

    def test_a_dead_server_does_not_defeat_the_context_probe(self):
        """A dead server does not defeat the context probe."""
        # With no context override, a refused connection must still surface as a
        # LocalBackendError from the chat call, not as a crash inside the probe.
        url = f"http://127.0.0.1:{_dead_port()}"
        generator = self._generator(url)
        with self.assertRaises(LocalBackendError):
            generator.propose("briefing", "question", 3)


# --- the bake-off -------------------------------------------------------------------


class _ScriptedBackend:
    """A stand-in for :class:`LocalGenerator` that replays scripted proposal rounds.

    Args:
        rounds: One entry per ``propose`` call. A list is returned; an exception is
            raised.
        **kwargs: Accepted and ignored, so this matches the factory signature
            :func:`~llm_insights.agent.bakeoff.run_model` calls.
    """

    def __init__(self, rounds: Sequence[Any], **kwargs: Any) -> None:
        self.rounds = list(rounds)
        self.model = kwargs.get("model", "scripted")
        self.tokens = {"input": 0, "output": 0}
        self.context_length = 32768
        self.closed = False
        self._index = 0

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Return the next scripted round.

        Args:
            briefing: Ignored.
            question: Ignored.
            n: Ignored.

        Returns:
            The scripted mappings.

        Raises:
            Exception: Whatever the script says this round raises.
        """
        del briefing, question, n
        item = self.rounds[min(self._index, len(self.rounds) - 1)]
        self._index += 1
        self.tokens["input"] += 3200
        self.tokens["output"] += 500
        if isinstance(item, Exception):
            raise item
        return [dict(entry) for entry in item]

    def complete(self, user_text: str, system: str | None = None) -> str:
        """Return the scripted canary reply.

        Args:
            user_text: The user turn; the scripted reply may be derived from it.
            system: Ignored.

        Returns:
            The reply text.
        """
        del system
        self.tokens["input"] += 3200
        reply = self.rounds[0]
        return reply(user_text) if callable(reply) else str(reply)

    def close(self) -> None:
        """Mark this backend closed."""
        self.closed = True


class BakeoffTests(unittest.TestCase):
    """The bake-off counts what the strict validator did, and nothing else."""

    def test_score_batch_splits_accepted_from_rejected(self):
        """Score batch splits accepted from rejected."""
        report = ModelReport(model="m")
        score_batch(
            [VALID_HYPOTHESIS, EXTRA_FIELD_HYPOTHESIS, UNKNOWN_PRIMITIVE_HYPOTHESIS], report
        )
        self.assertEqual((report.proposed, report.accepted, report.rejected), (3, 1, 2))
        self.assertEqual(report.rejection_rate, 2 / 3)
        self.assertIn("unknown primitive", report.reasons)

    def test_rejection_rate_is_none_when_nothing_was_returned(self):
        """Rejection rate is none when nothing was returned."""
        report = ModelReport(model="m", rounds=3, asked=15, call_failures=3)
        self.assertIsNone(report.rejection_rate)
        self.assertEqual(report.usable_yield, 0.0)

    def test_usable_yield_counts_failed_calls_against_the_model(self):
        """Usable yield counts failed calls against the model."""
        report = run_model(
            "m",
            "briefing",
            "question",
            rounds=2,
            n=2,
            make_generator=lambda **kw: _ScriptedBackend(
                [[VALID_HYPOTHESIS, EXTRA_FIELD_HYPOTHESIS], LocalBackendError("boom")], **kw
            ),
        )
        self.assertEqual(report.rounds, 2)
        self.assertEqual(report.asked, 4)
        self.assertEqual(report.call_failures, 1)
        self.assertEqual(report.proposed, 2)
        self.assertEqual(report.rejection_rate, 0.5)
        # One valid claim out of four asked for: the failed call is not invisible.
        self.assertEqual(report.usable_yield, 0.25)
        self.assertEqual(len(report.latencies), 2)

    def test_run_model_records_tokens_per_call(self):
        """Run model records tokens per call."""
        report = run_model(
            "m",
            "b",
            "q",
            rounds=2,
            n=1,
            make_generator=lambda **kw: _ScriptedBackend([[VALID_HYPOTHESIS]], **kw),
        )
        self.assertEqual(report.prompt_tokens, [3200, 3200])
        self.assertEqual(report.mean_completion_tokens, 500.0)

    def test_run_model_closes_the_backend(self):
        """Run model closes the backend."""
        made: list[_ScriptedBackend] = []

        def factory(**kwargs: Any) -> _ScriptedBackend:
            backend = _ScriptedBackend([[VALID_HYPOTHESIS]], **kwargs)
            made.append(backend)
            return backend

        run_model("m", "b", "q", rounds=1, n=1, make_generator=factory)
        self.assertTrue(made[0].closed)

    def test_a_parse_failure_is_a_call_failure_not_a_crash(self):
        """A parse failure is a call failure not a crash."""
        report = run_model(
            "m",
            "b",
            "q",
            rounds=1,
            n=5,
            make_generator=lambda **kw: _ScriptedBackend(
                [ValueError("no JSON object or array could be found")], **kw
            ),
        )
        self.assertEqual(report.call_failures, 1)
        self.assertIsNone(report.rejection_rate)
        self.assertEqual(report.usable_yield, 0.0)

    def test_format_table_reports_n_a_rather_than_a_flattering_zero(self):
        """Format table reports n a rather than a flattering zero."""
        empty = ModelReport(model="silent", rounds=2, asked=10, call_failures=2)
        good = ModelReport(model="good", rounds=1, asked=2, proposed=2, accepted=2)
        table = format_table([empty, good])
        self.assertIn("| silent |", table)
        self.assertIn("n/a", table)
        self.assertIn("100.0%", table)
        self.assertIn("nothing reached the validator", table)

    def test_classify_rejection_buckets_by_what_broke(self):
        """Classify rejection buckets by what broke."""
        cases = {
            "hypothesis: unexpected field(s) ['confidence']; allowed": (
                "unexpected field(s) ['confidence']"
            ),
            "hypothesis: missing required field 'rationale'": "missing field 'rationale'",
            "hypothesis 'H' test: field 'primitive' names 'x', which is not a known test; k": (
                "unknown primitive"
            ),
        }
        for message, expected in cases.items():
            self.assertEqual(classify_rejection(message), expected)


class CanaryTests(unittest.TestCase):
    """The canary proves the end of the briefing arrived. It must be able to fail."""

    def test_the_passphrase_is_the_last_thing_before_the_instruction(self):
        """The passphrase is the last thing before the instruction."""
        seen: dict[str, str] = {}

        def reply(user_text: str) -> str:
            seen["user"] = user_text
            return "ELDERBERRY-1234"

        result = run_canary(
            "m",
            "BRIEFING BODY",
            make_generator=lambda **kw: _ScriptedBackend([reply], **kw),
            passphrase="ELDERBERRY-1234",
        )
        self.assertTrue(result.passed)
        body_end = seen["user"].index("BRIEFING BODY") + len("BRIEFING BODY")
        marker = seen["user"].index(CANARY_TEMPLATE.format(token="ELDERBERRY-1234"))
        self.assertGreater(marker, body_end)

    def test_a_model_that_cannot_quote_it_back_fails(self):
        """A model that cannot quote it back fails."""
        result = run_canary(
            "m",
            "BRIEFING BODY",
            make_generator=lambda **kw: _ScriptedBackend(["I do not see a passphrase."], **kw),
            passphrase="ELDERBERRY-1234",
        )
        self.assertFalse(result.passed)
        self.assertIn("FAIL", result.describe())

    def test_describe_reports_the_numbers_it_measured(self):
        """Describe reports the numbers it measured."""
        result = run_canary(
            "m",
            "BRIEFING BODY",
            make_generator=lambda **kw: _ScriptedBackend(["ELDERBERRY-1234"], **kw),
            passphrase="ELDERBERRY-1234",
        )
        described = result.describe()
        self.assertIn("PASS", described)
        self.assertIn("3200", described)


class NoNetworkTests(unittest.TestCase):
    """A guard on the rule the rest of this file relies on."""

    def test_the_default_url_is_loopback(self):
        """The default url is loopback."""
        from llm_insights.agent.local import DEFAULT_LOCAL_URL

        self.assertTrue(DEFAULT_LOCAL_URL.startswith("http://127.0.0.1"))

    def test_http_json_is_the_only_transport(self):
        """Http json is the only transport."""
        # If this ever stops being the injection point, every test above is fiction.
        self.assertIs(LocalGenerator(model="m", url="http://127.0.0.1:1").transport, http_json)


if __name__ == "__main__":
    unittest.main()
