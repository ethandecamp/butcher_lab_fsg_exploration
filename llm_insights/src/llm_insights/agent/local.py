"""A generator backed by a local open-weights model, over the OpenAI chat shape.

Every other live backend in this package reaches Anthropic. That is fine for a demo
on Ethan's laptop and wrong for two things this project now wants:

* **Portability.** A container that needs one person's Claude login is useless to
  anyone else. A local model needs no account, no key and no network.
* **A clean instrument.** The Claude Code CLI wraps the prompt in its own agent
  instructions, so any survival rate measured through it carries an uncontrolled
  variable. :class:`LocalGenerator` sends exactly the two messages this package built
  and nothing else.

:class:`LocalGenerator` speaks ``POST /v1/chat/completions`` — the OpenAI-compatible
shape that both Ollama and ``llama.cpp --server`` serve — so it is not tied to either.
It reuses :mod:`llm_insights.agent.generator`'s prompt builders and response parsers
verbatim, which is what makes it interchangeable with the other backends: nothing
downstream can tell which one wrote a claim.

Configuration is **environment variables only**: :data:`LOCAL_URL_ENV`,
:data:`LOCAL_MODEL_ENV`, and the optional :data:`LOCAL_CONTEXT_ENV`. There is no
credential anywhere in this module. No ``Authorization`` header is built, no key is
read from the environment, and :func:`request_headers` returns a fixed two-entry
mapping so there is nowhere for one to hide. See ``../INSTRUCTIONS.md`` §5b for the
exact ``export`` lines.

The context window is the hazard worth knowing about
----------------------------------------------------

Ollama derives a default context from available VRAM, and on a 16 GB machine that
default has been as low as **4096 tokens**. The briefing alone is around 3,200. A
prompt that does not fit is **not** rejected by the server: it is silently truncated,
and the model then answers about a briefing whose end it never saw. That failure is
invisible in the output — the claims still look plausible — so it has to be caught
before the request goes out.

:meth:`LocalGenerator.check_context` therefore reads the length the *server* reports
(``/api/ps`` on Ollama, ``/props`` on llama.cpp) and refuses to send a prompt that
would not fit, naming the fix. Token counts are estimated from character length with
:data:`CHARS_PER_TOKEN`, deliberately set low so the estimate errs high: refusing a
prompt that would have fitted costs a message, and sending one that does not fit costs
a wrong answer that looks right.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any, Final

from llm_insights.agent.generator import (
    build_narrow_prompt,
    build_system_prompt,
    build_user_prompt,
    parse_response,
    parse_single,
)

LOG = logging.getLogger(__name__)

#: Base URL of the local server. Read at call time, never stored as a credential.
LOCAL_URL_ENV: Final[str] = "LLM_INSIGHTS_LOCAL_URL"
#: Model tag the local server serves, e.g. ``"llama3.1:8b"``.
LOCAL_MODEL_ENV: Final[str] = "LLM_INSIGHTS_LOCAL_MODEL"
#: Optional override for the server's context length, in tokens. Set it only when the
#: server does not report one; a wrong value here re-enables silent truncation.
LOCAL_CONTEXT_ENV: Final[str] = "LLM_INSIGHTS_LOCAL_CONTEXT"

#: Where the local server is looked for when :data:`LOCAL_URL_ENV` is unset. Ollama's
#: default bind address. Loopback only: this backend never leaves the machine.
DEFAULT_LOCAL_URL: Final[str] = "http://127.0.0.1:11434"

#: Default model tag. Chosen by the TASK-016 bake-off; see that task's Log for the
#: measured rejection rates behind it.
DEFAULT_LOCAL_MODEL: Final[str] = "llama3.1:8b"

#: Per-request socket timeout in seconds. A local 8B model on a laptop generates at
#: roughly 20-30 tokens/s, so a 1,500-token reply is about a minute; this leaves room
#: for a cold model load on top of that.
DEFAULT_TIMEOUT: Final[float] = 600.0

#: Maximum tokens sampled per call, and the completion budget reserved by
#: :meth:`LocalGenerator.check_context`.
DEFAULT_MAX_TOKENS: Final[int] = 2048

#: Sampling temperature. Lower than the Anthropic backend's 1.0 because an 8B model
#: asked for strict JSON degrades into prose much faster as temperature rises.
DEFAULT_TEMPERATURE: Final[float] = 0.4

#: Characters per token used to estimate prompt size. Measured, not assumed, and the
#: measurement is why this is as low as it is. The 2026-09-08 proposal prompt is 18,980
#: characters (6,057 system + 12,923 user) and Ollama reported **6,248** prompt tokens
#: for ``llama3.1:8b`` and **7,277** for ``qwen3:8b`` -- 3.04 and 2.61 characters per
#: token, far below the ~4.0 that plain English gives, because the briefing is mostly
#: numbers, units and identifiers. At 3.0 the estimate came out *under* qwen3's real
#: count by 950 tokens, which is the wrong direction for a guard: an under-estimate
#: lets a prompt through that the server then truncates in silence. 2.0 clears both
#: measured tokenizers by about 30% and leaves room for one this project has not tried.
CHARS_PER_TOKEN: Final[float] = 2.0

#: Slack added to the estimate for chat-template markup — role headers, turn
#: delimiters and the generation prompt — which is not part of any message body.
TEMPLATE_TOKEN_OVERHEAD: Final[int] = 64

#: What to tell someone whose server reports a context too small for the briefing.
CONTEXT_HINT: Final[str] = (
    "restart the server with a real context window, e.g. "
    "`OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 "
    "ollama serve`, or set " + LOCAL_CONTEXT_ENV + " if the server cannot report one"
)


class LocalBackendError(RuntimeError):
    """A local server could not be reached, or answered with something unusable.

    Subclasses :class:`RuntimeError` so :mod:`llm_insights.agent.run`'s existing
    handling catches it unchanged, while a caller that wants to tell a local-server
    fault from a hypothesis-level failure still can.
    """


class ContextTooSmallError(LocalBackendError):
    """The assembled prompt does not fit the context the server reported.

    Raised *before* the request is sent. The alternative is worse than an error: a
    server given an over-long prompt truncates it silently and answers about a
    briefing whose end it never read.
    """


def http_json(
    url: str,
    payload: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    timeout: float,
) -> Any:
    """Make one HTTP request and decode a JSON body.

    This is the single transport all of :class:`LocalGenerator` goes through, and it
    is the injection point tests replace. Every failure mode a local server can
    produce -- refused connection, hang, HTTP 500, a body that is not JSON -- comes
    back as a :class:`LocalBackendError` naming what happened, rather than as an
    ``URLError`` or a ``JSONDecodeError`` from three frames down.

    Args:
        url: The full endpoint URL.
        payload: Request body, serialized to JSON. None issues a GET instead.
        headers: Request headers.
        timeout: Socket timeout in seconds.

    Returns:
        The decoded JSON body, of whatever shape the server sent.

    Raises:
        LocalBackendError: On any transport error, non-2xx status, or non-JSON body.
    """
    data = json.dumps(dict(payload)).encode("utf-8") if payload is not None else None
    method = "POST" if data is not None else "GET"
    request = urllib.request.Request(url, data=data, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise LocalBackendError(
            f"the local model server returned HTTP {exc.code} for {url}: {detail}"
        ) from None
    except urllib.error.URLError as exc:
        raise LocalBackendError(
            f"could not reach the local model server at {url} ({exc.reason}); start it, "
            f"or point {LOCAL_URL_ENV} at one that is running"
        ) from None
    except TimeoutError:
        raise LocalBackendError(
            f"the local model server at {url} did not respond within {timeout:g}s"
        ) from None
    except OSError as exc:
        raise LocalBackendError(f"could not reach the local model server at {url}: {exc}") from None

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        excerpt = body.strip().replace("\n", " ")[:200]
        raise LocalBackendError(
            f"the local model server at {url} returned a body that is not JSON ({exc}); "
            f"it began: {excerpt!r}"
        ) from None


def request_headers() -> dict[str, str]:
    """Build the headers for a local request.

    Fixed, and deliberately so. There is no ``Authorization`` header, no ``x-api-key``,
    and nothing read from the environment: a local server needs no credential, and a
    backend that never assembles one cannot leak one. ``tests/test_local.py`` asserts
    that no request this module makes contains ``ANTHROPIC_API_KEY``'s value even when
    that variable is exported.

    Returns:
        The complete header mapping.
    """
    return {"content-type": "application/json", "accept": "application/json"}


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens a string will occupy.

    No tokenizer is available -- adding one would mean a third-party runtime
    dependency this package does not have, and neither Ollama nor llama.cpp exposes a
    tokenizer over the OpenAI-compatible surface. The estimate is therefore made from
    character length at :data:`CHARS_PER_TOKEN`, which is set below the real ratio so
    the result errs high. See the module docstring for why erring high is the safe
    direction.

    Args:
        text: The string to measure.

    Returns:
        A deliberate over-estimate of the token count, never negative.
    """
    return math.ceil(max(len(text), 0) / CHARS_PER_TOKEN)


class LocalGenerator:
    """Propose and narrow hypotheses with a model served on this machine.

    One instance holds no session and no credential. Each call is a stateless
    ``POST /v1/chat/completions`` carrying the system prompt this package builds and
    one user turn, which is the whole reason this backend is a cleaner instrument than
    the CLI one.

    Args:
        model: Model tag the server serves. Defaults to :data:`LOCAL_MODEL_ENV`, then
            :data:`DEFAULT_LOCAL_MODEL`.
        url: Base URL of the server. Defaults to :data:`LOCAL_URL_ENV`, then
            :data:`DEFAULT_LOCAL_URL`.
        max_tokens: Maximum tokens sampled per call.
        temperature: Sampling temperature.
        timeout: Per-request socket timeout in seconds.
        json_mode: Ask the server for ``response_format: json_object``. Off by
            default: it constrains the reply to a JSON *object* while the proposal
            schema is a JSON *array*, so it trades one failure mode for another. The
            bake-off measures both.

    Attributes:
        name: Provenance string recorded in run metadata, ``"local:<model>"``.
        transport: Injection point for tests; signature matches :func:`http_json`.
    """

    def __init__(  # noqa: D107 - documented on the class
        self,
        model: str | None = None,
        url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        json_mode: bool = False,
    ) -> None:
        self.model = str(model or os.environ.get(LOCAL_MODEL_ENV) or DEFAULT_LOCAL_MODEL)
        self.url = str(url or os.environ.get(LOCAL_URL_ENV) or DEFAULT_LOCAL_URL).rstrip("/")
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.json_mode = bool(json_mode)
        self.name = f"local:{self.model}"
        self.transport = http_json

        #: Calls made by this instance.
        self.calls: int = 0
        #: Token totals the server reported, when it reports them.
        self.tokens: dict[str, int] = {"input": 0, "output": 0}
        #: Wall-clock seconds spent inside :meth:`complete`.
        self.seconds: float = 0.0
        #: Context length last read from the server, or None if it never reported one.
        self.context_length: int | None = None
        #: True once a warm-up request has been sent to make the server report one.
        self._warmed: bool = False

        LOG.info("local generator ready: model %s at %s", self.model, self.url)

    def __repr__(self) -> str:
        """Return a short representation of the backend and its sampling settings."""
        return (
            f"LocalGenerator(model={self.model!r}, url={self.url!r}, "
            f"max_tokens={self.max_tokens}, temperature={self.temperature})"
        )

    # --- context window ---------------------------------------------------------------

    def _ollama_context(self) -> int | None:
        """Read the serving context length from Ollama's ``/api/ps``.

        ``/api/ps`` lists models that are *loaded*, with the context length the server
        actually gave each one -- which is the number that matters, and is not the
        model's trained maximum. A model that is not loaded is simply absent, so this
        returns None rather than guessing.

        Returns:
            The reported context length, or None when the server does not report one.
        """
        try:
            body = self.transport(f"{self.url}/api/ps", None, request_headers(), 30.0)
        except LocalBackendError as exc:
            LOG.debug("no Ollama /api/ps on this server (%s)", exc)
            return None
        models = body.get("models") if isinstance(body, Mapping) else None
        if not isinstance(models, list):
            return None
        for entry in models:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("model") or entry.get("name") or "") != self.model:
                continue
            length = entry.get("context_length")
            if isinstance(length, int) and not isinstance(length, bool) and length > 0:
                return length
        return None

    def _llama_cpp_context(self) -> int | None:
        """Read the serving context length from ``llama.cpp --server``'s ``/props``.

        Returns:
            ``default_generation_settings.n_ctx``, or None when it is not reported.
        """
        try:
            body = self.transport(f"{self.url}/props", None, request_headers(), 30.0)
        except LocalBackendError as exc:
            LOG.debug("no llama.cpp /props on this server (%s)", exc)
            return None
        if not isinstance(body, Mapping):
            return None
        settings = body.get("default_generation_settings")
        length = settings.get("n_ctx") if isinstance(settings, Mapping) else body.get("n_ctx")
        if isinstance(length, int) and not isinstance(length, bool) and length > 0:
            return length
        return None

    def warm_up(self) -> None:
        """Send one trivial request so the server loads the model and can report on it.

        Ollama reports a serving context length only for a model it currently holds in
        memory, so a check made against a cold server learns nothing. One two-token
        request fixes that and costs a fraction of a second of laptop time. It runs at
        most once per instance, whether or not it succeeds -- a server that cannot
        answer a warm-up will not answer a proposal either, and that failure belongs to
        the real call, where the error message has context.
        """
        if self._warmed:
            return
        self._warmed = True
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            self.transport(
                f"{self.url}/v1/chat/completions", payload, request_headers(), self.timeout
            )
        except LocalBackendError as exc:
            LOG.warning("warm-up request to %s failed (%s)", self.url, exc)
            return
        LOG.debug("warm-up complete; %s should now be loaded", self.model)

    def server_context_length(self) -> int | None:
        """Report the context length this server will actually serve, in tokens.

        Consulted in order: the :data:`LOCAL_CONTEXT_ENV` override, Ollama's
        ``/api/ps``, llama.cpp's ``/props``, and finally the same two again after a
        :meth:`warm_up`, because Ollama reports nothing about a model it has not
        loaded.

        Returns:
            The context length, or None when no source reports one.
        """
        override = os.environ.get(LOCAL_CONTEXT_ENV, "").strip()
        if override:
            try:
                value = int(override)
            except ValueError:
                LOG.warning("%s is not an integer (%r); ignoring it", LOCAL_CONTEXT_ENV, override)
            else:
                if value > 0:
                    self.context_length = value
                    return value
                LOG.warning("%s must be positive, got %d; ignoring it", LOCAL_CONTEXT_ENV, value)

        for attempt in (0, 1):
            if attempt == 1:
                self.warm_up()
            length = self._ollama_context() or self._llama_cpp_context()
            if length is not None:
                self.context_length = length
                return length
            if self._warmed:
                break
        return None

    def check_context(self, system: str, user: str) -> None:
        """Refuse to send a prompt the server would silently truncate.

        Args:
            system: The system message about to be sent.
            user: The user message about to be sent.

        Raises:
            ContextTooSmallError: If the estimated prompt plus the reserved completion
                budget exceeds the length the server reported.
        """
        length = self.server_context_length()
        needed = (
            estimate_tokens(system)
            + estimate_tokens(user)
            + TEMPLATE_TOKEN_OVERHEAD
            + self.max_tokens
        )
        if length is None:
            LOG.warning(
                "the server at %s does not report a context length, so a prompt of "
                "about %d tokens is being sent unchecked; set %s to make this check "
                "real",
                self.url,
                needed,
                LOCAL_CONTEXT_ENV,
            )
            return
        if needed > length:
            raise ContextTooSmallError(
                f"refusing to send this prompt: it needs about {needed} tokens "
                f"({estimate_tokens(system)} system + {estimate_tokens(user)} user + "
                f"{TEMPLATE_TOKEN_OVERHEAD} template + {self.max_tokens} reserved for "
                f"the reply) but {self.url} reports a context of {length} tokens for "
                f"{self.model}. An over-long prompt is truncated silently, not "
                f"rejected, so the model would answer about a briefing whose end it "
                f"never saw. To fix: {CONTEXT_HINT}."
            )
        LOG.info("prompt fits: about %d tokens against a reported context of %d", needed, length)

    # --- one call ---------------------------------------------------------------------

    def payload(self, system: str, user: str) -> dict[str, Any]:
        """Build the chat-completions request body.

        Args:
            system: The system message.
            user: The user message.

        Returns:
            The request body.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def account(self, envelope: Mapping[str, Any], elapsed: float) -> None:
        """Record what one call consumed.

        Args:
            envelope: The decoded response body.
            elapsed: Wall-clock seconds the call took.
        """
        self.seconds += elapsed
        usage = envelope.get("usage")
        if isinstance(usage, Mapping):
            for key, field in (("input", "prompt_tokens"), ("output", "completion_tokens")):
                value = usage.get(field)
                if isinstance(value, int) and not isinstance(value, bool):
                    self.tokens[key] += value
        LOG.info(
            "call %d: %.1fs, %d in / %d out tokens cumulative",
            self.calls,
            elapsed,
            self.tokens["input"],
            self.tokens["output"],
        )

    @staticmethod
    def extract(envelope: Any) -> str:
        """Pull the assistant text out of a chat-completions envelope.

        Written to fail loudly on every shape a misbehaving server produces, because
        the alternative -- an empty string flowing on into the parser -- reports a
        transport fault as a modelling result.

        Args:
            envelope: The decoded response body.

        Returns:
            The assistant's text.

        Raises:
            LocalBackendError: If the envelope is not an object, carries no usable
                ``choices[0].message.content``, or that content is blank.
        """
        if not isinstance(envelope, Mapping):
            raise LocalBackendError(
                "the local model server returned JSON that is not an object, so it is "
                f"not a chat completion: got {type(envelope).__name__}"
            )
        if isinstance(envelope.get("error"), Mapping | str):
            detail = envelope["error"]
            message = detail.get("message") if isinstance(detail, Mapping) else detail
            raise LocalBackendError(f"the local model server reported an error: {message}")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LocalBackendError(
                "the local model server's reply has no 'choices' list, so it is not an "
                f"OpenAI-shaped chat completion; keys were {sorted(map(str, envelope))}"
            )
        first = choices[0]
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            reason = first.get("finish_reason") if isinstance(first, Mapping) else None
            raise LocalBackendError(
                "the local model server returned a choice with no message content; "
                f"finish_reason was {reason!r}"
            )
        return content

    def complete(self, user_text: str, system: str | None = None) -> str:
        """Make one call and return the assistant's text.

        Args:
            user_text: The varying half of the request.
            system: System prompt to send instead of
                :func:`~llm_insights.agent.generator.build_system_prompt`. Defaults to
                None, which sends that one.

        Returns:
            The assistant's text.

        Raises:
            ContextTooSmallError: If the prompt would not fit the server's context.
            LocalBackendError: On any transport, status or envelope failure.
        """
        head = system if system is not None else build_system_prompt()
        self.check_context(head, user_text)
        self.calls += 1
        started = time.monotonic()
        envelope = self.transport(
            f"{self.url}/v1/chat/completions",
            self.payload(head, user_text),
            request_headers(),
            self.timeout,
        )
        elapsed = time.monotonic() - started
        if isinstance(envelope, Mapping):
            self.account(envelope, elapsed)
        else:
            self.seconds += elapsed
        return self.extract(envelope)

    # --- the Generator protocol -------------------------------------------------------

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Ask the local model for ``n`` hypotheses.

        Args:
            briefing: The factual summary of the dataset.
            question: The question the run is answering.
            n: How many hypotheses to request.

        Returns:
            Raw mappings, unvalidated. Validation is deliberately the caller's job:
            see :class:`~llm_insights.agent.generator.Generator`.
        """
        LOG.info("asking %s for %d hypotheses", self.name, n)
        return parse_response(self.complete(build_user_prompt(briefing, question, n)))

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Ask the local model to narrow one falsified claim.

        Args:
            briefing: The factual summary of the dataset.
            failed: The falsified hypothesis, serialized.
            outcome: Its outcome, with every observed number.

        Returns:
            The successor mapping, or None if the model declined.
        """
        LOG.info("asking %s to narrow %s", self.name, failed.get("id"))
        return parse_single(self.complete(build_narrow_prompt(briefing, failed, outcome)))

    def synthesize(self, prompt: str) -> str | None:
        """Write the plain-English narration of a finished run.

        Made after every verdict is already fixed, and subject to the same numeric
        containment check as every other backend's -- see
        :mod:`llm_insights.agent.synthesis`. Nothing here judges a result.

        Args:
            prompt: The complete user turn, built by
                :func:`llm_insights.agent.synthesis.build_synthesis_prompt`.

        Returns:
            The summary text, or None if the server returned nothing usable.

        Raises:
            LocalBackendError: On any transport or envelope failure.
        """
        from llm_insights.agent.synthesis import SYNTHESIS_SYSTEM_PROMPT

        LOG.info("asking %s for a narrative summary", self.name)
        return self.complete(prompt, SYNTHESIS_SYSTEM_PROMPT) or None

    def close(self) -> None:
        """Release resources. There are none; the log line records what was spent.

        Present so :meth:`llm_insights.api.Study.close` finds it, and safe to call
        more than once. A local call holds no session, no temporary directory and no
        allowance, so there is genuinely nothing to tear down.
        """
        LOG.info("local generator finished: %s", self.usage_summary())

    def usage_summary(self) -> str:
        """Format what this backend consumed, for printing after an investigation.

        Returns:
            A one-line summary. No dollar figure appears, because a local call costs
            nothing beyond electricity and saying otherwise would be an invented
            number.
        """
        return (
            f"{self.calls} local call(s), {self.seconds:.1f}s, "
            f"{self.tokens['input']} in / {self.tokens['output']} out tokens, $0.00"
        )


__all__ = [
    "CHARS_PER_TOKEN",
    "CONTEXT_HINT",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_LOCAL_URL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
    "LOCAL_CONTEXT_ENV",
    "LOCAL_MODEL_ENV",
    "LOCAL_URL_ENV",
    "TEMPLATE_TOKEN_OVERHEAD",
    "ContextTooSmallError",
    "LocalBackendError",
    "LocalGenerator",
    "estimate_tokens",
    "http_json",
    "request_headers",
]
