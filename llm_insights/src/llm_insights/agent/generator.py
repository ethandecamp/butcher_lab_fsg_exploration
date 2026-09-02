"""Hypothesis generators: the swappable half of the agent loop.

A generator writes claims. It never judges them — that is
:mod:`llm_insights.harness.runner`'s job, and keeping the two apart is the whole
point of the design. Three backends implement the same :class:`Generator` protocol:

* :class:`AnthropicGenerator` calls the Anthropic Messages API over plain
  :mod:`urllib.request`, so the project has no third-party runtime dependency and
  works in an environment where ``anthropic`` cannot be installed.
* :class:`TranscriptGenerator` replays a recorded run from disk. It is what the demo
  runs on, and it names the file it replayed in :attr:`TranscriptGenerator.name` so a
  replay can never be mistaken for a live call.
* :class:`EchoGenerator` is a deterministic offline stub for tests.

The prompts are built here rather than stored as text files, because
:func:`build_system_prompt` *introspects* :data:`llm_insights.harness.primitives.PRIMITIVES`
to enumerate the tests, their exact parameter names, which are required, and their
defaults. A primitive that gains a parameter changes the prompt in the same commit;
the prompt cannot drift away from the code it describes.

Two rules apply to the API key throughout: it is read from the environment at call
time and never stored on an instance, and no ``__repr__``, log record, exception
message or serialized field in this package ever contains it.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from llm_insights.harness import primitives as _primitives
from llm_insights.harness.primitives import PRIMITIVES

LOG = logging.getLogger(__name__)

#: Environment variable holding the Anthropic API key. Read at call time, never stored.
API_KEY_ENV: Final[str] = "ANTHROPIC_API_KEY"
#: Anthropic Messages endpoint.
ANTHROPIC_URL: Final[str] = "https://api.anthropic.com/v1/messages"
#: API version header value the Messages API requires.
ANTHROPIC_VERSION: Final[str] = "2023-06-01"
#: Default model. Overridable per instance and from the CLI.
DEFAULT_MODEL: Final[str] = "claude-sonnet-4-5"

#: Where a primitive's per-parameter vocabulary lives in the primitives module.
#: Looked up with :func:`getattr` so a renamed tuple shows up as a build-time warning
#: instead of a prompt that quietly lists the wrong values.
_CHOICE_SOURCES: Final[dict[tuple[str, str], str]] = {
    ("peak_location", "field"): "_PEAK_FIELDS",
    ("fraction_above", "field"): "_FRACTION_FIELDS",
    ("profile_monotonic", "field"): "_PROFILE_FIELDS",
}

#: Vocabularies the primitives spell inline in ``_check_choice`` calls rather than
#: hoisting to a module constant. Kept here so the prompt still states them.
_INLINE_CHOICES: Final[dict[str, tuple[str, ...]]] = {
    "direction": ("increasing", "decreasing"),
    "method": ("pearson", "spearman"),
}

#: One-line human description of parameters whose vocabulary is open-ended.
_PARAM_NOTES: Final[dict[str, str]] = {
    "case": "a case folder name or condition label",
    "case_a": "a case folder name or condition label",
    "case_b": "a case folder name or condition label",
    "cases": "a list of two or more distinct case names, in the asserted order",
    "metric": "a metric name from the metric registry",
    "field_x": "a GRN table column",
    "field_y": "a GRN table column",
    "s_min": "lower edge of the arc window, 0 = atrial end, 1 = ventricular end",
    "s_max": "upper edge of the arc window, 0 = atrial end, 1 = ventricular end",
    "step": "growth step 0-14",
}

#: Keys a model sometimes wraps its array in. Unwrapped by :func:`parse_response`.
_WRAPPER_KEYS: Final[tuple[str, ...]] = ("hypotheses", "claims", "items", "results", "data")

#: Version stamp written into transcripts, so a future format change is detectable.
TRANSCRIPT_SCHEMA_VERSION: Final[str] = "1"


@runtime_checkable
class Generator(Protocol):
    """What the loop requires of a hypothesis backend.

    Both methods return *raw* mappings in the
    :func:`llm_insights.harness.spec.hypothesis_from_dict` schema. Validation is the
    caller's job on purpose: a generator that pre-filtered its own output could drop
    a malformed claim silently, and the loop is required to count those.

    Attributes:
        name: Provenance string recorded in the run metadata, e.g.
            ``"anthropic:claude-sonnet-4-5"`` or ``"transcript:data/run.json"``.
    """

    name: str

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Propose up to ``n`` hypotheses about the briefing."""
        ...

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Write a narrower successor to a falsified claim, or None to give up."""
        ...


# --- Prompt construction -----------------------------------------------------------


def _annotation_name(annotation: Any) -> str:
    """Render a parameter annotation as a short type word for the prompt."""
    text = getattr(annotation, "__name__", None) or str(annotation)
    text = text.replace("collections.abc.", "").replace("typing.", "")
    return {
        "str": "string",
        "float": "number",
        "int": "whole number",
        "bool": "true/false",
        "Sequence[str]": "list of strings",
    }.get(text, text)


def _allowed_values(primitive: str, param: str) -> tuple[str, ...] | None:
    """Look up the fixed vocabulary for one primitive parameter, if it has one.

    Args:
        primitive: Registered primitive name.
        param: Parameter name on that primitive.

    Returns:
        The allowed values, or None when the parameter is open-ended.
    """
    if param == "op":
        ops = getattr(_primitives, "_OPS", None)
        return tuple(sorted(ops)) if ops else None
    source = _CHOICE_SOURCES.get((primitive, param))
    if source is not None:
        values = getattr(_primitives, source, None)
        if values is None:
            LOG.warning(
                "primitives module has no %s, so the prompt cannot list the allowed "
                "values for %s.%s",
                source,
                primitive,
                param,
            )
            return None
        return tuple(values)
    return _INLINE_CHOICES.get(param)


def _metric_names() -> tuple[str, ...]:
    """Return the registered metric names, or an empty tuple if the registry is absent.

    The registry (``llm_insights.summary.metrics``) is written in parallel with this
    module. When it is missing the prompt says so rather than inventing metric names.
    """
    try:
        from llm_insights.summary.metrics import METRIC_NAMES
    except ImportError:
        LOG.warning("metric registry unavailable; the prompt will say so")
        return ()
    return tuple(sorted(METRIC_NAMES))


def _grn_columns() -> tuple[str, ...]:
    """Return the GRN table column names ``correlation`` accepts."""
    from llm_insights.io.dataset import GRN_NODES, GRN_SCENARIOS

    base = ("von_mises_pa", "wss_dyn_cm2", "mech_norm", "wss_norm")
    activities = tuple(f"{node}_{scen}" for node in GRN_NODES for scen in GRN_SCENARIOS)
    return base + activities


def _case_names() -> tuple[str, ...]:
    """Return every accepted case folder name and condition label."""
    from llm_insights.io.dataset import CASE_BY_CONDITION, CONDITION_BY_CASE

    return tuple(sorted(CONDITION_BY_CASE)) + tuple(sorted(CASE_BY_CONDITION))


def _describe_primitive(name: str, fn: Any) -> list[str]:
    """Describe one primitive by introspecting its signature and docstring.

    Args:
        name: The registered name, which is what a ``test.primitive`` field must say.
        fn: The primitive function.

    Returns:
        Prompt lines: the call signature, its one-line purpose, and one line per
        parameter naming its type, whether it is required, and its vocabulary.
    """
    sig = inspect.signature(fn)
    params = [(p, spec) for p, spec in sig.parameters.items() if p != "ds"]

    rendered: list[str] = []
    for param, spec in params:
        if spec.default is inspect.Parameter.empty:
            rendered.append(param)
        else:
            rendered.append(f"{param}={spec.default!r}")
    lines = [f"{name}({', '.join(rendered)})"]

    doc = inspect.getdoc(fn) or ""
    first = doc.splitlines()[0] if doc else ""
    if first:
        lines.append(f"  purpose: {first}")

    for param, spec in params:
        required = spec.default is inspect.Parameter.empty
        bits = [_annotation_name(spec.annotation)]
        bits.append("required" if required else f"optional, default {spec.default!r}")
        allowed = _allowed_values(name, param)
        if allowed is not None:
            bits.append("one of " + json.dumps(list(allowed)))
        elif param in _PARAM_NOTES:
            bits.append(_PARAM_NOTES[param])
        lines.append(f"  - {param}: {'; '.join(bits)}")
    return lines


def build_system_prompt() -> str:
    """Build the system prompt: the rules, and the six tests a claim may compile into.

    The prompt is byte-identical on every call in a process, which is what makes
    server-side prompt caching hit. Everything that varies — the briefing, the
    question, the falsified claim being narrowed — goes in the user turn instead.

    The primitive catalogue is generated by introspecting
    :data:`llm_insights.harness.primitives.PRIMITIVES`, so the parameter names and
    defaults quoted to the model are the ones the code will actually accept.

    Returns:
        The complete system prompt.
    """
    metrics = _metric_names()
    metric_line = (
        "  metric names currently registered: " + json.dumps(list(metrics))
        if metrics
        else "  the metric registry is not loaded in this environment, so avoid "
        "compare_metric and metric_ordering unless the briefing names a metric"
    )

    catalogue: list[str] = []
    for name in sorted(PRIMITIVES):
        catalogue.extend(_describe_primitive(name, PRIMITIVES[name]))
        catalogue.append("")

    return "\n".join(
        [
            "You propose falsifiable hypotheses about the output of a fluid-solid-growth "
            "simulation of a developing heart valve cushion, coupled to a gene-regulatory "
            "network model.",
            "",
            "You do not decide whether a claim is true. Executable code runs each test "
            "against the simulation output and returns the verdict. Your job is to write "
            "claims worth testing and to state, in advance, exactly what would refute them.",
            "",
            "RULES",
            "1. Propose only claims that compile into one of the six tests listed below. "
            "If an interesting idea cannot be expressed as one of them, discard it and "
            "propose something that can.",
            "2. State the decision rule before you can know the outcome. You have not seen "
            "the numbers the test will produce, and you must not write a rule that could be "
            "adjusted after seeing them.",
            "3. The 'claim' must be qualitative and plain English, readable by a biologist "
            "who has never seen this code. The 'test' must be mechanical: a primitive name "
            "and its parameters, with every threshold fixed as a number.",
            "4. Make each claim refutable. A claim the test cannot possibly contradict is "
            "worthless; prefer a threshold the data could plausibly fall on either side of.",
            "5. Return strict JSON only. No prose before or after it, no markdown fences, "
            "no comments, no trailing commas.",
            "",
            "OUTPUT SCHEMA",
            "A JSON array of objects. Each object has exactly these fields:",
            json.dumps(
                {
                    "id": "short unique id such as H1",
                    "claim": "the qualitative claim, one sentence, plain English",
                    "rationale": "why this claim is worth testing",
                    "test": {
                        "primitive": "one of the six names below",
                        "params": {"...": "exactly the parameters that primitive takes"},
                        "decision_rule": "what counts as passing, stated before the run",
                    },
                },
                indent=2,
            ),
            "No other top-level fields are allowed; an unexpected field is a hard error.",
            "",
            "THE SIX TESTS",
            "Every parameter below is passed by name. Required parameters must be present; "
            "optional ones may be omitted and will take the default shown.",
            "",
            *catalogue,
            "VOCABULARY",
            "  cases: " + json.dumps(list(_case_names())),
            metric_line,
            "  GRN columns for correlation: " + json.dumps(list(_grn_columns())),
            "  arc position s_norm runs 0 (atrial, upstream) to 1 (ventricular).",
            "  nodal von Mises exists only at step 14; wss is available at every step.",
            "  wss is missing (NaN) at interior solid nodes and those samples are dropped, "
            "never counted as zero.",
        ]
    )


def build_user_prompt(briefing: str, question: str, n: int) -> str:
    """Build the user turn for a proposal request.

    Args:
        briefing: The factual summary of the dataset the claims must be about.
        question: The question the run is trying to answer.
        n: How many hypotheses to propose.

    Returns:
        The user message text.
    """
    return "\n".join(
        [
            "BRIEFING",
            briefing.strip(),
            "",
            "QUESTION",
            question.strip(),
            "",
            f"Propose exactly {n} hypotheses that bear on this question, as a JSON array "
            "matching the schema in your instructions. Give each a distinct id. Return the "
            "JSON array and nothing else.",
        ]
    )


def build_narrow_prompt(
    briefing: str, failed: Mapping[str, Any], outcome: Mapping[str, Any]
) -> str:
    """Build the user turn asking for a narrower successor to a falsified claim.

    Args:
        briefing: The same factual summary used for the proposal.
        failed: The falsified hypothesis, serialized.
        outcome: Its outcome, including every number the decision rule consumed.

    Returns:
        The user message text.
    """
    return "\n".join(
        [
            "BRIEFING",
            briefing.strip(),
            "",
            "A claim you proposed was tested and refuted. Here is the claim:",
            json.dumps(dict(failed), indent=2, sort_keys=True, default=str),
            "",
            "Here is what the simulation actually returned, including every number the "
            "decision rule used:",
            json.dumps(dict(outcome), indent=2, sort_keys=True, default=str),
            "",
            "Write ONE narrower claim that is consistent with these observed numbers and "
            "is still refutable: it must be a smaller claim than the one that failed, not "
            "a restatement of what was just measured, and not a rule loosened until it "
            "cannot fail. Give it a new id. Return a single JSON object matching the "
            "schema in your instructions, and nothing else. If no honest narrower claim "
            "exists, return the JSON value null.",
        ]
    )


# --- Response parsing --------------------------------------------------------------


def _strip_fences(text: str) -> list[str]:
    """Return the contents of any markdown code fences found in ``text``."""
    blocks: list[str] = []
    parts = text.split("```")
    # Fenced content sits at odd indices: prose ``` code ``` prose ``` code ``` ...
    for index in range(1, len(parts), 2):
        block = parts[index]
        newline = block.find("\n")
        if newline != -1 and block[:newline].strip().lower() in {"json", "javascript", ""}:
            block = block[newline + 1 :]
        blocks.append(block)
    return blocks


def _normalize(obj: Any) -> list[dict] | None:
    """Coerce a decoded JSON value into a list of hypothesis mappings.

    Args:
        obj: Any decoded JSON value.

    Returns:
        A non-empty list of mappings, or None if this value is not one.
    """
    if isinstance(obj, Mapping):
        for key in _WRAPPER_KEYS:
            if key in obj and isinstance(obj[key], list):
                return _normalize(obj[key])
        return [dict(obj)] if obj else None
    if isinstance(obj, list):
        items = [dict(item) for item in obj if isinstance(item, Mapping)]
        if len(items) != len(obj):
            LOG.warning("dropped %d non-object entries from the response", len(obj) - len(items))
        return items or None
    return None


def _decode_candidates(text: str) -> list[Any]:
    """Yield every JSON value that can be decoded out of ``text``, best first.

    Tries the whole string, then any fenced block, then a scan that decodes from each
    ``[`` or ``{`` in turn, which is what rescues JSON buried in prose.
    """
    candidates: list[Any] = []
    decoder = json.JSONDecoder()

    for source in [text, *_strip_fences(text)]:
        stripped = source.strip()
        if not stripped:
            continue
        with contextlib.suppress(json.JSONDecodeError):
            candidates.append(json.loads(stripped))

    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(value)
    return candidates


def parse_response(text: str) -> list[dict]:
    """Extract hypothesis mappings from a model response, tolerating packaging.

    Handles a bare JSON array, an array inside ``json`` fences, an array with prose
    on either side, a single object instead of an array, and an object wrapping the
    array under a key such as ``"hypotheses"``. Nothing here validates the *content*
    of a mapping — :func:`llm_insights.harness.spec.hypothesis_from_dict` does that,
    and it must be the only place that does.

    Args:
        text: The raw assistant text.

    Returns:
        One or more mappings, in the order the model wrote them.

    Raises:
        ValueError: If no JSON object or array can be found at all.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the model returned an empty response, so there is nothing to parse")

    for candidate in _decode_candidates(text):
        items = _normalize(candidate)
        if items:
            LOG.debug("parsed %d object(s) from a %d-character response", len(items), len(text))
            return items

    excerpt = text.strip().replace("\n", " ")[:200]
    raise ValueError(
        "no JSON object or array could be found in the model response; the response "
        f"began: {excerpt!r}"
    )


def parse_single(text: str) -> dict | None:
    """Parse a narrowing response, which is one object or an explicit ``null``.

    Args:
        text: The raw assistant text.

    Returns:
        The single mapping, or None when the model declined to narrow.

    Raises:
        ValueError: If the text is neither ``null`` nor parseable as an object.
    """
    if isinstance(text, str) and text.strip().lower() in {"null", '"null"', "none"}:
        LOG.info("the generator declined to narrow this claim")
        return None
    items = parse_response(text)
    if len(items) > 1:
        LOG.warning("narrowing returned %d objects; using the first", len(items))
    return items[0]


# --- Anthropic ---------------------------------------------------------------------


def _http_post_json(
    url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: float
) -> dict:
    """POST JSON and decode the JSON response.

    Args:
        url: Endpoint.
        payload: Request body, serialized to JSON.
        headers: Request headers. Never logged: one of them is the API key.
        timeout: Socket timeout in seconds.

    Returns:
        The decoded response body.

    Raises:
        RuntimeError: On any transport or HTTP error. The message carries the status
            and the server's body, never the request headers.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"the Anthropic API returned HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach the Anthropic API: {exc.reason}") from None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"the Anthropic API returned a non-JSON body: {exc}") from None


def _require_api_key() -> str:
    """Read the API key from the environment.

    Returns:
        The key.

    Raises:
        RuntimeError: If the variable is unset or empty, naming the variable.
    """
    key = os.environ.get(API_KEY_ENV, "")
    if not key.strip():
        raise RuntimeError(
            f"the environment variable {API_KEY_ENV} is not set, so the Anthropic API "
            "cannot be called; set it, or run with --generator transcript to replay a "
            "recorded run instead"
        )
    return key


class AnthropicGenerator:
    """Propose and narrow hypotheses by calling the Anthropic Messages API.

    Implemented against the HTTP API with :mod:`urllib.request` rather than the
    ``anthropic`` package, so the project keeps a stdlib-only runtime.

    The API key is read from the environment on every call and is never stored on the
    instance, never logged, and never placed in an exception message or in run
    metadata. :meth:`__repr__` shows only the model and sampling settings.

    Args:
        model: Model id to call.
        max_tokens: Maximum tokens to sample per call.
        temperature: Sampling temperature.
        timeout: Per-request socket timeout in seconds.

    Raises:
        RuntimeError: At construction, if the API key variable is unset. Failing here
            rather than mid-run means a missing key costs nothing.
    """

    def __init__(  # noqa: D107 - documented on the class
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        timeout: float = 120.0,
    ) -> None:
        _require_api_key()  # fail fast; the value is deliberately not kept
        self.model = str(model)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.name = f"anthropic:{self.model}"
        #: Injection point for tests. Signature matches :func:`_http_post_json`.
        self.transport = _http_post_json
        LOG.info("Anthropic generator ready for model %s", self.model)

    def __repr__(self) -> str:
        """Return a representation that cannot leak the API key."""
        return (
            f"AnthropicGenerator(model={self.model!r}, max_tokens={self.max_tokens}, "
            f"temperature={self.temperature})"
        )

    def _headers(self) -> dict[str, str]:
        """Build request headers, reading the key from the environment each time."""
        return {
            "x-api-key": _require_api_key(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _payload(self, user_text: str) -> dict[str, Any]:
        """Build the request body.

        The system prompt is sent as a single cacheable block and is identical on
        every call, so the server-side cache can hit; only ``user_text`` varies.

        Args:
            user_text: The varying half of the request.

        Returns:
            The Messages API request body.
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": [
                {
                    "type": "text",
                    "text": build_system_prompt(),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_text}],
        }

    def _complete(self, user_text: str) -> str:
        """Call the API once and return the concatenated text of the reply.

        Args:
            user_text: The user turn.

        Returns:
            The assistant's text.

        Raises:
            RuntimeError: On a transport error, or a reply carrying no text block.
        """
        response = self.transport(
            ANTHROPIC_URL, self._payload(user_text), self._headers(), self.timeout
        )
        blocks = response.get("content") if isinstance(response, Mapping) else None
        if not isinstance(blocks, list):
            raise RuntimeError("the Anthropic API reply had no 'content' list")
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
        if not text.strip():
            raise RuntimeError("the Anthropic API reply contained no text block")
        usage = response.get("usage") if isinstance(response, Mapping) else None
        if isinstance(usage, Mapping):
            LOG.info(
                "model %s: %s input tokens, %s output tokens, %s cache read",
                self.model,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                usage.get("cache_read_input_tokens"),
            )
        return text

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Ask the model for ``n`` hypotheses.

        Args:
            briefing: The factual summary of the dataset.
            question: The question the run is answering.
            n: How many hypotheses to request.

        Returns:
            Raw mappings, unvalidated.
        """
        LOG.info("asking %s for %d hypotheses", self.model, n)
        return parse_response(self._complete(build_user_prompt(briefing, question, n)))

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Ask the model to narrow one falsified claim.

        Args:
            briefing: The factual summary of the dataset.
            failed: The falsified hypothesis, serialized.
            outcome: Its outcome, with every observed number.

        Returns:
            The successor mapping, or None if the model declined.
        """
        LOG.info("asking %s to narrow %s", self.model, failed.get("id"))
        return parse_single(self._complete(build_narrow_prompt(briefing, failed, outcome)))


# --- Transcript replay -------------------------------------------------------------


class TranscriptGenerator:
    """Replay generations recorded by an earlier run.

    A transcript file is either the object written by
    :func:`llm_insights.agent.loop.write_transcript` or a bare list of its entries.
    Proposals are replayed in recorded order; narrowings are matched by the id of the
    claim being narrowed, so a replay stays correct even if the loop visits failures
    in a different order.

    :attr:`name` names the file, and :attr:`recorded_from` carries the provenance of
    the backend that originally produced the text, so a report built from a replay
    can say what really generated it.

    Args:
        path: The transcript JSON file.

    Raises:
        ValueError: If the file is not a transcript.
        FileNotFoundError: If the file does not exist.
    """

    def __init__(self, path: str | Path) -> None:  # noqa: D107 - documented on the class
        self.path = Path(path)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            entries = raw.get("entries")
            self.recorded_from: dict[str, Any] = dict(raw.get("recorded_from") or {})
        else:
            entries = raw
            self.recorded_from = {}
        if not isinstance(entries, list):
            raise ValueError(
                f"{self.path} is not a transcript: expected a list of entries, or an "
                "object with an 'entries' list"
            )
        self._entries = [dict(e) for e in entries if isinstance(e, Mapping)]
        self._proposals = [e for e in self._entries if e.get("kind") == "propose"]
        self._narrowings = [e for e in self._entries if e.get("kind") == "narrow"]
        self._next_proposal = 0
        self.name = f"transcript:{self.path}"
        LOG.info(
            "replaying %s: %d proposal(s), %d narrowing(s)",
            self.path,
            len(self._proposals),
            len(self._narrowings),
        )

    def __repr__(self) -> str:
        """Return a short representation naming the replayed file."""
        return f"TranscriptGenerator(path={str(self.path)!r})"

    @property
    def source_model(self) -> str | None:
        """The model that produced the recorded text, when the transcript says."""
        model = self.recorded_from.get("model")
        return str(model) if model else None

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Return the next recorded proposal.

        ``briefing``, ``question`` and ``n`` are accepted to satisfy the protocol and
        are not consulted: a replay reproduces what was recorded, and pretending to
        respond to a new question would be a lie about provenance.

        Args:
            briefing: Ignored.
            question: Ignored.
            n: Ignored.

        Returns:
            The recorded raw mappings.

        Raises:
            RuntimeError: If the transcript holds no further proposals.
        """
        del briefing, question, n
        if self._next_proposal >= len(self._proposals):
            raise RuntimeError(
                f"{self.path} records {len(self._proposals)} proposal round(s), but the "
                "loop asked for another; the transcript does not match this run"
            )
        entry = self._proposals[self._next_proposal]
        self._next_proposal += 1
        items = _normalize(entry.get("response"))
        return items or []

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Return the recorded successor for this falsified claim, if any.

        Args:
            briefing: Ignored.
            failed: The falsified hypothesis; only its id is used, to find the record.
            outcome: Ignored.

        Returns:
            The recorded successor mapping, or None when the transcript recorded no
            successor for this claim.
        """
        del briefing, outcome
        failed_id = str(failed.get("id", ""))
        for entry in self._narrowings:
            if str(entry.get("failed_id", "")) == failed_id:
                response = entry.get("response")
                if response is None:
                    return None
                items = _normalize(response)
                return items[0] if items else None
        LOG.info("%s records no narrowing for %s", self.path, failed_id)
        return None


# --- Offline stub ------------------------------------------------------------------

#: The claim :class:`EchoGenerator` proposes when it is given no template. It uses a
#: primitive that needs no metric registry, so the stub works in isolation.
ECHO_TEMPLATE: Final[dict[str, Any]] = {
    "id": "E1",
    "claim": "Some of the cushion carries a non-zero von Mises stress at the final step.",
    "rationale": "A deterministic stub claim used to exercise the loop offline.",
    "test": {
        "primitive": "fraction_above",
        "params": {
            "field": "von_mises",
            "case": "Healthy",
            "threshold": 0.0,
            "op": "gt",
            "value": 0.0,
        },
        "decision_rule": "More than 0% of solid nodes have von Mises above 0 Pa.",
    },
}


class EchoGenerator:
    """A deterministic offline stub. For tests, never for results.

    Args:
        template: The hypothesis mapping to repeat. Ids are suffixed so a batch has
            distinct ids. Defaults to :data:`ECHO_TEMPLATE`.
        successor: The mapping returned by :meth:`narrow`, or None to always decline.
    """

    def __init__(  # noqa: D107 - documented on the class
        self,
        template: Mapping[str, Any] | None = None,
        successor: Mapping[str, Any] | None = None,
    ) -> None:
        self._template = json.loads(json.dumps(dict(template or ECHO_TEMPLATE)))
        self._successor = json.loads(json.dumps(dict(successor))) if successor else None
        self.name = "echo"

    def __repr__(self) -> str:
        """Return a short representation."""
        return "EchoGenerator()"

    def propose(self, briefing: str, question: str, n: int) -> list[dict]:
        """Return ``n`` copies of the template with distinct ids.

        Args:
            briefing: Ignored.
            question: Ignored.
            n: How many copies to return.

        Returns:
            The stub mappings.
        """
        del briefing, question
        base_id = str(self._template.get("id", "E"))
        out: list[dict] = []
        for index in range(max(int(n), 0)):
            item = json.loads(json.dumps(self._template))
            item["id"] = base_id if index == 0 else f"{base_id}_{index + 1}"
            out.append(item)
        return out

    def narrow(self, briefing: str, failed: dict, outcome: dict) -> dict | None:
        """Return the configured successor, or None.

        Args:
            briefing: Ignored.
            failed: The falsified hypothesis; its id seeds the successor id.
            outcome: Ignored.

        Returns:
            The successor mapping, or None when none was configured.
        """
        del briefing, outcome
        if self._successor is None:
            return None
        item = json.loads(json.dumps(self._successor))
        item.setdefault("id", f"{failed.get('id', 'E')}_narrowed")
        return item


# --- Factory -----------------------------------------------------------------------


def make_generator(
    kind: str, *, transcript: str | Path | None = None, model: str | None = None
) -> Generator:
    """Build a generator by name, for the CLI.

    Args:
        kind: ``"anthropic"``, ``"transcript"`` or ``"echo"``.
        transcript: Path to the transcript file, required for ``"transcript"``.
        model: Model id, used only by ``"anthropic"``.

    Returns:
        The generator.

    Raises:
        ValueError: On an unknown kind, or a transcript kind with no path.
    """
    if kind == "anthropic":
        return AnthropicGenerator(model=model or DEFAULT_MODEL)
    if kind == "transcript":
        if transcript is None:
            raise ValueError("--generator transcript requires --transcript PATH")
        return TranscriptGenerator(transcript)
    if kind == "echo":
        return EchoGenerator()
    raise ValueError(f"unknown generator {kind!r}; expected anthropic, transcript or echo")


def generator_model(generator: Any) -> str | None:
    """Report the model behind a generator, for run metadata.

    A transcript reports the model that originally produced its text, not a live
    model, because the run being described is a replay.

    Args:
        generator: Any generator.

    Returns:
        The model id, or None when the backend has no model.
    """
    model = getattr(generator, "model", None)
    if model:
        return str(model)
    source = getattr(generator, "source_model", None)
    return str(source) if source else None


__all__ = [
    "ANTHROPIC_URL",
    "ANTHROPIC_VERSION",
    "API_KEY_ENV",
    "DEFAULT_MODEL",
    "ECHO_TEMPLATE",
    "TRANSCRIPT_SCHEMA_VERSION",
    "AnthropicGenerator",
    "EchoGenerator",
    "Generator",
    "TranscriptGenerator",
    "build_narrow_prompt",
    "build_system_prompt",
    "build_user_prompt",
    "generator_model",
    "make_generator",
    "parse_response",
    "parse_single",
]
