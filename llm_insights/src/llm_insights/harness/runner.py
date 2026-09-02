"""Execute hypotheses against the dataset and record what happened.

:func:`run` is the only entry point that matters, and its contract is that it never
raises. Whatever goes wrong — an unknown primitive, a misspelled parameter, a case that
does not exist, a selection that turns out empty, a bug in a primitive — comes back as an
:class:`~llm_insights.harness.spec.Outcome` with ``error`` set and ``passed`` False.

That makes the distinction the rest of the project depends on: a claim that was tested
and refuted (``passed=False``, ``error=None``, summary starting ``FAIL``) is evidence
about the model, while a claim that could not be tested (``error`` set, summary starting
``COULD NOT RUN``) is evidence about the hypothesis or the harness and must never be
reported as a scientific result.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from typing import Any

from llm_insights.harness.primitives import PRIMITIVES, PrimitiveFn
from llm_insights.harness.spec import Hypothesis, Outcome
from llm_insights.io.dataset import Dataset

LOG = logging.getLogger(__name__)

#: Prefixes that make the three states unmistakable in a log or a table.
PASS_PREFIX = "PASS"
FAIL_PREFIX = "FAIL"
ERROR_PREFIX = "COULD NOT RUN"
#: Prefix for a failure that was the machine's fault rather than the hypothesis's.
ENVIRONMENT_PREFIX = "ENVIRONMENT FAULT"

#: Exceptions that mean "this machine could not run the test", as opposed to "this
#: hypothesis could not be evaluated". Kept narrow on purpose: widening it would start
#: laundering real harness bugs as somebody else's setup problem.
_ENVIRONMENT_ERRORS: tuple[type[BaseException], ...] = (ImportError, FileNotFoundError)


def classify_error(exc: BaseException) -> str | None:
    """Return ``"environment"`` when an exception is the machine's fault, else None.

    Args:
        exc: The exception the primitive raised.

    Returns:
        ``"environment"`` for a missing dependency or a missing file, otherwise None.
    """
    return "environment" if isinstance(exc, _ENVIRONMENT_ERRORS) else None


def _lookup(name: Any) -> PrimitiveFn:
    """Find a registered primitive by name.

    Args:
        name: The primitive name from the test spec.

    Returns:
        The registered callable.

    Raises:
        ValueError: If the name is not registered.
    """
    if not isinstance(name, str) or name not in PRIMITIVES:
        raise ValueError(
            f"there is no test called {name!r}; the available tests are {sorted(PRIMITIVES)}"
        )
    return PRIMITIVES[name]


def _check_params(fn: PrimitiveFn, params: Any) -> dict[str, Any]:
    """Validate parameter names against a primitive's signature before calling it.

    Checking up front turns an opaque ``TypeError`` into a message that names the bad
    parameter and lists the ones the test actually takes.

    Args:
        fn: The primitive about to be called.
        params: The mapping of keyword arguments from the test spec.

    Returns:
        The parameters as a plain dict.

    Raises:
        ValueError: If ``params`` is not a mapping, carries an unknown name, or omits a
            required one.
    """
    if not isinstance(params, dict):
        raise ValueError(
            f"'params' must be a set of named values, got {type(params).__name__} ({params!r})"
        )
    sig = inspect.signature(fn)
    accepted = [p for p in sig.parameters if p != "ds"]
    required = [
        p
        for p, spec in sig.parameters.items()
        if p != "ds" and spec.default is inspect.Parameter.empty
    ]
    unknown = sorted(k for k in params if k not in accepted)
    if unknown:
        raise ValueError(f"test {fn.__name__!r} does not take {unknown}; it takes {accepted}")
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"test {fn.__name__!r} is missing required setting(s) {missing}")
    return dict(params)


def _coerce_observed(observed: Any) -> dict[str, float]:
    """Force a primitive's ``observed`` mapping into plain floats.

    Args:
        observed: Whatever the primitive returned as its second element.

    Returns:
        A mapping of str to float.

    Raises:
        ValueError: If it is not a mapping of names to numbers.
    """
    if not isinstance(observed, dict):
        raise ValueError(f"the test returned {type(observed).__name__} instead of a set of numbers")
    out: dict[str, float] = {}
    for key, value in observed.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"the test reported a non-numeric value for {key!r}: {value!r}"
            ) from None
    return out


def run(ds: Dataset, h: Hypothesis) -> Outcome:
    """Run one hypothesis and return its outcome. Never raises.

    Args:
        ds: The :class:`~llm_insights.io.dataset.Dataset` to test against.
        h: The hypothesis to evaluate.

    Returns:
        An :class:`~llm_insights.harness.spec.Outcome`. ``error`` is set, ``passed`` is
        False and ``observed`` is empty whenever the test could not be evaluated.
    """
    hid = getattr(h, "id", "<unknown>")
    try:
        spec = h.test
        fn = _lookup(spec.primitive)
        params = _check_params(fn, spec.params)
        LOG.debug("running %s -> %s(%s)", hid, spec.primitive, sorted(params))
        result = fn(ds, **params)
        if not (isinstance(result, tuple) and len(result) == 3):
            raise ValueError(
                f"test {spec.primitive!r} returned {result!r} instead of "
                "(passed, observed, summary)"
            )
        passed_raw, observed_raw, summary_raw = result
        passed = bool(passed_raw)
        observed = _coerce_observed(observed_raw)
        summary = str(summary_raw)
    except Exception as exc:  # the whole point of this function is to contain everything
        message = f"{type(exc).__name__}: {exc}" if not isinstance(exc, ValueError) else str(exc)
        kind = classify_error(exc)
        prefix = ENVIRONMENT_PREFIX if kind == "environment" else ERROR_PREFIX
        LOG.warning("hypothesis %s %s: %s", hid, prefix.lower(), message)
        return Outcome(
            hypothesis_id=hid,
            passed=False,
            observed={},
            summary=f"{prefix}: {message}",
            error=message,
            error_kind=kind,
        )

    prefix = PASS_PREFIX if passed else FAIL_PREFIX
    LOG.info("hypothesis %s %s", hid, prefix)
    return Outcome(
        hypothesis_id=hid,
        passed=passed,
        observed=observed,
        summary=f"{prefix}: {summary}",
        error=None,
    )


def run_all(ds: Dataset, hs: Sequence[Hypothesis]) -> list[Outcome]:
    """Run a batch of hypotheses in order.

    Args:
        ds: The dataset to test against.
        hs: The hypotheses to evaluate.

    Returns:
        One outcome per hypothesis, in the same order.
    """
    outcomes = [run(ds, h) for h in hs]
    n_pass = sum(o.passed for o in outcomes)
    n_error = sum(o.error is not None for o in outcomes)
    LOG.info(
        "ran %d hypotheses: %d passed, %d refuted, %d could not run",
        len(outcomes),
        n_pass,
        len(outcomes) - n_pass - n_error,
        n_error,
    )
    return outcomes
