"""Executable falsification harness for hypotheses about the FSG + GRN model.

A hypothesis is admissible here only if it compiles into one of the six primitives in
:mod:`llm_insights.harness.primitives`, with every threshold fixed before the data is
touched. Nothing in this package asks a language model whether a claim is true; the
verdict is arithmetic on Dan's outputs, and every number that fed the verdict is returned
alongside it.

Typical use::

    from llm_insights.harness import hypothesis_from_dict, run
    from llm_insights.io.dataset import Dataset

    ds = Dataset("/path/to/one_way_fsg_model")
    h = hypothesis_from_dict({...})
    outcome = run(ds, h)
"""

from __future__ import annotations

from llm_insights.harness.primitives import (
    PRIMITIVES,
    arc_projection,
    compare,
    metric_value_or_raise,
    primitive,
)
from llm_insights.harness.runner import run, run_all
from llm_insights.harness.spec import (
    Hypothesis,
    Outcome,
    TestSpec,
    hypothesis_from_dict,
    hypothesis_to_dict,
    narrow,
    outcome_to_dict,
    test_spec_from_dict,
)

__all__ = [
    "PRIMITIVES",
    "Hypothesis",
    "Outcome",
    "TestSpec",
    "arc_projection",
    "compare",
    "hypothesis_from_dict",
    "hypothesis_to_dict",
    "metric_value_or_raise",
    "narrow",
    "outcome_to_dict",
    "primitive",
    "run",
    "run_all",
    "test_spec_from_dict",
]
