"""Tests for the falsification harness.

Written with :mod:`unittest` because pytest is not installed in this environment; pytest
collects ``unittest.TestCase`` classes unchanged, so nothing here needs rewriting later.

Almost every test runs against synthetic ``Dataset``-shaped fakes so the suite does not
depend on the 400 MB results tree. The real dataclasses from
:mod:`llm_insights.io.dataset` are used to build those fakes, so a change to their fields
breaks these tests rather than passing silently. :class:`TestRealDataset` at the bottom is
the exception: it asserts a handful of facts that are true of Dan's shipped output, to
catch schema drift, and skips itself if the tree is not mounted.

Run just this module::

    cd /home/claude/work/llm_insights/src
    python3 -m unittest discover -s ../tests -t .. -v -p "test_harness.py"

The top-level directory has to be the repository root (``-t ..``), not ``src``: unittest
refuses a start directory that sits outside the top-level one, and ``tests/`` is a sibling
of ``src/``. This module puts ``src`` on ``sys.path`` itself, so the command works from
any working directory. Drop the ``-p`` filter to run the whole suite.
"""

from __future__ import annotations

import contextlib
import logging
import math
import sys
import types
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_insights.harness import primitives as P  # noqa: E402, N812
from llm_insights.harness import runner as R  # noqa: E402, N812
from llm_insights.harness import spec as S  # noqa: E402, N812
from llm_insights.io.dataset import (  # noqa: E402
    ArcProfile,
    Dataset,
    GRNField,
    GrowthField,
    NodeField,
    arc_order_and_s,
)

def _resolve_root() -> Path:
    """Find the FSG results tree, preferring the checkout these tests live in.

    Hardcoding a sandbox path would make the real-data tests skip silently on a normal
    machine, which is worse than having no such tests at all: the suite would look green
    while verifying nothing.

    Returns:
        The first candidate containing ``FSG Results``, else the in-repo path.
    """
    in_repo = Path(__file__).resolve().parents[1].parent / "one_way_fsg_model"
    staged = Path("/mnt/user-data/uploads/butcher_lab_fsg_exploration/one_way_fsg_model")
    for candidate in (in_repo, staged):
        if (candidate / "FSG Results").is_dir():
            return candidate
    return in_repo


REAL_ROOT = _resolve_root()
METRICS_MODULE = "llm_insights.summary.metrics"
_MISSING = object()


def setUpModule() -> None:
    """Quieten the harness's own logging.

    Several tests deliberately drive the runner into its error path, which logs at
    WARNING. Suppressing that keeps the test output readable; the tests assert on the
    returned ``Outcome`` rather than on log lines.
    """
    logging.getLogger("llm_insights").setLevel(logging.CRITICAL)


# --- synthetic fixtures -----------------------------------------------------------


def make_arc(
    case: str = "flow_U0p0360",
    step: int = 14,
    s_norm: Any = (0.0, 0.25, 0.5, 0.75, 1.0),
    wss_dyn_cm2: Any = (1.0, 2.0, 9.0, 2.0, 1.0),
    pressure_pa: Any = (5.0, 4.0, 3.0, 2.0, 1.0),
    y_mm: Any = (0.0, 0.1, 0.2, 0.1, 0.0),
) -> ArcProfile:
    """Build a small synthetic :class:`ArcProfile`.

    Args:
        case: Case folder name to stamp on the profile.
        step: Growth step to stamp on the profile.
        s_norm: Ascending normalized arc positions.
        wss_dyn_cm2: Wall shear stress per arc point.
        pressure_pa: Fluid pressure per arc point.
        y_mm: Surface height per arc point.

    Returns:
        The synthetic profile.
    """
    s = np.asarray(s_norm, dtype=float)
    wss = np.asarray(wss_dyn_cm2, dtype=float)
    return ArcProfile(
        case=case,
        condition="Synthetic",
        step=step,
        s_norm=s,
        x_mm=s.copy(),
        y_mm=np.asarray(y_mm, dtype=float),
        wss_pa=wss / 10.0,
        wss_dyn_cm2=wss,
        pressure_pa=np.asarray(pressure_pa, dtype=float),
        arc_length_mm=1.0,
        is_pristine=step == 0,
    )


def make_nodes(
    case: str = "flow_U0p0360",
    x_mm: Any = (0.0, 1.0, 2.0, 1.0),
    y_mm: Any = (0.0, 1.0, 0.0, 0.5),
    von_mises_pa: Any = (10.0, 20.0, 30.0, 999.0),
    wss_dyn_cm2: Any = (1.0, 2.0, 3.0, np.nan),
    is_surface: Any = (True, True, True, False),
) -> NodeField:
    """Build a small synthetic :class:`NodeField`.

    The default is the hand-computed arc example: three surface points forming a
    triangle over a base line from x=0 to x=2, plus one interior node.

    Args:
        case: Case folder name to stamp on the field.
        x_mm: Node x coordinates in mm.
        y_mm: Node y coordinates in mm.
        von_mises_pa: Von Mises stress per node.
        wss_dyn_cm2: Wall shear stress per node, NaN at interior nodes.
        is_surface: Surface-band mask.

    Returns:
        The synthetic node field.
    """
    return NodeField(
        case=case,
        condition="Synthetic",
        step=14,
        x_mm=np.asarray(x_mm, dtype=float),
        y_mm=np.asarray(y_mm, dtype=float),
        von_mises_pa=np.asarray(von_mises_pa, dtype=float),
        wss_dyn_cm2=np.asarray(wss_dyn_cm2, dtype=float),
        is_surface=np.asarray(is_surface, dtype=bool),
        von_mises_raw_min_pa=0.0,
    )


def make_grn(
    case: str = "flow_U0p0360",
    von_mises_pa: Any = (0.0, 50.0, 100.0, 150.0),
    wss_dyn_cm2: Any = (1.0, 2.0, np.nan, np.nan),
    mech_norm: Any = (0.0, 0.5, 1.0, 1.0),
    wss_norm: Any = (0.1, 0.2, np.nan, np.nan),
    activities: dict[str, Any] | None = None,
) -> GRNField:
    """Build a small synthetic :class:`GRNField`.

    Args:
        case: Case folder name to stamp on the field.
        von_mises_pa: Mechanical input column.
        wss_dyn_cm2: Shear input column, NaN at interior nodes.
        mech_norm: Normalized mechanical input.
        wss_norm: Normalized shear input, NaN at interior nodes.
        activities: Optional ``"<Node>_<scenario>"`` activity columns.

    Returns:
        The synthetic GRN field.
    """
    n = len(tuple(von_mises_pa))
    return GRNField(
        case=case,
        condition="Synthetic",
        x_mm=np.arange(n, dtype=float),
        y_mm=np.zeros(n, dtype=float),
        von_mises_pa=np.asarray(von_mises_pa, dtype=float),
        wss_dyn_cm2=np.asarray(wss_dyn_cm2, dtype=float),
        mech_norm=np.asarray(mech_norm, dtype=float),
        wss_norm=np.asarray(wss_norm, dtype=float),
        activities={k: np.asarray(v, dtype=float) for k, v in (activities or {}).items()},
    )


def make_growth(
    case: str = "flow_U0p0360", step: int = 14, g: Any = (1.0, 1.1, 1.2)
) -> GrowthField:
    """Build a small synthetic :class:`GrowthField`.

    Args:
        case: Case folder name to stamp on the field.
        step: Growth step.
        g: Per-cell growth multipliers.

    Returns:
        The synthetic growth field.
    """
    arr = np.asarray(g, dtype=float)
    return GrowthField(
        case=case,
        condition="Synthetic",
        step=step,
        g=arr,
        centroid_x_mm=np.arange(arr.size, dtype=float),
        centroid_y_mm=np.zeros(arr.size, dtype=float),
    )


class FakeDataset:
    """A stand-in for :class:`~llm_insights.io.dataset.Dataset` backed by in-memory objects.

    Only the five members the primitives touch are implemented. Anything not supplied
    raises, so a test that reaches for data it did not stage fails loudly.

    Args:
        arcs: Maps ``(case, step)`` to an :class:`ArcProfile`.
        nodes: Maps case to a :class:`NodeField`.
        grns: Maps case to a :class:`GRNField`.
        growths: Maps ``(case, step)`` to a :class:`GrowthField`.
        known_cases: Case names ``resolve_case`` accepts. Defaults to whatever was staged.
    """

    def __init__(
        self,
        arcs: dict[tuple[str, int], ArcProfile] | None = None,
        nodes: dict[str, NodeField] | None = None,
        grns: dict[str, GRNField] | None = None,
        growths: dict[tuple[str, int], GrowthField] | None = None,
        known_cases: tuple[str, ...] | None = None,
    ) -> None:
        """Stage the in-memory fields this fake will serve."""
        self._arcs = arcs or {}
        self._nodes = nodes or {}
        self._grns = grns or {}
        self._growths = growths or {}
        staged = (
            {c for c, _ in self._arcs}
            | set(self._nodes)
            | set(self._grns)
            | {c for c, _ in self._growths}
        )
        self._known = set(known_cases) if known_cases is not None else staged

    def resolve_case(self, name: str) -> str:
        """Resolve a case name, mimicking ``Dataset.resolve_case``'s ``KeyError``."""
        if name not in self._known:
            raise KeyError(f"unknown case {name!r}; expected one of {sorted(self._known)}")
        return name

    def arc(self, case: str, step: int) -> ArcProfile:
        """Return the staged arc profile for a case and step."""
        return self._arcs[(case, step)]

    def nodes(self, case: str) -> NodeField:
        """Return the staged node field for a case."""
        return self._nodes[case]

    def grn(self, case: str) -> GRNField:
        """Return the staged GRN field for a case."""
        return self._grns[case]

    def growth(self, case: str, step: int) -> GrowthField:
        """Return the staged growth field for a case and step."""
        return self._growths[(case, step)]


@contextlib.contextmanager
def fake_metric_registry(
    values: dict[tuple[str, str], float] | Callable[[Any, str, str], float],
    names: tuple[str, ...] | None = None,
) -> Iterator[None]:
    """Install a stand-in ``llm_insights.summary.metrics`` for the duration of a block.

    The harness imports the registry lazily inside a function, so replacing the
    ``sys.modules`` entry is enough; no import of the real module ever happens.

    Args:
        values: Either a ``{(metric, case): value}`` mapping or a callable with the
            registry's ``metric_value(ds, name, case)`` signature.
        names: The ``METRIC_NAMES`` tuple to advertise. Defaults to the mapping's keys.

    Yields:
        None, with the fake registry installed.
    """
    module = types.ModuleType(METRICS_MODULE)
    if callable(values):
        module.metric_value = values
        module.METRIC_NAMES = tuple(names or ())
    else:
        module.METRIC_NAMES = (
            tuple(names) if names is not None else tuple(sorted({metric for metric, _ in values}))
        )

        def metric_value(ds: Any, name: str, case: str) -> float:
            """Look the value up in the staged mapping."""
            return float(values[(name, case)])

        module.metric_value = metric_value

    previous = sys.modules.get(METRICS_MODULE, _MISSING)
    sys.modules[METRICS_MODULE] = module
    try:
        yield
    finally:
        if previous is _MISSING:
            del sys.modules[METRICS_MODULE]
        else:
            sys.modules[METRICS_MODULE] = previous  # type: ignore[assignment]


@contextlib.contextmanager
def broken_metric_registry() -> Iterator[None]:
    """Make importing the metric registry fail, the way a missing module would.

    Yields:
        None, with the registry import poisoned.
    """
    previous = sys.modules.get(METRICS_MODULE, _MISSING)
    sys.modules[METRICS_MODULE] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        if previous is _MISSING:
            del sys.modules[METRICS_MODULE]
        else:
            sys.modules[METRICS_MODULE] = previous  # type: ignore[assignment]


def hyp(primitive: str, params: dict[str, Any], hid: str = "H1") -> S.Hypothesis:
    """Build a hypothesis around one primitive call.

    Args:
        primitive: Primitive name.
        params: Parameters for it.
        hid: Hypothesis id.

    Returns:
        The hypothesis.
    """
    return S.Hypothesis(
        id=hid,
        claim="synthetic claim",
        rationale="synthetic rationale",
        test=S.TestSpec(primitive=primitive, params=params, decision_rule="fixed before running"),
    )


# --- arc projection ---------------------------------------------------------------


class TestArcProjection(unittest.TestCase):
    """The nodal-to-arc projection, checked against a hand-computed example."""

    def test_hand_computed_triangle(self) -> None:
        """Three surface points on a triangle give s = [0, 0.5, 1] in atrial-first order."""
        # Surface points (0,0), (1,1), (2,0); interior point (1,0.5) must be dropped.
        # x_center = 1, so angles are atan2(0,-1)=pi, atan2(1,0)=pi/2, atan2(0,1)=0.
        # Descending angle orders them (0,0) -> (1,1) -> (2,0); each leg is sqrt(2) long,
        # so cumulative s = [0, sqrt2, 2*sqrt2] and s_norm = [0, 0.5, 1].
        nodes = make_nodes()
        s, values = P.arc_projection(nodes, nodes.von_mises_pa)
        np.testing.assert_allclose(s, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(values, [10.0, 20.0, 30.0])
        self.assertEqual(s.size, 3, "the interior node must not appear on the arc")

    def test_reorders_scrambled_input(self) -> None:
        """Input order does not matter; the angle rule decides the arc order."""
        nodes = make_nodes(
            x_mm=(2.0, 0.0, 1.0, 1.0),
            y_mm=(0.0, 0.0, 1.0, 0.5),
            von_mises_pa=(30.0, 10.0, 20.0, 999.0),
            wss_dyn_cm2=(3.0, 1.0, 2.0, np.nan),
            is_surface=(True, True, True, False),
        )
        s, values = P.arc_projection(nodes, nodes.von_mises_pa)
        np.testing.assert_allclose(s, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(values, [10.0, 20.0, 30.0])

    def test_matches_dataset_helper(self) -> None:
        """The helper reuses ``arc_order_and_s`` rather than reimplementing it."""
        nodes = make_nodes()
        mask = nodes.is_surface
        order, s_norm, _ = arc_order_and_s(nodes.x_mm[mask], nodes.y_mm[mask])
        s, values = P.arc_projection(nodes, nodes.von_mises_pa)
        np.testing.assert_allclose(s, s_norm)
        np.testing.assert_allclose(values, nodes.von_mises_pa[mask][order])

    def test_length_mismatch_raises(self) -> None:
        """A values array of the wrong length is rejected, not broadcast."""
        nodes = make_nodes()
        with self.assertRaises(ValueError) as ctx:
            P.arc_projection(nodes, np.array([1.0, 2.0]))
        self.assertIn("line up", str(ctx.exception))

    def test_no_surface_nodes_raises(self) -> None:
        """An empty surface band is an error, not an empty result."""
        nodes = make_nodes(is_surface=(False, False, False, False))
        with self.assertRaises(ValueError) as ctx:
            P.arc_projection(nodes, nodes.von_mises_pa)
        self.assertIn("no surface-band nodes", str(ctx.exception))


# --- operators --------------------------------------------------------------------


class TestCompareOperators(unittest.TestCase):
    """The four ordering operators and the rejection of anything else."""

    def test_gt(self) -> None:
        """``gt`` is strict greater-than."""
        self.assertTrue(P.compare("gt", 2.0, 1.0))
        self.assertFalse(P.compare("gt", 1.0, 1.0))

    def test_ge(self) -> None:
        """``ge`` admits equality."""
        self.assertTrue(P.compare("ge", 1.0, 1.0))
        self.assertFalse(P.compare("ge", 0.9, 1.0))

    def test_lt(self) -> None:
        """``lt`` is strict less-than."""
        self.assertTrue(P.compare("lt", 0.9, 1.0))
        self.assertFalse(P.compare("lt", 1.0, 1.0))

    def test_le(self) -> None:
        """``le`` admits equality."""
        self.assertTrue(P.compare("le", 1.0, 1.0))
        self.assertFalse(P.compare("le", 1.1, 1.0))

    def test_unknown_op_raises(self) -> None:
        """An unknown operator names the four that are allowed."""
        with self.assertRaises(ValueError) as ctx:
            P.compare("approximately", 1.0, 1.0)
        self.assertIn("'approximately'", str(ctx.exception))
        self.assertIn("ge", str(ctx.exception))

    def test_registry_has_all_six(self) -> None:
        """All six contracted primitives are registered."""
        self.assertEqual(
            set(P.PRIMITIVES),
            {
                "compare_metric",
                "metric_ordering",
                "peak_location",
                "fraction_above",
                "correlation",
                "profile_monotonic",
            },
        )


# --- compare_metric ---------------------------------------------------------------


class TestCompareMetric(unittest.TestCase):
    """Two-case metric comparison and its noise guard."""

    def setUp(self) -> None:
        """Stage a dataset with three known cases."""
        self.ds = FakeDataset(known_cases=("A", "B"))
        self.values = {("wss_peak_dyn_cm2", "A"): 43.0, ("wss_peak_dyn_cm2", "B"): 33.0}

    def test_passing_comparison(self) -> None:
        """A true ordering with a real margin passes and reports both operands."""
        with fake_metric_registry(self.values):
            passed, observed, summary = P.compare_metric(
                self.ds, "wss_peak_dyn_cm2", "A", "B", "gt"
            )
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["value_a"], 43.0)
        self.assertAlmostEqual(observed["value_b"], 33.0)
        self.assertAlmostEqual(observed["rel_diff"], 10.0 / 33.0)
        self.assertIn("holds", summary)

    def test_failing_comparison(self) -> None:
        """A false ordering fails and says so."""
        with fake_metric_registry(self.values):
            passed, observed, summary = P.compare_metric(
                self.ds, "wss_peak_dyn_cm2", "A", "B", "lt"
            )
        self.assertFalse(passed)
        self.assertEqual(observed["ordering_ok"], 0.0)
        self.assertIn("ordering does not hold", summary)

    def test_min_rel_diff_blocks_noise(self) -> None:
        """A difference of one part in 10^13 does not pass a 1% margin."""
        noisy = {("wss_mean_dyn_cm2", "A"): 10.000000000001, ("wss_mean_dyn_cm2", "B"): 10.0}
        with fake_metric_registry(noisy):
            passed, observed, summary = P.compare_metric(
                self.ds, "wss_mean_dyn_cm2", "A", "B", "gt", min_rel_diff=0.01
            )
        self.assertFalse(passed)
        self.assertEqual(observed["ordering_ok"], 1.0, "the ordering itself does hold")
        self.assertEqual(observed["margin_ok"], 0.0, "but the margin is noise-level")
        self.assertEqual(observed["min_rel_diff"], 0.01)
        self.assertIn("below the required", summary)

    def test_min_rel_diff_admits_real_difference(self) -> None:
        """The same guard passes a 30% difference."""
        with fake_metric_registry(self.values):
            passed, observed, _ = P.compare_metric(
                self.ds, "wss_peak_dyn_cm2", "A", "B", "gt", min_rel_diff=0.01
            )
        self.assertTrue(passed)
        self.assertGreater(observed["rel_diff"], 0.01)

    def test_zero_denominator_does_not_explode(self) -> None:
        """A reference of exactly zero uses the epsilon floor instead of dividing by zero."""
        zeros = {("growth_mean", "A"): 1.0, ("growth_mean", "B"): 0.0}
        with fake_metric_registry(zeros):
            passed, observed, _ = P.compare_metric(
                self.ds, "growth_mean", "A", "B", "gt", min_rel_diff=0.5
            )
        self.assertTrue(passed)
        self.assertTrue(math.isfinite(observed["rel_diff"]))

    def test_negative_min_rel_diff_rejected(self) -> None:
        """A negative margin is meaningless and is rejected up front."""
        with fake_metric_registry(self.values), self.assertRaises(ValueError) as ctx:
            P.compare_metric(self.ds, "wss_peak_dyn_cm2", "A", "B", "gt", min_rel_diff=-0.1)
        self.assertIn("min_rel_diff", str(ctx.exception))

    def test_unknown_metric_lists_known_ones(self) -> None:
        """An unknown metric name is reported with the list of real ones."""
        with fake_metric_registry(self.values), self.assertRaises(ValueError) as ctx:
            P.compare_metric(self.ds, "wss_peek", "A", "B", "gt")
        self.assertIn("unknown metric 'wss_peek'", str(ctx.exception))
        self.assertIn("wss_peak_dyn_cm2", str(ctx.exception))

    def test_unknown_case_is_a_value_error(self) -> None:
        """An unknown case comes back as a ValueError, not a KeyError."""
        with fake_metric_registry(self.values), self.assertRaises(ValueError) as ctx:
            P.compare_metric(self.ds, "wss_peak_dyn_cm2", "A", "Nope", "gt")
        self.assertIn("unknown case", str(ctx.exception))

    def test_missing_registry_explains_itself(self) -> None:
        """If the metric registry cannot be imported the message says exactly that."""
        with broken_metric_registry(), self.assertRaises(ValueError) as ctx:
            P.compare_metric(self.ds, "wss_peak_dyn_cm2", "A", "B", "gt")
        self.assertIn("metric registry", str(ctx.exception))


# --- metric_ordering --------------------------------------------------------------


class TestMetricOrdering(unittest.TestCase):
    """Ordering of one metric across a sequence of cases."""

    def setUp(self) -> None:
        """Stage three cases with a strictly increasing metric."""
        self.ds = FakeDataset(known_cases=("U", "H", "O"))
        self.rising = {
            ("wss_peak_dyn_cm2", "U"): 22.6,
            ("wss_peak_dyn_cm2", "H"): 33.6,
            ("wss_peak_dyn_cm2", "O"): 43.2,
        }

    def test_increasing_passes(self) -> None:
        """A strictly rising metric passes with zero violations."""
        with fake_metric_registry(self.rising):
            passed, observed, summary = P.metric_ordering(
                self.ds, "wss_peak_dyn_cm2", ["U", "H", "O"], "increasing"
            )
        self.assertTrue(passed)
        self.assertEqual(observed["n_violations"], 0.0)
        self.assertEqual(observed["value_U"], 22.6)
        self.assertIn("increasing", summary)

    def test_decreasing_fails_on_rising_metric(self) -> None:
        """Asking for the opposite direction fails and counts every violation."""
        with fake_metric_registry(self.rising):
            passed, observed, _ = P.metric_ordering(
                self.ds, "wss_peak_dyn_cm2", ["U", "H", "O"], "decreasing"
            )
        self.assertFalse(passed)
        self.assertEqual(observed["n_violations"], 2.0)

    def test_strictness_matters_on_ties(self) -> None:
        """A tie violates a strict ordering but satisfies a non-strict one."""
        tied = {
            ("growth_mean", "U"): 1.0,
            ("growth_mean", "H"): 1.0,
            ("growth_mean", "O"): 2.0,
        }
        with fake_metric_registry(tied):
            strict_pass, _, _ = P.metric_ordering(
                self.ds, "growth_mean", ["U", "H", "O"], "increasing", strict=True
            )
            loose_pass, _, _ = P.metric_ordering(
                self.ds, "growth_mean", ["U", "H", "O"], "increasing", strict=False
            )
        self.assertFalse(strict_pass)
        self.assertTrue(loose_pass)

    def test_needs_two_cases(self) -> None:
        """A one-case ordering is not a claim about anything."""
        with fake_metric_registry(self.rising), self.assertRaises(ValueError) as ctx:
            P.metric_ordering(self.ds, "wss_peak_dyn_cm2", ["U"], "increasing")
        self.assertIn("at least 2", str(ctx.exception))

    def test_duplicate_cases_rejected(self) -> None:
        """Listing a case twice would make the ordering trivially decidable."""
        with fake_metric_registry(self.rising), self.assertRaises(ValueError) as ctx:
            P.metric_ordering(self.ds, "wss_peak_dyn_cm2", ["U", "U"], "increasing")
        self.assertIn("same case twice", str(ctx.exception))

    def test_bad_direction_rejected(self) -> None:
        """Only the two directions are accepted."""
        with fake_metric_registry(self.rising), self.assertRaises(ValueError) as ctx:
            P.metric_ordering(self.ds, "wss_peak_dyn_cm2", ["U", "H"], "sideways")
        self.assertIn("direction", str(ctx.exception))


# --- peak_location ----------------------------------------------------------------


class TestPeakLocation(unittest.TestCase):
    """Where along the arc a field's maximum sits."""

    def setUp(self) -> None:
        """Stage one arc peaking at s=0.5 and one node field peaking at s=1."""
        self.ds = FakeDataset(
            arcs={("A", 14): make_arc(case="A")},
            nodes={"A": make_nodes(case="A")},
        )

    def test_wss_peak_inside_window_passes(self) -> None:
        """The synthetic arc peaks at s=0.5, inside [0.4, 0.6]."""
        passed, observed, summary = P.peak_location(self.ds, "wss", "A", 0.4, 0.6)
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["s_at_peak"], 0.5)
        self.assertAlmostEqual(observed["peak_value"], 9.0)
        self.assertIn("inside", summary)

    def test_wss_peak_outside_window_fails(self) -> None:
        """The same peak is outside [0.0, 0.2]."""
        passed, observed, summary = P.peak_location(self.ds, "wss", "A", 0.0, 0.2)
        self.assertFalse(passed)
        self.assertAlmostEqual(observed["s_at_peak"], 0.5)
        self.assertIn("outside", summary)

    def test_von_mises_uses_arc_projection(self) -> None:
        """Nodal von Mises is placed on the arc; the synthetic peak is at s=1."""
        passed, observed, _ = P.peak_location(self.ds, "von_mises", "A", 0.9, 1.0)
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["s_at_peak"], 1.0)
        self.assertAlmostEqual(observed["peak_value"], 30.0)
        self.assertEqual(observed["n_used"], 3.0, "interior node must be excluded")

    def test_nans_are_skipped_not_maximised(self) -> None:
        """A NaN never wins the argmax."""
        ds = FakeDataset(
            arcs={("A", 14): make_arc(case="A", wss_dyn_cm2=(1.0, np.nan, 5.0, np.nan, 2.0))}
        )
        passed, observed, _ = P.peak_location(ds, "wss", "A", 0.4, 0.6)
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["peak_value"], 5.0)
        self.assertEqual(observed["n_dropped"], 2.0)

    def test_all_nan_profile_raises(self) -> None:
        """A profile with nothing finite has no peak, and says so."""
        ds = FakeDataset(arcs={("A", 14): make_arc(case="A", wss_dyn_cm2=(np.nan,) * 5)})
        with self.assertRaises(ValueError) as ctx:
            P.peak_location(ds, "wss", "A", 0.0, 1.0)
        self.assertIn("missing (NaN)", str(ctx.exception))

    def test_von_mises_off_step_14_raises(self) -> None:
        """Nodal von Mises exists only at step 14; asking elsewhere is refused."""
        with self.assertRaises(ValueError) as ctx:
            P.peak_location(self.ds, "von_mises", "A", 0.0, 1.0, step=7)
        self.assertIn("only stored at step 14", str(ctx.exception))

    def test_inverted_window_raises(self) -> None:
        """A window with s_min above s_max can never contain anything."""
        with self.assertRaises(ValueError) as ctx:
            P.peak_location(self.ds, "wss", "A", 0.8, 0.2)
        self.assertIn("must not exceed", str(ctx.exception))

    def test_unknown_field_raises(self) -> None:
        """Only the two arc-resolvable fields are accepted."""
        with self.assertRaises(ValueError) as ctx:
            P.peak_location(self.ds, "pressure", "A", 0.0, 1.0)
        self.assertIn("field", str(ctx.exception))


# --- fraction_above ---------------------------------------------------------------


class TestFractionAbove(unittest.TestCase):
    """Threshold-exceedance fractions, and what happens to missing samples."""

    def setUp(self) -> None:
        """Stage node, GRN and growth fields for one case."""
        self.ds = FakeDataset(
            nodes={"A": make_nodes(case="A", von_mises_pa=(10.0, 20.0, 200.0, 300.0))},
            grns={"A": make_grn(case="A")},
            growths={("A", 14): make_growth(case="A", g=(0.9, 1.0, 1.1, 1.4))},
        )

    def test_passing_fraction(self) -> None:
        """Half the synthetic nodes exceed 100 Pa, which is more than 0.4."""
        passed, observed, summary = P.fraction_above(self.ds, "von_mises", "A", 100.0, "gt", 0.4)
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["fraction"], 0.5)
        self.assertEqual(observed["n_above"], 2.0)
        self.assertEqual(observed["n_used"], 4.0)
        self.assertIn("2/4", summary)

    def test_failing_fraction(self) -> None:
        """The same fraction is not above 0.9."""
        passed, observed, _ = P.fraction_above(self.ds, "von_mises", "A", 100.0, "gt", 0.9)
        self.assertFalse(passed)
        self.assertAlmostEqual(observed["fraction"], 0.5)

    def test_nans_are_dropped_not_treated_as_zero(self) -> None:
        """WSS is NaN at interior nodes; those samples leave the denominator entirely.

        With values [nan, nan, nan, 5, 20] and a threshold of 10 the honest answer is
        1/2 = 0.5. Coercing the NaNs to 0 would give 1/5 = 0.2, so this test pins the
        difference rather than trusting it.
        """
        ds = FakeDataset(
            nodes={
                "A": make_nodes(
                    case="A",
                    x_mm=(0.0, 1.0, 2.0, 3.0, 4.0),
                    y_mm=(0.0, 1.0, 0.0, 1.0, 0.0),
                    von_mises_pa=(1.0, 2.0, 3.0, 4.0, 5.0),
                    wss_dyn_cm2=(np.nan, np.nan, np.nan, 5.0, 20.0),
                    is_surface=(True, True, True, True, True),
                )
            }
        )
        passed, observed, summary = P.fraction_above(ds, "wss", "A", 10.0, "gt", 0.4)
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["fraction"], 0.5)
        self.assertNotAlmostEqual(observed["fraction"], 0.2, msg="NaNs must not count as zeros")
        self.assertEqual(observed["n_used"], 2.0)
        self.assertEqual(observed["n_dropped"], 3.0)
        self.assertIn("dropped", summary)

    def test_all_nan_field_raises(self) -> None:
        """A field with no finite samples yields an error, not a fraction of zero."""
        ds = FakeDataset(
            nodes={"A": make_nodes(case="A", wss_dyn_cm2=(np.nan, np.nan, np.nan, np.nan))}
        )
        with self.assertRaises(ValueError) as ctx:
            P.fraction_above(ds, "wss", "A", 1.0, "gt", 0.0)
        self.assertIn("missing (NaN)", str(ctx.exception))

    def test_mech_norm_uses_grn_table(self) -> None:
        """``mech_norm`` is sourced from the GRN table, where half the rows saturate."""
        passed, observed, _ = P.fraction_above(self.ds, "mech_norm", "A", 0.99, "ge", 0.5)
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["fraction"], 0.5)

    def test_growth_uses_the_requested_step(self) -> None:
        """``growth`` reads the per-cell multiplier at the given step."""
        passed, observed, _ = P.fraction_above(self.ds, "growth", "A", 1.05, "gt", 0.4, step=14)
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["fraction"], 0.5)

    def test_threshold_is_strict(self) -> None:
        """A sample exactly at the threshold does not count as above it."""
        _, observed, _ = P.fraction_above(self.ds, "von_mises", "A", 200.0, "gt", 0.0)
        self.assertAlmostEqual(observed["fraction"], 0.25)

    def test_unknown_field_lists_the_valid_ones(self) -> None:
        """A misspelled field names the four that exist."""
        with self.assertRaises(ValueError) as ctx:
            P.fraction_above(self.ds, "vonmises", "A", 1.0, "gt", 0.5)
        self.assertIn("mech_norm", str(ctx.exception))


# --- correlation ------------------------------------------------------------------


class TestCorrelation(unittest.TestCase):
    """Correlations between GRN-table columns, and pairwise NaN removal."""

    def setUp(self) -> None:
        """Stage a GRN table with a clean linear pair and a NaN-laden shear column."""
        self.ds = FakeDataset(
            grns={
                "A": make_grn(
                    case="A",
                    von_mises_pa=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                    wss_dyn_cm2=(0.0, 1.0, 2.0, 3.0, np.nan, np.nan),
                    mech_norm=(0.0, 1.0, 2.0, 3.0, 10.0, 10.0),
                    wss_norm=(0.0, 1.0, 4.0, 9.0, 16.0, 25.0),
                    activities={"EndMT_combined": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)},
                )
            }
        )

    def test_passing_correlation(self) -> None:
        """A perfectly linear pair correlates at r=1."""
        passed, observed, summary = P.correlation(
            self.ds, "von_mises_pa", "EndMT_combined", "A", "gt", 0.9
        )
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["r"], 1.0, places=10)
        self.assertEqual(observed["n_used"], 6.0)
        self.assertIn("pearson", summary)

    def test_failing_correlation(self) -> None:
        """The same r=1 is not less than 0."""
        passed, observed, _ = P.correlation(
            self.ds, "von_mises_pa", "EndMT_combined", "A", "lt", 0.0
        )
        self.assertFalse(passed)
        self.assertAlmostEqual(observed["r"], 1.0, places=10)

    def test_nan_pairs_are_dropped_not_zero_filled(self) -> None:
        """Rows where either column is NaN leave the calculation entirely.

        ``wss_dyn_cm2`` is [0,1,2,3,nan,nan] and ``mech_norm`` is [0,1,2,3,10,10]. On the
        four usable rows the relationship is exactly linear, so r must be 1.0 and
        ``n_used`` must be 4. Zero-filling the NaNs would pair 0 with 10 twice and drag
        r well below 1, so this is a direct test of the NaN policy.
        """
        passed, observed, summary = P.correlation(
            self.ds, "wss_dyn_cm2", "mech_norm", "A", "gt", 0.99
        )
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["r"], 1.0, places=10)
        self.assertEqual(observed["n_used"], 4.0)
        self.assertEqual(observed["n_dropped"], 2.0)
        self.assertIn("2 dropped", summary)

    def test_zero_filling_would_have_given_a_different_answer(self) -> None:
        """Confirms the previous test is actually discriminating, not vacuous."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 10.0])
        r_if_zero_filled = float(np.corrcoef(x, y)[0, 1])
        self.assertLess(r_if_zero_filled, 0.5)

    def test_spearman_sees_monotone_nonlinearity(self) -> None:
        """A monotone but curved relationship ranks at 1 while Pearson falls short."""
        pearson_r = P.correlation(self.ds, "von_mises_pa", "wss_norm", "A", "gt", -2.0)[1]["r"]
        spearman_r = P.correlation(
            self.ds, "von_mises_pa", "wss_norm", "A", "gt", -2.0, method="spearman"
        )[1]["r"]
        self.assertAlmostEqual(spearman_r, 1.0, places=10)
        self.assertLess(pearson_r, 1.0)

    def test_unknown_field_lists_available_columns(self) -> None:
        """A misspelled GRN column names the ones that exist, activities included."""
        with self.assertRaises(ValueError) as ctx:
            P.correlation(self.ds, "von_mises", "mech_norm", "A", "gt", 0.0)
        message = str(ctx.exception)
        self.assertIn("unknown GRN field 'von_mises'", message)
        self.assertIn("EndMT_combined", message)

    def test_too_few_usable_pairs_raises(self) -> None:
        """Two surviving rows cannot support a correlation."""
        ds = FakeDataset(
            grns={
                "A": make_grn(
                    case="A",
                    von_mises_pa=(0.0, 1.0, 2.0, 3.0),
                    wss_dyn_cm2=(0.0, 1.0, np.nan, np.nan),
                    mech_norm=(0.0, 1.0, 2.0, 3.0),
                    wss_norm=(0.0, 1.0, 2.0, 3.0),
                )
            }
        )
        with self.assertRaises(ValueError) as ctx:
            P.correlation(ds, "wss_dyn_cm2", "mech_norm", "A", "gt", 0.0)
        self.assertIn("at least 3", str(ctx.exception))

    def test_constant_column_raises(self) -> None:
        """A column with no variance makes the coefficient undefined."""
        ds = FakeDataset(
            grns={
                "A": make_grn(
                    case="A",
                    von_mises_pa=(1.0, 1.0, 1.0, 1.0),
                    wss_dyn_cm2=(1.0, 2.0, 3.0, 4.0),
                    mech_norm=(0.0, 1.0, 2.0, 3.0),
                    wss_norm=(0.0, 1.0, 2.0, 3.0),
                )
            }
        )
        with self.assertRaises(ValueError) as ctx:
            P.correlation(ds, "von_mises_pa", "mech_norm", "A", "gt", 0.0)
        self.assertIn("constant", str(ctx.exception))

    def test_unknown_method_raises(self) -> None:
        """Only pearson and spearman are supported."""
        with self.assertRaises(ValueError) as ctx:
            P.correlation(self.ds, "von_mises_pa", "mech_norm", "A", "gt", 0.0, method="kendall")
        self.assertIn("method", str(ctx.exception))


# --- profile_monotonic ------------------------------------------------------------


class TestProfileMonotonic(unittest.TestCase):
    """Shape claims over a window of arc position."""

    def setUp(self) -> None:
        """Stage an arc that rises to s=0.5 and falls after it."""
        self.ds = FakeDataset(
            arcs={("A", 14): make_arc(case="A")},
            nodes={"A": make_nodes(case="A")},
        )

    def test_rising_half_passes(self) -> None:
        """WSS rises monotonically over the first half of the arc."""
        passed, observed, summary = P.profile_monotonic(self.ds, "wss", "A", 0.0, 0.5, "increasing")
        self.assertTrue(passed)
        self.assertAlmostEqual(observed["fraction_monotonic"], 1.0)
        self.assertEqual(observed["n_diffs"], 2.0)
        self.assertIn("meeting", summary)

    def test_wrong_direction_fails(self) -> None:
        """Over the same window it is not decreasing at all."""
        passed, observed, summary = P.profile_monotonic(self.ds, "wss", "A", 0.0, 0.5, "decreasing")
        self.assertFalse(passed)
        self.assertAlmostEqual(observed["fraction_monotonic"], 0.0)
        self.assertIn("short of", summary)

    def test_min_fraction_is_the_deciding_threshold(self) -> None:
        """Three of four differences rise: enough at 0.75, not enough at 0.8."""
        ds = FakeDataset(
            arcs={
                ("A", 14): make_arc(
                    case="A",
                    s_norm=(0.0, 0.25, 0.5, 0.75, 1.0),
                    wss_dyn_cm2=(1.0, 2.0, 3.0, 2.5, 4.0),
                )
            }
        )
        loose, observed, _ = P.profile_monotonic(
            ds, "wss", "A", 0.0, 1.0, "increasing", min_fraction=0.75
        )
        tight, _, _ = P.profile_monotonic(ds, "wss", "A", 0.0, 1.0, "increasing", min_fraction=0.8)
        self.assertAlmostEqual(observed["fraction_monotonic"], 0.75)
        self.assertTrue(loose)
        self.assertFalse(tight)

    def test_ties_count_against_the_claim(self) -> None:
        """A flat profile is neither increasing nor decreasing."""
        ds = FakeDataset(
            arcs={("A", 14): make_arc(case="A", wss_dyn_cm2=(2.0, 2.0, 2.0, 2.0, 2.0))}
        )
        up, observed, _ = P.profile_monotonic(ds, "wss", "A", 0.0, 1.0, "increasing")
        down, _, _ = P.profile_monotonic(ds, "wss", "A", 0.0, 1.0, "decreasing")
        self.assertFalse(up)
        self.assertFalse(down)
        self.assertAlmostEqual(observed["fraction_monotonic"], 0.0)

    def test_height_and_pressure_are_available(self) -> None:
        """Pressure falls along the synthetic arc and height rises then falls."""
        p_down, _, _ = P.profile_monotonic(self.ds, "pressure", "A", 0.0, 1.0, "decreasing")
        h_up, _, _ = P.profile_monotonic(self.ds, "height", "A", 0.0, 0.5, "increasing")
        self.assertTrue(p_down)
        self.assertTrue(h_up)

    def test_von_mises_window_uses_projection(self) -> None:
        """Nodal von Mises rises along the synthetic surface band."""
        passed, observed, _ = P.profile_monotonic(self.ds, "von_mises", "A", 0.0, 1.0, "increasing")
        self.assertTrue(passed)
        self.assertEqual(observed["n_used"], 3.0)

    def test_nans_inside_the_window_are_dropped(self) -> None:
        """Missing samples are removed before differencing and are reported."""
        ds = FakeDataset(
            arcs={("A", 14): make_arc(case="A", wss_dyn_cm2=(1.0, np.nan, 3.0, np.nan, 5.0))}
        )
        passed, observed, _ = P.profile_monotonic(ds, "wss", "A", 0.0, 1.0, "increasing")
        self.assertTrue(passed)
        self.assertEqual(observed["n_used"], 3.0)
        self.assertEqual(observed["n_dropped"], 2.0)

    def test_empty_window_raises(self) -> None:
        """A window holding fewer than two samples cannot show a trend."""
        with self.assertRaises(ValueError) as ctx:
            P.profile_monotonic(self.ds, "wss", "A", 0.51, 0.52, "increasing")
        self.assertIn("at least 2 are needed", str(ctx.exception))

    def test_min_fraction_out_of_range_raises(self) -> None:
        """A required fraction above 1 could never be met."""
        with self.assertRaises(ValueError) as ctx:
            P.profile_monotonic(self.ds, "wss", "A", 0.0, 1.0, "increasing", min_fraction=1.5)
        self.assertIn("between 0 and 1", str(ctx.exception))


# --- spec parsing -----------------------------------------------------------------


def valid_payload() -> dict[str, Any]:
    """Return a well-formed serialized hypothesis.

    Returns:
        A mapping that :func:`hypothesis_from_dict` accepts.
    """
    return {
        "id": "H-001",
        "claim": "Peak wall shear stress is higher under overflow than under healthy flow.",
        "rationale": "Higher inlet velocity should raise near-wall velocity gradients.",
        "test": {
            "primitive": "compare_metric",
            "params": {
                "metric": "wss_peak_dyn_cm2",
                "case_a": "Overflow",
                "case_b": "Healthy",
                "op": "gt",
                "min_rel_diff": 0.05,
            },
            "decision_rule": "Passes iff overflow exceeds healthy by at least 5%.",
        },
    }


class TestSpecParsing(unittest.TestCase):
    """Strict validation of serialized hypotheses."""

    def test_round_trip(self) -> None:
        """Parsing then serializing gives back an equivalent payload."""
        h = S.hypothesis_from_dict(valid_payload())
        self.assertEqual(h.id, "H-001")
        self.assertEqual(h.test.primitive, "compare_metric")
        self.assertIsNone(h.parent_id)
        again = S.hypothesis_from_dict(S.hypothesis_to_dict(h))
        self.assertEqual(again, h)

    def test_parent_id_is_preserved(self) -> None:
        """A narrowed revision keeps its link to the claim it replaces."""
        payload = valid_payload()
        payload["parent_id"] = "H-000"
        self.assertEqual(S.hypothesis_from_dict(payload).parent_id, "H-000")

    def test_narrow_links_to_parent(self) -> None:
        """``narrow`` builds a revision whose parent is the original."""
        h = S.hypothesis_from_dict(valid_payload())
        revised = S.narrow(h, "H-002", "narrower claim", "because", h.test)
        self.assertEqual(revised.parent_id, "H-001")
        self.assertEqual(revised.id, "H-002")

    def test_missing_top_level_field(self) -> None:
        """A missing field is named in the error."""
        payload = valid_payload()
        del payload["rationale"]
        with self.assertRaises(ValueError) as ctx:
            S.hypothesis_from_dict(payload)
        self.assertIn("missing required field 'rationale'", str(ctx.exception))

    def test_missing_test_block(self) -> None:
        """A hypothesis without a test is not falsifiable and is rejected."""
        payload = valid_payload()
        del payload["test"]
        with self.assertRaises(ValueError) as ctx:
            S.hypothesis_from_dict(payload)
        self.assertIn("must be an object/dict", str(ctx.exception))

    def test_missing_decision_rule(self) -> None:
        """The decision rule is mandatory; it is the pre-registration."""
        payload = valid_payload()
        del payload["test"]["decision_rule"]
        with self.assertRaises(ValueError) as ctx:
            S.hypothesis_from_dict(payload)
        self.assertIn("decision_rule", str(ctx.exception))

    def test_unknown_primitive(self) -> None:
        """An unregistered primitive is rejected with the list of real ones."""
        payload = valid_payload()
        payload["test"]["primitive"] = "vibe_check"
        with self.assertRaises(ValueError) as ctx:
            S.hypothesis_from_dict(payload)
        message = str(ctx.exception)
        self.assertIn("'vibe_check'", message)
        self.assertIn("compare_metric", message)

    def test_wrong_type_for_claim(self) -> None:
        """A non-string claim names the field and both types."""
        payload = valid_payload()
        payload["claim"] = 42
        with self.assertRaises(ValueError) as ctx:
            S.hypothesis_from_dict(payload)
        message = str(ctx.exception)
        self.assertIn("'claim'", message)
        self.assertIn("must be str", message)
        self.assertIn("int", message)

    def test_wrong_type_for_params(self) -> None:
        """Params must be a mapping, not a list of positional values."""
        payload = valid_payload()
        payload["test"]["params"] = ["Overflow", "Healthy"]
        with self.assertRaises(ValueError) as ctx:
            S.hypothesis_from_dict(payload)
        self.assertIn("'params'", str(ctx.exception))

    def test_wrong_type_for_parent_id(self) -> None:
        """``parent_id`` is a string or null, nothing else."""
        payload = valid_payload()
        payload["parent_id"] = 7
        with self.assertRaises(ValueError) as ctx:
            S.hypothesis_from_dict(payload)
        self.assertIn("parent_id", str(ctx.exception))

    def test_empty_string_rejected(self) -> None:
        """A blank claim is as unfalsifiable as a missing one."""
        payload = valid_payload()
        payload["claim"] = "   "
        with self.assertRaises(ValueError) as ctx:
            S.hypothesis_from_dict(payload)
        self.assertIn("must not be empty", str(ctx.exception))

    def test_unexpected_field_rejected(self) -> None:
        """A stray field usually means a typo, so it is not silently ignored."""
        payload = valid_payload()
        payload["confidence"] = 0.9
        with self.assertRaises(ValueError) as ctx:
            S.hypothesis_from_dict(payload)
        self.assertIn("'confidence'", str(ctx.exception))

    def test_non_dict_input_rejected(self) -> None:
        """A JSON list is not a hypothesis."""
        with self.assertRaises(ValueError) as ctx:
            S.hypothesis_from_dict([1, 2, 3])
        self.assertIn("must be an object/dict", str(ctx.exception))

    def test_outcome_to_dict_is_json_ready(self) -> None:
        """Outcome serialization gives plain scalars only."""
        outcome = S.Outcome(
            hypothesis_id="H-001",
            passed=True,
            observed={"r": np.float64(0.5), "n_used": 12},
            summary="PASS: something",
        )
        d = S.outcome_to_dict(outcome)
        self.assertIs(type(d["passed"]), bool)
        self.assertIs(type(d["observed"]["r"]), float)
        self.assertIs(type(d["observed"]["n_used"]), float)
        self.assertIsNone(d["error"])


# --- runner -----------------------------------------------------------------------


class TestRunner(unittest.TestCase):
    """Containment of every failure mode, and the pass/fail/error distinction."""

    def setUp(self) -> None:
        """Stage a dataset whose arc peaks at s=0.5."""
        self.ds = FakeDataset(arcs={("A", 14): make_arc(case="A")})

    def test_pass_outcome(self) -> None:
        """A satisfied claim is marked PASS and carries its numbers."""
        outcome = R.run(
            self.ds, hyp("peak_location", {"field": "wss", "case": "A", "s_min": 0.4, "s_max": 0.6})
        )
        self.assertTrue(outcome.passed)
        self.assertIsNone(outcome.error)
        self.assertTrue(outcome.ran)
        self.assertTrue(outcome.summary.startswith("PASS:"))
        self.assertAlmostEqual(outcome.observed["s_at_peak"], 0.5)

    def test_fail_outcome_is_not_an_error(self) -> None:
        """A refuted claim ran successfully: FAIL, no error, numbers present."""
        outcome = R.run(
            self.ds, hyp("peak_location", {"field": "wss", "case": "A", "s_min": 0.0, "s_max": 0.1})
        )
        self.assertFalse(outcome.passed)
        self.assertIsNone(outcome.error)
        self.assertTrue(outcome.ran)
        self.assertTrue(outcome.summary.startswith("FAIL:"))
        self.assertIn("s_at_peak", outcome.observed)

    def test_unknown_primitive_becomes_an_error_outcome(self) -> None:
        """An unregistered primitive is contained, not raised."""
        outcome = R.run(self.ds, hyp("vibe_check", {}))
        self.assertFalse(outcome.passed)
        self.assertIsNotNone(outcome.error)
        self.assertFalse(outcome.ran)
        self.assertTrue(outcome.summary.startswith("COULD NOT RUN:"))
        self.assertIn("no test called 'vibe_check'", outcome.error or "")
        self.assertEqual(outcome.observed, {})

    def test_misspelled_parameter_names_the_real_ones(self) -> None:
        """An unknown parameter is reported with the accepted parameter list."""
        outcome = R.run(
            self.ds,
            hyp("peak_location", {"field": "wss", "case": "A", "smin": 0.0, "s_max": 1.0}),
        )
        self.assertIsNotNone(outcome.error)
        self.assertIn("'smin'", outcome.error or "")
        self.assertIn("s_min", outcome.error or "")

    def test_missing_required_parameter(self) -> None:
        """A missing required setting is named."""
        outcome = R.run(self.ds, hyp("peak_location", {"field": "wss", "case": "A"}))
        self.assertIsNotNone(outcome.error)
        self.assertIn("missing required setting", outcome.error or "")
        self.assertIn("s_min", outcome.error or "")

    def test_unknown_case_becomes_an_error_outcome(self) -> None:
        """An unknown case is a setup problem, reported as such."""
        outcome = R.run(
            self.ds,
            hyp("peak_location", {"field": "wss", "case": "Z", "s_min": 0.0, "s_max": 1.0}),
        )
        self.assertIsNotNone(outcome.error)
        self.assertIn("unknown case", outcome.error or "")

    def test_arbitrary_exception_is_contained(self) -> None:
        """Even a bug inside a primitive comes back as an outcome, with its type named."""
        name = "_exploding_test_primitive"
        original = dict(P.PRIMITIVES)

        @P.primitive(name)
        def _boom(ds: Any) -> tuple[bool, dict[str, float], str]:
            """Always raise, to prove the runner contains it."""
            raise ZeroDivisionError("division by zero in a primitive")

        try:
            outcome = R.run(self.ds, hyp(name, {}))
        finally:
            P.PRIMITIVES.clear()
            P.PRIMITIVES.update(original)

        self.assertFalse(outcome.passed)
        self.assertIsNotNone(outcome.error)
        self.assertIn("ZeroDivisionError", outcome.error or "")
        self.assertTrue(outcome.summary.startswith("COULD NOT RUN:"))

    def test_malformed_hypothesis_object_is_contained(self) -> None:
        """A hypothesis whose test block is nonsense still yields an outcome."""
        bad = S.Hypothesis(
            id="H-bad",
            claim="c",
            rationale="r",
            test=S.TestSpec(primitive="peak_location", params="not a mapping", decision_rule="d"),
        )
        outcome = R.run(self.ds, bad)
        self.assertFalse(outcome.passed)
        self.assertIn("named values", outcome.error or "")
        self.assertEqual(outcome.hypothesis_id, "H-bad")

    def test_missing_data_is_contained(self) -> None:
        """A case that resolves but has no staged data does not crash the run."""
        ds = FakeDataset(arcs={("A", 14): make_arc(case="A")}, known_cases=("A", "B"))
        outcome = R.run(
            ds, hyp("peak_location", {"field": "wss", "case": "B", "s_min": 0.0, "s_max": 1.0})
        )
        self.assertFalse(outcome.passed)
        self.assertIsNotNone(outcome.error)
        self.assertIn("KeyError", outcome.error or "")

    def test_run_all_preserves_order_and_length(self) -> None:
        """One outcome per hypothesis, in order, mixing all three states."""
        hs = [
            hyp("peak_location", {"field": "wss", "case": "A", "s_min": 0.4, "s_max": 0.6}, "ok"),
            hyp("peak_location", {"field": "wss", "case": "A", "s_min": 0.0, "s_max": 0.1}, "bad"),
            hyp("nope", {}, "broken"),
        ]
        outcomes = R.run_all(self.ds, hs)
        self.assertEqual([o.hypothesis_id for o in outcomes], ["ok", "bad", "broken"])
        self.assertEqual([o.passed for o in outcomes], [True, False, False])
        self.assertEqual([o.ran for o in outcomes], [True, True, False])

    def test_run_all_on_empty_input(self) -> None:
        """An empty batch is an empty list, not an error."""
        self.assertEqual(R.run_all(self.ds, []), [])


# --- real dataset -----------------------------------------------------------------


@unittest.skipUnless(REAL_ROOT.is_dir(), f"real dataset not mounted at {REAL_ROOT}")
class TestRealDataset(unittest.TestCase):
    """Facts that are true of Dan's shipped output, to catch schema drift.

    These are deliberately coarse. They exist so that a change in file layout, column
    naming or unit convention breaks the suite loudly, not so that they pin exact values.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Open the real results tree once for the whole class."""
        cls.ds = Dataset(REAL_ROOT)

    def test_overflow_peak_von_mises_exceeds_healthy(self) -> None:
        """Overflow drives higher peak von Mises stress than healthy flow."""

        def metric_value(ds: Dataset, name: str, case: str) -> float:
            """Minimal real-data metric, standing in for the parallel registry."""
            return float(ds.nodes(case).von_mises_pa.max())

        with fake_metric_registry(metric_value, names=("von_mises_peak_pa",)):
            passed, observed, summary = P.compare_metric(
                self.ds,
                "von_mises_peak_pa",
                "flow_U0p0540",
                "flow_U0p0360",
                "gt",
                min_rel_diff=0.10,
            )
        self.assertTrue(passed, summary)
        self.assertGreater(observed["value_a"], observed["value_b"])
        self.assertGreater(observed["rel_diff"], 0.10)

    def test_mechanical_saturation_rises_with_flow(self) -> None:
        """More of the mesh exceeds the 100 Pa normalization ceiling as flow increases."""
        fractions = [
            P.fraction_above(self.ds, "von_mises", case, 100.0, "gt", 0.0)[1]["fraction"]
            for case in ("flow_U0p0180", "flow_U0p0360", "flow_U0p0540")
        ]
        self.assertLess(fractions[0], fractions[1])
        self.assertLess(fractions[1], fractions[2])
        self.assertGreater(fractions[2], 0.4, "overflow saturates a large minority of nodes")

    def test_healthy_wss_peak_sits_downstream_of_midpoint(self) -> None:
        """The healthy peak WSS is past the crest, in the ventricular half of the arc."""
        passed, observed, summary = P.peak_location(self.ds, "wss", "Healthy", 0.5, 0.8, step=14)
        self.assertTrue(passed, summary)
        self.assertGreater(observed["peak_value"], 20.0)
        outside, _, _ = P.peak_location(self.ds, "wss", "Healthy", 0.0, 0.3, step=14)
        self.assertFalse(outside, "the peak is not in the atrial third")

    def test_real_wss_column_is_mostly_nan_and_those_rows_are_dropped(self) -> None:
        """Interior nodes carry no WSS; the correlation must run on the surface rows only."""
        grn = self.ds.grn("Healthy")
        n_rows = int(grn.wss_dyn_cm2.size)
        n_finite = int(np.count_nonzero(np.isfinite(grn.wss_dyn_cm2)))
        self.assertLess(n_finite, n_rows // 2, "most GRN rows are interior nodes with NaN WSS")

        _, observed, _ = P.correlation(self.ds, "wss_dyn_cm2", "wss_norm", "Healthy", "gt", 0.9)
        self.assertEqual(observed["n_used"], float(n_finite))
        self.assertEqual(observed["n_dropped"], float(n_rows - n_finite))
        self.assertGreater(observed["r"], 0.9, "wss_norm is a clipped rescaling of wss")

    def test_cushion_rises_from_the_atrial_end(self) -> None:
        """Surface height increases monotonically over the first 45% of the arc."""
        for case in ("Healthy", "Overflow"):
            with self.subTest(case=case):
                passed, observed, summary = P.profile_monotonic(
                    self.ds, "height", case, 0.0, 0.45, "increasing", min_fraction=0.95
                )
                self.assertTrue(passed, summary)
                self.assertGreater(observed["n_diffs"], 10)

    def test_arc_projection_on_the_real_mesh(self) -> None:
        """Projection keeps exactly the surface band and returns an ascending arc."""
        nodes = self.ds.nodes("Overflow")
        s, values = P.arc_projection(nodes, nodes.von_mises_pa)
        self.assertEqual(s.size, nodes.n_surface)
        self.assertEqual(values.size, nodes.n_surface)
        self.assertLess(nodes.n_surface, nodes.n_nodes)
        self.assertAlmostEqual(float(s[0]), 0.0)
        self.assertAlmostEqual(float(s[-1]), 1.0)
        self.assertTrue(bool(np.all(np.diff(s) >= 0.0)), "arc position must be non-decreasing")
        self.assertTrue(bool(np.all(np.isfinite(values))))

    def test_end_to_end_through_the_runner(self) -> None:
        """A serialized hypothesis parses, runs on real data, and passes."""
        payload = {
            "id": "H-real-1",
            "claim": "Under overflow, more than 40% of the solid mesh exceeds 100 Pa.",
            "rationale": "The 100 Pa normalization ceiling appears to saturate at high flow.",
            "test": {
                "primitive": "fraction_above",
                "params": {
                    "field": "von_mises",
                    "case": "Overflow",
                    "threshold": 100.0,
                    "op": "gt",
                    "value": 0.4,
                },
                "decision_rule": "Passes iff the exceedance fraction is strictly above 0.40.",
            },
        }
        outcome = R.run(self.ds, S.hypothesis_from_dict(payload))
        self.assertTrue(outcome.passed, outcome.summary)
        self.assertIsNone(outcome.error)
        self.assertGreater(outcome.observed["n_used"], 10000)


if __name__ == "__main__":
    unittest.main()
