"""TASK-013: the re-run of Dan's GRN is the same computation, on more nodes.

The value of ``llm_insights.grn`` rests entirely on it being *Dan's* network and not a
re-implementation that drifted. Four things are checked here, and each one is checking a
different way that could go wrong:

* **The wiring was copied correctly and stays copied.** ``networkinput.py`` cannot be
  imported -- it reads a hardcoded ``/Users/sophiewang/...`` path and runs ~27,000 ODE
  solves at import time -- so its scenario flags, its integration horizon and its two
  pinned mid-range drives are restated as data in ``runner.py``. Here they are read back
  out of Dan's file with :mod:`ast` and compared. A future edit on either side fails a
  test instead of silently producing numbers that no longer mean what they claim.
* **Nothing is dropped.** The whole point of the task is that all 24 nodes reach disk,
  under all three scenarios.
* **Resume is not a lie.** An interrupted run is restarted and must produce output
  *identical* to an uninterrupted one -- not close, identical. A checkpoint that
  silently loses or re-orders a chunk would otherwise look like success.
* **The solver reaches the fixed point the algebra predicts.** Six of the 24 nodes have
  a steady state that can be written down by hand from ``networkpoint.rhs``, because
  their target depends only on the external drives and the fixed parameters. Those are
  the closest thing this network has to an analytic solution, and they pin the runner's
  input plumbing: get the normalization or the scenario branch wrong and ``YAP_TAZ`` and
  ``JAG`` land somewhere else.

The real faithfulness gate -- recomputed ``Snai1``/``Snai2``/``EndMT``/``NICD``/
``YAP_TAZ`` against the shipped ``results_spatial.csv`` to 1e-6 -- is a full-dataset run
and lives in TASK-013's Log, not here: it costs ~40,000 ODE solves. What runs here is
its cheap shadow, on a handful of points, plus the structural checks above.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_insights.grn import runner  # noqa: E402


def _resolve_root() -> Path:
    """Find the model tree, preferring the checkout these tests live in.

    Returns:
        The first candidate containing ``FSG Results``, else the in-repo path, so the
        skip message names a location a person would actually look in.
    """
    in_repo = Path(__file__).resolve().parents[1].parent / "one_way_fsg_model"
    staged = Path("/mnt/user-data/uploads/butcher_lab_fsg_exploration/one_way_fsg_model")
    for candidate in (in_repo, staged):
        if (candidate / "FSG Results").is_dir():
            return candidate
    return in_repo


ROOT = _resolve_root()
HAS_MODEL = (ROOT / "networkpoint.py").is_file()
HAS_DATA = (ROOT / "FSG Results").is_dir()
CASES = ("flow_U0p0180", "flow_U0p0360", "flow_U0p0540")


# --- Reading constants back out of networkinput.py --------------------------------


def _module_constants(path: Path) -> dict[str, Any]:
    """Evaluate the module-level constant assignments of a file, without importing it.

    ``networkinput.py`` must never be imported: it opens a path that exists on nobody's
    machine and then runs the whole GRN. Only assignments whose right-hand side is a
    literal, or arithmetic over literals and names already bound, are evaluated; every
    other statement is skipped. That is enough for the flags and the constants, and it
    executes none of Dan's code.

    Args:
        path: The ``.py`` file to read.

    Returns:
        The names it was able to bind, to their values.
    """
    tree = ast.parse(path.read_text())
    env: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            env[target.id] = _const_eval(node.value, env)
        except (ValueError, TypeError, KeyError):
            continue
    return env


def _const_eval(node: ast.expr, env: dict[str, Any]) -> Any:
    """Evaluate a constant expression node against already-bound names.

    Args:
        node: The expression.
        env: Names bound so far.

    Returns:
        The value.

    Raises:
        ValueError: If the expression is anything but a literal, a bound name, or
            arithmetic over those. Nothing is called, so nothing of Dan's runs.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Dict):
        return {
            _const_eval(k, env): _const_eval(v, env)
            for k, v in zip(node.keys, node.values, strict=True)
            if k is not None
        }
    if isinstance(node, ast.List):
        return [_const_eval(e, env) for e in node.elts]
    if isinstance(node, ast.BinOp):
        left = _const_eval(node.left, env)
        right = _const_eval(node.right, env)
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mult):
            return left * right
        raise ValueError("unsupported operator")
    raise ValueError(f"not a constant expression: {type(node).__name__}")


@unittest.skipUnless(HAS_MODEL, "one_way_fsg_model not mounted")
class WiringMatchesDansFile(unittest.TestCase):
    """The constants copied out of ``networkinput.py`` still match ``networkinput.py``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = _module_constants(ROOT / "networkinput.py")

    def test_scenario_flags_match(self):
        self.assertEqual(runner.SCENARIO_FLAGS, self.env["scenarios"])

    def test_scenario_order_matches(self):
        # Column order in activities.csv follows this, so it is not merely cosmetic.
        self.assertEqual(list(runner.SCENARIO_FLAGS), list(self.env["scenarios"]))

    def test_integration_horizon_matches(self):
        # 400.0, not networkpoint.simulate's own 300.0 default.
        self.assertEqual(runner.T_END, self.env["T_END"])
        self.assertNotEqual(runner.T_END, 300.0)

    def test_normalization_ceilings_match(self):
        self.assertEqual(runner.WSS_MAX_DYN_CM2, self.env["WSS_MAX"])
        self.assertEqual(runner.MECH_MAX_PA, self.env["MECH_MAX"])

    def test_pinned_mid_range_drives_match(self):
        self.assertEqual(runner.SHEAR_MID_NORM, self.env["SHEAR_MID_NORM"])
        self.assertEqual(runner.MECH_MID_NORM, self.env["MECH_MID_NORM"])

    def test_the_extractor_is_not_vacuous(self):
        # If _module_constants silently returned {}, every assertion above would still
        # need a KeyError to fail -- so prove it actually read the file.
        self.assertIn("scenarios", self.env)
        self.assertEqual(self.env["OUTPUT_NODES"], ["Snai1", "Snai2", "EndMT"])

    def test_networkinput_is_never_imported(self):
        # Importing it would execute a hardcoded /Users/sophiewang path and ~27,000
        # solves. The runner's source must not name it outside prose.
        source = (Path(runner.__file__)).read_text()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                self.assertNotIn("networkinput", stripped)
        self.assertNotIn("networkinput", sys.modules)


@unittest.skipUnless(HAS_MODEL, "one_way_fsg_model not mounted")
class NodeSet(unittest.TestCase):
    """All 24 nodes, in the solver's own order."""

    def test_twenty_four_nodes(self):
        self.assertEqual(len(runner.node_names(ROOT)), 24)

    def test_the_five_shipped_nodes_are_a_subset(self):
        nodes = set(runner.node_names(ROOT))
        self.assertTrue({"Snai1", "Snai2", "EndMT", "NICD", "YAP_TAZ"} <= nodes)

    def test_the_nineteen_that_used_to_be_discarded_are_present(self):
        nodes = set(runner.node_names(ROOT))
        # SMAD23 is the one demo card H6 had to be refused over.
        for name in ("SMAD23", "SMAD4", "SMAD67", "SMAD158", "beta_catenin", "HEY"):
            self.assertIn(name, nodes)


# --- A synthetic case, with drives whose fixed point can be written down ----------


def _write_synthetic_case(root: Path, case: str, rows: list[tuple[float, float, float, float]]) -> None:
    """Write a minimal GRN input grid in the real layout.

    Args:
        root: Directory to build ``FSG Results/<case>/...`` under.
        case: Case folder name.
        rows: ``(x_m, y_m, von_mises_Pa, wss_mag_dyn_cm2)`` per point; ``wss`` may be
            NaN to mark an interior (non-surface) point.
    """
    dest = root / "FSG Results" / case / runner.GRN_INPUT_RELPATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A root holds both the data and the model, exactly as the real tree does, so
    # run_case resolves networkpoint from the same place it resolves the grid.
    link = root / "networkpoint.py"
    if not link.exists():
        link.symlink_to(ROOT / "networkpoint.py")
    frame = pd.DataFrame(rows, columns=["x_m", "y_m", "von_mises_Pa", "wss_mag_dyn_cm2"])
    frame["is_surface"] = (~frame["wss_mag_dyn_cm2"].isna()).astype(float)
    frame.to_csv(dest, index=False)


def _hill(x: float, n: float, ec50: float) -> float:
    """``networkpoint.hill``, restated so the expected value is not the code under test.

    Args:
        x: Input, clipped to [0, 1].
        n: Hill coefficient.
        ec50: Half-activation constant.

    Returns:
        The Hill response.
    """
    x = min(max(x, 0.0), 1.0)
    return (x**n) / (ec50**n + x**n + 1e-12)


@unittest.skipUnless(HAS_MODEL, "one_way_fsg_model not mounted")
class SyntheticFixedPoint(unittest.TestCase):
    """Six nodes whose steady state is algebra, computed here independently.

    In ``networkpoint.rhs`` these six have a target that depends only on the external
    drives and on fixed :class:`Params` values -- not on any other state variable -- so
    ``dy/dt = 0`` gives them in closed form. Everything else in the network is coupled
    and has no such expression, which is exactly why these six are the ones worth
    asserting on: they are checkable without re-deriving the model.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.p = runner.load_networkpoint(ROOT).Params()

    def _expected(self, shear: float, mech: float) -> dict[str, float]:
        """Closed-form steady states at a given normalized drive.

        Args:
            shear: Normalized shear in [0, 1].
            mech: Normalized mech in [0, 1].

        Returns:
            Node name to expected activity.
        """
        p = self.p
        return {
            "JAG": _hill(shear, p.n, p.ec50_input),
            "DLL": 1.0 - _hill(shear, p.n, p.ec50_input),
            "YAP_TAZ": _hill(mech, p.n, p.ec50_input),
            "LRP": 1.0 - _hill(p.DKK, p.n, p.ec50),
            "MAML": 1.0,
            "RBPJ": 0.3,
            "TCF_LEF": 1.0,
        }

    def test_solver_lands_on_the_closed_form(self):
        nodes = runner.node_names(ROOT)
        shear, mech = 0.4, 0.7
        values = runner.solve_points(
            ROOT, np.array([shear]), np.array([mech]), n_jobs=1, chunk_size=1
        )
        for name, want in self._expected(shear, mech).items():
            got = values[0, nodes.index(name)]
            self.assertAlmostEqual(got, want, delta=1e-6, msg=name)

    def test_a_wrong_drive_would_be_caught(self):
        # The negative control for the assertion above: YAP_TAZ at mech=0.7 must not
        # also satisfy the prediction for mech=0.2, or the test proves nothing.
        self.assertGreater(
            abs(self._expected(0.4, 0.7)["YAP_TAZ"] - self._expected(0.4, 0.2)["YAP_TAZ"]),
            0.1,
        )

    def test_scenarios_pin_the_unused_drive(self):
        root = Path(self.enterContext(_tempdir()))
        # One surface point (wss present) and one interior point (wss NaN).
        _write_synthetic_case(root, "synth", [(0.0, 0.0, 70.0, 12.0), (1.0, 0.0, 70.0, float("nan"))])
        inputs = runner.load_case_inputs(root, "synth")

        shear, mech = runner.scenario_inputs(inputs, "shear_only")
        self.assertAlmostEqual(shear[0], 12.0 / 30.0)
        self.assertAlmostEqual(mech[0], runner.MECH_MID_NORM)
        self.assertTrue(np.isnan(shear[1]), "an interior point is skipped under shear_only")

        shear, mech = runner.scenario_inputs(inputs, "mech_only")
        self.assertTrue(np.all(shear == runner.SHEAR_MID_NORM))
        self.assertAlmostEqual(mech[0], 0.70)

        shear, mech = runner.scenario_inputs(inputs, "combined")
        self.assertAlmostEqual(shear[0], 12.0 / 30.0)
        self.assertEqual(shear[1], 0.0, "an interior point is driven at zero shear, not skipped")
        self.assertAlmostEqual(mech[1], 0.70)


@unittest.skipUnless(HAS_MODEL, "one_way_fsg_model not mounted")
class WrittenOutput(unittest.TestCase):
    """``activities.csv`` carries every node under every scenario."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = _tempdir()
        cls.root = Path(cls._tmp.__enter__())
        _write_synthetic_case(
            cls.root,
            "synth",
            [(0.0, 0.0, 70.0, 12.0), (1.0, 0.0, 20.0, 3.0), (2.0, 0.0, 55.0, float("nan"))],
        )
        cls.path = runner.run_case(
            cls.root, "synth", cls.root / "out", n_jobs=1, chunk_size=2, resume=False
        )
        cls.frame = pd.read_csv(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.__exit__(None, None, None)

    def test_every_node_and_scenario_has_a_column(self):
        nodes = runner.node_names(ROOT)
        want = {f"{n}_{s}" for n in nodes for s in runner.SCENARIO_FLAGS}
        self.assertTrue(want <= set(self.frame.columns))
        self.assertEqual(len(want), 24 * 3)

    def test_input_columns_are_carried_through(self):
        self.assertEqual(list(self.frame.columns[:6]), list(runner.INPUT_COLUMNS))

    def test_interior_points_are_nan_under_shear_only_and_finite_otherwise(self):
        self.assertTrue(np.isnan(self.frame["EndMT_shear_only"].to_numpy()[2]))
        self.assertTrue(np.isfinite(self.frame["EndMT_combined"].to_numpy()[2]))
        self.assertTrue(np.isfinite(self.frame["SMAD23_mech_only"].to_numpy()).all())

    def test_activities_are_in_range(self):
        node_cols = [c for c in self.frame.columns if c not in runner.INPUT_COLUMNS]
        values = self.frame[node_cols].to_numpy()
        finite = values[np.isfinite(values)]
        self.assertGreaterEqual(finite.min(), 0.0)
        self.assertLessEqual(finite.max(), 1.0)


@unittest.skipUnless(HAS_MODEL, "one_way_fsg_model not mounted")
class CheckpointResume(unittest.TestCase):
    """An interrupted run resumes to a bit-identical answer."""

    def test_resume_matches_an_uninterrupted_run(self):
        tmp = Path(self.enterContext(_tempdir()))
        shear = np.linspace(0.1, 0.9, 6)
        mech = np.linspace(0.9, 0.1, 6)

        clean = runner.solve_pairs(ROOT, shear, mech, n_jobs=1, chunk_size=2, group_size=1)

        ckpt = tmp / "resume.npz"
        boom = _FailAfter(1)
        original = runner.Parallel
        runner.Parallel = boom.wrap(original)
        try:
            with self.assertRaises(_Interrupted):
                runner.solve_pairs(
                    ROOT, shear, mech, n_jobs=1, chunk_size=2, group_size=1, checkpoint=ckpt
                )
        finally:
            runner.Parallel = original

        self.assertTrue(ckpt.is_file(), "the interrupted run must leave a checkpoint")
        with np.load(ckpt) as data:
            partial = data["done"]
        self.assertEqual(int(partial.sum()), 1, "exactly one chunk finished before the kill")
        self.assertLess(int(partial.sum()), len(partial), "the kill must have left work undone")

        resumed = runner.solve_pairs(
            ROOT, shear, mech, n_jobs=1, chunk_size=2, group_size=1, checkpoint=ckpt
        )
        np.testing.assert_array_equal(resumed, clean)

    def test_a_checkpoint_for_other_inputs_is_ignored(self):
        tmp = Path(self.enterContext(_tempdir()))
        ckpt = tmp / "stale.npz"
        first = np.array([0.3])
        runner.solve_pairs(ROOT, first, first, n_jobs=1, chunk_size=1, checkpoint=ckpt)
        # Same shape, different values: a checkpoint keyed only on shape would hand
        # back the first job's answers for the second job's inputs.
        second = np.array([0.8])
        again = runner.solve_pairs(ROOT, second, second, n_jobs=1, chunk_size=1, checkpoint=ckpt)
        direct = runner.solve_pairs(ROOT, second, second, n_jobs=1, chunk_size=1)
        np.testing.assert_array_equal(again, direct)


class _Interrupted(RuntimeError):
    """Stands in for the overnight kill an operator cannot schedule inside a test."""


class _FailAfter:
    """Wrap :class:`joblib.Parallel` so it raises after N successful dispatches.

    Attributes:
        limit: How many ``Parallel(...)`` calls succeed before one raises.
        seen: How many have been dispatched so far.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.seen = 0

    def wrap(self, original: Any) -> Any:
        """Return a drop-in replacement for the ``Parallel`` class.

        Args:
            original: The real :class:`joblib.Parallel`.

        Returns:
            A factory producing a context manager whose call raises after ``limit``.
        """
        outer = self

        class _Wrapped:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._inner = original(*args, **kwargs)

            def __enter__(self) -> Any:
                self._inner.__enter__()
                return self

            def __exit__(self, *exc: Any) -> Any:
                return self._inner.__exit__(*exc)

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                if outer.seen >= outer.limit:
                    raise _Interrupted("simulated overnight kill")
                outer.seen += 1
                return self._inner(*args, **kwargs)

        return _Wrapped


def _tempdir() -> Any:
    """Return a temporary-directory context manager.

    Returns:
        A :class:`tempfile.TemporaryDirectory`.
    """
    import tempfile

    return tempfile.TemporaryDirectory()


@unittest.skipUnless(HAS_DATA, "FSG results tree not mounted")
class RealInputGrids(unittest.TestCase):
    """The input grid this runner reads is the one the shipped results were made from."""

    def test_row_counts_match_the_shipped_results(self):
        from llm_insights.io.dataset import GRN_RESULT_DIR_BY_CASE

        for case in CASES:
            inputs = runner.load_case_inputs(ROOT, case)
            shipped = pd.read_csv(
                ROOT / "AHA GRN Plots" / "3scenarios_results" / GRN_RESULT_DIR_BY_CASE[case]
                / "results_spatial.csv"
            )
            self.assertEqual(inputs.n_points, len(shipped), case)

    def test_coordinates_and_drives_match_the_shipped_results(self):
        from llm_insights.io.dataset import GRN_RESULT_DIR_BY_CASE

        for case in CASES:
            inputs = runner.load_case_inputs(ROOT, case)
            shipped = pd.read_csv(
                ROOT / "AHA GRN Plots" / "3scenarios_results" / GRN_RESULT_DIR_BY_CASE[case]
                / "results_spatial.csv"
            )
            np.testing.assert_allclose(inputs.x_m, shipped["x_m"].to_numpy(), atol=1e-12)
            np.testing.assert_allclose(inputs.mech_norm, shipped["mech_norm"].to_numpy(), atol=1e-12)
            np.testing.assert_array_equal(
                inputs.wss_has_data, np.isfinite(shipped["wss_mag_dyn_cm2"].to_numpy())
            )


if __name__ == "__main__":
    unittest.main()
