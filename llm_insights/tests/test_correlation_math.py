"""Verify the numpy correlation math that replaced scipy.

``harness.primitives.correlation`` used to call ``scipy.stats``. Importing scipy deep
inside a primitive meant a missing dependency surfaced only when a hypothesis happened
to use that primitive, and then arrived as ``COULD NOT RUN`` -- the same channel a
genuinely unevaluable claim uses. Since ``correlation`` is the only primitive whose
answer is not already printed in the briefing, that failure mode silently removed the
one test capable of a real prediction. See TASK-009.

The replacement is verified in three layers, deliberately:

1. **Golden values**, chosen so the exact answer is computable by hand (0.8, sqrt(0.9),
   +/-1). These run everywhere, including a sandbox with no scipy, and they are the
   layer that would catch an algebra error.
2. **Rank behaviour**, checked against ranks written out by hand. Tie averaging is the
   part most likely to be wrong, and this dataset is full of exact ties -- every node
   clipped at the 100 Pa ceiling reports the identical activity.
3. **An oracle**, comparing against ``scipy.stats`` itself on both synthetic and real
   data. This layer *skips* where scipy is absent, so it proves nothing in the sandbox
   this code was written in; it is the evidence that has to be produced on a machine
   with the full dependency set.

Layer 3 skipping is why layers 1 and 2 state exact expected numbers rather than
comparing two implementations to each other.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from llm_insights.harness.primitives import (
    average_ranks,
    correlation,
    pearson_r,
    spearman_r,
)
from llm_insights.io.dataset import Dataset

try:  # the oracle is optional by design; see the module docstring
    from scipy import stats as _scipy_stats

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment, not the code
    _scipy_stats = None
    HAVE_SCIPY = False


def _resolve_root() -> Path:
    """Find the FSG results tree, preferring the checkout these tests live in."""
    in_repo = Path(__file__).resolve().parents[1].parent / "one_way_fsg_model"
    staged = Path("/mnt/user-data/uploads/butcher_lab_fsg_exploration/one_way_fsg_model")
    for candidate in (in_repo, staged):
        if (candidate / "FSG Results").is_dir():
            return candidate
    return in_repo


REAL_ROOT = _resolve_root()


class TestAverageRanks(unittest.TestCase):
    """Layer 2: ranks written out by hand."""

    def test_distinct_values_rank_one_to_n(self):
        """No ties means the ranks are a permutation of 1..n."""
        np.testing.assert_allclose(average_ranks(np.array([3.0, 1.0, 2.0])), [3.0, 1.0, 2.0])

    def test_a_tied_pair_shares_the_average_of_the_ranks_it_spans(self):
        """[10, 20, 20, 30] occupies ranks 1, 2, 3, 4; the pair splits 2 and 3."""
        np.testing.assert_allclose(
            average_ranks(np.array([10.0, 20.0, 20.0, 30.0])), [1.0, 2.5, 2.5, 4.0]
        )

    def test_an_all_tied_series_collapses_to_one_rank(self):
        """Three equal values span ranks 1..3, so each is 2.0."""
        np.testing.assert_allclose(average_ranks(np.array([5.0, 5.0, 5.0])), [2.0, 2.0, 2.0])

    def test_a_tie_group_at_the_end_is_averaged_too(self):
        """The final group is the one an off-by-one in the boundary walk would miss."""
        np.testing.assert_allclose(
            average_ranks(np.array([1.0, 2.0, 9.0, 9.0, 9.0])), [1.0, 2.0, 4.0, 4.0, 4.0]
        )

    def test_ranks_are_stable_under_input_order(self):
        """Ranking is a property of the values, not of the order they arrive in."""
        values = np.array([4.0, 1.0, 4.0, 2.0])
        permutation = np.array([2, 0, 3, 1])
        direct = average_ranks(values)[permutation]
        permuted = average_ranks(values[permutation])
        np.testing.assert_allclose(direct, permuted)

    def test_the_ranks_sum_to_the_triangular_number(self):
        """Averaging ties must not change the total; n(n+1)/2 holds however they clump."""
        for values in (
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([1.0, 2.0, 2.0, 5.0, 5.0, 5.0]),
            np.linspace(0.0, 1.0, 17),
        ):
            n = values.size
            self.assertAlmostEqual(float(average_ranks(values).sum()), n * (n + 1) / 2.0)


class TestPearsonGoldenValues(unittest.TestCase):
    """Layer 1: exact answers, computable without a computer."""

    def test_a_perfect_positive_line_is_one(self):
        """y = 2x gives 1.0 to floating point, and never above it.

        Not *exactly* 1.0: centring and normalising a perfectly correlated pair lands a
        unit or two below, which is ordinary double-precision behaviour and is what the
        reference implementation does too. The clamp exists to stop the value drifting
        *above* 1.0, not to manufacture an exact one.
        """
        r = pearson_r(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
        self.assertAlmostEqual(r, 1.0, places=12)
        self.assertLessEqual(r, 1.0)

    def test_a_perfect_negative_line_is_minus_one(self):
        """A reversed series is anticorrelated, and never below -1.0."""
        r = pearson_r(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
        self.assertAlmostEqual(r, -1.0, places=12)
        self.assertGreaterEqual(r, -1.0)

    def test_a_hand_computable_intermediate_case(self):
        """x=[1,2,3,4], y=[1,3,2,4]: centred dot 4.0 over norms sqrt(5)*sqrt(5) = 0.8."""
        r = pearson_r(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3.0, 2.0, 4.0]))
        self.assertAlmostEqual(r, 0.8, places=12)

    def test_it_is_symmetric(self):
        """Correlation does not care which series is named first."""
        x = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        y = np.array([2.0, 1.0, 7.0, 3.0, 11.0])
        self.assertAlmostEqual(pearson_r(x, y), pearson_r(y, x), places=15)

    def test_it_is_invariant_under_positive_affine_rescaling(self):
        """Changing units must not change a correlation."""
        x = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        y = np.array([2.0, 1.0, 7.0, 3.0, 11.0])
        self.assertAlmostEqual(pearson_r(x, y), pearson_r(3.0 * x + 7.0, 0.5 * y - 2.0), places=12)

    def test_a_constant_series_is_refused(self):
        """An undefined coefficient is an error, never a number."""
        with self.assertRaises(ValueError):
            pearson_r(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))

    def test_the_result_never_escapes_the_unit_interval(self):
        """However much rounding accumulates, |r| must not exceed 1.

        A coefficient of 1.0000000000000002 reaching a decision rule of ``r >= 1.0``
        would be a real bug: it would let a claim pass on rounding. Undershoot is
        harmless and expected; overshoot is what the clamp is for.
        """
        for n in (3, 500, 5000):
            x = np.linspace(0.0, 1.0, n)
            for a, b in ((x, x), (x, -x), (x, 2.0 * x + 1.0)):
                r = pearson_r(a, b)
                self.assertLessEqual(abs(r), 1.0, f"n={n} produced |r|>1: {r!r}")
                self.assertAlmostEqual(abs(r), 1.0, places=12)


class TestSpearmanGoldenValues(unittest.TestCase):
    """Layer 1, for the rank coefficient."""

    def test_a_monotone_curve_ranks_perfectly_even_though_pearson_does_not(self):
        """y = x^3 is monotone but not linear: Spearman 1.0, Pearson short of it."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = x**3
        self.assertEqual(spearman_r(x, y), 1.0)
        self.assertLess(pearson_r(x, y), 1.0)

    def test_a_tied_series_gives_a_hand_computable_value(self):
        """Ranks [1,2,3,4] against [1,2.5,2.5,4] give 4.5/sqrt(5*4.5) = sqrt(0.9)."""
        r = spearman_r(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 2.0, 3.0]))
        self.assertAlmostEqual(r, math.sqrt(0.9), places=12)

    def test_it_is_invariant_under_any_monotone_transform(self):
        """Rank correlation depends on order alone, so exp() must not move it."""
        x = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        y = np.array([2.0, 1.0, 7.0, 3.0, 11.0, 4.0])
        self.assertAlmostEqual(spearman_r(x, y), spearman_r(np.exp(x), y), places=12)

    def test_a_series_with_no_variation_is_refused(self):
        """All-tied ranks are constant, so the coefficient stays undefined."""
        with self.assertRaises(ValueError):
            spearman_r(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]))


@unittest.skipUnless(HAVE_SCIPY, "scipy is not installed; the oracle layer cannot run here")
class TestAgreesWithScipy(unittest.TestCase):
    """Layer 3: the reference implementation, where it is available.

    These are the tests that must be run on a machine with the full dependency set. In
    a sandbox without scipy they skip, and layers 1 and 2 carry the verification.
    """

    def test_pearson_matches_on_random_data(self):
        """Twenty random pairs, agreeing to floating-point noise."""
        rng = np.random.default_rng(20260902)
        for _ in range(20):
            x = rng.normal(size=500)
            y = 0.3 * x + rng.normal(size=500)
            self.assertAlmostEqual(pearson_r(x, y), float(_scipy_stats.pearsonr(x, y).statistic), places=12)

    def test_spearman_matches_on_random_data(self):
        """Same, for ranks."""
        rng = np.random.default_rng(20260903)
        for _ in range(20):
            x = rng.normal(size=500)
            y = np.exp(0.3 * x) + rng.normal(size=500)
            self.assertAlmostEqual(spearman_r(x, y), float(_scipy_stats.spearmanr(x, y).statistic), places=12)

    def test_spearman_matches_on_heavily_tied_data(self):
        """The case this dataset actually produces: many exactly-equal values."""
        rng = np.random.default_rng(20260904)
        for _ in range(20):
            x = rng.integers(0, 4, size=400).astype(float)
            y = rng.integers(0, 3, size=400).astype(float)
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            self.assertAlmostEqual(spearman_r(x, y), float(_scipy_stats.spearmanr(x, y).statistic), places=12)

    def test_average_ranks_matches_rankdata(self):
        """Directly against the function this replaced."""
        rng = np.random.default_rng(20260905)
        for _ in range(20):
            values = rng.integers(0, 6, size=300).astype(float)
            np.testing.assert_allclose(
                average_ranks(values), _scipy_stats.rankdata(values, method="average")
            )

    @unittest.skipUnless(REAL_ROOT.is_dir(), f"real dataset not mounted at {REAL_ROOT}")
    def test_matches_scipy_on_the_real_grn_columns(self):
        """The claim that matters: identical answers on Dan's actual output.

        This is the pairing card H4 of the 2026-09-02 run asked for, plus the WSS
        pairing whose interior NaNs exercise the pairwise drop.
        """
        ds = Dataset(REAL_ROOT)
        pairs = [
            ("mech_norm", "EndMT_combined"),
            ("von_mises_pa", "EndMT_combined"),
            ("wss_dyn_cm2", "EndMT_combined"),
        ]
        for case in ds.cases():
            for field_x, field_y in pairs:
                for method in ("pearson", "spearman"):
                    with self.subTest(case=case, x=field_x, y=field_y, method=method):
                        try:
                            _, observed, _ = correlation(
                                ds, field_x, field_y, case, "ge", -2.0, method=method
                            )
                        except ValueError:
                            continue  # constant column in this case; nothing to compare
                        grn = ds.grn(case)
                        from llm_insights.harness.primitives import _finite_pair, _grn_series

                        x, y, _ = _finite_pair(_grn_series(grn, field_x), _grn_series(grn, field_y))
                        fn = (
                            _scipy_stats.pearsonr
                            if method == "pearson"
                            else _scipy_stats.spearmanr
                        )
                        self.assertAlmostEqual(observed["r"], float(fn(x, y).statistic), places=12)


@unittest.skipUnless(REAL_ROOT.is_dir(), f"real dataset not mounted at {REAL_ROOT}")
class TestCorrelationRunsWithoutScipy(unittest.TestCase):
    """The regression that started TASK-009: this must work with no scipy installed."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the dataset once."""
        cls.ds = Dataset(REAL_ROOT)

    def test_the_h4_pairing_now_evaluates(self):
        """Card H4 of the 2026-09-02 run returned COULD NOT RUN. It must return a number."""
        passed, observed, summary = correlation(
            self.ds, "mech_norm", "EndMT_combined", "Underflow", "ge", 0.4
        )
        self.assertIn("r", observed)
        self.assertTrue(-1.0 <= observed["r"] <= 1.0)
        self.assertGreater(observed["n_used"], 0.0)
        self.assertIsInstance(passed, bool)
        self.assertIn("pearson correlation", summary)

    def test_spearman_also_evaluates_on_real_data(self):
        """The rank path has its own numpy code, so it needs its own real-data run."""
        _, observed, _ = correlation(
            self.ds, "mech_norm", "EndMT_combined", "Underflow", "ge", 0.4, method="spearman"
        )
        self.assertTrue(-1.0 <= observed["r"] <= 1.0)

    def test_interior_nans_are_still_dropped_pairwise(self):
        """WSS is NaN at interior nodes; those rows must be dropped, never zeroed."""
        _, observed, _ = correlation(
            self.ds, "wss_dyn_cm2", "EndMT_combined", "Healthy", "ge", -2.0
        )
        self.assertGreater(observed["n_dropped"], observed["n_used"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
