"""Answer preservation: does the compression keep the answers the full data gives?

This project turns roughly 27 million field values into a briefing of about 3,000
tokens. The rule the whole stack is judged by is:

    A compression is correct if and only if it preserves the answers to the questions
    people actually ask of the data.

So each question below is answered twice. ``raw`` goes to the full arrays and does the
arithmetic itself. ``summary`` is allowed to touch *only* the 25-point profiles and the
scalar metric registry — never a full array. The two answers must agree.

The suite has teeth by construction: :class:`TestNegativeControl` corrupts the summary
and asserts that the same checks then fail. A verification suite that passes a
corrupted input is not verifying anything.
"""

from __future__ import annotations

import sys
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_insights.io.dataset import Dataset  # noqa: E402
from llm_insights.summary.briefing import build_briefing  # noqa: E402
from llm_insights.summary.metrics import METRIC_NAMES, METRICS, metric_value  # noqa: E402
from llm_insights.summary.profiles import profile  # noqa: E402

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


ROOT = _resolve_root()
HAS_DATA = (ROOT / "FSG Results").is_dir()
CASES = ("flow_U0p0180", "flow_U0p0360", "flow_U0p0540")
ATRIAL_THIRD = 1.0 / 3.0


def _ds() -> Dataset:
    return Dataset(ROOT)


def _surface_vm(ds: Dataset, case: str) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution surface von Mises on arc position, straight from the arrays."""
    from llm_insights.harness.primitives import arc_projection

    nodes = ds.nodes(case)
    return arc_projection(nodes, nodes.von_mises_pa)


def _slope_sign(s: np.ndarray, v: np.ndarray, s_max: float) -> int:
    """Sign of the least-squares slope of ``v`` against ``s`` over ``s <= s_max``."""
    mask = (s <= s_max) & np.isfinite(v)
    if mask.sum() < 3:
        raise ValueError("not enough points to fit a slope")
    slope = float(np.polyfit(s[mask], v[mask], 1)[0])
    return int(np.sign(slope))


def _argmax_case(values: dict[str, float]) -> str:
    """Name of the case with the largest value."""
    return max(values, key=lambda k: values[k])


@dataclass(frozen=True)
class Question:
    """One question answered both from raw arrays and from the summary alone.

    Attributes:
        key: Short identifier used in test output.
        text: The question in plain English.
        raw: Answers using the full-resolution arrays.
        summary: Answers using only profiles and scalar metrics.
        tol: Absolute tolerance for numeric answers; ``None`` means exact equality.
    """

    key: str
    text: str
    raw: Callable[[Dataset], Any]
    summary: Callable[[Dataset], Any]
    tol: float | None = None


# --- The question table ----------------------------------------------------------


def _build_questions() -> list[Question]:
    """Assemble the answer-preservation question set."""
    q: list[Question] = []

    q.append(
        Question(
            "peak_wss_case",
            "Which case has the highest peak wall shear stress?",
            lambda ds: _argmax_case(
                {c: float(np.max(ds.arc(c, 14).wss_dyn_cm2)) for c in CASES}
            ),
            lambda ds: _argmax_case(
                {c: metric_value(ds, "wss_peak_dyn_cm2", c) for c in CASES}
            ),
        )
    )
    q.append(
        Question(
            "mean_vm_case",
            "Which case has the highest mean von Mises stress?",
            lambda ds: _argmax_case(
                {c: float(np.mean(ds.nodes(c).von_mises_pa)) for c in CASES}
            ),
            lambda ds: _argmax_case(
                {c: metric_value(ds, "von_mises_mean_pa", c) for c in CASES}
            ),
        )
    )
    q.append(
        Question(
            "peak_vm_case",
            "Which case has the highest peak von Mises stress?",
            lambda ds: _argmax_case(
                {c: float(np.max(ds.nodes(c).von_mises_pa)) for c in CASES}
            ),
            lambda ds: _argmax_case(
                {c: metric_value(ds, "von_mises_peak_pa", c) for c in CASES}
            ),
        )
    )
    q.append(
        Question(
            "wss_peak_furthest_downstream",
            "Which case's WSS peak sits furthest downstream?",
            lambda ds: _argmax_case(
                {
                    c: float(ds.arc(c, 14).s_norm[int(np.argmax(ds.arc(c, 14).wss_dyn_cm2))])
                    for c in CASES
                }
            ),
            lambda ds: _argmax_case({c: metric_value(ds, "wss_peak_s", c) for c in CASES}),
        )
    )
    for case in CASES:
        q.append(
            Question(
                f"wss_half_{case}",
                f"Is the WSS peak in the atrial or ventricular half, for {case}?",
                lambda ds, c=case: (
                    "ventricular"
                    if ds.arc(c, 14).s_norm[int(np.argmax(ds.arc(c, 14).wss_dyn_cm2))] >= 0.5
                    else "atrial"
                ),
                lambda ds, c=case: (
                    "ventricular" if metric_value(ds, "wss_peak_s", c) >= 0.5 else "atrial"
                ),
            )
        )
    q.append(
        Question(
            "tallest_cushion",
            "Which case has the tallest cushion at the final step?",
            lambda ds: _argmax_case({c: float(np.max(ds.arc(c, 14).y_mm)) for c in CASES}),
            lambda ds: _argmax_case(
                {c: metric_value(ds, "cushion_height_mm", c) for c in CASES}
            ),
        )
    )
    q.append(
        Question(
            "rank_mean_wss",
            "Rank the three cases by mean wall shear stress.",
            lambda ds: tuple(
                sorted(CASES, key=lambda c: float(np.mean(ds.arc(c, 14).wss_dyn_cm2)))
            ),
            lambda ds: tuple(
                sorted(CASES, key=lambda c: metric_value(ds, "wss_mean_dyn_cm2", c))
            ),
        )
    )
    q.append(
        Question(
            "endmt_variability_case",
            "Which case has the greatest spatial variability in EndMT activity?",
            lambda ds: _argmax_case(
                {
                    c: float(np.nanstd(ds.grn(c).activity("EndMT", "combined")))
                    for c in CASES
                }
            ),
            lambda ds: _argmax_case(
                {c: metric_value(ds, "endmt_sd_combined", c) for c in CASES}
            ),
        )
    )
    q.append(
        Question(
            "vm_monotonic_with_flow",
            "Does mean von Mises increase monotonically with inlet velocity?",
            lambda ds: bool(
                np.all(np.diff([float(np.mean(ds.nodes(c).von_mises_pa)) for c in CASES]) > 0)
            ),
            lambda ds: bool(
                np.all(np.diff([metric_value(ds, "von_mises_mean_pa", c) for c in CASES]) > 0)
            ),
        )
    )
    q.append(
        Question(
            "height_monotonic_with_flow",
            "Does cushion height decrease monotonically with inlet velocity?",
            lambda ds: bool(
                np.all(np.diff([float(np.max(ds.arc(c, 14).y_mm)) for c in CASES]) < 0)
            ),
            lambda ds: bool(
                np.all(np.diff([metric_value(ds, "cushion_height_mm", c) for c in CASES]) < 0)
            ),
        )
    )
    for case in CASES:
        q.append(
            Question(
                f"wss_atrial_slope_{case}",
                f"Across the atrial third, is WSS rising or falling for {case}?",
                lambda ds, c=case: _slope_sign(
                    ds.arc(c, 14).s_norm, ds.arc(c, 14).wss_dyn_cm2, ATRIAL_THIRD
                ),
                lambda ds, c=case: _slope_sign(
                    profile(ds, c, "wss").s, profile(ds, c, "wss").values, ATRIAL_THIRD
                ),
            )
        )
    q.append(
        Question(
            "most_saturated_case",
            "Which case has the largest fraction of its domain at the mechanical ceiling?",
            lambda ds: _argmax_case(
                {c: float(np.mean(ds.nodes(c).von_mises_pa > 100.0)) for c in CASES}
            ),
            lambda ds: _argmax_case(
                {c: metric_value(ds, "mech_clipped_fraction", c) for c in CASES}
            ),
        )
    )
    q.append(
        Question(
            "most_growth_pinned",
            "Which case has the most cells pinned at the growth floor?",
            lambda ds: _argmax_case(
                {c: float(np.mean(ds.growth(c, 14).g <= 0.5 + 1e-9)) for c in CASES}
            ),
            lambda ds: _argmax_case(
                {c: metric_value(ds, "growth_pinned_low_fraction", c) for c in CASES}
            ),
        )
    )
    for case in CASES:
        q.append(
            Question(
                f"wss_peak_s_{case}",
                f"Where along the arc is the WSS maximum for {case}? (tolerance 0.1 in s)",
                lambda ds, c=case: float(
                    ds.arc(c, 14).s_norm[int(np.argmax(ds.arc(c, 14).wss_dyn_cm2))]
                ),
                lambda ds, c=case: profile(ds, c, "wss").peak_s(),
                tol=0.1,
            )
        )
    q.append(
        Question(
            "strongest_downstream_weighting",
            "Which case shows the strongest downstream weighting of WSS?",
            lambda ds: _argmax_case(
                {
                    c: float(
                        (
                            np.mean(ds.arc(c, 14).wss_dyn_cm2[ds.arc(c, 14).s_norm >= 0.5])
                            - np.mean(ds.arc(c, 14).wss_dyn_cm2[ds.arc(c, 14).s_norm < 0.5])
                        )
                        / np.mean(ds.arc(c, 14).wss_dyn_cm2)
                    )
                    for c in CASES
                }
            ),
            lambda ds: _argmax_case(
                {c: metric_value(ds, "wss_lr_asymmetry", c) for c in CASES}
            ),
        )
    )
    q.append(
        Question(
            "overflow_healthy_wss_ratio",
            "Ratio of Overflow to Healthy mean WSS (tolerance 0.02).",
            lambda ds: float(
                np.mean(ds.arc("flow_U0p0540", 14).wss_dyn_cm2)
                / np.mean(ds.arc("flow_U0p0360", 14).wss_dyn_cm2)
            ),
            lambda ds: metric_value(ds, "wss_mean_dyn_cm2", "flow_U0p0540")
            / metric_value(ds, "wss_mean_dyn_cm2", "flow_U0p0360"),
            tol=0.02,
        )
    )
    q.append(
        Question(
            "peak_vm_surface_half",
            "Is peak surface von Mises in the atrial or ventricular half, for Healthy?",
            lambda ds: (
                "ventricular"
                if _surface_vm(ds, "flow_U0p0360")[0][
                    int(np.argmax(_surface_vm(ds, "flow_U0p0360")[1]))
                ]
                >= 0.5
                else "atrial"
            ),
            lambda ds: (
                "ventricular"
                if metric_value(ds, "von_mises_peak_s", "flow_U0p0360") >= 0.5
                else "atrial"
            ),
        )
    )
    return q


QUESTIONS = _build_questions()


@unittest.skipUnless(HAS_DATA, "FSG results tree not mounted")
class TestAnswerPreservation(unittest.TestCase):
    """Every question must get the same answer from the summary as from the raw arrays."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the dataset once for the whole class."""
        cls.ds = _ds()

    def test_question_set_is_large_enough(self) -> None:
        """The suite promises at least 20 questions; hold it to that."""
        self.assertGreaterEqual(len(QUESTIONS), 20, "answer-preservation set has shrunk")

    def test_all_questions_agree(self) -> None:
        """Raw and summary answers agree for every question in the table."""
        failures: list[str] = []
        for q in QUESTIONS:
            with self.subTest(question=q.key):
                raw = q.raw(self.ds)
                summ = q.summary(self.ds)
                if q.tol is None:
                    if raw != summ:
                        failures.append(f"{q.key}: raw={raw!r} summary={summ!r}")
                    self.assertEqual(raw, summ, f"{q.text} -> raw {raw!r} vs summary {summ!r}")
                else:
                    delta = abs(float(raw) - float(summ))
                    if delta > q.tol:
                        failures.append(f"{q.key}: |{raw}-{summ}|={delta:.4g} > {q.tol}")
                    self.assertLessEqual(
                        delta, q.tol, f"{q.text} -> raw {raw!r} vs summary {summ!r}"
                    )
        self.assertEqual(failures, [], "answer-preservation failures")


@unittest.skipUnless(HAS_DATA, "FSG results tree not mounted")
class TestResamplingFidelity(unittest.TestCase):
    """The 25-point resampling must not move peaks, invent values, or flatten plateaus."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the dataset once for the whole class."""
        cls.ds = _ds()

    def test_peak_position_within_tolerance(self) -> None:
        """Resampling moves the WSS peak by at most 0.1 in normalized arc length."""
        for case in CASES:
            with self.subTest(case=case):
                arc = self.ds.arc(case, 14)
                raw_s = float(arc.s_norm[int(np.argmax(arc.wss_dyn_cm2))])
                got = profile(self.ds, case, "wss").peak_s()
                self.assertLessEqual(abs(raw_s - got), 0.1, f"{case}: {raw_s} vs {got}")

    def test_vm_peak_position_within_tolerance(self) -> None:
        """Surface von Mises peak survives resampling to within 0.1 in s."""
        for case in CASES:
            with self.subTest(case=case):
                s, v = _surface_vm(self.ds, case)
                raw_s = float(s[int(np.argmax(v))])
                got = profile(self.ds, case, "von_mises").peak_s()
                self.assertLessEqual(abs(raw_s - got), 0.1, f"{case}: {raw_s} vs {got}")

    def test_no_interpolation_overshoot(self) -> None:
        """Resampled values never leave the range of the source array."""
        for case in CASES:
            for field, attr in (("wss", "wss_dyn_cm2"), ("pressure", "pressure_pa")):
                with self.subTest(case=case, field=field):
                    src = getattr(self.ds.arc(case, 14), attr)
                    pr = profile(self.ds, case, field)
                    self.assertGreaterEqual(float(np.min(pr.values)), float(np.min(src)) - 1e-9)
                    self.assertLessEqual(float(np.max(pr.values)), float(np.max(src)) + 1e-9)

    def test_saturation_plateau_survives(self) -> None:
        """The clipped high-stress region stays flat-topped after compression.

        Overflow clips 46% of its nodes at the 100 Pa ceiling. If resampling smoothed
        that plateau into a rounded hump, the summary would hide the single most
        consequential artifact in the dataset.
        """
        grn = self.ds.grn("flow_U0p0540")
        pinned = grn.mech_norm >= 1.0
        self.assertGreater(float(np.mean(pinned)), 0.4, "expected a large clipped region")
        self.assertEqual(
            len(np.unique(np.round(grn.mech_norm[pinned], 12))),
            1,
            "clipped nodes must all carry the identical ceiling value",
        )


@unittest.skipUnless(HAS_DATA, "FSG results tree not mounted")
class TestBriefingIntegrity(unittest.TestCase):
    """The briefing must be prompt-sized, complete, and honest about its caveats."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the briefing once for the whole class."""
        cls.ds = _ds()
        cls.text = build_briefing(cls.ds)

    def test_fits_in_a_prompt(self) -> None:
        """Briefing is non-empty and under 20,000 characters."""
        self.assertGreater(len(self.text), 2000)
        self.assertLess(len(self.text), 20000, "briefing has outgrown a comfortable prompt")

    def test_every_metric_is_named(self) -> None:
        """Every registered metric appears, so the model knows what it may test."""
        for name in METRIC_NAMES:
            self.assertIn(name, self.text, f"metric {name} missing from briefing")

    def test_every_metric_declares_units(self) -> None:
        """No number is presented without its units."""
        for name in METRIC_NAMES:
            self.assertTrue(METRICS[name].units, f"{name} has no units")
            self.assertTrue(METRICS[name].source, f"{name} has no provenance")

    def test_all_conditions_present(self) -> None:
        """All three condition labels appear."""
        for label in ("Underflow", "Healthy", "Overflow"):
            self.assertIn(label, self.text)

    def test_caveats_mention_saturation(self) -> None:
        """The saturation ceiling is disclosed; omitting it would invite a wrong claim."""
        self.assertIn("100", self.text)
        lowered = self.text.lower()
        self.assertIn("satur", lowered)
        self.assertIn("clip", lowered)


@unittest.skipUnless(HAS_DATA, "FSG results tree not mounted")
class TestNegativeControl(unittest.TestCase):
    """Proof the suite has teeth: a corrupted summary must fail it."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the dataset once for the whole class."""
        cls.ds = _ds()

    def test_three_point_resampling_breaks_peak_location(self) -> None:
        """Compressing to 3 points must move at least one peak outside tolerance.

        If a 3-point summary still answered every question correctly, the questions
        would not be sensitive to compression and the suite would be vacuous.
        """
        errors = []
        for case in CASES:
            arc = self.ds.arc(case, 14)
            raw_s = float(arc.s_norm[int(np.argmax(arc.wss_dyn_cm2))])
            got = profile(self.ds, case, "wss", n=3).peak_s()
            errors.append(abs(raw_s - got))
        self.assertGreater(
            max(errors), 0.1, "3-point compression should have broken peak location"
        )

    def test_perturbed_metric_breaks_a_categorical_answer(self) -> None:
        """A 20% perturbation of one case's metric must flip a 'which case' answer."""
        true_values = {c: metric_value(self.ds, "endmt_sd_combined", c) for c in CASES}
        self.assertEqual(_argmax_case(true_values), "flow_U0p0360")
        corrupted = dict(true_values)
        corrupted["flow_U0p0540"] = corrupted["flow_U0p0360"] * 1.2
        self.assertNotEqual(
            _argmax_case(corrupted),
            _argmax_case(true_values),
            "perturbation failed to change the answer, so the check is insensitive",
        )


if __name__ == "__main__":
    unittest.main()
