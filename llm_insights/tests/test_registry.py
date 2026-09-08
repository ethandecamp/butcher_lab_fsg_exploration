"""The two extension points TASK-012 created, and the property that makes them safe.

The briefing-section registry and the metric registry both exist so that adding
content is adding a file rather than editing a shared one. That trade is only worth
making if the registry itself is trustworthy, which here means three things:

* registering works, and a section reaches the briefing without ``briefing.py`` naming
  it;
* iteration is sorted by ``(order, name)`` and is **independent of import order** --
  asserted by re-importing the five real section modules in reverse and rebuilding the
  briefing, which must come out byte-identical;
* a duplicate registration **raises** rather than silently replacing, because a
  silently replaced section is one that vanishes from every briefing with nothing to
  notice it.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_insights.summary import registry
from llm_insights.summary.briefing import build_briefing
from llm_insights.summary.metrics import (
    METRIC_REGISTRY,
    METRICS,
    MetricDef,
    metric_value,
    register_metric,
)


def _resolve_root() -> Path:
    """Find the FSG results tree, preferring the checkout these tests live in.

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

#: The five sections TASK-012 moved out of ``briefing.py``, in render order.
EXPECTED_SECTIONS = ("metrics", "profiles", "trajectory", "caveats", "protocol")

#: The same five modules, in the order ``sections/__init__.py`` does *not* use.
REVERSE_IMPORT_ORDER = ("protocol", "caveats", "trajectory", "profiles", "metrics")


class _StubDataset:
    """The smallest thing :func:`metric_value` will accept, for registry tests."""

    @staticmethod
    def resolve_case(name: str) -> str:
        """Return the name unchanged.

        Args:
            name: A case name or condition label.

        Returns:
            ``name``.
        """
        return name


class _IsolatedRegistry:
    """Context manager giving a test the section registry to itself.

    The registry is module-level state populated at import time, so a test that
    registers into it has to put it back or every later test sees the leftovers.
    """

    def __enter__(self) -> dict[str, registry.SectionDef]:
        """Save and clear the registry.

        Returns:
            The live (now empty) registry dict, for the test to register into.
        """
        self._saved = dict(registry._SECTIONS)
        registry._SECTIONS.clear()
        return registry._SECTIONS

    def __exit__(self, *exc: object) -> None:
        """Restore whatever was registered before.

        Args:
            *exc: The exception triple, unused; the registry is restored either way.
        """
        del exc
        registry._SECTIONS.clear()
        registry._SECTIONS.update(self._saved)


class TestSectionRegistration(unittest.TestCase):
    """Registering a section, and what happens when two collide."""

    def test_a_registered_section_is_returned(self) -> None:
        """A decorated function appears in the registry under its declared name."""
        with _IsolatedRegistry():

            @registry.briefing_section(order=5, name="scratch")
            def scratch(ds: Any, n_profile_points: int) -> list[str]:
                """Return a fixed line."""
                del ds, n_profile_points
                return ["scratch line"]

            found = registry.registered_sections()
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].name, "scratch")
            self.assertEqual(found[0].order, 5)
            self.assertIs(found[0].fn, scratch)

    def test_the_decorator_returns_the_function_unchanged(self) -> None:
        """Decorating must not wrap; the module keeps a directly callable function."""
        with _IsolatedRegistry():

            def raw(ds: Any, n_profile_points: int) -> list[str]:
                """Return a fixed line."""
                del ds, n_profile_points
                return ["x"]

            decorated = registry.briefing_section(order=1, name="raw")(raw)
            self.assertIs(decorated, raw)

    def test_duplicate_name_raises_rather_than_overwriting(self) -> None:
        """The second registration is an error, not a silent replacement."""
        with _IsolatedRegistry():

            @registry.briefing_section(order=1, name="dup")
            def first(ds: Any, n_profile_points: int) -> list[str]:
                """Return the first line."""
                del ds, n_profile_points
                return ["first"]

            with self.assertRaises(ValueError) as ctx:

                @registry.briefing_section(order=1, name="dup")
                def second(ds: Any, n_profile_points: int) -> list[str]:
                    """Return the second line."""
                    del ds, n_profile_points
                    return ["second"]

            self.assertIn("dup", str(ctx.exception))
            # The first registration is still the one that stands.
            self.assertIs(registry.registered_sections()[0].fn, first)

    def test_a_duplicate_name_at_a_different_order_also_raises(self) -> None:
        """The key is the name; a different order does not make it a new section."""
        with _IsolatedRegistry():
            registry.briefing_section(order=1, name="dup")(lambda *_: ["a"])
            with self.assertRaises(ValueError):
                registry.briefing_section(order=99, name="dup")(lambda *_: ["b"])

    def test_iteration_is_sorted_by_order_then_name(self) -> None:
        """Registration order does not survive into iteration order."""
        with _IsolatedRegistry():
            for order, name in ((30, "c"), (10, "b"), (10, "a"), (20, "d")):
                registry.briefing_section(order=order, name=name)(lambda *_: [])
            self.assertEqual(registry.section_names(), ("a", "b", "d", "c"))

    def test_render_sections_concatenates_in_order(self) -> None:
        """The composer's helper joins the sections' lines, in render order."""
        with _IsolatedRegistry():
            registry.briefing_section(order=20, name="second")(lambda *_: ["B"])
            registry.briefing_section(order=10, name="first")(lambda *_: ["A"])
            self.assertEqual(registry.render_sections(None, 25), ["A", "B"])


class TestRealSections(unittest.TestCase):
    """The five sections TASK-012 moved really are registered, in the right order."""

    def test_the_five_original_sections_are_registered(self) -> None:
        """No section was lost in the move out of briefing.py."""
        self.assertEqual(registry.section_names(), EXPECTED_SECTIONS)

    def test_every_section_declares_a_distinct_order(self) -> None:
        """Distinct orders keep the render order stable under any tie-break."""
        orders = [section.order for section in registry.registered_sections()]
        self.assertEqual(len(orders), len(set(orders)), f"orders collide: {orders}")


@unittest.skipUnless(HAS_DATA, "the FSG dataset is not present")
class TestImportOrderIndependence(unittest.TestCase):
    """Importing the section modules in reverse must change nothing at all."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the briefing once, in the normal import order."""
        from llm_insights.io.dataset import Dataset

        cls.ds = Dataset(ROOT)
        cls.reference = build_briefing(cls.ds)

    def test_reverse_import_order_produces_an_identical_briefing(self) -> None:
        """The registry sorts; it does not remember who registered first."""
        saved = dict(registry._SECTIONS)
        try:
            registry._SECTIONS.clear()
            for name in REVERSE_IMPORT_ORDER:
                module = importlib.import_module(f"llm_insights.summary.sections.{name}")
                importlib.reload(module)
            self.assertEqual(registry.section_names(), EXPECTED_SECTIONS)
            rebuilt = build_briefing(self.ds)
        finally:
            registry._SECTIONS.clear()
            registry._SECTIONS.update(saved)
        self.assertEqual(rebuilt, self.reference, "briefing changed with import order")

    def test_the_briefing_is_reproducible_within_one_import_order(self) -> None:
        """Two builds of the same tree agree, so the comparison above has meaning."""
        self.assertEqual(build_briefing(self.ds), self.reference)


class TestMetricRegistry(unittest.TestCase):
    """``METRIC_REGISTRY`` / ``register_metric`` are the metric extension point."""

    def test_metric_registry_is_the_same_object_as_metrics(self) -> None:
        """The public alias must not be a copy, or registrations would not be seen."""
        self.assertIs(METRIC_REGISTRY, METRICS)

    def test_the_sixteen_original_metrics_are_still_registered(self) -> None:
        """TASK-012 was a refactor: no metric was added, moved or lost."""
        self.assertEqual(len(METRIC_REGISTRY), 16)
        for name, definition in METRIC_REGISTRY.items():
            self.assertIsInstance(definition, MetricDef)
            self.assertTrue(definition.units, f"{name} has no units")
            self.assertTrue(definition.source, f"{name} has no provenance")

    def test_registering_a_metric_updates_the_sorted_name_view(self) -> None:
        """A metric added from outside this module reaches METRIC_NAMES immediately.

        This is what lets a new metric appear in the generator's allowed-value list and
        pass the harness's admissibility check without either of them being edited.
        """
        import llm_insights.summary.metrics as metrics_module

        saved_registry = dict(metrics_module.METRIC_REGISTRY)
        saved_names = metrics_module.METRIC_NAMES
        try:

            @register_metric(
                "aaa_scratch_metric",
                "dimensionless",
                "A scratch metric registered by a test.",
                "test_registry.py",
            )
            def scratch(ds: Any, case: str) -> float:
                """Return a constant."""
                del ds, case
                return 42.0

            self.assertIn("aaa_scratch_metric", metrics_module.METRIC_NAMES)
            # Sorted, and the new name sorts first, so this also proves it is re-sorted
            # rather than appended.
            self.assertEqual(metrics_module.METRIC_NAMES[0], "aaa_scratch_metric")
            self.assertEqual(
                list(metrics_module.METRIC_NAMES), sorted(metrics_module.METRIC_REGISTRY)
            )
            self.assertEqual(metric_value(_StubDataset(), "aaa_scratch_metric", "Healthy"), 42.0)
        finally:
            metrics_module.METRIC_REGISTRY.clear()
            metrics_module.METRIC_REGISTRY.update(saved_registry)
            metrics_module.METRIC_NAMES = saved_names

    def test_registering_a_metric_twice_raises(self) -> None:
        """Overwriting would change what an already-written hypothesis is tested on."""
        with self.assertRaises(RuntimeError):
            register_metric("wss_peak_dyn_cm2", "x", "y", "z")(lambda *_: 0.0)


if __name__ == "__main__":  # pragma: no cover - test entry point
    unittest.main()
