"""TASK-014: the gene-network briefing section says what the network actually does.

This is the TASK-005 answer-preservation pattern applied to structure rather than to
fields. Every question below is answered twice -- once from the model and the full
per-point arrays, once from **the rendered briefing section text and the registered
metrics alone** -- and the two answers must agree. If they ever stop agreeing, the
briefing is telling a reader something the simulation does not do, which is the failure
mode this whole project exists to prevent.

Two of the raw answerers are worth describing, because they are the reason the
structural questions are checks rather than assertions of belief:

* :func:`_direct_readers` finds which nodes an input drives *directly* by evaluating
  ``networkpoint.rhs`` twice at the same state with only that input changed, at several
  random states. A node whose derivative moves is reading the input itself, not through
  a neighbour. That is how "mechanical stress reaches exactly one node" is verified
  instead of asserted.
* :func:`_regulators_of` does the same in the state variables, so "EndMT is a readout of
  Snai1 and Snai2 with no dynamics of its own" is a computed fact.

The negative control at the bottom corrupts the section text and requires the suite to
notice. Without it, a suite of questions whose two sides happened to call the same code
would pass while checking nothing.
"""

from __future__ import annotations

import re
import sys
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_insights.grn import runner, store  # noqa: E402
from llm_insights.io.dataset import Dataset  # noqa: E402
from llm_insights.summary.metrics import METRIC_REGISTRY, metric_value  # noqa: E402
from llm_insights.summary.sections.grn import PATHWAYS, grn_section  # noqa: E402


def _resolve_root() -> Path:
    """Find the model tree, preferring the checkout these tests live in.

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
HAS_MODEL = (ROOT / "networkpoint.py").is_file()
CASES = ("flow_U0p0180", "flow_U0p0360", "flow_U0p0540")
HAS_GRN_FULL = store.available(CASES)

#: The briefing's hard ceiling, from TASK-014.
BRIEFING_CHAR_CEILING = 20_000


# --- Raw-side oracles --------------------------------------------------------------


def _rhs_at(state: np.ndarray, shear: float, mech: float) -> np.ndarray:
    """Evaluate ``networkpoint.rhs`` at one state and drive.

    Args:
        state: The 24-vector.
        shear: Normalized shear.
        mech: Normalized mechanical stress.

    Returns:
        The derivative vector.
    """
    networkpoint = runner.load_networkpoint(ROOT)
    return networkpoint.rhs(0.0, state, shear, mech, networkpoint.Params())


def _direct_readers(which: str) -> frozenset[str]:
    """Find the nodes one physical input drives directly.

    Holds the state fixed and changes only the named input. Any node whose derivative
    moves is reading that input itself; a node that only responds through a neighbour
    cannot move, because the neighbour's value did not change. Several random states are
    probed so that a node whose response happens to vanish at one state is still found.

    Args:
        which: ``"shear"`` or ``"mech"``.

    Returns:
        The node names, as a frozen set.
    """
    networkpoint = runner.load_networkpoint(ROOT)
    nodes = list(networkpoint.NODES)
    rng = np.random.default_rng(20260908)
    hits: set[str] = set()
    for _ in range(16):
        state = rng.uniform(0.05, 0.95, len(nodes))
        lo = (0.2, 0.5) if which == "shear" else (0.5, 0.2)
        hi = (0.8, 0.5) if which == "shear" else (0.5, 0.8)
        delta = np.abs(_rhs_at(state, *lo) - _rhs_at(state, *hi))
        hits |= {nodes[i] for i in np.flatnonzero(delta > 1e-12)}
    return frozenset(hits)


def _regulators_of(node: str) -> frozenset[str]:
    """Find which state variables a node's derivative depends on.

    Args:
        node: The node whose regulators are wanted.

    Returns:
        Node names whose value changes ``d<node>/dt``, including the node itself (every
        node decays toward its own target, so it always appears).
    """
    networkpoint = runner.load_networkpoint(ROOT)
    nodes = list(networkpoint.NODES)
    index = nodes.index(node)
    rng = np.random.default_rng(20260908)
    hits: set[str] = set()
    for _ in range(16):
        base = rng.uniform(0.05, 0.95, len(nodes))
        reference = _rhs_at(base, 0.4, 0.6)[index]
        for j, name in enumerate(nodes):
            probe = base.copy()
            probe[j] = 0.95 if base[j] < 0.5 else 0.05
            if abs(_rhs_at(probe, 0.4, 0.6)[index] - reference) > 1e-12:
                hits.add(name)
    return frozenset(hits)


def _activity(case: str, node: str, scenario: str = "combined") -> np.ndarray:
    """Read one node's full per-point array straight from TASK-013's table.

    Args:
        case: Case folder name.
        node: Node name.
        scenario: Scenario name.

    Returns:
        The activities, NaN included.
    """
    return store.load_activities(case)[f"{node}_{scenario}"].to_numpy(dtype=float)


def _argmax_case(values: dict[str, float]) -> str:
    """Name of the case with the largest value.

    Args:
        values: Case name to value.

    Returns:
        The winning case name.
    """
    return max(values, key=lambda k: values[k])


# --- Summary-side parsers ----------------------------------------------------------
#
# These read the rendered section the way a language model would: as text. Nothing here
# touches the arrays, which is the whole point -- if a parser can answer the question,
# the briefing carried the answer.


def _section_text(ds: Dataset) -> str:
    """Render the GRN briefing section.

    Args:
        ds: The dataset to describe.

    Returns:
        The section as one string.
    """
    return "\n".join(grn_section(ds, 25))


def _stated_node_count(text: str) -> int:
    """The node count the section states.

    Args:
        text: The section text.

    Returns:
        The integer in ``**N-state**``.
    """
    return int(re.search(r"\*\*(\d+)-state\*\*", text).group(1))


def _stated_params(text: str) -> dict[str, float]:
    """The parameter values the section lists.

    Args:
        text: The section text.

    Returns:
        Parameter name to value.
    """
    line = next(ln for ln in text.splitlines() if ln.startswith("**Parameters**"))
    return {name: float(value) for name, value in re.findall(r"`(\w+)`=([-\d.eE+]+)", line)}


def _stated_ceilings(text: str) -> tuple[float, float]:
    """The two normalization ceilings the section states.

    Args:
        text: The section text.

    Returns:
        ``(shear ceiling in dyn/cm^2, mech ceiling in Pa)``.
    """
    shear = float(re.search(r"shear against ([\d.]+) dyn/cm\^2", text).group(1))
    mech = float(re.search(r"stress against ([\d.]+) Pa", text).group(1))
    return shear, mech


def _stated_t_end(text: str) -> float:
    """The integration horizon the section states.

    Args:
        text: The section text.

    Returns:
        Hours.
    """
    return float(re.search(r"Integrated to t = ([\d.]+) h", text).group(1))


def _bullet(text: str, opener: str) -> str:
    """Return one wrapped bullet of the section, joined onto a single line.

    Args:
        text: The section text.
        opener: How the bullet starts, e.g. ``"- **Shear**"``.

    Returns:
        The bullet's text.
    """
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(opener))
    out = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("- ") or not line.strip():
            break
        out.append(line)
    return " ".join(out)


def _nodes_named_in(fragment: str) -> frozenset[str]:
    """Which node names appear as whole words in a fragment of the section.

    Args:
        fragment: Section text.

    Returns:
        The node names found.
    """
    nodes = runner.node_names(ROOT)
    return frozenset(n for n in nodes if re.search(rf"\b{re.escape(n)}\b", fragment))


def _stated_pathways(text: str) -> dict[str, tuple[str, ...]]:
    """The pathway groupings the section lists.

    Args:
        text: The section text.

    Returns:
        Pathway name to node names.
    """
    out: dict[str, tuple[str, ...]] = {}
    for line in text.splitlines():
        match = re.match(r"- \*\*(\w+)\*\* \(\d+\): (.+)$", line)
        if match:
            out[match.group(1)] = tuple(n.strip() for n in match.group(2).split(","))
    return out


def _stated_constant_nodes(text: str) -> dict[str, float]:
    """The nodes the section calls pinned by construction, and their values.

    Args:
        text: The section text.

    Returns:
        Node name to pinned value; empty if the statistics are absent.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("**Pinned by construction"))
    except StopIteration:
        return {}
    body = lines[start + 1].strip().rstrip(".")
    return {
        name: float(value)
        for name, value in (item.strip().split(" = ") for item in body.split(", "))
    }


def _stated_table(text: str, heading: str) -> list[tuple[str, float]]:
    """Read a ranked node table out of the section.

    Args:
        text: The section text.
        heading: The bold heading the table follows.

    Returns:
        ``(node, statistic)`` in the order the table lists them; empty if absent.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith(heading))
    except StopIteration:
        return []
    out: list[tuple[str, float]] = []
    for line in lines[start:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("node", "") or set(cells[1]) <= {"-"}:
            if out:
                break
            continue
        out.append((cells[0], float(cells[1])))
    return out


# --- The question table ------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One question answered from the full arrays and from the section alone.

    Attributes:
        key: Short identifier used in test output.
        text: The question in plain English.
        raw: Answers from the model source and the per-point arrays.
        summary: Answers from the rendered section text and the metric registry.
        needs_data: True if the question needs TASK-013's full-node tables.
        tol: Absolute tolerance for numeric answers; None means exact equality.
    """

    key: str
    text: str
    raw: Callable[[Dataset], Any]
    summary: Callable[[Dataset, str], Any]
    needs_data: bool = False
    tol: float | None = None


def _build_questions() -> list[Question]:
    """Assemble the GRN answer-preservation question set.

    Returns:
        The questions.
    """
    q: list[Question] = []

    # --- Structure: answerable with no re-run at all ---
    q.append(
        Question(
            "node_count",
            "How many state variables does the gene network have?",
            lambda ds: len(runner.node_names(ROOT)),  # noqa: ARG005 - uniform signature
            lambda ds, t: _stated_node_count(t),  # noqa: ARG005
        )
    )
    q.append(
        Question(
            "node_roster",
            "Which nodes are in the network?",
            lambda ds: frozenset(runner.node_names(ROOT)),  # noqa: ARG005
            lambda ds, t: frozenset(  # noqa: ARG005
                n for nodes in _stated_pathways(t).values() for n in nodes
            ),
        )
    )
    q.append(
        Question(
            "mech_direct_readers",
            "Which nodes does mechanical stress drive directly?",
            lambda ds: _direct_readers("mech"),  # noqa: ARG005
            lambda ds, t: _nodes_named_in(_bullet(t, "- **Mechanical stress**")),  # noqa: ARG005
        )
    )
    q.append(
        Question(
            "shear_direct_readers",
            "Which nodes does wall shear stress drive directly?",
            lambda ds: _direct_readers("shear"),  # noqa: ARG005
            lambda ds, t: _nodes_named_in(_bullet(t, "- **Shear**")),  # noqa: ARG005
        )
    )
    q.append(
        Question(
            "how_many_mech_readers",
            "How many nodes read mechanical stress directly?",
            lambda ds: len(_direct_readers("mech")),  # noqa: ARG005
            lambda ds, t: len(_nodes_named_in(_bullet(t, "- **Mechanical stress**"))),  # noqa: ARG005
        )
    )
    q.append(
        Question(
            "endmt_regulators",
            "What regulates EndMT?",
            lambda ds: _regulators_of("EndMT"),  # noqa: ARG005
            lambda ds, t: _nodes_named_in(_bullet(t, "- **Outputs**")),  # noqa: ARG005
        )
    )
    q.append(
        Question(
            "hill_coefficient",
            "What Hill coefficient does the network use?",
            lambda ds: runner.load_networkpoint(ROOT).Params().n,  # noqa: ARG005
            lambda ds, t: _stated_params(t)["n"],  # noqa: ARG005
        )
    )
    q.append(
        Question(
            "input_half_activation",
            "At what normalized drive is a physical input half-activating?",
            lambda ds: runner.load_networkpoint(ROOT).Params().ec50_input,  # noqa: ARG005
            lambda ds, t: _stated_params(t)["ec50_input"],  # noqa: ARG005
        )
    )
    q.append(
        Question(
            "all_params",
            "What are every one of the network's parameter values?",
            lambda ds: {  # noqa: ARG005
                k: float(v)
                for k, v in vars(runner.load_networkpoint(ROOT).Params()).items()
            },
            lambda ds, t: _stated_params(t),  # noqa: ARG005
        )
    )
    q.append(
        Question(
            "slowest_time_constant",
            "Which time constant is the slowest, and what is it?",
            lambda ds: max(  # noqa: ARG005
                (v, k)
                for k, v in vars(runner.load_networkpoint(ROOT).Params()).items()
                if k.startswith("tau_")
            ),
            lambda ds, t: max(  # noqa: ARG005
                (v, k) for k, v in _stated_params(t).items() if k.startswith("tau_")
            ),
        )
    )
    q.append(
        Question(
            "normalization_ceilings",
            "What are the shear and mechanical-stress normalization ceilings?",
            lambda ds: (  # noqa: ARG005
                runner.load_networkpoint(ROOT).SHEAR_MAX,
                runner.load_networkpoint(ROOT).MECH_MAX,
            ),
            lambda ds, t: _stated_ceilings(t),  # noqa: ARG005
        )
    )
    q.append(
        Question(
            "integration_horizon",
            "How long is the network integrated for?",
            lambda ds: runner.T_END,  # noqa: ARG005
            lambda ds, t: _stated_t_end(t),  # noqa: ARG005
        )
    )
    q.append(
        Question(
            "half_maximal_shear_dyn",
            "How many dyn/cm^2 of shear is half-maximal drive?",
            lambda ds: (  # noqa: ARG005
                runner.load_networkpoint(ROOT).SHEAR_MAX
                * runner.load_networkpoint(ROOT).Params().ec50_input
            ),
            lambda ds, t: _stated_ceilings(t)[0] * _stated_params(t)["ec50_input"],  # noqa: ARG005
            tol=1e-9,
        )
    )

    # --- Statistics: these need TASK-013's re-run ---
    q.append(
        Question(
            "constant_nodes",
            "Which nodes are the same everywhere, in every case?",
            lambda ds: frozenset(  # noqa: ARG005
                n
                for n, s in store.summarize_nodes(CASES).items()
                if s.is_constant
            ),
            lambda ds, t: frozenset(_stated_constant_nodes(t)),  # noqa: ARG005
            needs_data=True,
        )
    )
    q.append(
        Question(
            "constant_node_count",
            "How many nodes are pinned by construction?",
            lambda ds: sum(  # noqa: ARG005
                1
                for case_node in runner.node_names(ROOT)
                if float(np.nanstd(_activity(CASES[0], case_node))) < store.CONSTANT_TOL
                and max(
                    abs(
                        float(np.nanmean(_activity(a, case_node)))
                        - float(np.nanmean(_activity(b, case_node)))
                    )
                    for a in CASES
                    for b in CASES
                )
                < store.CONSTANT_TOL
            ),
            lambda ds, t: len(_stated_constant_nodes(t)),  # noqa: ARG005
            needs_data=True,
        )
    )
    q.append(
        Question(
            "rbpj_value",
            "What activity does RBPJ hold?",
            lambda ds: float(np.nanmean(_activity(CASES[1], "RBPJ"))),  # noqa: ARG005
            lambda ds, t: _stated_constant_nodes(t)["RBPJ"],  # noqa: ARG005
            needs_data=True,
            tol=1e-4,
        )
    )
    q.append(
        Question(
            "most_spatially_variable_node",
            "Which node varies most across the cushion?",
            lambda ds: max(  # noqa: ARG005
                runner.node_names(ROOT),
                key=lambda n: float(
                    np.mean([np.nanstd(_activity(c, n)) for c in CASES])
                ),
            ),
            lambda ds, t: _stated_table(t, "**Most spatially variable")[0][0],  # noqa: ARG005
            needs_data=True,
        )
    )
    q.append(
        Question(
            "most_flow_sensitive_node",
            "Which node's domain-mean changes most across the three inlet velocities?",
            lambda ds: max(  # noqa: ARG005
                runner.node_names(ROOT),
                key=lambda n: (
                    max(float(np.nanmean(_activity(c, n))) for c in CASES)
                    - min(float(np.nanmean(_activity(c, n))) for c in CASES)
                ),
            ),
            lambda ds, t: _stated_table(t, "**Most flow-sensitive")[0][0],  # noqa: ARG005
            needs_data=True,
        )
    )
    q.append(
        Question(
            "is_smad23_constant",
            "Is SMAD23 the same everywhere, or does it vary?",
            lambda ds: float(np.nanstd(_activity(CASES[1], "SMAD23")))  # noqa: ARG005
            < store.CONSTANT_TOL,
            lambda ds, t: "SMAD23" in _stated_constant_nodes(t),  # noqa: ARG005
            needs_data=True,
        )
    )
    q.append(
        Question(
            "smad23_mean_healthy",
            "What is the mean SMAD23 activity in the healthy case?",
            lambda ds: float(np.nanmean(_activity("flow_U0p0360", "SMAD23"))),  # noqa: ARG005
            lambda ds, t: metric_value(ds, "smad23_mean_combined", "flow_U0p0360"),  # noqa: ARG005
            needs_data=True,
            tol=1e-9,
        )
    )
    q.append(
        Question(
            "highest_nicd_case",
            "Which case has the highest mean NICD activity?",
            lambda ds: _argmax_case(  # noqa: ARG005
                {c: float(np.nanmean(_activity(c, "NICD"))) for c in CASES}
            ),
            lambda ds, t: _argmax_case(  # noqa: ARG005
                {c: metric_value(ds, "nicd_mean_combined", c) for c in CASES}
            ),
            needs_data=True,
        )
    )
    q.append(
        Question(
            "yap_taz_monotonic_with_flow",
            "Does mean YAP/TAZ rise monotonically with inlet velocity?",
            lambda ds: bool(  # noqa: ARG005
                np.all(np.diff([float(np.nanmean(_activity(c, "YAP_TAZ"))) for c in CASES]) > 0)
            ),
            lambda ds, t: bool(  # noqa: ARG005
                np.all(np.diff([metric_value(ds, "yap_taz_mean_combined", c) for c in CASES]) > 0)
            ),
            needs_data=True,
        )
    )
    q.append(
        Question(
            "rank_cases_by_smad67",
            "Rank the three cases by mean inhibitory SMAD6/7.",
            lambda ds: tuple(  # noqa: ARG005
                sorted(CASES, key=lambda c: float(np.nanmean(_activity(c, "SMAD67"))))
            ),
            lambda ds, t: tuple(  # noqa: ARG005
                sorted(CASES, key=lambda c: metric_value(ds, "smad67_mean_combined", c))
            ),
            needs_data=True,
        )
    )
    q.append(
        Question(
            "bmp_repressed_by_flow",
            "Is mean BMP_2456 lower in the overflow case than in the underflow case?",
            lambda ds: bool(  # noqa: ARG005
                float(np.nanmean(_activity("flow_U0p0540", "BMP_2456")))
                < float(np.nanmean(_activity("flow_U0p0180", "BMP_2456")))
            ),
            lambda ds, t: bool(  # noqa: ARG005
                metric_value(ds, "bmp2456_mean_combined", "flow_U0p0540")
                < metric_value(ds, "bmp2456_mean_combined", "flow_U0p0180")
            ),
            needs_data=True,
        )
    )
    q.append(
        Question(
            "smad23_spread_healthy",
            "How spread out is SMAD23 across the healthy cushion?",
            lambda ds: float(np.nanstd(_activity("flow_U0p0360", "SMAD23"))),  # noqa: ARG005
            lambda ds, t: metric_value(ds, "smad23_sd_combined", "flow_U0p0360"),  # noqa: ARG005
            needs_data=True,
            tol=1e-9,
        )
    )
    return q


QUESTIONS = _build_questions()


def _compare(question: Question, raw: Any, summ: Any) -> str | None:
    """Compare one question's two answers.

    Args:
        question: The question, for its tolerance.
        raw: The array-side answer.
        summ: The briefing-side answer.

    Returns:
        A failure description, or None if they agree.
    """
    if question.tol is None:
        return None if raw == summ else f"{question.key}: raw={raw!r} summary={summ!r}"
    delta = abs(float(raw) - float(summ))
    return None if delta <= question.tol else f"{question.key}: |raw-summary|={delta:.4g}"


@unittest.skipUnless(HAS_MODEL and HAS_DATA, "one_way_fsg_model not mounted")
class TestGrnAnswerPreservation(unittest.TestCase):
    """Every GRN question gets the same answer from the section as from the arrays."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = Dataset(ROOT)
        cls.text = _section_text(cls.ds)

    def test_question_set_is_large_enough(self):
        # TASK-014 asks for at least 12.
        self.assertGreaterEqual(len(QUESTIONS), 12, "GRN answer-preservation set has shrunk")

    def test_structural_questions_do_not_need_the_rerun(self):
        # The topology half of the section must be answerable in a fresh checkout.
        self.assertGreaterEqual(sum(1 for q in QUESTIONS if not q.needs_data), 12)

    def test_all_questions_agree(self):
        failures: list[str] = []
        skipped = 0
        for q in QUESTIONS:
            if q.needs_data and not HAS_GRN_FULL:
                skipped += 1
                continue
            with self.subTest(question=q.key):
                problem = _compare(q, q.raw(self.ds), q.summary(self.ds, self.text))
                if problem:
                    failures.append(problem)
                self.assertIsNone(problem, q.text)
        self.assertEqual(failures, [], "GRN answer-preservation failures")
        if skipped:
            print(
                f"\n  note: {skipped} of {len(QUESTIONS)} GRN questions were skipped -- "
                "data/grn_full is absent; run `python3 -m llm_insights.grn.runner`"
            )

    def test_the_suite_is_not_vacuous(self):
        """A corrupted section must make at least one question disagree.

        Every question above answers from parsed text on one side and from the model or
        the arrays on the other, but a suite can still be vacuous by accident -- if the
        parser were lenient enough, or the two sides quietly shared a code path, wrong
        text would pass. So: damage the text in four ways a real editing mistake could
        damage it, and require the suite to catch each one.
        """
        corruptions = {
            "wrong node count": lambda t: t.replace("**24-state**", "**23-state**"),
            "wrong Hill coefficient": lambda t: t.replace("`n`=2,", "`n`=3,"),
            "wrong ceiling": lambda t: t.replace(
                "shear against 30 dyn/cm^2", "shear against 25 dyn/cm^2"
            ),
            "a node dropped from the roster": lambda t: t.replace(", SMAD23", ""),
        }
        for label, corrupt in corruptions.items():
            with self.subTest(corruption=label):
                bad = corrupt(self.text)
                self.assertNotEqual(bad, self.text, f"corruption {label!r} changed nothing")
                caught = [
                    q.key
                    for q in QUESTIONS
                    if not q.needs_data
                    and _compare(q, q.raw(self.ds), q.summary(self.ds, bad)) is not None
                ]
                self.assertTrue(caught, f"no question noticed: {label}")


@unittest.skipUnless(HAS_MODEL and HAS_DATA, "one_way_fsg_model not mounted")
class TestSectionIntegrity(unittest.TestCase):
    """The section's own claims about itself hold."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = Dataset(ROOT)
        cls.text = _section_text(cls.ds)

    def test_pathway_grouping_partitions_the_node_set(self):
        listed = [n for _, nodes in PATHWAYS for n in nodes]
        self.assertEqual(sorted(listed), sorted(runner.node_names(ROOT)))
        self.assertEqual(len(listed), len(set(listed)), "a node is listed under two pathways")

    def test_it_supersedes_the_stale_caveat(self):
        # Caveat 4 still says only 5 of 23 nodes were written out. That file is frozen,
        # so this section has to say so out loud or the briefing contradicts itself.
        self.assertIn("supersedes caveat 4", self.text)

    def test_briefing_stays_under_the_ceiling(self):
        from llm_insights.summary.briefing import build_briefing

        text = build_briefing(self.ds)
        self.assertLess(len(text), BRIEFING_CHAR_CEILING, "briefing exceeded its char ceiling")
        self.assertLess(len(text.encode()), BRIEFING_CHAR_CEILING, "briefing exceeded it in bytes")

    def test_the_section_is_in_the_briefing(self):
        from llm_insights.summary.briefing import build_briefing

        self.assertIn("## 4b. The gene network", build_briefing(self.ds))


class TestNewMetricsAreWellFormed(unittest.TestCase):
    """Every metric this task adds carries units and provenance and reaches the generator."""

    NEW = (
        "beta_catenin_mean_combined",
        "bmp2456_mean_combined",
        "hey_mean_combined",
        "nicd_mean_combined",
        "smad23_mean_combined",
        "smad23_sd_combined",
        "smad67_mean_combined",
        "yap_taz_mean_combined",
    )

    def test_all_registered(self):
        for name in self.NEW:
            self.assertIn(name, METRIC_REGISTRY, name)

    def test_units_and_provenance(self):
        for name in self.NEW:
            md = METRIC_REGISTRY[name]
            self.assertTrue(md.units, name)
            self.assertIn("activities.csv", md.source, name)
            self.assertIn("GRN grid", md.source, name)
            self.assertTrue(md.description.strip().endswith("."), name)

    def test_they_reach_the_generators_allowed_value_list(self):
        """Confirmed against the rendered prompt, not against the registry twice over.

        TASK-014 criterion 4 says to confirm rather than assume, and asserting that a
        registered name is in the registry would confirm nothing. So the actual system
        prompt is built and the names are looked for in its text -- the same string a
        model is handed.
        """
        from llm_insights.agent.generator import build_system_prompt

        prompt = build_system_prompt()
        for name in self.NEW:
            self.assertIn(name, prompt, f"{name} never reaches the model")

    def test_the_harness_admits_them(self):
        """The metric primitive accepts them, so a hypothesis about one can be tested."""
        from llm_insights.summary.metrics import METRIC_NAMES

        for name in self.NEW:
            # primitives.metric() rejects anything not in METRIC_NAMES; that tuple is
            # rebound by register_metric, which is the mechanism under test here.
            self.assertIn(name, METRIC_NAMES, name)


if __name__ == "__main__":
    unittest.main()
