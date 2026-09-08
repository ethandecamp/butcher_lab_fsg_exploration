"""Read the full 24-node activity tables TASK-013 writes, and summarize them.

``runner.py`` produces ``data/grn_full/<case>/activities.csv``: ~8,900 rows by 78
columns per case, about 12 MB each. That is regenerable output, so it is gitignored and
**may simply not be there** -- in a fresh checkout, in CI, or before anybody has run
``grn-rerun``. Every function here is written for that: :func:`available` says whether
the store exists, and the summaries return empty rather than raising, so a briefing
built without it is a briefing missing some numbers rather than a traceback.

The summaries are deliberately few. There are 24 nodes x 3 scenarios x 3 cases of raw
material here and the briefing cannot carry it; what it can carry is which nodes vary
in space, which vary with inlet flow, and which are constant by construction. Those
three questions are what :class:`NodeSummary` answers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from llm_insights.grn.runner import INPUT_COLUMNS, SCENARIO_FLAGS, activities_path

LOG = logging.getLogger(__name__)

#: Environment variable overriding where the full-node tables are looked for.
GRN_FULL_ENV: Final[str] = "LLM_INSIGHTS_GRN_FULL"

#: Below this spread, a node is called constant. The constant nodes are constant by
#: construction -- ``MAML_inf = 1.0`` and friends -- so the only spread they can show is
#: solver noise, which lands many orders of magnitude below this.
CONSTANT_TOL: Final[float] = 1e-6

#: The nodes ``networkinput.py`` already wrote. Everything else was recomputed by
#: TASK-013 and had never been on disk before.
SHIPPED_NODES: Final[tuple[str, ...]] = ("Snai1", "Snai2", "EndMT", "NICD", "YAP_TAZ")

#: The scenario the summaries describe unless told otherwise. It is the only one where
#: both drives are the point's own.
REFERENCE_SCENARIO: Final[str] = "combined"


def default_dir() -> Path:
    """Return where ``activities.csv`` files are looked for.

    Returns:
        ``$LLM_INSIGHTS_GRN_FULL`` if set, else ``<package root>/data/grn_full``, which
        is where ``grn-rerun`` writes when run from ``llm_insights/`` as documented.
    """
    override = os.environ.get(GRN_FULL_ENV)
    if override:
        return Path(override)
    # .../llm_insights/src/llm_insights/grn/store.py -> .../llm_insights
    return Path(__file__).resolve().parents[3] / "data" / "grn_full"


def available(cases: tuple[str, ...], directory: Path | None = None) -> bool:
    """Say whether every named case has a full-node table on disk.

    Args:
        cases: Case folder names.
        directory: Where to look. Defaults to :func:`default_dir`.

    Returns:
        True only if all of them are present; a partial store is treated as absent
        because a summary built from two cases out of three would quietly describe a
        different dataset than it claims to.
    """
    root = directory or default_dir()
    return all(activities_path(root, case).is_file() for case in cases)


@lru_cache(maxsize=8)
def _read(path: str) -> pd.DataFrame:
    """Read and cache one activities table.

    Args:
        path: Absolute path to an ``activities.csv``.

    Returns:
        The table.
    """
    frame = pd.read_csv(path)
    LOG.info("read %s (%d rows, %d columns)", path, len(frame), frame.shape[1])
    return frame


def load_activities(case: str, directory: Path | None = None) -> pd.DataFrame:
    """Load one case's full-node activity table.

    Args:
        case: Case folder name, e.g. ``"flow_U0p0360"``.
        directory: Where to look. Defaults to :func:`default_dir`.

    Returns:
        The table, cached across calls.

    Raises:
        FileNotFoundError: If the case has no table. Call :func:`available` first if
            the caller needs to degrade instead.
    """
    path = activities_path(directory or default_dir(), case)
    if not path.is_file():
        raise FileNotFoundError(
            f"no full-node GRN table at {path}; regenerate it with "
            "`PYTHONPATH=src python3 -m llm_insights.grn.runner --out data/grn_full`"
        )
    return _read(str(path))


def node_list(case: str, directory: Path | None = None) -> tuple[str, ...]:
    """Return the node names present in a case's table, in column order.

    Args:
        case: Case folder name.
        directory: Where to look.

    Returns:
        The node names.
    """
    frame = load_activities(case, directory)
    suffix = f"_{REFERENCE_SCENARIO}"
    return tuple(c[: -len(suffix)] for c in frame.columns if c.endswith(suffix))


@dataclass(frozen=True)
class NodeSummary:
    """What one node does across space, across scenarios and across the three cases.

    Attributes:
        node: Node name.
        mean_by_case: Domain-mean activity under :data:`REFERENCE_SCENARIO`, per case.
        sd_by_case: Spatial standard deviation under the same scenario, per case.
        scenario_mean: Domain-mean activity per scenario, averaged over the cases.
        spatial_sd: ``sd_by_case`` averaged over cases. The "how much does this node
            vary across the cushion" number.
        flow_span: Largest minus smallest ``mean_by_case``. The "how much does this
            node care about inlet flow" number. **n = 3**, always.
        is_constant: True when both ``spatial_sd`` and ``flow_span`` are below
            :data:`CONSTANT_TOL`, i.e. the node is pinned by construction.
        constant_value: The pinned value when ``is_constant``, else NaN.
    """

    node: str
    mean_by_case: dict[str, float]
    sd_by_case: dict[str, float]
    scenario_mean: dict[str, float]
    spatial_sd: float
    flow_span: float
    is_constant: bool
    constant_value: float


def summarize_nodes(
    cases: tuple[str, ...], directory: Path | None = None
) -> dict[str, NodeSummary]:
    """Summarize every node over every case.

    Args:
        cases: Case folder names, in the order the briefing lists them.
        directory: Where to look. Defaults to :func:`default_dir`.

    Returns:
        Node name to :class:`NodeSummary`, or an empty mapping if the store is absent.
        Empty rather than raising: a missing regenerable artifact must degrade the
        briefing, not break it.
    """
    if not available(cases, directory):
        LOG.warning("no full-node GRN store under %s; node statistics omitted", directory or default_dir())
        return {}

    frames = {case: load_activities(case, directory) for case in cases}
    nodes = node_list(cases[0], directory)
    out: dict[str, NodeSummary] = {}
    for node in nodes:
        mean_by_case = {}
        sd_by_case = {}
        for case, frame in frames.items():
            values = frame[f"{node}_{REFERENCE_SCENARIO}"].to_numpy(dtype=float)
            mean_by_case[case] = float(np.nanmean(values))
            sd_by_case[case] = float(np.nanstd(values))
        scenario_mean = {
            scenario: float(
                np.mean(
                    [
                        np.nanmean(frames[c][f"{node}_{scenario}"].to_numpy(dtype=float))
                        for c in cases
                    ]
                )
            )
            for scenario in SCENARIO_FLAGS
        }
        spatial_sd = float(np.mean(list(sd_by_case.values())))
        means = list(mean_by_case.values())
        flow_span = float(max(means) - min(means))
        is_constant = spatial_sd < CONSTANT_TOL and flow_span < CONSTANT_TOL
        out[node] = NodeSummary(
            node=node,
            mean_by_case=mean_by_case,
            sd_by_case=sd_by_case,
            scenario_mean=scenario_mean,
            spatial_sd=spatial_sd,
            flow_span=flow_span,
            is_constant=is_constant,
            constant_value=float(np.mean(means)) if is_constant else float("nan"),
        )
    return out


def node_statistic(
    case: str,
    node: str,
    statistic: str,
    scenario: str = REFERENCE_SCENARIO,
    directory: Path | None = None,
) -> float:
    """Compute one scalar for one node, one case and one scenario.

    This is what the registered metrics call. It returns NaN rather than raising when
    the store is absent, because a metric that raises takes the whole briefing down
    with it, and the store is regenerable output that may legitimately not be there.

    Args:
        case: Case folder name.
        node: Node name, e.g. ``"SMAD23"``.
        statistic: ``"mean"`` or ``"sd"``, computed over grid points, ignoring NaN.
        scenario: One of the three scenarios.
        directory: Where to look.

    Returns:
        The value, or NaN if the store or the column is missing.

    Raises:
        ValueError: If ``statistic`` is not one of the two.
    """
    if statistic not in ("mean", "sd"):
        raise ValueError(f"statistic must be 'mean' or 'sd', got {statistic!r}")
    try:
        frame = load_activities(case, directory)
    except FileNotFoundError:
        return float("nan")
    column = f"{node}_{scenario}"
    if column not in frame.columns:
        LOG.warning("no column %s in the full-node table for %s", column, case)
        return float("nan")
    values = frame[column].to_numpy(dtype=float)
    return float(np.nanmean(values) if statistic == "mean" else np.nanstd(values))


__all__ = [
    "CONSTANT_TOL",
    "GRN_FULL_ENV",
    "INPUT_COLUMNS",
    "REFERENCE_SCENARIO",
    "SHIPPED_NODES",
    "NodeSummary",
    "available",
    "default_dir",
    "load_activities",
    "node_list",
    "node_statistic",
    "summarize_nodes",
]
