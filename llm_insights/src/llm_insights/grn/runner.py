"""Re-run the GRN at every spatial point and keep all 24 nodes.

What this module reproduces, and why it reproduces rather than imports
---------------------------------------------------------------------

``one_way_fsg_model/networkinput.py`` cannot be imported. Its first statements do work:
it reads ``/Users/sophiewang/Desktop/...``, a path that exists on nobody's machine, and
then runs ~27,000 ODE solves at import time. So only ``networkpoint`` is imported here
-- the solver itself -- and ``networkinput``'s *wiring* is restated below as data:
:data:`SCENARIO_FLAGS`, :data:`T_END`, :data:`SHEAR_MID_NORM`, :data:`MECH_MID_NORM`.

Restating it is a duplication, and duplication drifts. ``tests/test_grn_runner.py``
parses ``networkinput.py`` with :mod:`ast` and asserts every one of those constants
still matches the file it was copied from, so drift fails a test rather than silently
producing numbers that no longer mean what they claim to.

The three scenarios, in ``networkinput.py``'s own terms
------------------------------------------------------

============  ==========  =========  ================================================
scenario      use_shear   use_mech   what a point is given
============  ==========  =========  ================================================
shear_only    True        False      its own WSS; mech pinned at 40 Pa. Interior
                                     points (no WSS) return NaN and are not solved.
mech_only     False       True       its own von Mises; shear pinned at 5 dyn/cm^2.
combined      True        True       its own WSS *or 0.0 if it has none*, and its own
                                     von Mises.
============  ==========  =========  ================================================

The asymmetry in the last two rows is Dan's, not a transcription error: a point with no
WSS is skipped entirely under ``shear_only`` but is driven at zero shear under
``combined``.

Cost, and the two things that make it affordable
------------------------------------------------

Naively this is 3 cases x 3 scenarios x ~8,900 points = ~80,000 solves. Two exact
reductions apply, neither of which changes a single output value:

1. ``shear_only`` solves only the surface points (609 / 655 / 640 of them); the rest are
   NaN by construction.
2. Inputs repeat. The solver is deterministic, so identical ``(shear, mech)`` pairs are
   solved once and the answer is scattered back. Pairs are compared bit-exactly, so this
   is caching, not approximation.

Solver tolerances (``rtol=1e-8``, ``atol=1e-10``) and ``t_end`` are **not** tuning
knobs. They are what the shipped results were produced with, and the faithfulness gate
in TASK-013 is the only thing making this re-run trustworthy. Do not loosen them to make
the run finish sooner.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

LOG = logging.getLogger(__name__)

# --- Constants copied from one_way_fsg_model/networkinput.py ----------------------
# Every one of these is checked against that file by tests/test_grn_runner.py.

#: Integration horizon in hours. ``networkinput.T_END``; note it is 400.0, not the
#: 300.0 default that ``networkpoint.simulate`` uses when called directly.
T_END: Final[float] = 400.0

#: Wall-shear normalization ceiling, dyn/cm^2. ``networkinput.WSS_MAX``.
WSS_MAX_DYN_CM2: Final[float] = 30.0
#: Mechanical-stress normalization ceiling, Pa. ``networkinput.MECH_MAX``.
MECH_MAX_PA: Final[float] = 100.0

#: Shear held fixed at 5 dyn/cm^2 when a scenario does not use shear.
SHEAR_MID_NORM: Final[float] = 5.0 / WSS_MAX_DYN_CM2
#: Mech held fixed at 40 Pa when a scenario does not use mech.
MECH_MID_NORM: Final[float] = 40.0 / MECH_MAX_PA

#: ``networkinput.scenarios``, verbatim. Insertion order is the column order written to
#: ``activities.csv``, matching ``networkinput``'s own ``for scenario: for node:`` loop.
SCENARIO_FLAGS: Final[dict[str, dict[str, bool]]] = {
    "shear_only": {"use_shear": True, "use_mech": False},
    "mech_only": {"use_shear": False, "use_mech": True},
    "combined": {"use_shear": True, "use_mech": True},
}

# --- Where the inputs and the outputs live ---------------------------------------

#: The per-case GRN input grid. ``080`` in the GRN result folder names is this 80%
#: downsample, which is why this is the file whose row count matches
#: ``AHA GRN Plots/3scenarios_results/<dir>/results_spatial.csv`` exactly.
GRN_INPUT_RELPATH: Final[str] = "grn_inputs/downsampled_080pct/grn_input.csv"

#: Columns carried through from the input grid to the output, in order.
INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "x_m",
    "y_m",
    "wss_mag_dyn_cm2",
    "von_mises_Pa",
    "wss_norm",
    "mech_norm",
)

#: Points per parallel task. Large enough that per-task overhead is noise against a
#: ~0.15 s solve, small enough that a checkpoint never loses much work.
DEFAULT_CHUNK_SIZE: Final[int] = 64


# --- Importing the model ----------------------------------------------------------


_NETWORKPOINT: dict[str, Any] = {}


def load_networkpoint(model_dir: str | Path) -> Any:
    """Import ``networkpoint`` from the read-only model tree.

    ``networkpoint`` is a top-level module in a directory that is not a package, so it
    is reached by putting that directory on ``sys.path``. It is imported by name (not
    via a file-location spec) so that the module has a stable, importable identity in
    every worker process -- ``joblib``'s loky backend pickles by reference, and a module
    loaded under a synthetic name does not survive that round trip.

    ``networkinput`` is deliberately never imported: see this module's docstring.

    Args:
        model_dir: The ``one_way_fsg_model`` directory.

    Returns:
        The imported ``networkpoint`` module, cached per directory.

    Raises:
        FileNotFoundError: If ``networkpoint.py`` is not in ``model_dir``.
    """
    key = str(Path(model_dir).resolve())
    cached = _NETWORKPOINT.get(key)
    if cached is not None:
        return cached
    if not (Path(key) / "networkpoint.py").is_file():
        raise FileNotFoundError(f"no networkpoint.py under {key}")
    if key not in sys.path:
        sys.path.insert(0, key)
    import networkpoint

    _NETWORKPOINT[key] = networkpoint
    return networkpoint


def node_names(model_dir: str | Path) -> tuple[str, ...]:
    """Return the 24 node names, in ``networkpoint``'s own state-vector order.

    Args:
        model_dir: The ``one_way_fsg_model`` directory.

    Returns:
        The node names. Ordering matters: it is the row order of the state vector, and
        the column order of the output CSV depends on it.
    """
    return tuple(load_networkpoint(model_dir).NODES)


# --- Inputs -----------------------------------------------------------------------


@dataclass(frozen=True)
class CaseInputs:
    """The mechanical input grid one case's GRN was evaluated on.

    Attributes:
        case: FSG case folder name, e.g. ``"flow_U0p0360"``.
        path: The CSV these arrays were read from.
        x_m: Point x coordinate in metres. Shape (N,).
        y_m: Point y coordinate in metres. Shape (N,).
        wss_dyn_cm2: Wall shear stress magnitude, dyn/cm^2. NaN at interior points.
        von_mises_pa: Von Mises stress, Pa.
        wss_norm: ``clip(wss / 30, 0, 1)``. NaN where ``wss_dyn_cm2`` is NaN.
        mech_norm: ``clip(von_mises / 100, 0, 1)``.
        wss_has_data: True where the point is on the fluid-contacting surface.
        mech_has_data: True where von Mises is finite.
    """

    case: str
    path: Path
    x_m: np.ndarray
    y_m: np.ndarray
    wss_dyn_cm2: np.ndarray
    von_mises_pa: np.ndarray
    wss_norm: np.ndarray
    mech_norm: np.ndarray
    wss_has_data: np.ndarray
    mech_has_data: np.ndarray

    @property
    def n_points(self) -> int:
        """Number of grid points."""
        return int(self.x_m.size)


def load_case_inputs(root: str | Path, case: str) -> CaseInputs:
    """Read one case's GRN input grid and normalize it exactly as ``networkinput`` does.

    Args:
        root: The ``one_way_fsg_model`` directory.
        case: FSG case folder name, e.g. ``"flow_U0p0180"``.

    Returns:
        The normalized grid.

    Raises:
        FileNotFoundError: If the case's GRN input CSV is absent.
    """
    path = Path(root) / "FSG Results" / case / GRN_INPUT_RELPATH
    if not path.is_file():
        raise FileNotFoundError(f"no GRN input grid for {case} at {path}")
    frame = pd.read_csv(path)
    wss_raw = frame["wss_mag_dyn_cm2"].to_numpy(dtype=float)
    mech_raw = frame["von_mises_Pa"].to_numpy(dtype=float)
    LOG.info("loaded %d GRN input points for %s", len(frame), case)
    return CaseInputs(
        case=case,
        path=path,
        x_m=frame["x_m"].to_numpy(dtype=float),
        y_m=frame["y_m"].to_numpy(dtype=float),
        wss_dyn_cm2=wss_raw,
        von_mises_pa=mech_raw,
        wss_norm=np.clip(wss_raw / WSS_MAX_DYN_CM2, 0.0, 1.0),
        mech_norm=np.clip(mech_raw / MECH_MAX_PA, 0.0, 1.0),
        wss_has_data=~np.isnan(wss_raw),
        mech_has_data=~np.isnan(mech_raw),
    )


def scenario_inputs(inputs: CaseInputs, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(shear, mech)`` drive each point sees under one scenario.

    This is ``networkinput.run_point``'s branch structure, vectorized. Points the
    scenario does not solve are marked with NaN in both arrays.

    Args:
        inputs: The case's normalized input grid.
        scenario: One of :data:`SCENARIO_FLAGS`.

    Returns:
        ``(shear, mech)``, each shape (N,) and normalized to [0, 1]. Both are NaN at
        points the scenario skips, which happens only under ``shear_only``.

    Raises:
        KeyError: If ``scenario`` is not one of the three.
    """
    flags = SCENARIO_FLAGS[scenario]
    use_shear = flags["use_shear"]
    use_mech = flags["use_mech"]
    surface_only = use_shear and not use_mech
    n = inputs.n_points

    if use_shear:
        # Missing WSS means "no data" under combined -> driven at 0.0; under the
        # surface-only scenario it means the point is skipped altogether.
        shear = np.where(inputs.wss_has_data, inputs.wss_norm, np.nan if surface_only else 0.0)
    else:
        shear = np.full(n, SHEAR_MID_NORM)

    if use_mech:
        mech = np.where(inputs.mech_has_data, inputs.mech_norm, MECH_MID_NORM)
    else:
        mech = np.full(n, MECH_MID_NORM)

    skipped = np.isnan(shear) | np.isnan(mech)
    return np.where(skipped, np.nan, shear), np.where(skipped, np.nan, mech)


# --- The solve --------------------------------------------------------------------


def _solve_chunk(
    model_dir: str,
    shear: list[float],
    mech: list[float],
    t_end: float,
    params: dict[str, float],
) -> list[list[float]]:
    """Integrate the network to ``t_end`` for a batch of input pairs.

    Runs in a worker process, so it takes and returns plain builtins and re-imports the
    model itself rather than closing over anything.

    Args:
        model_dir: The ``one_way_fsg_model`` directory.
        shear: Normalized shear drives.
        mech: Normalized mech drives, same length as ``shear``.
        t_end: Integration horizon in hours.
        params: Keyword overrides for ``networkpoint.Params``. Empty means the defaults.

    Returns:
        One 24-element steady-state vector per input pair, in ``networkpoint.NODES``
        order.
    """
    networkpoint = load_networkpoint(model_dir)
    p = networkpoint.Params(**params)
    out: list[list[float]] = []
    for s, m in zip(shear, mech, strict=True):
        _, _, yss = networkpoint.simulate(float(s), float(m), p=p, t_end=t_end)
        out.append([float(v) for v in yss])
    return out


def solve_pairs(
    model_dir: str | Path,
    shear: np.ndarray,
    mech: np.ndarray,
    *,
    t_end: float = T_END,
    params: dict[str, float] | None = None,
    n_jobs: int = -1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    checkpoint: Path | None = None,
    group_size: int | None = None,
    label: str = "",
) -> np.ndarray:
    """Solve the network for a list of ``(shear, mech)`` pairs, in parallel.

    Args:
        model_dir: The ``one_way_fsg_model`` directory.
        shear: Normalized shear drives. Shape (M,). Must contain no NaN.
        mech: Normalized mech drives. Shape (M,).
        t_end: Integration horizon in hours.
        params: Keyword overrides for ``networkpoint.Params``.
        n_jobs: Passed to :class:`joblib.Parallel`. ``-1`` means every core.
        chunk_size: Pairs per parallel task.
        checkpoint: If given, an ``.npz`` written after every completed batch and read
            back on a later call, so an interrupted overnight run resumes. The file
            records the inputs it was computed for and is ignored if they differ.
        group_size: Chunks dispatched between two checkpoint writes. Defaults to eight
            per worker, which keeps the pool saturated. Exposed so a test can force
            several checkpoint writes on a job small enough to run in a second.
        label: Text used in the progress log lines.

    Returns:
        Steady states, shape (M, 24), in ``networkpoint.NODES`` order.

    Raises:
        ValueError: If ``shear`` and ``mech`` differ in length.
    """
    if shear.shape != mech.shape:
        raise ValueError(f"shear {shear.shape} and mech {mech.shape} must match")
    n_nodes = len(node_names(model_dir))
    total = int(shear.size)
    values = np.full((total, n_nodes), np.nan)
    starts = list(range(0, total, chunk_size))
    done = np.zeros(len(starts), dtype=bool)

    fingerprint = _fingerprint(shear, mech, t_end, params or {}, chunk_size)
    if checkpoint is not None:
        restored = _load_checkpoint(checkpoint, fingerprint, values.shape, len(starts))
        if restored is not None:
            values, done = restored
            LOG.info("%s: resumed with %d/%d chunks already done", label, done.sum(), len(starts))

    model_dir = str(Path(model_dir).resolve())
    params = dict(params or {})
    pending = [i for i, flag in enumerate(done) if not flag]
    if not pending:
        return values

    # One Parallel context for the whole run so the worker pool is created once, but
    # dispatched in groups so there is a checkpoint and a progress line in between.
    if group_size is None:
        workers = cpu_count() if n_jobs in (-1, 0) else abs(n_jobs)
        group_size = max(1, 8 * workers)
    group_size = max(1, int(group_size))
    started = time.monotonic()
    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        for group_start in range(0, len(pending), group_size):
            group = pending[group_start : group_start + group_size]
            results = parallel(
                delayed(_solve_chunk)(
                    model_dir,
                    shear[starts[c] : starts[c] + chunk_size].tolist(),
                    mech[starts[c] : starts[c] + chunk_size].tolist(),
                    t_end,
                    params,
                )
                for c in group
            )
            for c, block in zip(group, results, strict=True):
                values[starts[c] : starts[c] + chunk_size] = np.asarray(block, dtype=float)
                done[c] = True
            if checkpoint is not None:
                _save_checkpoint(checkpoint, fingerprint, values, done)
            n_done = int(done.sum())
            elapsed = time.monotonic() - started
            LOG.info(
                "%s: %d/%d chunks (%.1f%%), %.0fs elapsed",
                label,
                n_done,
                len(starts),
                100.0 * n_done / len(starts),
                elapsed,
            )
    return values


def solve_points(
    model_dir: str | Path,
    shear: np.ndarray,
    mech: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """Solve per point, skipping NaN inputs and reusing repeated ones.

    The network is deterministic, so two points with bit-identical drives have
    bit-identical steady states. Solving the unique pairs and scattering the answer back
    is therefore exact, not an approximation -- and on the ``shear_only`` scenario it is
    most of the saving, because thousands of interior points share the same drive.

    Args:
        model_dir: The ``one_way_fsg_model`` directory.
        shear: Normalized shear per point, NaN where the point is skipped. Shape (N,).
        mech: Normalized mech per point, NaN where skipped. Shape (N,).
        **kwargs: Forwarded to :func:`solve_pairs`.

    Returns:
        Steady states, shape (N, 24), NaN-filled at skipped points.
    """
    n_nodes = len(node_names(model_dir))
    out = np.full((shear.size, n_nodes), np.nan)
    live = np.flatnonzero(np.isfinite(shear) & np.isfinite(mech))
    if live.size == 0:
        return out
    pairs = np.column_stack([shear[live], mech[live]])
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    LOG.info(
        "%s: %d points, %d live, %d distinct input pairs",
        kwargs.get("label", ""),
        shear.size,
        live.size,
        len(uniq),
    )
    solved = solve_pairs(model_dir, uniq[:, 0], uniq[:, 1], **kwargs)
    out[live] = solved[inverse.ravel()]
    return out


# --- Checkpointing ----------------------------------------------------------------


def _fingerprint(
    shear: np.ndarray,
    mech: np.ndarray,
    t_end: float,
    params: dict[str, float],
    chunk_size: int,
) -> str:
    """Describe a job precisely enough that a stale checkpoint is never reused.

    Args:
        shear: The shear drives.
        mech: The mech drives.
        t_end: Integration horizon.
        params: Parameter overrides.
        chunk_size: Pairs per task, which sets the chunk boundaries a checkpoint
            records.

    Returns:
        A hex digest.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(shear, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(mech, dtype=np.float64).tobytes())
    h.update(repr((float(t_end), sorted(params.items()), int(chunk_size))).encode())
    return h.hexdigest()


def _load_checkpoint(
    path: Path, fingerprint: str, shape: tuple[int, int], n_chunks: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a checkpoint, or return None if it is absent, stale or unreadable.

    A checkpoint that does not match is ignored rather than raising: the cost of
    ignoring it is a re-run, and the cost of trusting it is wrong numbers.

    Args:
        path: The ``.npz`` file.
        fingerprint: The digest the current job expects.
        shape: The expected ``values`` shape.
        n_chunks: The expected number of chunk flags.

    Returns:
        ``(values, done)`` if the checkpoint matches, else None.
    """
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            if str(data["fingerprint"]) != fingerprint:
                LOG.warning("ignoring checkpoint %s: it was written for other inputs", path)
                return None
            values = data["values"]
            done = data["done"]
    except (OSError, KeyError, ValueError) as exc:
        LOG.warning("ignoring unreadable checkpoint %s: %s", path, exc)
        return None
    if values.shape != shape or done.shape != (n_chunks,):
        LOG.warning("ignoring checkpoint %s: shape changed", path)
        return None
    return values, done.astype(bool)


def _save_checkpoint(path: Path, fingerprint: str, values: np.ndarray, done: np.ndarray) -> None:
    """Write a checkpoint atomically, so a kill mid-write cannot corrupt it.

    Args:
        path: The ``.npz`` file to write.
        fingerprint: The job digest to stamp into it.
        values: Steady states so far, NaN where not yet computed.
        done: Per-chunk completion flags.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # The suffix must still end in ``.npz``: ``np.savez`` appends one otherwise, and
    # the atomic rename would then be chasing a file that was never written.
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, fingerprint=np.str_(fingerprint), values=values, done=done)
    tmp.replace(path)


# --- Per-case driver --------------------------------------------------------------


def activities_path(out_dir: str | Path, case: str) -> Path:
    """Return where one case's full-node activity table is written.

    Args:
        out_dir: The ``grn_full`` output directory.
        case: FSG case folder name.

    Returns:
        ``<out_dir>/<case>/activities.csv``.
    """
    return Path(out_dir) / case / "activities.csv"


def run_case(
    root: str | Path,
    case: str,
    out_dir: str | Path,
    *,
    t_end: float = T_END,
    params: dict[str, float] | None = None,
    n_jobs: int = -1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    subsample: int | None = None,
    resume: bool = True,
) -> Path:
    """Run all three scenarios for one case and write ``activities.csv``.

    Args:
        root: The ``one_way_fsg_model`` directory.
        case: FSG case folder name, e.g. ``"flow_U0p0360"``.
        out_dir: Directory to write ``<case>/activities.csv`` under.
        t_end: Integration horizon in hours.
        params: Keyword overrides for ``networkpoint.Params``. Used by the negative
            control; leave empty for a faithful run.
        n_jobs: Passed to :class:`joblib.Parallel`.
        chunk_size: Points per parallel task.
        subsample: If given, keep only the first N grid points. For smoke tests and the
            negative control; a subsampled run is not comparable to the shipped results.
        resume: Whether to read and write per-scenario checkpoints.

    Returns:
        The path written.
    """
    inputs = load_case_inputs(root, case)
    nodes = node_names(root)
    keep = slice(None) if subsample is None else slice(0, subsample)

    frame = pd.DataFrame(
        {
            "x_m": inputs.x_m[keep],
            "y_m": inputs.y_m[keep],
            "wss_mag_dyn_cm2": inputs.wss_dyn_cm2[keep],
            "von_mises_Pa": inputs.von_mises_pa[keep],
            "wss_norm": inputs.wss_norm[keep],
            "mech_norm": inputs.mech_norm[keep],
        }
    )

    dest = activities_path(out_dir, case)
    ckpt_dir = dest.parent / "_checkpoints"
    for scenario in SCENARIO_FLAGS:
        shear, mech = scenario_inputs(inputs, scenario)
        values = solve_points(
            root,
            shear[keep],
            mech[keep],
            t_end=t_end,
            params=params,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            checkpoint=(ckpt_dir / f"{scenario}.npz") if resume else None,
            label=f"{case}/{scenario}",
        )
        for j, node in enumerate(nodes):
            frame[f"{node}_{scenario}"] = values[:, j]

    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    LOG.info("wrote %s (%d rows, %d columns)", dest, len(frame), frame.shape[1])
    return dest


# --- CLI ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the ``grn-rerun`` argument parser.

    Returns:
        The parser.
    """
    from llm_insights.api import DEFAULT_ROOT

    parser = argparse.ArgumentParser(
        prog="grn-rerun",
        description=(
            "Re-run Dan's 24-node GRN over every spatial point of every case and keep "
            "all 24 nodes, instead of the 5 that results_spatial.csv carries."
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="one_way_fsg_model dir")
    parser.add_argument("--out", type=Path, default=Path("data/grn_full"), help="output directory")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="case to run; repeatable. Default: all three.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="points per parallel task"
    )
    parser.add_argument(
        "--subsample", type=int, default=None, help="run only the first N grid points"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="ignore and do not write checkpoints"
    )
    parser.add_argument(
        "--param",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help=(
            "override one networkpoint.Params field; repeatable. This exists for the "
            "negative control (TASK-013 criterion 2) -- a run with any override is NOT "
            "comparable to the shipped results and must not be written to data/grn_full."
        ),
    )
    parser.add_argument(
        "--time-only",
        type=int,
        default=None,
        help="time N solves, print the extrapolated full cost, and exit without running",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the re-computation from the command line.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    from llm_insights.io.dataset import CONDITION_BY_CASE

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cases = args.case or list(CONDITION_BY_CASE)

    if args.time_only is not None:
        report = time_solves(args.root, cases, n=args.time_only)
        print(json.dumps(report, indent=2))
        return 0

    params = _parse_params(args.param or [])
    if params:
        LOG.warning("running with perturbed parameters %s; this is NOT a faithful run", params)

    started = time.monotonic()
    for case in cases:
        run_case(
            args.root,
            case,
            args.out,
            params=params,
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
            subsample=args.subsample,
            resume=not args.no_resume,
        )
    print(f"all cases done in {time.monotonic() - started:.0f}s")
    return 0


def _parse_params(items: list[str]) -> dict[str, float]:
    """Turn ``NAME=VALUE`` strings into ``Params`` keyword overrides.

    Args:
        items: Raw ``--param`` values.

    Returns:
        The overrides, empty if none were given.

    Raises:
        ValueError: If an item is not ``NAME=VALUE`` with a float value.
    """
    out: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--param expects NAME=VALUE, got {item!r}")
        out[name.strip()] = float(value)
    return out


def time_solves(root: str | Path, cases: list[str], n: int = 20) -> dict[str, Any]:
    """Time ``n`` single-point solves and extrapolate the cost of the full run.

    Args:
        root: The ``one_way_fsg_model`` directory.
        cases: Cases the extrapolation should cover.
        n: How many points to time, serially, on one core.

    Returns:
        A JSON-serializable report: the per-solve time, the number of solves the full
        run needs after de-duplication, and the serial extrapolation.
    """
    networkpoint = load_networkpoint(root)
    p = networkpoint.Params()
    inputs = load_case_inputs(root, cases[0])
    shear, mech = scenario_inputs(inputs, "combined")
    live = np.flatnonzero(np.isfinite(shear))[:n]
    started = time.monotonic()
    for i in live:
        networkpoint.simulate(float(shear[i]), float(mech[i]), p=p, t_end=T_END)
    elapsed = time.monotonic() - started

    per_solve = elapsed / len(live)
    naive = 0
    unique = 0
    for case in cases:
        case_inputs = load_case_inputs(root, case)
        for scenario in SCENARIO_FLAGS:
            s, m = scenario_inputs(case_inputs, scenario)
            ok = np.isfinite(s) & np.isfinite(m)
            naive += int(ok.sum())
            unique += len(np.unique(np.column_stack([s[ok], m[ok]]), axis=0))
    return {
        "n_timed": len(live),
        "seconds_total": round(elapsed, 3),
        "seconds_per_solve": round(per_solve, 4),
        "solves_naive": naive,
        "solves_after_dedup": unique,
        "serial_hours_naive": round(naive * per_solve / 3600.0, 2),
        "serial_hours_after_dedup": round(unique * per_solve / 3600.0, 2),
    }


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "GRN_INPUT_RELPATH",
    "INPUT_COLUMNS",
    "MECH_MAX_PA",
    "MECH_MID_NORM",
    "SCENARIO_FLAGS",
    "SHEAR_MID_NORM",
    "T_END",
    "WSS_MAX_DYN_CM2",
    "CaseInputs",
    "activities_path",
    "build_parser",
    "load_case_inputs",
    "load_networkpoint",
    "main",
    "node_names",
    "run_case",
    "scenario_inputs",
    "solve_pairs",
    "solve_points",
    "time_solves",
]
