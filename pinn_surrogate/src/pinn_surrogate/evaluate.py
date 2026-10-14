"""Evaluate a trained surrogate on the held-out splits (val, test_B) -> eval.json + figures.

Per solve, at the 256 wall points (the outline vertices, where Dan's ``*_256`` targets are):

* ``tau_rel_l2``: ||tau_pred - tau|| / ||tau|| of the signed wall shear tau_t.
* ``p_rel_l2``: same for wall pressure, after subtracting each field's own mean outlet
  pressure (pred from the network at the outlet nodes, true from Dan's outlet nodes). A
  constant pressure offset does not change the flow, so this scores the *shape* of the wall
  pressure independent of how well the gauge (set by the outlet condition) was learned.
  The gauge error still shows up in ``p_field_rel_l2``, which uses raw pressures.
* ``peak_loc_err``: |s(argmax |tau_pred|) - s(argmax |tau|)| / arc length.
* ``sign_agree``: fraction of points with matching sign of tau_t where |tau| > 1 % of max.

Plus field relative L2 of u, v, p on the 3072 packed nodes, the mean normalized PDE
residual at the interior nodes (same normalization as the training loss), and the time to
predict p and tau at the 256 wall points for one solve.

Usage::

    PYTHONPATH=src python3 -m pinn_surrogate.evaluate --run runs/n80_physon_s0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from pinn_surrogate import metrics, physics
from pinn_surrogate.data import Packed, load_packed
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.losses import pde_residual_magnitude
from pinn_surrogate.model import ConditionedPINN, load_checkpoint

log = logging.getLogger(__name__)

SPLITS = ("val", "test_B")
METRICS = [
    "tau_rel_l2",
    "p_rel_l2",
    "peak_loc_err",
    "sign_agree",
    "u_rel_l2",
    "v_rel_l2",
    "p_field_rel_l2",
    "pde_res",
    "infer_ms",
]


@contextlib.contextmanager
def frozen(model: ConditionedPINN) -> Iterator[None]:
    """Disable parameter gradients (and eval mode) inside the block.

    Autograd w.r.t. *positions* is still needed for wall shear and the PDE residual, but
    building graphs through the weights would only waste time and memory.
    """
    was_training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in zip(model.parameters(), flags):
            p.requires_grad_(f)
        model.train(was_training)


def predict_wall(model: ConditionedPINN, s: SolveSet, i: int) -> dict[str, np.ndarray]:
    """Wall p* and tau_t* at the outline vertices of solve ``i`` (model must be frozen)."""
    sl = slice(i, i + 1)
    fn = model.field_fn(s.outline[sl], s.log_re[sl])
    wq = physics.wall_quantities(fn, s.outline[sl], s.normals[sl], offset=model.cfg.wall_offset)
    return {"p": wq["p"][0].detach().numpy(), "tau_t": wq["tau_t"][0].detach().numpy()}


def outlet_mean(p: np.ndarray, s: SolveSet, i: int) -> float:
    """Mean of a nodal pressure array over solve ``i``'s outlet nodes (corners excluded)."""
    pos = s.outlet[i].numpy()
    mask = s.outlet_mask[i].numpy() > 0
    return float(p[pos][mask].mean())


def evaluate_solve(model: ConditionedPINN, s: SolveSet, i: int) -> dict:
    """All metrics for one solve; keys starting with '_' hold arrays for the figures."""
    sl = slice(i, i + 1)
    true = s.uvp[i].numpy()
    with torch.no_grad():
        u, v, p = (t[0].numpy() for t in model(s.xy[sl], s.outline[sl], s.log_re[sl]))

    t0 = time.perf_counter()
    wall = predict_wall(model, s, i)
    infer_ms = 1e3 * (time.perf_counter() - t0)

    tau_true, p_true = s.wall_tau_t[i].numpy(), s.wall_p[i].numpy()
    p_wall_pred = wall["p"] - outlet_mean(p, s, i)
    p_wall_true = p_true - outlet_mean(true[:, 2], s, i)
    s_frac = metrics.arc_fraction(s.outline[i].numpy())

    xy_int = s.xy[sl][:, s.interior[i]]
    res = physics.ns_residual(model.field_fn(s.outline[sl], s.log_re[sl]), xy_int, float(s.re[i]))
    ints = s.interior[i]
    pde = pde_residual_magnitude(
        res, s.c[sl][:, None], s.mom_scale[sl][:, ints], s.div_scale[sl][:, ints]
    ).detach().median()  # median: the same statistic as Dan's r_rel summaries

    return {
        "solve": int(s.index[i]),
        "geom_id": int(s.geom_id[i]),
        "k": int(s.flow_k[i]),
        "family": str(s.family[i]),
        "Re": float(s.re[i]),
        "tau_rel_l2": metrics.rel_l2(wall["tau_t"], tau_true),
        "p_rel_l2": metrics.rel_l2(p_wall_pred, p_wall_true),
        "peak_loc_err": metrics.peak_location_error(wall["tau_t"], tau_true, s_frac),
        "sign_agree": metrics.sign_agreement(wall["tau_t"], tau_true),
        "u_rel_l2": metrics.rel_l2(u, true[:, 0]),
        "v_rel_l2": metrics.rel_l2(v, true[:, 1]),
        "p_field_rel_l2": metrics.rel_l2(p, true[:, 2]),
        "pde_res": float(pde),
        "infer_ms": infer_ms,
        "_s": s_frac,
        "_tau_pred": wall["tau_t"],
        "_tau_true": tau_true,
    }


def evaluate_split(model: ConditionedPINN, s: SolveSet) -> list[dict]:
    """Per-solve rows for every solve in ``s``."""
    with frozen(model):
        return [evaluate_solve(model, s, i) for i in range(len(s))]


def quick_check(model: ConditionedPINN, s: SolveSet) -> dict[str, float]:
    """Cheap periodic check during training: median wall tau / p errors on a few solves."""
    rows = []
    with frozen(model):
        for i in range(len(s)):
            wall = predict_wall(model, s, i)
            rows.append(metrics.rel_l2(wall["tau_t"], s.wall_tau_t[i].numpy()))
    return {"quick_val_tau_rel_l2": float(np.median(rows))}


def summarize_rows(rows: list[dict]) -> dict:
    """Aggregate per-solve rows overall, per flow index k and per family."""
    return {
        "all": metrics.aggregate(rows, METRICS),
        "by_k": metrics.aggregate(rows, METRICS, "k"),
        "by_family": metrics.aggregate(rows, METRICS, "family"),
    }


def _public(row: dict) -> dict:
    """Row without the array-valued '_' entries (for JSON)."""
    return {k: v for k, v in row.items() if not k.startswith("_")}


def make_figures(rows_by_split: dict[str, list[dict]], out_dir: Path) -> list[Path]:
    """Wall-shear profiles for 4 representative test_B solves and error vs Re.

    "Representative" = for each flow index k, the solve whose tau error is the median of
    that k (not the best or worst case).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    rows_b = rows_by_split.get("test_B", [])
    if rows_b:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, k in zip(axes.ravel(), range(4)):
            rk = sorted((r for r in rows_b if r["k"] == k), key=lambda r: r["tau_rel_l2"])
            if not rk:
                ax.set_axis_off()
                continue
            r = rk[len(rk) // 2]
            ax.plot(r["_s"], r["_tau_true"], "k-", lw=1.5, label="Dan (FEM)")
            ax.plot(r["_s"], r["_tau_pred"], "C1--", lw=1.5, label="surrogate")
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_title(
                f"test_B g{r['geom_id']:05d} k={k} Re={r['Re']:.2g}  relL2={r['tau_rel_l2']:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("arc length s / L")
            ax.set_ylabel("wall shear tau_t*")
            ax.legend(fontsize=8)
        paths.append(out_dir / "wall_tau_test_B.png")
        fig.savefig(paths[-1], dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for split, rows in rows_by_split.items():
        ax.scatter([r["Re"] for r in rows], [r["tau_rel_l2"] for r in rows], s=12, label=split)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re")
    ax.set_ylabel("wall tau_t relative L2 error")
    ax.legend()
    paths.append(out_dir / "error_vs_re.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def evaluate_model(
    model: ConditionedPINN,
    pk: Packed,
    nd: dict[str, np.ndarray],
    out_dir: Path,
    max_solves: int = 0,
    splits: tuple[str, ...] = SPLITS,
) -> dict:
    """Evaluate on ``splits``, write ``eval.json`` and figures into ``out_dir``.

    Args:
        model: Trained model.
        pk, nd: Packed data and its nondimensional view.
        out_dir: Where to write.
        max_solves: If > 0, only the first ``max_solves`` solves per split (smoke tests).
        splits: Split names.

    Returns:
        The report dict that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {}
    for split in splits:
        idx = pk.split_indices(split)
        if max_solves > 0:
            idx = idx[:max_solves]
        t0 = time.perf_counter()
        rows_by_split[split] = evaluate_split(model, SolveSet.from_packed(pk, nd, idx))
        log.info("evaluated %s: %d solves in %.1f s", split, len(idx), time.perf_counter() - t0)
    report = {
        "splits": {
            split: {"aggregate": summarize_rows(rows), "per_solve": [_public(r) for r in rows]}
            for split, rows in rows_by_split.items()
        },
        "metric_notes": (__doc__ or "").split("Usage::")[0].strip(),
    }
    (out_dir / "eval.json").write_text(json.dumps(report, indent=1))
    figs = make_figures(rows_by_split, out_dir)
    for split in splits:
        agg = report["splits"][split]["aggregate"]["all"]
        log.info(
            "%-6s median: tau %.4f  p %.4f  peak %.4f  sign %.3f  pde %.3g  infer %.1f ms",
            split,
            *(agg[m]["median"] for m in ("tau_rel_l2", "p_rel_l2", "peak_loc_err")),
            agg["sign_agree"]["median"],
            agg["pde_res"]["median"],
            agg["infer_ms"]["median"],
        )
    log.info("wrote %s and %s", out_dir / "eval.json", ", ".join(p.name for p in figs))
    return report


def main() -> None:
    """CLI: evaluate the ``model.pt`` of a run directory."""
    ap = argparse.ArgumentParser(description="Evaluate a trained run on val and test_B.")
    ap.add_argument("--run", type=Path, required=True, help="run dir containing model.pt")
    ap.add_argument("--max-solves", type=int, default=0, help="0 = all solves per split")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    model, _ = load_checkpoint(args.run / "model.pt")
    pk = load_packed()
    evaluate_model(model, pk, pk.nondim(), args.run, args.max_solves)


if __name__ == "__main__":
    main()
