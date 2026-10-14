"""Stage 3: the textbook PINN on one solve -- physics only, no data.

Same network class as the surrogate, but the conditioning (outline, Re) is fixed to one
solve and the only losses are the NS residual at interior mesh nodes and the outlet
traction. Walls and inlet are exact through the hard-constraint ansatz. Dan's solution is
used only to *score* the result: wall tau_t and p at the 256 outline points, reported every
``--report-every`` steps along with seconds per step.

No data enters the loss -- not even the pressure scale: the residual is normalized by
c = 12 (the empty-channel Poiseuille gradient) instead of the solve's measured value.

Usage::

    PYTHONPATH=src python3 -m pinn_surrogate.single_case --steps 3000 --threads 2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pinn_surrogate import metrics, physics
from pinn_surrogate.config import PROJECT_ROOT, ModelConfig
from pinn_surrogate.data import (
    TAG_OUTLET,
    load_full,
    mean_inlet_speed,
    to_nondim,
)
from pinn_surrogate.evaluate import frozen
from pinn_surrogate.losses import AdaptiveWeights, outlet_loss, pde_loss
from pinn_surrogate.model import ConditionedPINN, save_checkpoint
from pinn_surrogate.train import JsonlWriter, set_threads

log = logging.getLogger(__name__)

DEFAULT_FULL = PROJECT_ROOT / "data" / "full" / "full_g_00071_k0.npz"
C_FIXED = 12.0  # residual scale: Poiseuille gradient (no data used)


def load_case(path: Path) -> dict[str, torch.Tensor | np.ndarray | float]:
    """Nondimensional tensors for one full dump.

    Returns:
        ``xy_int`` (1, P, 2) interior nodes, ``xy_out`` (1, K, 2) outlet nodes without the
        corners, ``outline`` (1, 256, 2), ``normals`` (1, 256, 2), ``log_re`` (1,), ``re``,
        and numpy targets ``tau_t``, ``p_wall`` (256,) and the true outlet mean pressure.
    """
    f = load_full(path)
    h, mu = float(f["H"]), float(f["mu"])
    ub = float(mean_inlet_speed(f["u_inlet_max"]))
    xy = f["xy"] / h
    on_boundary = np.zeros(len(xy), dtype=bool)
    on_boundary[np.unique(f["lines"])] = True
    out = np.unique(f["lines"][f["line_tags"] == TAG_OUTLET])
    out = out[(xy[out, 1] > 1e-6) & (xy[out, 1] < 1 - 1e-6)]  # corners are no-slip
    p_true = to_nondim(f["p"], "pressure", ub, h, mu)

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.asarray(x), dtype=torch.float32)

    return {
        "xy_int": t(xy[~on_boundary][None]),
        "xy_out": t(xy[out][None]),
        "outline": t(f["outline"][None] / h),
        "normals": t(np.stack([f["arc_nx_256"], f["arc_ny_256"]], axis=-1)[None]),
        "log_re": t([np.log(float(f["Re"]))]),
        "re": float(f["Re"]),
        "tau_t": to_nondim(f["arc_tau_t_256"], "stress", ub, h, mu),
        "p_wall": to_nondim(f["arc_p_256"], "pressure", ub, h, mu),
        "p_out_true": float(p_true[out].mean()),
    }


def wall_report(model: ConditionedPINN, case: dict) -> dict[str, float]:
    """Wall tau and gauge-free wall p errors vs Dan's ``arc_*_256`` arrays."""
    fn = model.field_fn(case["outline"], case["log_re"])
    with frozen(model):
        wq = physics.wall_quantities(
            fn, case["outline"], case["normals"], offset=model.cfg.wall_offset
        )
        with torch.no_grad():
            p_out = float(fn(case["xy_out"])[2].mean())
    tau = wq["tau_t"][0].detach().numpy()
    p = wq["p"][0].detach().numpy() - p_out
    s_frac = metrics.arc_fraction(case["outline"][0].numpy())
    return {
        "tau_rel_l2": metrics.rel_l2(tau, case["tau_t"]),
        "p_rel_l2": metrics.rel_l2(p, case["p_wall"] - case["p_out_true"]),
        "peak_loc_err": metrics.peak_location_error(tau, case["tau_t"], s_frac),
        "sign_agree": metrics.sign_agreement(tau, case["tau_t"]),
    }


def run(args: argparse.Namespace) -> dict:
    """Train on one case and return the final report."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    case = load_case(Path(args.full))
    log.info(
        "case %s: Re %.3g, %d interior nodes, %d outlet nodes",
        args.full,
        case["re"],
        case["xy_int"].shape[1],
        case["xy_out"].shape[1],
    )

    cfg = ModelConfig(delta=args.delta, seed=args.seed)
    model = ConditionedPINN(cfg)
    model.set_normalization(case["outline"].numpy(), case["log_re"].numpy())
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    weights = AdaptiveWeights(["pde", "out"], ref="pde")
    writer = JsonlWriter(out / "metrics.jsonl")
    c = torch.full((1, 1), C_FIXED)
    n_int = case["xy_int"].shape[1]
    fn = model.field_fn(case["outline"], case["log_re"])

    t_start = time.perf_counter()
    report_s, t_last, step_last = 0.0, t_start, 0
    row: dict = {}
    for step in range(1, args.steps + 1):
        pick = torch.tensor(rng.choice(n_int, size=min(args.colloc, n_int), replace=False))
        losses = {
            "pde": pde_loss(physics.ns_residual(fn, case["xy_int"][:, pick], case["re"]), c),
            "out": outlet_loss(
                physics.outlet_residual(fn, case["xy_out"]), torch.ones(case["xy_out"].shape[:2])
            ),
        }
        if step == 1 or step % args.balance_every == 0:
            weights.update(losses, params)
        loss = weights.total(losses)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % args.report_every == 0 or step == args.steps:
            now = time.perf_counter()
            row = {
                "step": step,
                "train_s": now - t_start - report_s,
                "sec_per_step": (now - t_last) / (step - step_last),
                **{f"loss_{n}": float(v) for n, v in losses.items()},
                **{f"lambda_{n}": v for n, v in weights.lambdas.items()},
            }
            t0 = time.perf_counter()
            row.update(wall_report(model, case))
            report_s += time.perf_counter() - t0
            t_last, step_last = time.perf_counter(), step
            writer.write(row)
            log.info(
                "step %5d  pde %.3g  out %.3g (x%.3g)  tau relL2 %.4f  p relL2 %.4f  %.3f s/step",
                step,
                row["loss_pde"],
                row["loss_out"],
                weights.lambdas["out"],
                row["tau_rel_l2"],
                row["p_rel_l2"],
                row["sec_per_step"],
            )

    summary = {
        "full": str(args.full),
        "Re": case["re"],
        "steps": args.steps,
        "train_s": time.perf_counter() - t_start - report_s,
        "final": row,
        "model_config": asdict(cfg),
    }
    summary["sec_per_step"] = summary["train_s"] / args.steps
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    save_checkpoint(model, out / "model.pt", {"single_case": str(args.full)})
    log.info("done: %s", json.dumps(row))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the single-case run."""
    ap = argparse.ArgumentParser(description="Stage 3: physics-only PINN on one solve.")
    ap.add_argument("--full", type=Path, default=DEFAULT_FULL)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--colloc", type=int, default=1024, help="interior points per step")
    ap.add_argument("--report-every", type=int, default=250)
    ap.add_argument("--balance-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--delta", type=float, default=ModelConfig().delta)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "single_case")
    return ap


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "single_case.log")],
    )
    run(args)


if __name__ == "__main__":
    main()
