"""Train one arm of the ablation: conditioned PINN with or without the physics terms.

Each step draws ``batch_solves`` solves from the first ``n_train_geoms`` train geometries
(all 4 flows each) and ``nodes_per_solve`` of their 3072 packed nodes, then minimizes

    physics on:   L = L_data + lambda_pde L_pde + lambda_out L_out   (adaptive lambdas)
    physics off:  L = L_data

with Adam and a cosine learning-rate decay. The step budget is fixed so both arms get the
same number of updates; wall-clock is recorded separately. The full training state is
checkpointed every ``ckpt_every`` steps (``--resume`` continues from it). At the end the
model is saved and evaluated on val and test_B (``evaluate.py``).

Usage::

    PYTHONPATH=src python3 -m pinn_surrogate.train --n-train-geoms 80 --physics on \
        --steps 3000 --threads 2
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from pinn_surrogate import physics
from pinn_surrogate.config import TrainConfig, data_scales, parse_config
from pinn_surrogate.data import load_packed
from pinn_surrogate.dataset import Batch, SolveSet, sample_batch
from pinn_surrogate.evaluate import evaluate_model, quick_check
from pinn_surrogate.losses import AdaptiveWeights, data_loss, outlet_loss, pde_loss, wall_tau_loss
from pinn_surrogate.model import ConditionedPINN, atomic_save, save_checkpoint, torch_load

log = logging.getLogger(__name__)

CKPT_NAME = "checkpoint.pt"  # full training state, for --resume
MODEL_NAME = "model.pt"  # final weights + config, for evaluation and later use


def set_threads(n: int) -> None:
    """Limit torch to ``n`` CPU threads (2 runs x 2 threads share the 4-core VM)."""
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # can only be set once, before any parallel work
        pass


def compute_losses(
    model: ConditionedPINN,
    batch: Batch,
    scales: tuple[float, float, float],
    use_physics: bool,
    use_tau: bool = False,
) -> dict[str, torch.Tensor]:
    """All loss terms for one batch (physics only if ``use_physics``, wall shear if ``use_tau``)."""
    pred = model(batch.xy, batch.outline, batch.log_re)
    losses = {"data": data_loss(pred, batch.uvp, batch.weight, scales)}
    if use_tau:
        wq = physics.wall_quantities(
            model.field_fn(batch.outline, batch.log_re),
            batch.outline,
            batch.normals,
            offset=model.cfg.wall_offset,
        )
        losses["tau"] = wall_tau_loss(wq["tau_t"], batch.wall_tau_t)
    if use_physics:
        fn = model.field_fn(batch.outline, batch.log_re)
        losses["pde"] = pde_loss(
            physics.ns_residual(fn, batch.xy_pde, batch.re), batch.c, batch.c_mom, batch.c_div
        )
        losses["out"] = outlet_loss(physics.outlet_residual(fn, batch.xy_out), batch.out_mask)
    return losses


@dataclass
class Clock:
    """Wall-clock bookkeeping, accumulated over all sessions of a (resumed) run.

    Attributes:
        train_s: Seconds inside optimization steps only (the number the ablation compares).
        wall_s: Seconds from the start of each session to its last checkpoint, summed
            (includes periodic quick checks and checkpoint writes; excludes data loading
            and any time the machine spent asleep between sessions).
        sessions: How many times training was started or resumed.
    """

    train_s: float = 0.0
    wall_s: float = 0.0
    sessions: int = 0


def save_state(
    path: Path,
    step: int,
    model: ConditionedPINN,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    weights: AdaptiveWeights,
    rng: np.random.Generator,
    clock: Clock,
) -> None:
    """Write everything needed to continue training exactly where it stopped (atomically).

    The numpy RNG state (it drives batch sampling) is stored as a JSON string so the
    checkpoint holds only tensors and plain types, as ``weights_only`` loading requires.
    """
    atomic_save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "lambdas": dict(weights.lambdas),
            "rng": json.dumps(rng.bit_generator.state),
            "clock": asdict(clock),
        },
        path,
    )


def load_state(
    path: Path,
    model: ConditionedPINN,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    weights: AdaptiveWeights,
    rng: np.random.Generator,
) -> tuple[int, Clock]:
    """Restore the objects saved by ``save_state`` in place.

    Returns:
        ``(last completed step, clock)``.
    """
    state = torch_load(path)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    weights.lambdas.update(state["lambdas"])
    weights.n_updates = 1  # restored lambdas are already past their first (jump) update
    rng.bit_generator.state = json.loads(state["rng"])
    return int(state["step"]), Clock(**state["clock"])


def checkpoint_step(path: Path) -> int:
    """Last completed step recorded in a checkpoint (0 if there is none)."""
    return int(torch_load(path)["step"]) if path.exists() else 0


class JsonlWriter:
    """Append one JSON object per line (``metrics.jsonl``)."""

    def __init__(self, path: Path, keep_until_step: int = 0) -> None:
        """Open the log, keeping only existing rows with ``step <= keep_until_step``.

        On a fresh run that empties the file; on resume it drops rows logged after the
        checkpoint, because the resumed run is about to redo those steps.
        """
        self.path = path
        rows: list[str] = []
        if keep_until_step > 0 and path.exists():
            rows = [ln for ln in path.read_text().splitlines() if ln.strip()]
            rows = [ln for ln in rows if json.loads(ln)["step"] <= keep_until_step]
        path.write_text("".join(ln + "\n" for ln in rows))

    def write(self, row: dict) -> None:
        """Append ``row``."""
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def prepare_run_dir(cfg: TrainConfig) -> tuple[Path, bool]:
    """Create the run directory and decide whether this is a fresh run or a resume.

    Refuses to silently overwrite an existing run (no ``--resume``), and refuses to resume
    a run whose saved config differs in anything that changes the optimization.

    Returns:
        ``(run dir, resuming)``.
    """
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / CKPT_NAME
    if ckpt.exists() and not cfg.resume:
        raise SystemExit(f"{ckpt} exists: pass --resume to continue it, or choose a new --out")
    resuming = cfg.resume and ckpt.exists()
    if resuming:
        saved = json.loads((out / "config.json").read_text())
        bad = cfg.resume_mismatch(saved)
        if bad:
            raise SystemExit(f"cannot resume {out}: {bad} differ from its config.json")
    cfg.save(out / "config.json")
    return out, resuming


def train(cfg: TrainConfig) -> Path:
    """Run (or resume) training, then the final evaluation; returns the run directory.

    Args:
        cfg: Full configuration (``config.parse_config``).
    """
    out, resuming = prepare_run_dir(cfg)
    ckpt_path = out / CKPT_NAME
    if resuming and checkpoint_step(ckpt_path) >= cfg.steps and (out / "eval.json").exists():
        log.info("%s is already complete (%d steps, eval.json present)", out, cfg.steps)
        return out
    set_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)

    t_load = time.perf_counter()
    pk = load_packed()
    nd = pk.nondim()
    train_set = SolveSet.from_packed(pk, nd, pk.train_subset(cfg.n_train_geoms))
    val_quick = SolveSet.from_packed(pk, nd, pk.split_indices("val")[: cfg.eval_solves])
    scales = data_scales()
    log.info(
        "data: %d train solves, loss scales u/v/p = %.3f/%.3f/%.1f (%.1f s)",
        len(train_set),
        *scales,
        time.perf_counter() - t_load,
    )

    model = ConditionedPINN(cfg.model)
    model.set_normalization(train_set.outline.numpy(), train_set.log_re.numpy())
    log.info("model: %d trainable parameters", model.n_params())
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.lr_min)
    weights = AdaptiveWeights(
        ["data", "pde", "out"] if cfg.physics else ["data"], ref="data", share=cfg.physics_share
    )

    done, clock = 0, Clock()
    if resuming:
        done, clock = load_state(ckpt_path, model, opt, sched, weights, rng)
        log.info("resumed %s at step %d (%.0f s trained before)", ckpt_path, done, clock.train_s)
    clock.sessions += 1
    writer = JsonlWriter(out / "metrics.jsonl", keep_until_step=done)

    t_session = time.perf_counter()
    wall_before = clock.wall_s
    train_at_log, step_at_log = clock.train_s, done
    for step in range(done + 1, cfg.steps + 1):
        t_step = time.perf_counter()
        batch = sample_batch(
            train_set, rng, cfg.batch_solves, cfg.nodes_per_solve, cfg.pde_nodes_per_solve
        )
        losses = compute_losses(model, batch, scales, cfg.physics, cfg.tau_weight > 0)
        if cfg.physics and (step == 1 or step % cfg.balance_every == 0):
            weights.update(losses, params)
        loss = weights.total(losses)  # adaptive terms only; "tau" has a fixed weight
        if "tau" in losses:
            loss = loss + cfg.tau_weight * losses["tau"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {losses}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        clock.train_s += time.perf_counter() - t_step

        if step % cfg.log_every == 0 or step == cfg.steps:
            row = {
                "step": step,
                "wall_s": wall_before + time.perf_counter() - t_session,
                "train_s": clock.train_s,
                "sec_per_step": (clock.train_s - train_at_log) / (step - step_at_log),
                "lr": sched.get_last_lr()[0],
                "loss": float(loss.detach()),
                **{f"loss_{n}": float(v) for n, v in losses.items()},
                **{f"lambda_{n}": v for n, v in weights.lambdas.items()},
                **({"lambda_tau": cfg.tau_weight} if "tau" in losses else {}),
            }
            train_at_log, step_at_log = clock.train_s, step
            if cfg.eval_every > 0 and step % cfg.eval_every == 0:
                row.update(quick_check(model, val_quick))
            writer.write(row)
            terms = "  ".join(
                f"{n} {float(v):.3g} (x{weights.lambdas.get(n, cfg.tau_weight):.3g})"
                for n, v in losses.items()
            )
            log.info(
                "step %5d  loss %.4g  %s  %.3f s/step",
                step,
                row["loss"],
                terms,
                row["sec_per_step"],
            )
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            clock.wall_s = wall_before + time.perf_counter() - t_session
            save_state(ckpt_path, step, model, opt, sched, weights, rng, clock)

    summary = {
        "steps": cfg.steps,
        "train_s": clock.train_s,
        "wall_s": clock.wall_s,
        "sessions": clock.sessions,
        "sec_per_step": clock.train_s / cfg.steps,
        "n_params": model.n_params(),
        "final_lambdas": weights.lambdas,
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    save_checkpoint(model, out / MODEL_NAME, {"train_config": cfg.to_dict()})
    log.info(
        "trained %d steps: %.0f s optimizing (%.3f s/step), %d session(s)",
        cfg.steps,
        clock.train_s,
        summary["sec_per_step"],
        clock.sessions,
    )

    evaluate_model(model, pk, nd, out, cfg.max_eval_solves)
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (see ``config.build_parser`` for the arguments)."""
    cfg = parse_config(argv)
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(Path(cfg.out) / "train.log")],
    )
    train(cfg)


if __name__ == "__main__":
    main()
