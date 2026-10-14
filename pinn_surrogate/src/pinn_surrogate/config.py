"""Model and training configuration (torch-free, so it can be parsed and tested anywhere).

Every number that shapes the network or the loss lives here, with the reason for its value.
``train.py`` turns command-line arguments into a ``TrainConfig``; the model is built from the
``ModelConfig`` nested inside it. Both are saved as JSON next to every run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = PROJECT_ROOT / "data" / "nondim_stats.json"
TRAIN_SIZES = (20, 80, 160)


@dataclass
class ModelConfig:
    """Architecture and hard-constraint settings of ``model.ConditionedPINN``.

    The output scales ``s_u, s_v, s_p`` are chosen so that the raw network outputs
    (N_u, N_v, N_p) the data require are O(1): a freshly initialized MLP outputs O(1)
    numbers, so S ~ the typical size of the quantity it multiplies means no output has to
    grow by orders of magnitude during training. Solving the ansatz for what the data
    require, over the 640 packed train solves (all interior nodes for p; for u, v only where
    m > 0.2 and w g > 0.01, since dividing by a near-zero factor is meaningless there),
    delta = 0.05 (reproduce: ``python3 -m pinn_surrogate.geometry_np``, LOG.md Stage 3-5):

    * S_u N_u = (u/m - u_p)/(w g): std 8.8, |.| median 0.75, p95 12, p99 27, max 385 (the
      max is in narrow gaps under the top wall, where w is small but u is large) -> s_u = 8.
    * S_v N_v = v/(m w g): std 2.9, |.| p95 5.4, p99 12.5 -> s_v = 3.
    * p - 12(10 - x) (correction to the Poiseuille baseline): std 223, |.| p95 482,
      p99 1210 -> s_p = 200. The baseline already carries the 240 empty-channel drop; the
      correction is the extra drop across the cushion.
    """

    # Shape encoder: 256-point outline -> 512 numbers -> 128 -> 128 -> latent.
    n_outline: int = 256
    enc_width: int = 128
    latent: int = 32
    # Trunk MLP on (Fourier features, raw coords, z, normalized log Re).
    n_fourier: int = 32
    # Std of the Gaussian frequency matrix, applied to (x*/10, y*). sigma = 2 gives ~2
    # oscillations across the channel height; larger risks fitting noise (spectral bias).
    fourier_sigma: float = 2.0
    width: int = 96
    depth: int = 5  # hidden layers
    # Hard constraint m = tanh(d_c / delta). delta = 0.05 H = 20 um: the velocity mask
    # reaches 0.76 at one delta, so the network only has to represent u/m, which is smooth.
    delta: float = 0.05
    s_u: float = 8.0
    s_v: float = 3.0
    s_p: float = 200.0
    # Wall gradients are taken this far (nondim) into the fluid along the normal: on the
    # polyline the distance d_c has no gradient (|r| at r = 0). 1e-4 H = 40 nm; the bias it
    # adds to tau is ~2 * offset / (length scale of u/m) ~ 0.2-0.4 %, below the FEM's own
    # 0.7-1.2 % wall-shear discretization error (LOG.md, Stage 2).
    wall_offset: float = 1e-4
    seed: int = 0


@dataclass
class TrainConfig:
    """One training run. Defaults target ~10-15 min per physics-on run of 3000 steps.

    That target is an estimate (torch was not available where this was written): the cost
    is dominated by second derivatives through the trunk at
    ``batch_solves * pde_nodes_per_solve`` = 2048 points per step.
    """

    n_train_geoms: int = 20
    physics: bool = True
    steps: int = 3000
    batch_solves: int = 8
    nodes_per_solve: int = 512
    # PDE collocation points per solve, drawn from the solve's interior nodes. Capped
    # separately from nodes_per_solve because second derivatives are ~10x the cost of the
    # data forward pass.
    pde_nodes_per_solve: int = 256
    lr: float = 1e-3
    lr_min: float = 1e-5
    threads: int = 2
    seed: int = 0
    out: str = ""
    log_every: int = 50
    eval_every: int = 500
    eval_solves: int = 16  # val solves used by the periodic quick check
    balance_every: int = 100
    # Gradient budget of each physics term relative to the data term. Balancing every term
    # to *equal* gradient magnitude (share = 1, Wang et al.'s default) let the PDE gradient
    # override the data fit in the pilot run (data loss never fell; wall-tau error 0.96).
    # share = 0.1 targets physics gradients at 10 % of the data gradient: physics acts as a
    # regularizer on the data fit instead of competing with it. See LOG.md, Stage 4 pilot.
    # 0.1 is what the report's grid used; it is untuned (the PDE weight then sat near its
    # 1e-3 floor for most of training, so a sweep of share and floor is still open).
    physics_share: float = 0.1
    max_eval_solves: int = 0  # 0 = all solves in the final evaluation (smoke tests use >0)
    # Full training state (weights, Adam moments, LR schedule, loss weights, RNG) is saved
    # every ckpt_every steps, so a laptop sleep or VM restart loses at most that much work.
    ckpt_every: int = 200
    resume: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)

    # Fields that may differ between the original run and a --resume of it (they do not
    # change the optimization, only bookkeeping or speed).
    RESUME_MAY_DIFFER = ("out", "threads", "resume", "log_every", "ckpt_every", "eval_every")

    def resume_mismatch(self, saved: dict) -> list[str]:
        """Names of fields that differ from a saved config and would change the result.

        Args:
            saved: ``to_dict()`` of the run being resumed (its ``config.json``).

        Returns:
            Sorted field names (empty if resuming is safe).
        """
        mine = self.to_dict()
        keys = (set(mine) | set(saved)) - set(self.RESUME_MAY_DIFFER)
        return sorted(k for k in keys if mine.get(k) != saved.get(k))

    def run_name(self) -> str:
        """Default run directory name, e.g. ``n80_physon_s0``."""
        return f"n{self.n_train_geoms}_phys{'on' if self.physics else 'off'}_s{self.seed}"

    def to_dict(self) -> dict:
        """Plain-JSON form (nested model config included)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        """Inverse of ``to_dict``; unknown keys are rejected so typos do not pass silently."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown TrainConfig keys: {sorted(unknown)}")
        kw = dict(d)
        kw["model"] = ModelConfig(**kw.get("model", {}))
        return cls(**kw)

    def save(self, path: Path) -> None:
        """Write as indented JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=1))


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of ``train.py``."""
    d, m = TrainConfig(), ModelConfig()
    ap = argparse.ArgumentParser(description="Train the conditioned PINN (one ablation arm).")
    ap.add_argument("--n-train-geoms", type=int, choices=TRAIN_SIZES, default=d.n_train_geoms)
    ap.add_argument("--physics", choices=("on", "off"), default="on")
    ap.add_argument("--steps", type=int, default=d.steps)
    ap.add_argument("--batch-solves", type=int, default=d.batch_solves)
    ap.add_argument("--nodes-per-solve", type=int, default=d.nodes_per_solve)
    ap.add_argument("--pde-nodes-per-solve", type=int, default=d.pde_nodes_per_solve)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--lr-min", type=float, default=d.lr_min)
    ap.add_argument("--threads", type=int, default=d.threads)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", type=str, default="", help="run dir (default runs/<name>)")
    ap.add_argument("--log-every", type=int, default=d.log_every)
    ap.add_argument("--eval-every", type=int, default=d.eval_every, help="0 disables")
    ap.add_argument("--eval-solves", type=int, default=d.eval_solves)
    ap.add_argument("--balance-every", type=int, default=d.balance_every)
    ap.add_argument("--physics-share", type=float, default=d.physics_share)
    ap.add_argument("--max-eval-solves", type=int, default=d.max_eval_solves)
    ap.add_argument("--ckpt-every", type=int, default=d.ckpt_every)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue from <out>/checkpoint.pt if it exists (else start fresh)",
    )
    ap.add_argument("--width", type=int, default=m.width)
    ap.add_argument("--depth", type=int, default=m.depth)
    ap.add_argument("--delta", type=float, default=m.delta)
    ap.add_argument("--fourier-sigma", type=float, default=m.fourier_sigma)
    ap.add_argument("--n-fourier", type=int, default=m.n_fourier)
    return ap


def parse_config(argv: list[str] | None = None) -> TrainConfig:
    """Parse command-line arguments into a validated ``TrainConfig``.

    Args:
        argv: Argument list (``None`` = ``sys.argv[1:]``).

    Returns:
        The config; ``out`` defaults to ``runs/<run_name>`` under the project root.
    """
    a = build_parser().parse_args(argv)
    positive = ("steps", "batch_solves", "nodes_per_solve", "pde_nodes_per_solve", "threads")
    for name in (*positive, "ckpt_every", "log_every", "balance_every"):
        if getattr(a, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if not 0 < a.lr_min <= a.lr:
        raise SystemExit("need 0 < --lr-min <= --lr")
    model = ModelConfig(
        width=a.width,
        depth=a.depth,
        delta=a.delta,
        fourier_sigma=a.fourier_sigma,
        n_fourier=a.n_fourier,
        seed=a.seed,
    )
    cfg = TrainConfig(
        n_train_geoms=a.n_train_geoms,
        physics=a.physics == "on",
        steps=a.steps,
        batch_solves=a.batch_solves,
        nodes_per_solve=a.nodes_per_solve,
        pde_nodes_per_solve=a.pde_nodes_per_solve,
        lr=a.lr,
        lr_min=a.lr_min,
        threads=a.threads,
        seed=a.seed,
        out=a.out,
        log_every=a.log_every,
        eval_every=a.eval_every,
        eval_solves=a.eval_solves,
        balance_every=a.balance_every,
        physics_share=a.physics_share,
        max_eval_solves=a.max_eval_solves,
        ckpt_every=a.ckpt_every,
        resume=a.resume,
        model=model,
    )
    if not cfg.out:
        cfg.out = str(PROJECT_ROOT / "runs" / cfg.run_name())
    return cfg


def data_scales(path: Path = STATS_PATH) -> tuple[float, float, float]:
    """Fixed per-channel scales (std of u*, v*, p*) that normalize the data loss.

    Without them p* (std 238) would dominate u* (std 0.66) by ~10^5 in a plain MSE.

    Args:
        path: ``data/nondim_stats.json`` written by ``stats.py``.

    Returns:
        ``(std_u, std_v, std_p)`` over all packed nodes.
    """
    s = json.loads(Path(path).read_text())["all"]
    return float(s["u"]["std"]), float(s["v"]["std"]), float(s["p"]["std"])
