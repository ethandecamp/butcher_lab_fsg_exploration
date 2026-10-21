"""Torch-free checks of command-line parsing and config serialization."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pinn_surrogate.config import (
    PROJECT_ROOT,
    STATS_PATH,
    ModelConfig,
    TrainConfig,
    data_scales,
    parse_config,
)


def _parse_fails(argv: list[str]) -> bool:
    """True if parsing exits (argparse errors print to stderr; silence them)."""
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parse_config(argv)
        except SystemExit:
            return True
    return False


class TestParse(unittest.TestCase):
    """CLI -> TrainConfig."""

    def test_defaults(self) -> None:
        """Defaults match the task spec and name the run directory."""
        cfg = parse_config([])
        self.assertEqual(
            (cfg.n_train_geoms, cfg.physics, cfg.steps, cfg.batch_solves, cfg.nodes_per_solve),
            (20, True, 3000, 8, 512),
        )
        self.assertEqual((cfg.lr, cfg.lr_min, cfg.balance_every), (1e-3, 1e-5, 100))
        self.assertEqual(Path(cfg.out), PROJECT_ROOT / "runs" / "n20_physon_s0")
        self.assertEqual(cfg.model, ModelConfig())

    def test_arguments(self) -> None:
        """Every argument lands in the right field (model args in the nested config)."""
        cfg = parse_config(
            ["--n-train-geoms", "160", "--physics", "off", "--steps", "7", "--seed", "3"]
            + ["--threads", "1", "--out", "/tmp/x", "--width", "64", "--delta", "0.1"]
        )
        self.assertEqual((cfg.n_train_geoms, cfg.physics, cfg.steps), (160, False, 7))
        self.assertEqual((cfg.seed, cfg.threads, cfg.out), (3, 1, "/tmp/x"))
        self.assertEqual((cfg.model.width, cfg.model.delta, cfg.model.seed), (64, 0.1, 3))
        self.assertEqual(cfg.run_name(), "n160_physoff_s3")

    def test_rejects_bad_values(self) -> None:
        """Train size < 1, unknown physics, zero steps, lr_min > lr.

        (v1 also rejected sizes outside {20, 80, 160}; v2 allows any size up to the pack's
        train count, which ``Packed.train_subset`` enforces.)
        """
        self.assertTrue(_parse_fails(["--n-train-geoms", "0"]))
        self.assertFalse(_parse_fails(["--n-train-geoms", "50"]))
        self.assertTrue(_parse_fails(["--physics", "maybe"]))
        self.assertTrue(_parse_fails(["--steps", "0"]))
        self.assertTrue(_parse_fails(["--lr", "1e-4", "--lr-min", "1e-3"]))

    def test_json_round_trip(self) -> None:
        """save -> load gives an equal config; unknown keys are rejected."""
        cfg = parse_config(["--physics", "off", "--width", "48"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            cfg.save(path)
            back = TrainConfig.from_dict(json.loads(path.read_text()))
        self.assertEqual(back, cfg)
        with self.assertRaises(KeyError):
            TrainConfig.from_dict({**cfg.to_dict(), "lr_typo": 1})

    def test_checkpoint_arguments(self) -> None:
        """--ckpt-every and --resume parse; --resume is off unless given."""
        self.assertFalse(parse_config([]).resume)
        cfg = parse_config(["--ckpt-every", "50", "--resume"])
        self.assertEqual((cfg.ckpt_every, cfg.resume), (50, True))
        self.assertTrue(_parse_fails(["--ckpt-every", "0"]))

    def test_resume_mismatch(self) -> None:
        """Bookkeeping fields may change on resume; anything affecting the math may not."""
        saved = parse_config(["--steps", "3000"]).to_dict()
        ok = parse_config(["--steps", "3000", "--threads", "1", "--resume", "--out", "/x"])
        self.assertEqual(ok.resume_mismatch(saved), [])
        bad = parse_config(["--steps", "500", "--width", "64", "--resume"])
        self.assertEqual(bad.resume_mismatch(saved), ["model", "steps"])

    def test_data_scales(self) -> None:
        """Loss scales are the stds in nondim_stats.json (u 0.656, v 0.197, p 238)."""
        s_u, s_v, s_p = data_scales(STATS_PATH)
        self.assertAlmostEqual(s_u, 0.6557, places=3)
        self.assertAlmostEqual(s_v, 0.1967, places=3)
        self.assertAlmostEqual(s_p, 237.93, places=1)


if __name__ == "__main__":
    unittest.main()
