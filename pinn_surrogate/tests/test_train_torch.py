"""End-to-end smoke tests of train.py (incl. --resume) and single_case.py (need torch).

The model and batches are tiny so the whole file runs in about a minute on a laptop CPU;
these check plumbing (shapes, files, checkpoint/resume), not accuracy.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pinn_surrogate.config import parse_config

try:
    import torch

    from pinn_surrogate import single_case
    from pinn_surrogate import train as train_mod
    from pinn_surrogate.model import torch_load

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

TINY = [
    "--steps", "6", "--batch-solves", "2", "--nodes-per-solve", "64",
    "--pde-nodes-per-solve", "32", "--width", "16", "--depth", "2", "--log-every", "1",
    "--ckpt-every", "3", "--eval-every", "0", "--max-eval-solves", "1", "--threads", "1",
    "--balance-every", "2",
]  # fmt: skip


def _steps_logged(run: Path) -> list[int]:
    lines = (run / "metrics.jsonl").read_text().splitlines()
    return [json.loads(ln)["step"] for ln in lines]


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTrain(unittest.TestCase):
    """train.train on a tiny configuration."""

    def test_resume_reproduces_uninterrupted_run(self) -> None:
        """Crash after step 4 (last checkpoint at 3), resume -> same weights as no crash."""
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            train_mod.train(parse_config([*TINY, "--out", str(a)]))
            for name in ("config.json", "metrics.jsonl", "model.pt", "eval.json"):
                self.assertTrue((a / name).exists(), name)

            real, calls = train_mod.sample_batch, {"n": 0}

            def crash_on_step_5(*args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 5:
                    raise KeyboardInterrupt  # stands in for a VM restart
                return real(*args, **kwargs)

            with (
                mock.patch.object(train_mod, "sample_batch", crash_on_step_5),
                self.assertRaises(KeyboardInterrupt),
            ):
                train_mod.train(parse_config([*TINY, "--out", str(b)]))
            self.assertEqual(train_mod.checkpoint_step(b / train_mod.CKPT_NAME), 3)
            self.assertEqual(_steps_logged(b), [1, 2, 3, 4])

            with self.assertRaises(SystemExit):  # no --resume: refuse to overwrite
                train_mod.train(parse_config([*TINY, "--out", str(b)]))
            with self.assertRaises(SystemExit):  # different step budget: refuse to mix
                train_mod.train(parse_config([*TINY, "--out", str(b), "--resume", "--steps", "9"]))

            train_mod.train(parse_config([*TINY, "--out", str(b), "--resume"]))
            self.assertEqual(_steps_logged(b), [1, 2, 3, 4, 5, 6])
            self.assertEqual(json.loads((b / "train_summary.json").read_text())["sessions"], 2)
            sa = torch_load(a / train_mod.MODEL_NAME)["state_dict"]
            sb = torch_load(b / train_mod.MODEL_NAME)["state_dict"]
            for key in sa:
                torch.testing.assert_close(sb[key], sa[key], rtol=1e-6, atol=1e-7, msg=key)

            mtime = (b / "eval.json").stat().st_mtime_ns  # complete run + --resume: no-op
            train_mod.train(parse_config([*TINY, "--out", str(b), "--resume"]))
            self.assertEqual((b / "eval.json").stat().st_mtime_ns, mtime)

    def test_tau_loss_runs_and_zero_weight_is_v1(self) -> None:
        """--tau-weight > 0 logs a tau term; --tau-weight 0 gives the exact v1 weights."""
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = (Path(tmp) / n for n in "abc")
            off = [*TINY, "--physics", "off"]
            train_mod.train(parse_config([*off, "--out", str(a)]))
            train_mod.train(parse_config([*off, "--tau-weight", "0", "--out", str(b)]))
            train_mod.train(parse_config([*off, "--tau-weight", "1", "--out", str(c)]))
            wa, wb, wc = (torch_load(r / "model.pt")["state_dict"] for r in (a, b, c))
            for k in wa:
                self.assertTrue(torch.equal(wa[k], wb[k]), k)
            self.assertFalse(all(torch.equal(wa[k], wc[k]) for k in wa))
            row = json.loads((c / "metrics.jsonl").read_text().splitlines()[-1])
            self.assertIn("loss_tau", row)
            self.assertEqual(row["lambda_tau"], 1.0)

    def test_physics_off_and_eval_contents(self) -> None:
        """Data-only arm logs only the data loss; eval.json has both splits and all metrics."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "off"
            train_mod.train(parse_config([*TINY, "--physics", "off", "--out", str(run)]))
            row = json.loads((run / "metrics.jsonl").read_text().splitlines()[-1])
            self.assertIn("loss_data", row)
            self.assertNotIn("loss_pde", row)
            ev = json.loads((run / "eval.json").read_text())
            for split in ("val", "test_B"):
                agg = ev["splits"][split]["aggregate"]
                self.assertEqual(set(agg), {"all", "by_k", "by_family"})
                self.assertIn("tau_rel_l2", agg["all"])
            self.assertTrue((run / "error_vs_re.png").exists())


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestSingleCase(unittest.TestCase):
    """single_case.run on two steps."""

    def test_runs_and_reports(self) -> None:
        """Writes summary.json with the wall errors and seconds per step."""
        with tempfile.TemporaryDirectory() as tmp:
            args = single_case.build_parser().parse_args(
                ["--steps", "2", "--colloc", "64", "--report-every", "1", "--threads", "1"]
                + ["--out", tmp]
            )
            summary = single_case.run(args)
            self.assertEqual(summary["steps"], 2)
            for key in ("tau_rel_l2", "p_rel_l2", "sec_per_step"):
                self.assertIn(key, summary["final"])
            self.assertTrue((Path(tmp) / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
