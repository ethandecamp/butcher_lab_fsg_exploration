"""Torch-free check of the ablation summary on fake run directories."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pinn_surrogate import summarize


def _fake_run(root: Path, n: int, physics: bool, tau: float, with_summary: bool = True) -> None:
    """Write the three JSON files a finished run leaves behind."""
    run = root / f"n{n}_phys{'on' if physics else 'off'}_s0"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps({"n_train_geoms": n, "physics": physics, "seed": 0, "steps": 100})
    )
    if with_summary:
        (run / "train_summary.json").write_text(
            json.dumps({"steps": 100, "train_s": 600.0, "sec_per_step": 6.0})
        )
    agg = {
        m: {"median": tau if m == "tau_rel_l2" else 0.5, "mean": 1.0, "n": 4}
        for m in ("tau_rel_l2", "p_rel_l2", "peak_loc_err", "sign_agree", "pde_res")
    }
    splits = {s: {"aggregate": {"all": agg}, "per_solve": []} for s in ("val", "test_B")}
    (run / "eval.json").write_text(json.dumps({"splits": splits}))


class TestSummarize(unittest.TestCase):
    """collect -> table -> files."""

    def test_collect_and_table(self) -> None:
        """Runs are sorted by n then physics on before off; medians and times are shown."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_run(root, 80, False, 0.3)
            _fake_run(root, 20, True, 0.2, with_summary=False)
            _fake_run(root, 80, True, 0.1)
            (root / "not_a_run").mkdir()  # no eval.json -> ignored
            records = summarize.collect(root)
            self.assertEqual(
                [(r["n_train_geoms"], r["physics"]) for r in records],
                [(20, True), (80, True), (80, False)],
            )
            self.assertEqual(records[1]["medians"]["test_B"]["tau_rel_l2"], 0.1)
            self.assertIsNone(records[0]["train_s"])
            md = summarize.to_markdown(records)
            lines = md.splitlines()
            self.assertEqual(len([ln for ln in lines if ln.startswith("| n")]), 3)
            self.assertIn("10.0 min", md)
            self.assertIn("| n20_physon_s0 | 20 | on | - |", md)
            with contextlib.redirect_stderr(io.StringIO()):
                summarize.main(["--runs", str(root)])
            self.assertEqual(len(json.loads((root / "ablation_summary.json").read_text())), 3)
            self.assertTrue((root / "ablation_summary.md").read_text().startswith("| run"))

    def test_no_runs_exits(self) -> None:
        """An empty runs directory is an error, not an empty table."""
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            summarize.main(["--runs", tmp])


if __name__ == "__main__":
    unittest.main()
