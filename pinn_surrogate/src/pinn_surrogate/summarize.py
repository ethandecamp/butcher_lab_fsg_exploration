"""Collect ``runs/*/eval.json`` into the ablation table (torch-free).

Writes ``runs/ablation_summary.json`` (all aggregates) and ``runs/ablation_summary.md``
(one row per run: n, physics, train wall-clock, val & test_B medians).

Usage::

    PYTHONPATH=src python3 -m pinn_surrogate.summarize --runs runs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pinn_surrogate.config import PROJECT_ROOT

log = logging.getLogger(__name__)

SPLITS = ("val", "test_B")
# (column header, metric key) for the per-split columns of the markdown table.
COLUMNS = (
    ("tau relL2", "tau_rel_l2"),
    ("p relL2", "p_rel_l2"),
    ("peak loc", "peak_loc_err"),
    ("sign agree", "sign_agree"),
    ("PDE res", "pde_res"),
)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def collect(runs_dir: Path) -> list[dict]:
    """One record per run directory that has an ``eval.json``, sorted by (n, physics)."""
    records = []
    for ev_path in sorted(runs_dir.glob("*/eval.json")):
        run = ev_path.parent
        cfg = _read_json(run / "config.json") or {}
        summary = _read_json(run / "train_summary.json") or {}
        ev = json.loads(ev_path.read_text())
        records.append(
            {
                "run": run.name,
                "n_train_geoms": cfg.get("n_train_geoms"),
                "physics": cfg.get("physics"),
                "seed": cfg.get("seed"),
                "steps": summary.get("steps", cfg.get("steps")),
                "train_s": summary.get("train_s"),
                "sec_per_step": summary.get("sec_per_step"),
                "medians": {
                    split: {
                        key: ev["splits"][split]["aggregate"]["all"][key]["median"]
                        for _, key in COLUMNS
                    }
                    for split in SPLITS
                    if split in ev.get("splits", {})
                },
                "aggregate": {s: v["aggregate"] for s, v in ev.get("splits", {}).items()},
            }
        )
    return sorted(
        records,
        key=lambda r: (r["n_train_geoms"] or 0, not r["physics"], r["seed"] or 0, r["run"]),
    )


def _fmt(x: float | None, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def to_markdown(records: list[dict]) -> str:
    """The ablation table as GitHub markdown."""
    head = ["run", "n geoms", "physics", "train time"]
    for split in SPLITS:
        head += [f"{split} {name}" for name, _ in COLUMNS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in records:
        t = r["train_s"]
        cells = [
            r["run"],
            str(r["n_train_geoms"]),
            {True: "on", False: "off"}.get(r["physics"], "?"),
            "-" if t is None else f"{t / 60:.1f} min",
        ]
        for split in SPLITS:
            med = r["medians"].get(split, {})
            for name, key in COLUMNS:
                spec = ".3f" if key == "sign_agree" else ".3g"
                cells.append(_fmt(med.get(key), spec))
        lines.append("| " + " | ".join(cells) + " |")
    note = (
        "\nMedians over solves. tau/p relL2 at the 256 wall points (p gauge-free: each "
        "relative to its own outlet mean); peak loc = |arc-length error of peak |tau||/L; "
        "sign agree = fraction of points with |tau| > 1% max where sign matches; PDE res = "
        "median per solve of the normalized interior residual sqrt((|r_mom| / s_mom)^2 + (r_div / s_div)^2), s = local term size from the data; the momentum part is Dan's r_rel (FEM median 1.3e-4).\n"
    )
    return "\n".join(lines) + "\n" + note


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Summarize ablation runs.")
    ap.add_argument("--runs", type=Path, default=PROJECT_ROOT / "runs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = collect(args.runs)
    if not records:
        raise SystemExit(f"no */eval.json under {args.runs}")
    (args.runs / "ablation_summary.json").write_text(json.dumps(records, indent=1))
    md = to_markdown(records)
    (args.runs / "ablation_summary.md").write_text(md)
    log.info("%s", md)
    log.info("wrote %s and .md (%d runs)", args.runs / "ablation_summary.json", len(records))


if __name__ == "__main__":
    main()
