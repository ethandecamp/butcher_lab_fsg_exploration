"""Paired comparison of physics-on vs physics-off runs across training sizes and seeds.

For every (training size n, seed) the two arms are evaluated on the *same* held-out solves,
so the comparison is paired per solve: for each solve, did the physics-on model have the
lower error? A two-sided sign test on those wins (exact binomial, no scipy) says whether
the split is distinguishable from a coin flip. Seeds are reported separately and pooled.

Usage::

    python3 -m pinn_surrogate.compare --runs runs/grid
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

METRICS = ("tau_rel_l2", "p_rel_l2", "peak_loc_err", "pde_res")


def sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact sign test: P(a fair coin gives a split at least this uneven)."""
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def per_solve(run: Path, split: str) -> dict[int, dict]:
    """Per-solve metric rows of ``run``'s eval.json for ``split``, keyed by solve index."""
    rows = json.loads((run / "eval.json").read_text())["splits"][split]["per_solve"]
    return {r["solve"]: r for r in rows}


def paired(on: dict[int, dict], off: dict[int, dict], metric: str) -> dict[str, float]:
    """Paired statistics of ``metric`` over the solves both runs evaluated."""
    keys = sorted(set(on) & set(off))
    a = np.array([on[k][metric] for k in keys])
    b = np.array([off[k][metric] for k in keys])
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    ties = int(np.sum(a == b))
    wins = int(np.sum(a < b))
    n = len(a) - ties
    return {
        "n": len(a),
        "median_on": float(np.median(a)),
        "median_off": float(np.median(b)),
        "median_diff": float(np.median(a - b)),
        "on_better_frac": wins / n if n else float("nan"),
        "sign_p": sign_test_p(wins, n),
    }


def by_k(on: dict[int, dict], off: dict[int, dict], metric: str) -> dict[int, tuple]:
    """Median ``metric`` per flow index k (0 = lowest Re) for both arms."""
    out = {}
    for k in range(4):
        a = [r[metric] for r in on.values() if r["k"] == k]
        b = [r[metric] for r in off.values() if r["k"] == k]
        out[k] = (float(np.median(a)), float(np.median(b)))
    return out


def compare(runs: Path) -> dict:
    """Build the comparison for every (n, seed) pair found under ``runs``."""
    result: dict = {"pairs": [], "pooled": {}}
    seeds = sorted({int(p.name.rsplit("_s", 1)[1]) for p in runs.glob("n*_phys*_s*") if p.is_dir()})
    for n in (20, 80, 160):
        for split in ("val", "test_B"):
            pool: dict[str, list] = {m: [] for m in METRICS}
            for seed in seeds:
                on_dir, off_dir = runs / f"n{n}_physon_s{seed}", runs / f"n{n}_physoff_s{seed}"
                if not ((on_dir / "eval.json").exists() and (off_dir / "eval.json").exists()):
                    continue
                on, off = per_solve(on_dir, split), per_solve(off_dir, split)
                row = {"n": n, "seed": seed, "split": split}
                for m in METRICS:
                    row[m] = paired(on, off, m)
                    keys = sorted(set(on) & set(off))
                    pool[m] += [(on[k][m], off[k][m]) for k in keys]
                row["tau_by_k"] = by_k(on, off, "tau_rel_l2")
                result["pairs"].append(row)
            pooled = {}
            for m, pairs in pool.items():
                if not pairs:
                    continue
                a, b = np.array(pairs).T
                wins, n_eff = int(np.sum(a < b)), int(np.sum(a != b))
                pooled[m] = {
                    "median_on": float(np.median(a)),
                    "median_off": float(np.median(b)),
                    "on_better_frac": wins / n_eff if n_eff else float("nan"),
                    "sign_p": sign_test_p(wins, n_eff),
                    "n_pairs": len(pairs),
                }
            result["pooled"][f"n{n}_{split}"] = pooled
    return result


def to_markdown(res: dict) -> str:
    """Two tables: pooled over seeds, then each seed's paired numbers for test_B."""
    lines = [
        "## Physics on vs off, pooled over seeds (paired per solve)",
        "",
        "| n | split | metric | median on | median off | on better | sign-test p |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, pooled in res["pooled"].items():
        n, split = key.split("_", 1)
        for m, s in pooled.items():
            lines.append(
                f"| {n[1:]} | {split} | {m} | {s['median_on']:.3g} | {s['median_off']:.3g} | "
                f"{100 * s['on_better_frac']:.0f}% of {s['n_pairs']} | {s['sign_p']:.2g} |"
            )
    lines += [
        "",
        "## Per seed, test_B wall shear (median relL2)",
        "",
        "| n | seed | on | off | on better | sign-test p | by k (on/off), k=0..3 low->high Re |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in res["pairs"]:
        if row["split"] != "test_B":
            continue
        t = row["tau_rel_l2"]
        ks = ", ".join(f"{a:.2f}/{b:.2f}" for a, b in row["tau_by_k"].values())
        lines.append(
            f"| {row['n']} | {row['seed']} | {t['median_on']:.3f} | {t['median_off']:.3f} | "
            f"{100 * t['on_better_frac']:.0f}% | {t['sign_p']:.2g} | {ks} |"
        )
    lines += [
        "",
        (
            "'on better' = share of solves (ties excluded) where the physics-on model had the "
            "lower error. Sign test: two-sided exact binomial against 50%."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs/grid"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = compare(args.runs)
    (args.runs / "comparison.json").write_text(json.dumps(res, indent=1))
    (args.runs / "comparison.md").write_text(to_markdown(res))
    log.info("wrote %s/comparison.{json,md}", args.runs)


if __name__ == "__main__":
    main()
