"""CumulativeQuantilePolicy sweep across alpha (and coverage)."""
from __future__ import annotations

from benchmark.cv_harness import build_extended_sales
from policies import CumulativeQuantilePolicy
from experiments._shared import print_summary_table, run_and_log, sacred_eval_and_log


def _mk(cov=2, alpha=0.65, multiplier=1.0):
    def f():
        return CumulativeQuantilePolicy(
            coverage=cov, alpha=alpha, multiplier=multiplier,
            censoring_strategy="mean_impute", ensemble=True,
        )
    return f


def run():
    fs, fis, master = build_extended_sales()
    grid = [
        # Sweep alpha at cov=2 (our best coverage)
        (2, 0.50, 1.00),
        (2, 0.55, 1.00),
        (2, 0.60, 1.00),
        (2, 0.65, 1.00),
        (2, 0.70, 1.00),
        (2, 0.75, 1.00),
        (2, 0.833, 1.00),
        # Also cov=3 for safety
        (3, 0.55, 1.00),
        (3, 0.65, 1.00),
    ]
    rows = []
    for cov, alpha, mult in grid:
        name = f"CumQuantile cov={cov} alpha={alpha} x{mult}"
        factory = _mk(cov=cov, alpha=alpha, multiplier=mult)
        res = run_and_log(name, factory, fs, fis, master, coverage_weeks=cov, alpha=alpha)
        sres = sacred_eval_and_log(name, factory)
        rows.append((name, res, sres))
    print_summary_table(rows)
