"""k*RMSE safety stock sweep, RMSE computed by in-sample backtest."""
from __future__ import annotations

from benchmark.cv_harness import build_extended_sales
from policies import RMSESafetyPolicy
from experiments._shared import print_summary_table, run_and_log, sacred_eval_and_log


def _mk(master, k=0.5, multiplier=1.0):
    def f():
        return RMSESafetyPolicy(
            coverage_weeks=2, k=k, multiplier=multiplier, master=master,
            censoring_strategy="mean_impute",
        )
    return f


def run():
    fs, fis, master = build_extended_sales()
    grid = [
        (0.00, 1.05),  # just point forecast + multiplier (= baseline)
        (0.25, 1.00),
        (0.50, 1.00),
        (0.75, 1.00),
        (1.00, 1.00),
        (0.25, 1.05),
        (0.50, 1.05),
    ]
    rows = []
    for k, mult in grid:
        name = f"RMSE-safety k={k} cov=2 x{mult}"
        factory = _mk(master, k=k, multiplier=mult)
        res = run_and_log(name, factory, fs, fis, master, coverage_weeks=2)
        sres = sacred_eval_and_log(name, factory)
        rows.append((name, res, sres))
    print_summary_table(rows)
