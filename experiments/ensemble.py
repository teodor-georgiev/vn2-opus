"""Ensemble forecaster sweep: SeasonalMA weighted with ML Point."""
from __future__ import annotations

from benchmark.cv_harness import build_extended_sales
from policies import EnsemblePolicy
from experiments._shared import print_summary_table, run_and_log, sacred_eval_and_log


def _mk(master, w_ma=0.5, multiplier=1.05, censoring="mean_impute"):
    def f():
        return EnsemblePolicy(
            coverage_weeks=2, w_ma=w_ma, multiplier=multiplier,
            master=master, censoring_strategy=censoring,
        )
    return f


def run():
    fs, fis, master = build_extended_sales()
    grid = [
        (0.00, 1.05),   # pure ML Point (baseline reference)
        (0.25, 1.05),
        (0.50, 1.05),
        (0.75, 1.05),
        (1.00, 1.05),   # pure seasonal MA
        (0.50, 1.10),
    ]
    rows = []
    for w_ma, mult in grid:
        name = f"Ensemble w_ma={w_ma} cov=2 x{mult}"
        factory = _mk(master, w_ma=w_ma, multiplier=mult)
        res = run_and_log(name, factory, fs, fis, master, coverage_weeks=2)
        sres = sacred_eval_and_log(name, factory)
        rows.append((name, res, sres))
    print_summary_table(rows)
