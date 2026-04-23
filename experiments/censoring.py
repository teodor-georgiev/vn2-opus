"""Censoring-strategy comparison for ML Point cov=2 x1.05."""
from __future__ import annotations

from benchmark.cv_harness import build_extended_sales, log_experiment
from policies import MLPointPolicy
from experiments._shared import print_summary_table, run_and_log, sacred_eval_and_log


def _mk(master, strategy, mult=1.05):
    def f():
        return MLPointPolicy(
            coverage_weeks=2, master=master, multiplier=mult,
            censoring_strategy=strategy,
        )
    return f


def run():
    fs, fis, master = build_extended_sales()
    strategies = ("interpolate", "mean_impute", "seasonal_impute", "zero")
    rows = []
    for strat in strategies:
        name = f"ML Point cov=2 x1.05 [censoring={strat}]"
        factory = _mk(master, strat)
        res = run_and_log(name, factory, fs, fis, master, coverage_weeks=2)
        sres = sacred_eval_and_log(name, factory)
        rows.append((name, res, sres))
    print_summary_table(rows)
