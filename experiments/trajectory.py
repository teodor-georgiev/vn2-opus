"""Trajectory-simulation order optimization."""
from __future__ import annotations

from benchmark.cv_harness import build_extended_sales
from policies import TrajectoryPolicy
from experiments._shared import print_summary_table, run_and_log, sacred_eval_and_log


def _mk(master, n_samples=30, multiplier=1.0):
    def f():
        return TrajectoryPolicy(
            coverage_weeks=2, n_samples=n_samples, multiplier=multiplier,
            master=master, censoring_strategy="mean_impute",
        )
    return f


def run():
    fs, fis, master = build_extended_sales()
    # Keep n_samples modest for speed (per-SKU loop is Python-slow).
    grid = [
        (20, 1.00),
        (50, 1.00),
    ]
    rows = []
    for ns, mult in grid:
        name = f"Trajectory nsamples={ns} x{mult}"
        factory = _mk(master, n_samples=ns, multiplier=mult)
        res = run_and_log(name, factory, fs, fis, master, coverage_weeks=2)
        sres = sacred_eval_and_log(name, factory)
        rows.append((name, res, sres))
    print_summary_table(rows)
