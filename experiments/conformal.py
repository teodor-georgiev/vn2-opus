"""Conformal ensemble: per-SKU additive offset from backtest residuals."""
from __future__ import annotations

from benchmark.cv_harness import build_extended_sales
from policies import ConformalEnsemblePolicy
from experiments._shared import print_summary_table, run_and_log, sacred_eval_and_log


def _mk(master, *, alpha_cp=0.70, multiplier=1.0, w_ma=0.25, backtest_window=26,
        min_offset=0.0):
    def f():
        return ConformalEnsemblePolicy(
            coverage_weeks=2, w_ma=w_ma, alpha_cp=alpha_cp,
            multiplier=multiplier, backtest_window=backtest_window,
            min_offset=min_offset, master=master,
            censoring_strategy="mean_impute",
        )
    return f


def run():
    fs, fis, master = build_extended_sales()

    experiments = [
        ("Conformal alpha_cp=0.50 x1.00",      dict(alpha_cp=0.50, multiplier=1.00)),
        ("Conformal alpha_cp=0.60 x1.00",      dict(alpha_cp=0.60, multiplier=1.00)),
        ("Conformal alpha_cp=0.70 x1.00",      dict(alpha_cp=0.70, multiplier=1.00)),
        ("Conformal alpha_cp=0.80 x1.00",      dict(alpha_cp=0.80, multiplier=1.00)),
        ("Conformal alpha_cp=0.833 x1.00",     dict(alpha_cp=0.833, multiplier=1.00)),
        # Combined with small multiplier.
        ("Conformal alpha_cp=0.70 x1.05",      dict(alpha_cp=0.70, multiplier=1.05)),
        # Only positive offsets (min_offset=0: never subtract).
        ("Conformal alpha_cp=0.60 x1.05 clip",
            dict(alpha_cp=0.60, multiplier=1.05, min_offset=0.0)),
        # Larger backtest window.
        ("Conformal alpha_cp=0.70 K=52",
            dict(alpha_cp=0.70, multiplier=1.00, backtest_window=52)),
    ]
    rows = []
    for name, kwargs in experiments:
        factory = _mk(master, **kwargs)
        res = run_and_log(name, factory, fs, fis, master, coverage_weeks=2)
        sres = sacred_eval_and_log(name, factory)
        rows.append((name, res, sres))
    print_summary_table(rows)
