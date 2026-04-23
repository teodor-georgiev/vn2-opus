"""Next-gen ensemble: per-SKU w_ma (backtest) + per-round multiplier."""
from __future__ import annotations

from benchmark.cv_harness import build_extended_sales
from policies import EnsemblePolicy
from experiments._shared import print_summary_table, run_and_log, sacred_eval_and_log


def _mk(master, *, w_ma=0.25, multiplier=1.05, per_round=None,
        learn_w_ma=False, censoring="mean_impute"):
    def f():
        return EnsemblePolicy(
            coverage_weeks=2, w_ma=w_ma, multiplier=multiplier,
            per_round_multiplier=per_round,
            learn_w_ma_by_backtest=learn_w_ma,
            master=master, censoring_strategy=censoring,
        )
    return f


def run():
    fs, fis, master = build_extended_sales()

    experiments = [
        # (name, kwargs)
        ("Ensemble w_ma=0.25 x1.05 [baseline]",
            dict(w_ma=0.25, multiplier=1.05)),
        # Per-round multipliers: ramp up for later rounds (tail risk).
        ("Ensemble w_ma=0.25 per_round=[1.05,1.05,1.10,1.10,1.10,1.10]",
            dict(w_ma=0.25, per_round=[1.05, 1.05, 1.10, 1.10, 1.10, 1.10])),
        ("Ensemble w_ma=0.25 per_round=[1.00,1.05,1.05,1.10,1.10,1.15]",
            dict(w_ma=0.25, per_round=[1.00, 1.05, 1.05, 1.10, 1.10, 1.15])),
        ("Ensemble w_ma=0.25 per_round=[1.10,1.10,1.10,1.05,1.05,1.00]",
            dict(w_ma=0.25, per_round=[1.10, 1.10, 1.10, 1.05, 1.05, 1.00])),
        # Per-SKU w_ma via backtest MAE.
        ("Ensemble per-SKU w_ma [backtest-MAE] x1.05",
            dict(learn_w_ma=True, multiplier=1.05)),
        ("Ensemble per-SKU w_ma [backtest-MAE] x1.10",
            dict(learn_w_ma=True, multiplier=1.10)),
        # Combined: per-SKU + per-round.
        ("Ensemble per-SKU per_round=[1.05,1.05,1.10,1.10,1.10,1.10]",
            dict(learn_w_ma=True, per_round=[1.05, 1.05, 1.10, 1.10, 1.10, 1.10])),
    ]

    rows = []
    for name, kwargs in experiments:
        factory = _mk(master, **kwargs)
        res = run_and_log(name, factory, fs, fis, master, coverage_weeks=2)
        sres = sacred_eval_and_log(name, factory)
        rows.append((name, res, sres))
    print_summary_table(rows)
