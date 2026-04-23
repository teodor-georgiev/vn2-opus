"""Evaluate CostAwarePolicy (Bartosz's Stage 2) on CV+VAL with the LGB+CB ensemble.

Usage:
    python run_costaware.py --alpha 0.833 --mult 1.0 --name cap_a0.833_m1.0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark.cv_harness import (  # noqa: E402
    FOLDS, FOLDS_8, VAL_START, build_extended_sales, format_summary,
    log_experiment, score_policy_on_folds,
)
from policies import CostAwarePolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.833, help="critical ratio (default 1/1.2)")
    ap.add_argument("--mult", type=float, default=1.0, help="extra multiplier on target Q_alpha(d3)")
    ap.add_argument("--safety-floor", type=float, default=0.5)
    ap.add_argument("--backtest-window", type=int, default=26)
    ap.add_argument("--w-ma", type=float, default=0.25, help="ensemble w_ma for d1..d3 blend")
    ap.add_argument("--folds", default="all", help="all | cv0 | cv0,cv1")
    ap.add_argument("--folds-8", action="store_true", help="use 8-fold CV (weeks 85..141)")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    folds_dict = FOLDS_8 if args.folds_8 else FOLDS
    if args.folds != "all":
        names = set(args.folds.split(","))
        folds_dict = {k: v for k, v in folds_dict.items() if k in names}
    include_val = not args.no_val

    name = args.name or f"cap_a{args.alpha}_m{args.mult}"
    full_sales, full_in_stock, master = build_extended_sales()

    def factory():
        return CostAwarePolicy(
            alpha=args.alpha,
            backtest_window=args.backtest_window,
            safety_floor=args.safety_floor,
            multiplier=args.mult,
            ensemble_cfg={
                "coverage_weeks": 3,
                "w_ma": args.w_ma,
                "censoring_strategy": "mean_impute",
                "random_state": 42,
            },
        )

    print(f"[{name}] folds={list(folds_dict)} val={include_val} alpha={args.alpha} mult={args.mult}")
    t0 = time.time()
    results = score_policy_on_folds(
        policy_factory=factory,
        full_sales=full_sales,
        full_in_stock=full_in_stock,
        master=master,
        folds=folds_dict,
        include_val=include_val,
        coverage_weeks=3,   # we report MAE/WAPE over next 3 weeks
        alpha=args.alpha,
        n_workers=args.workers,
    )
    wall = time.time() - t0

    print(format_summary(results, name))
    print(f"\n[{name}] total wall-time: {wall:.1f}s")

    log_experiment(name, results, extra={
        "alpha": args.alpha, "mult": args.mult, "w_ma": args.w_ma,
        "safety_floor": args.safety_floor, "backtest_window": args.backtest_window,
        "wall_seconds": round(wall, 1),
    })


if __name__ == "__main__":
    main()
