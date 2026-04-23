"""Evaluate DiverseCostAwarePolicy (5-model equal-weight ensemble + MA blend) on 8-fold CV+VAL."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark.cv_harness import (  # noqa: E402
    FOLDS, FOLDS_8, build_extended_sales, format_summary, log_experiment,
    score_policy_on_folds,
)
from policies import DiverseCostAwarePolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.65)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--w-ma", type=float, default=0.25)
    ap.add_argument("--n-variants", type=int, default=5, choices=[5, 9])
    ap.add_argument("--folds-8", action="store_true")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    folds_dict = FOLDS_8 if args.folds_8 else FOLDS
    if args.folds != "all":
        names = set(args.folds.split(","))
        folds_dict = {k: v for k, v in folds_dict.items() if k in names}
    include_val = not args.no_val

    name = args.name or f"diverse{args.n_variants}_a{args.alpha}_m{args.mult}_w{args.w_ma}"
    full_sales, full_in_stock, master = build_extended_sales()

    def factory():
        return DiverseCostAwarePolicy(
            alpha=args.alpha, multiplier=args.mult, w_ma=args.w_ma,
            backtest_window=26, safety_floor=0.5, rmse_horizons=1,
            censoring_strategy="mean_impute", random_state=42, master=master,
            n_variants=args.n_variants,
        )

    print(f"[{name}] folds={list(folds_dict)} val={include_val} alpha={args.alpha} mult={args.mult} w_ma={args.w_ma}")
    t0 = time.time()
    results = score_policy_on_folds(
        policy_factory=factory,
        full_sales=full_sales, full_in_stock=full_in_stock, master=master,
        folds=folds_dict, include_val=include_val,
        coverage_weeks=3, alpha=args.alpha, n_workers=1,
    )
    wall = time.time() - t0
    print(format_summary(results, name))
    print(f"\n[{name}] wall-time: {wall:.1f}s")

    log_experiment(name, results, extra={
        "alpha": args.alpha, "mult": args.mult, "w_ma": args.w_ma,
        "approach": "diverse_5model_eqwt", "wall_seconds": round(wall, 1),
    })


if __name__ == "__main__":
    main()
