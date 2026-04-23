"""Per-round multiplier sweep on CostAwarePolicy. Coordinate descent: fix
all-rounds-mult=1.0 except sweep one round at a time.

Usage:
    python run_cap_per_round.py --round 5 --mults 0.80,0.85,0.90,0.95,1.00,1.05
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark.cv_harness import (  # noqa: E402
    FOLDS_8, build_extended_sales, format_summary, log_experiment,
    score_policy_on_folds,
)
from policies import CostAwarePolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.65)
    ap.add_argument("--round", type=int, required=True, help="0-5: which round to sweep")
    ap.add_argument("--mults", default="0.80,0.85,0.90,0.95,1.00,1.05")
    ap.add_argument("--frozen", default=None,
                    help="comma-separated 6-element template e.g. 1.0,1.0,1.0,1.0,1.0,0.90 "
                         "(values for rounds 0..5; the swept round is overwritten)")
    args = ap.parse_args()

    full_sales, full_in_stock, master = build_extended_sales()

    if args.frozen:
        base = [float(x) for x in args.frozen.split(",")]
        assert len(base) == 6
    else:
        base = [1.0] * 6

    mults = [float(m) for m in args.mults.split(",")]
    grid = []
    for m in mults:
        prm = list(base)
        prm[args.round] = m
        name = f"cap2_a{args.alpha}_r{args.round}_m{m}"
        print(f"\n>>> {name}  per_round_multiplier={prm}")

        def factory(prm=tuple(prm)):
            return CostAwarePolicy(
                alpha=args.alpha, multiplier=1.0,
                per_round_multiplier=list(prm),
                backtest_window=26, safety_floor=0.5, rmse_horizons=1,
                ensemble_cfg={
                    "coverage_weeks": 3, "w_ma": 0.25,
                    "censoring_strategy": "mean_impute", "random_state": 42,
                },
            )

        t0 = time.time()
        results = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales, full_in_stock=full_in_stock, master=master,
            folds=FOLDS_8, include_val=True, coverage_weeks=3,
            alpha=args.alpha, n_workers=1,
        )
        wall = time.time() - t0
        print(format_summary(results, name))
        print(f"[{name}] wall-time: {wall:.1f}s")

        log_experiment(name, results, extra={
            "alpha": args.alpha, "swept_round": args.round, "mult": m,
            "per_round_mult": ";".join(f"{x:.2f}" for x in prm),
            "wall_seconds": round(wall, 1),
        })
        grid.append({
            "mult_r{}".format(args.round): m,
            "cv": results["mean"]["competition_cost"],
            "val": (results.get("val") or {}).get("competition_cost"),
        })

    print("\n=== SUMMARY (sweep round={}) ===".format(args.round))
    print(f"{'mult':>6} {'cv':>10} {'val':>10}")
    for r in sorted(grid, key=lambda r: r["cv"]):
        print(f"{r[f'mult_r{args.round}']:>6.3f} {r['cv']:>10,.2f} {r['val']:>10,.2f}")


if __name__ == "__main__":
    main()
