"""Sweep alpha for ConformalCostAwarePolicy on 8-fold CV+VAL.

Uses mlforecast's built-in conformal prediction intervals (no Normal assumption).
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
    FOLDS_8, build_extended_sales, format_summary, log_experiment,
    score_policy_on_folds,
)
from policies import ConformalCostAwarePolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.55,0.60,0.65,0.70,0.75")
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--n-windows", type=int, default=4)
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    full_sales, full_in_stock, master = build_extended_sales()

    grid = []
    for alpha in alphas:
        name = f"conf_cap_a{alpha}_m{args.mult}"
        print(f"\n>>> {name}  (n_windows={args.n_windows}, h={args.h})")
        def factory(a=alpha):
            return ConformalCostAwarePolicy(
                alpha=a, multiplier=args.mult,
                prediction_intervals_n_windows=args.n_windows,
                prediction_intervals_h=args.h,
            )

        t0 = time.time()
        results = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales, full_in_stock=full_in_stock, master=master,
            folds=FOLDS_8, include_val=True, coverage_weeks=3,
            alpha=alpha, n_workers=args.workers,
        )
        wall = time.time() - t0
        print(format_summary(results, name))
        print(f"[{name}] wall-time: {wall:.1f}s")
        log_experiment(name, results, extra={
            "alpha": alpha, "mult": args.mult, "method": "conformal_distribution",
            "n_windows": args.n_windows, "h": args.h,
            "wall_seconds": round(wall, 1),
        })
        grid.append({
            "alpha": alpha,
            "cv": results["mean"]["competition_cost"],
            "val": (results.get("val") or {}).get("competition_cost"),
        })

    print("\n=== SUMMARY ===")
    print(f"{'alpha':>6} {'cv':>10} {'val':>10}")
    for r in sorted(grid, key=lambda r: r["cv"]):
        print(f"{r['alpha']:>6.3f} {r['cv']:>10,.2f} {r['val']:>10,.2f}")


if __name__ == "__main__":
    main()
