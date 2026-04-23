"""Alpha sweep for 5-model Diverse policy on 8-fold CV+VAL."""
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
    FOLDS_8, build_extended_sales, format_summary, log_experiment,
    score_policy_on_folds,
)
from policies import DiverseCostAwarePolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.60,0.65,0.70")
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--w-ma", type=float, default=0.25)
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    full_sales, full_in_stock, master = build_extended_sales()
    summary = []
    for alpha in alphas:
        name = f"diverse5_a{alpha}_m{args.mult}_w{args.w_ma}"
        print(f"\n>>> {name}")

        def factory(a=alpha):
            return DiverseCostAwarePolicy(
                alpha=a, multiplier=args.mult, w_ma=args.w_ma,
                backtest_window=26, safety_floor=0.5, rmse_horizons=1,
                censoring_strategy="mean_impute", random_state=42, master=master,
                n_variants=5,
            )

        t0 = time.time()
        results = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales, full_in_stock=full_in_stock, master=master,
            folds=FOLDS_8, include_val=True,
            coverage_weeks=3, alpha=alpha, n_workers=1,
        )
        wall = time.time() - t0
        print(format_summary(results, name))
        print(f"[{name}] wall-time: {wall:.1f}s")
        log_experiment(name, results, extra={
            "alpha": alpha, "mult": args.mult, "w_ma": args.w_ma,
            "approach": "diverse_5model_eqwt", "wall_seconds": round(wall, 1),
        })
        summary.append({
            "alpha": alpha,
            "cv": results["mean"]["competition_cost"],
            "val": (results.get("val") or {}).get("competition_cost"),
        })

    print(f"\n=== ALPHA SWEEP SUMMARY (5-model) ===")
    print(f"{'alpha':>6} {'cv':>10} {'val':>10}")
    for r in sorted(summary, key=lambda r: r["cv"]):
        print(f"{r['alpha']:>6.3f} {r['cv']:>10,.2f} {r['val']:>10,.2f}")


if __name__ == "__main__":
    main()
