"""CostAwarePolicy + tuned Ensemble base (the 3,581 sacred config) on 8-fold CV+VAL.

Sweeps (alpha, mult). Picks CV winner under 8-fold rule; logs each variant.
"""
from __future__ import annotations

import argparse
import json
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


def load_tuned_cfg() -> dict:
    """Map best_winner.json to an EnsemblePolicy kwargs dict for CostAware base."""
    with open("best_winner.json") as f:
        bw = json.load(f)
    cfg = bw["config"]
    # CostAware needs at least cov=2 (for d1, d2 = next & arrival week).
    # We override `multiplier` and `safety_units` because CostAware does its own ordering.
    return {
        "coverage_weeks": max(2, int(cfg.get("coverage_weeks", 2))),
        "w_ma": float(cfg.get("w_ma", 0.25)),
        "w_lgb_share": cfg.get("w_lgb_share", None),
        "censoring_strategy": cfg.get("censoring_strategy", "mean_impute"),
        "per_series_scaling": bool(cfg.get("per_series_scaling", False)),
        "demand_cluster_k": cfg.get("demand_cluster_k", None),
        "extended_features": bool(cfg.get("extended_features", False)),
        "direct_forecast": bool(cfg.get("direct_forecast", False)),
        "recency_decay": cfg.get("recency_decay", None),
        "categorical_features": bool(cfg.get("categorical_features", False)),
        "intermittency_features": bool(cfg.get("intermittency_features", False)),
        "w_stats": float(cfg.get("w_stats", 0.0)),
        "random_state": 42,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.60,0.65,0.70")
    ap.add_argument("--mults", default="1.0")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    mults = [float(m) for m in args.mults.split(",")]

    base_cfg = load_tuned_cfg()
    print(f"Tuned ensemble cfg: {base_cfg}")

    full_sales, full_in_stock, master = build_extended_sales()

    grid = []
    for alpha in alphas:
        for mult in mults:
            name = f"cap2_8f_TUNED_a{alpha}_m{mult}"
            print(f"\n>>> {name}")

            def factory(a=alpha, m=mult):
                return CostAwarePolicy(
                    alpha=a, multiplier=m,
                    backtest_window=26, safety_floor=0.5,
                    rmse_horizons=1,
                    ensemble_cfg=dict(base_cfg),
                )

            t0 = time.time()
            results = score_policy_on_folds(
                policy_factory=factory,
                full_sales=full_sales, full_in_stock=full_in_stock, master=master,
                folds=FOLDS_8, include_val=True,
                coverage_weeks=3, alpha=a if False else alpha,  # for harness reporting
                n_workers=args.workers,
            )
            wall = time.time() - t0
            print(format_summary(results, name))
            print(f"[{name}] wall-time: {wall:.1f}s")

            log_experiment(name, results, extra={
                "alpha": alpha, "mult": mult, "tuned_base": True,
                "wall_seconds": round(wall, 1),
            })
            grid.append({
                "alpha": alpha, "mult": mult,
                "cv": results["mean"]["competition_cost"],
                "val": (results.get("val") or {}).get("competition_cost"),
            })

    print("\n=== SUMMARY ===")
    print(f"{'alpha':>6} {'mult':>6} {'cv':>10} {'val':>10}")
    for r in sorted(grid, key=lambda r: r["cv"]):
        print(f"{r['alpha']:>6.3f} {r['mult']:>6.3f} {r['cv']:>10,.2f} {r['val']:>10,.2f}")


if __name__ == "__main__":
    main()
