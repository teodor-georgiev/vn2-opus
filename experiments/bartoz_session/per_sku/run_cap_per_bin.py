"""Per-SKU bin multiplier sweep on CostAwarePolicy.

Loads sku_bins.csv (mean_bin, zero_bin), assigns a per-SKU mult based on
{intermittent: m_int, smooth: m_smooth, lumpy: m_lumpy} (zero_bin) OR
{Q4_high: m_q4, ...} (mean_bin). Sweeps one bin at a time (coordinate descent).

Usage:
    python run_cap_per_bin.py --bin-col zero_bin --bin intermittent \
        --mults 1.0,1.05,1.10,1.15,1.20
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

import pandas as pd  # noqa: E402

from benchmark.cv_harness import (  # noqa: E402
    FOLDS_8, build_extended_sales, format_summary, log_experiment,
    score_policy_on_folds,
)
from policies import CostAwarePolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.65)
    ap.add_argument("--bin-col", choices=["mean_bin", "zero_bin"], required=True)
    ap.add_argument("--bin", required=True, help="bin label being swept (e.g., intermittent)")
    ap.add_argument("--mults", default="0.95,1.00,1.05,1.10,1.15,1.20")
    ap.add_argument("--frozen", default=None,
                    help="comma-separated 'bin=mult' pairs for OTHER bins, e.g. smooth=0.95,lumpy=1.10")
    args = ap.parse_args()

    bins = pd.read_csv("sku_bins.csv", index_col=[0, 1])
    print(f"Bin column '{args.bin_col}' value counts:")
    print(bins[args.bin_col].value_counts())

    frozen_map: dict[str, float] = {}
    if args.frozen:
        for kv in args.frozen.split(","):
            k, v = kv.split("=")
            frozen_map[k.strip()] = float(v)

    full_sales, full_in_stock, master = build_extended_sales()
    mults = [float(m) for m in args.mults.split(",")]
    grid = []

    for m in mults:
        bin_mults = {**frozen_map, args.bin: m}
        # Build per-SKU mult Series.
        mp = bins[args.bin_col].map(bin_mults).astype(float).fillna(1.0)
        mp.index = bins.index  # already MultiIndex
        unique_settings = ", ".join(f"{k}={v:.2f}" for k, v in bin_mults.items())
        name = f"cap2_a{args.alpha}_{args.bin_col}_{args.bin}_m{m}"
        print(f"\n>>> {name}  ({unique_settings})  per_sku_mult.unique={sorted(mp.unique())}")

        def factory(mp=mp.copy()):
            return CostAwarePolicy(
                alpha=args.alpha, multiplier=1.0,
                multiplier_per_sku=mp,
                backtest_window=26, safety_floor=0.5, rmse_horizons=1,
                ensemble_cfg={"coverage_weeks": 3, "w_ma": 0.25,
                              "censoring_strategy": "mean_impute", "random_state": 42},
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
            "alpha": args.alpha, "bin_col": args.bin_col,
            "swept_bin": args.bin, "mult": m,
            "bin_mults": ";".join(f"{k}={v:.2f}" for k, v in bin_mults.items()),
            "wall_seconds": round(wall, 1),
        })
        grid.append({
            "mult": m,
            "cv": results["mean"]["competition_cost"],
            "val": (results.get("val") or {}).get("competition_cost"),
        })

    print(f"\n=== SUMMARY ({args.bin_col}={args.bin}) ===")
    print(f"{'mult':>6} {'cv':>10} {'val':>10}")
    for r in sorted(grid, key=lambda r: r["cv"]):
        print(f"{r['mult']:>6.2f} {r['cv']:>10,.2f} {r['val']:>10,.2f}")


if __name__ == "__main__":
    main()
