"""Sweep recency_decay at current champion config. Tests if down-weighting old data helps."""
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
    ap.add_argument("--decays", default="None,0.5,0.7,0.9,0.95,0.99",
                    help="'None' for no decay, floats for decay factor")
    args = ap.parse_args()

    decays = []
    for x in args.decays.split(","):
        x = x.strip()
        decays.append(None if x == "None" else float(x))

    full_sales, full_in_stock, master = build_extended_sales()
    summary = []
    for rd in decays:
        name = f"diverse5_recency_{rd}"
        print(f"\n>>> {name}")

        def factory(rd=rd):
            return DiverseCostAwarePolicy(
                alpha=0.5813428492464154,
                mult_low=1.1731901672544072,
                mult_high=1.1918987771673257,
                high_demand_quantile=0.75,
                w_ma=0.33800656001887974,
                backtest_window=26, safety_floor=0.5, rmse_horizons=1,
                censoring_strategy="mean_impute", random_state=42, master=master,
                n_variants=5,
                recency_decay=rd,
            )

        t0 = time.time()
        results = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales, full_in_stock=full_in_stock, master=master,
            folds=FOLDS_8, include_val=True,
            coverage_weeks=3, alpha=0.58, n_workers=1,
        )
        wall = time.time() - t0
        print(format_summary(results, name))
        print(f"[{name}] wall-time: {wall:.1f}s")
        log_experiment(name, results, extra={
            "recency_decay": rd, "approach": "diverse_5model_recency_sweep",
            "wall_seconds": round(wall, 1),
        })
        summary.append({
            "decay": rd,
            "cv": results["mean"]["competition_cost"],
            "val": (results.get("val") or {}).get("competition_cost"),
        })

    print("\n=== RECENCY SWEEP SUMMARY ===")
    print(f"{'decay':>8} {'cv':>10} {'val':>10}")
    for r in sorted(summary, key=lambda r: r["cv"]):
        print(f"{str(r['decay']):>8} {r['cv']:>10,.2f} {r['val']:>10,.2f}")


if __name__ == "__main__":
    main()
