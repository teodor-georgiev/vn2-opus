"""Test hierarchical / categorical / intermittency features on the champion config."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark.cv_harness import (  # noqa: E402
    FOLDS_8, build_extended_sales, format_summary, log_experiment,
    score_policy_on_folds,
)
from run_lgb_share_sweep import DiverseLGBShareCostAwarePolicy  # noqa: E402


# Champion base config — all variants inherit these.
CHAMP = dict(
    lgb_share=0.90,
    alpha=0.5813428492464154,
    mult_low=1.1731901672544072,
    mult_high=1.1918987771673257,
    high_demand_quantile=0.75,
    w_ma=0.33800656001887974,
    backtest_window=26, safety_floor=0.5, rmse_horizons=1,
    censoring_strategy="mean_impute", random_state=42,
    n_variants=5,
)

# Feature-flag combinations to try. Keep it small since each variant = ~6 min.
VARIANTS = [
    {"name": "champion_baseline",  "hier": False, "cat": False, "intm": False},
    {"name": "only_hierarchical",  "hier": True,  "cat": False, "intm": False},
    {"name": "only_categorical",   "hier": False, "cat": True,  "intm": False},
    {"name": "only_intermittency", "hier": False, "cat": False, "intm": True},
    {"name": "all_three",          "hier": True,  "cat": True,  "intm": True},
]


def main():
    full_sales, full_in_stock, master = build_extended_sales()
    results_summary = []

    for v in VARIANTS:
        name = f"champ_features_{v['name']}"
        print(f"\n>>> {name}  hier={v['hier']} cat={v['cat']} intm={v['intm']}")

        def factory(v=v):
            return DiverseLGBShareCostAwarePolicy(
                **CHAMP, master=master,
                hierarchical_features=v["hier"],
                categorical_features=v["cat"],
                intermittency_features=v["intm"],
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
            **CHAMP, "hier_features": v["hier"],
            "cat_features": v["cat"], "intm_features": v["intm"],
            "approach": "diverse5_lgbshare_feature_sweep",
            "wall_seconds": round(wall, 1),
        })
        results_summary.append({
            "name": v["name"],
            "cv": results["mean"]["competition_cost"],
            "val": (results.get("val") or {}).get("competition_cost"),
        })

    print("\n=== FEATURE SWEEP SUMMARY ===")
    print(f"{'variant':<20} {'cv':>10} {'val':>10}")
    for r in sorted(results_summary, key=lambda r: r["cv"]):
        print(f"{r['name']:<20} {r['cv']:>10,.2f} {r['val']:>10,.2f}")


if __name__ == "__main__":
    main()
