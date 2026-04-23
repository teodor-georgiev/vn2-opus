"""Run the 4-fold CV + VAL for benchmark, ML-point, and quantile policies.

Usage:  python run_cv.py

Reports per-fold and mean competition_cost alongside forecast accuracy
(MAE, WAPE, bias, pinball@0.833 where applicable) so you can see whether a
policy is limited by its forecast or by its ordering logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.cv_harness import build_extended_sales, format_summary, score_policy_on_folds
from policies import MLPointPolicy, QuantilePolicy, SeasonalBenchmarkPolicy


def main():
    print("Loading data...")
    full_sales, full_in_stock, master = build_extended_sales()
    print(f"  full_sales shape: {full_sales.shape}  "
          f"(history columns: {len(full_sales.columns) - 8}, hidden competition: 8)")

    experiments = [
        ("Seasonal Benchmark (4wk cov)", lambda: SeasonalBenchmarkPolicy(coverage_weeks=4), 4, None),
        ("ML Point (LGB+CB, 3wk cov)", lambda: MLPointPolicy(coverage_weeks=3, master=master), 3, None),
        ("Quantile (LGB+CB, alpha=0.833, 3wk cov)",
         lambda: QuantilePolicy(coverage_weeks=3, alpha=0.833, master=master), 3, 0.833),
    ]

    all_summaries = []
    means = {}
    vals = {}
    for name, factory, cov, alpha in experiments:
        print(f"\nRunning: {name}")
        res = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales,
            full_in_stock=full_in_stock,
            master=master,
            coverage_weeks=cov,
            alpha=alpha,
        )
        all_summaries.append((name, res))
        means[name] = res["mean"]
        vals[name] = res["val"]
        print(format_summary(res, name))

    # Final comparative table.
    print("\n\n" + "=" * 78)
    print("  SUMMARY — CV mean competition_cost + forecast accuracy")
    print("=" * 78)
    header = f"  {'Policy':<42s} {'CV mean':>10s} {'VAL':>10s} {'MAE':>8s} {'WAPE':>6s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, _, _, _ in experiments:
        m = means[name]
        v = vals[name] or {}
        mae = m.get("fc_mae", float("nan"))
        wape = m.get("fc_wape", float("nan"))
        print(
            f"  {name:<42s} {m['competition_cost']:>10,.0f} {v.get('competition_cost', float('nan')):>10,.0f} "
            f"{mae:>8.2f} {wape:>6.3f}"
        )
    print()
    print("Reference points (sacred — do not tune against):")
    print(f"  Official benchmark target  :  4,334 EUR")
    print(f"  Matias 2nd place           :  3,765 EUR")
    print(f"  Matias replayed through us :  3,737 EUR")


if __name__ == "__main__":
    main()
