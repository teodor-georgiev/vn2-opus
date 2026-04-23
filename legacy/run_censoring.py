"""Compare censored-demand handling strategies on CV + VAL + sacred.

Strategies:
  interpolate — current default
  drop        — out-of-stock weeks dropped from training (NaN then dropna)
  zero        — raw sales kept (stockout weeks contribute 0 to training)

Best policy: ML Point cov=2 x1.05.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "2")

from benchmark.cv_harness import (  # noqa: E402
    build_extended_sales, format_summary, log_experiment, score_policy_on_folds,
)
from policies import MLPointPolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


def mk(master, strategy, mult=1.05):
    def f():
        return MLPointPolicy(
            coverage_weeks=2, master=master, multiplier=mult,
            censoring_strategy=strategy,
        )
    return f


def main():
    full_sales, full_in_stock, master = build_extended_sales()
    N = int(os.environ.get("CV_WORKERS", "3"))

    strategies = ["interpolate", "mean_impute", "seasonal_impute", "zero"]
    rows = []
    for strat in strategies:
        name = f"ML Point cov=2 x1.05 [censoring={strat}]"
        print(f"\nRunning: {name}")
        res = score_policy_on_folds(
            policy_factory=mk(master, strat),
            full_sales=full_sales, full_in_stock=full_in_stock, master=master,
            coverage_weeks=2, alpha=None, n_workers=N, include_val=True,
        )
        log_experiment(name, res)
        print(format_summary(res, name))

        sim = InventorySimulator()
        sres = sim.run_simulation(mk(master, strat)())
        print(f"  SACRED: {sres['competition_cost']:,.2f}  "
              f"(h={sres['competition_holding']:,.2f}, s={sres['competition_shortage']:,.2f})")
        log_experiment(f"{name} [SACRED]",
                       {"mean": {"competition_cost": sres["competition_cost"],
                                 "comp_holding": sres["competition_holding"],
                                 "comp_shortage": sres["competition_shortage"]},
                        "val": None, "per_fold": None},
                       extra={"sacred": True})
        rows.append((name, res, sres))

    print("\n\n" + "=" * 80)
    print("  CENSORING STRATEGY COMPARISON (cov=2 x1.05)")
    print("=" * 80)
    print(f"  {'Strategy':<45s} {'CV mean':>9s} {'VAL':>9s} {'SACRED':>9s}")
    print("  " + "-" * 78)
    for name, res, sres in rows:
        m = res["mean"]
        v = res.get("val") or {}
        print(f"  {name:<45s} {m['competition_cost']:>9,.0f} "
              f"{v.get('competition_cost', float('nan')):>9,.0f} "
              f"{sres['competition_cost']:>9,.0f}")


if __name__ == "__main__":
    main()
