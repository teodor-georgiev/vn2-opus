"""Sweep 2: ML Point with coverage_weeks, multiplier, safety_units variations.

Appends each result to benchmark/experiments.csv via log_experiment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from benchmark.cv_harness import (  # noqa: E402
    build_extended_sales,
    format_summary,
    log_experiment,
    score_policy_on_folds,
)
from policies import MLPointPolicy  # noqa: E402


def mk(master, cov=3, multiplier=1.0, safety_units=0.0):
    def f():
        return MLPointPolicy(
            coverage_weeks=cov, master=master,
            multiplier=multiplier, safety_units=safety_units,
        )
    return f


def main():
    full_sales, full_in_stock, master = build_extended_sales()
    N = int(os.environ.get("CV_WORKERS", "3"))

    experiments = [
        # baseline
        ("ML Point cov=2",                     mk(master, cov=2),                                  2),
        ("ML Point cov=3",                     mk(master, cov=3),                                  3),
        ("ML Point cov=4",                     mk(master, cov=4),                                  4),
        # multiplicative bias correction on cov=3
        ("ML Point cov=3 x1.05",               mk(master, cov=3, multiplier=1.05),                 3),
        ("ML Point cov=3 x1.10",               mk(master, cov=3, multiplier=1.10),                 3),
        ("ML Point cov=3 x1.20",               mk(master, cov=3, multiplier=1.20),                 3),
        ("ML Point cov=3 x1.30",               mk(master, cov=3, multiplier=1.30),                 3),
        # additive safety on cov=3
        ("ML Point cov=3 +1u",                 mk(master, cov=3, safety_units=1.0),                3),
        ("ML Point cov=3 +2u",                 mk(master, cov=3, safety_units=2.0),                3),
        ("ML Point cov=3 +3u",                 mk(master, cov=3, safety_units=3.0),                3),
    ]

    summary_rows = []
    for name, factory, cov in experiments:
        print(f"\nRunning: {name}")
        res = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales,
            full_in_stock=full_in_stock,
            master=master,
            coverage_weeks=cov,
            alpha=None,
            n_workers=N,
        )
        log_experiment(name, res)
        print(format_summary(res, name))
        summary_rows.append((name, res))

    print("\n\n" + "=" * 80)
    print("  SWEEP 2 SUMMARY  (target 4,334 | Matias 3,765)")
    print("=" * 80)
    print(f"  {'Policy':<40s} {'CV mean':>10s} {'VAL':>10s} {'holding':>8s} {'shortage':>8s}")
    print("  " + "-" * 78)
    for name, res in summary_rows:
        m = res["mean"]
        v = res.get("val") or {}
        print(
            f"  {name:<40s} {m['competition_cost']:>10,.0f} "
            f"{v.get('competition_cost', float('nan')):>10,.0f} "
            f"{m['comp_holding']:>8,.0f} {m['comp_shortage']:>8,.0f}"
        )


if __name__ == "__main__":
    main()
