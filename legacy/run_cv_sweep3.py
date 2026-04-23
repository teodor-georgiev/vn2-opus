"""Sweep 3: refine around cov=2 winner. Try cov=1, cov=2 with small
multipliers (<, =, > 1.0), and small additive safety units.
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


def mk(master, cov=2, multiplier=1.0, safety_units=0.0):
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
        # explore cov=1 (could be too aggressive but cheap to test)
        ("ML Point cov=1",                       mk(master, cov=1),                                 1),
        ("ML Point cov=1 x1.10",                 mk(master, cov=1, multiplier=1.10),                1),
        ("ML Point cov=1 x1.20",                 mk(master, cov=1, multiplier=1.20),                1),
        # cov=2 baseline (retune around winner)
        ("ML Point cov=2 x0.90",                 mk(master, cov=2, multiplier=0.90),                2),
        ("ML Point cov=2 x0.95",                 mk(master, cov=2, multiplier=0.95),                2),
        ("ML Point cov=2 x1.00",                 mk(master, cov=2, multiplier=1.00),                2),
        ("ML Point cov=2 x1.05",                 mk(master, cov=2, multiplier=1.05),                2),
        ("ML Point cov=2 x1.10",                 mk(master, cov=2, multiplier=1.10),                2),
        # additive safety on cov=2
        ("ML Point cov=2 +1u",                   mk(master, cov=2, safety_units=1.0),               2),
        ("ML Point cov=2 -1u",                   mk(master, cov=2, safety_units=-1.0),              2),
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
    print("  SWEEP 3 SUMMARY  (target 4,334 | Matias 3,765 | cov=2 baseline 3,782)")
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
