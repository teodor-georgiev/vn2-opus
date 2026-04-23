"""Sweep Gaussian-safety alpha on top of ML point forecaster.

Runs each (policy, alpha) across 4 CV folds + VAL in parallel.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cap inner BLAS / LGBM / CB threads per worker so 4 workers don't contend.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from benchmark.cv_harness import (  # noqa: E402
    build_extended_sales,
    format_summary,
    score_policy_on_folds,
)
from policies import GaussianSafetyPolicy, MLPointPolicy  # noqa: E402


def make_mlpoint(master):
    def f():
        return MLPointPolicy(coverage_weeks=3, master=master)
    return f


def make_gauss(master, alpha, cov=3, sigma_source="residuals"):
    def f():
        return GaussianSafetyPolicy(
            coverage_weeks=cov, alpha=alpha, sigma_source=sigma_source, master=master
        )
    return f


def main():
    print("Loading data...")
    full_sales, full_in_stock, master = build_extended_sales()
    print(f"  full_sales shape: {full_sales.shape}")

    N_WORKERS = int(os.environ.get("CV_WORKERS", "5"))

    experiments = [
        ("ML Point (cov=3)",                     make_mlpoint(master),                           3, None),
        ("Gaussian Safety cov=3 alpha=0.60",     make_gauss(master, 0.60),                      3, 0.60),
        ("Gaussian Safety cov=3 alpha=0.75",     make_gauss(master, 0.75),                      3, 0.75),
        ("Gaussian Safety cov=3 alpha=0.833",    make_gauss(master, 0.833),                     3, 0.833),
        ("Gaussian Safety cov=3 alpha=0.90",     make_gauss(master, 0.90),                      3, 0.90),
        ("Gaussian Safety cov=3 alpha=0.833 [history sigma]",
            make_gauss(master, 0.833, sigma_source="history"),                                  3, 0.833),
    ]

    summary = []
    for name, factory, cov, alpha in experiments:
        print(f"\nRunning: {name}")
        res = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales,
            full_in_stock=full_in_stock,
            master=master,
            coverage_weeks=cov,
            alpha=alpha,
            n_workers=N_WORKERS,
        )
        summary.append((name, res))
        print(format_summary(res, name))

    print("\n\n" + "=" * 86)
    print("  SWEEP SUMMARY — CV mean vs VAL (lower is better; target 4,334; Matias ~3,765)")
    print("=" * 86)
    header = f"  {'Policy':<52s} {'CV mean':>10s} {'VAL':>10s} {'MAE':>6s} {'bias':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, res in summary:
        m = res["mean"]
        v = res.get("val") or {}
        print(
            f"  {name:<52s} {m['competition_cost']:>10,.0f} "
            f"{v.get('competition_cost', float('nan')):>10,.0f} "
            f"{m.get('fc_mae', float('nan')):>6.2f} {m.get('fc_bias', float('nan')):>+7.2f}"
        )


if __name__ == "__main__":
    main()
