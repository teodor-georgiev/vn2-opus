"""Evaluate NegBinomPolicy across a grid of (coverage, alpha) on CV + sacred."""
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
from negbinom_policy import NegBinomPolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


def main():
    full_sales, full_in_stock, master = build_extended_sales()
    N = int(os.environ.get("CV_WORKERS", "3"))

    grid = [
        # (cov, alpha)
        (2, 0.50),
        (2, 0.60),
        (2, 0.70),
        (2, 0.833),
        (3, 0.50),
        (3, 0.60),
        (3, 0.70),
    ]

    rows = []
    for cov, alpha in grid:
        name = f"NegBinom cov={cov} alpha={alpha}"
        print(f"\nRunning: {name}")
        def factory(c=cov, a=alpha):
            return NegBinomPolicy(coverage_weeks=c, alpha=a, multiplier=1.0)
        res = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales,
            full_in_stock=full_in_stock,
            master=master,
            coverage_weeks=cov,
            alpha=alpha,
            n_workers=N,
        )
        log_experiment(name, res)
        print(format_summary(res, name))

        sim = InventorySimulator()
        sres = sim.run_simulation(factory())
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
    print("  NEGBINOM SUMMARY")
    print("=" * 80)
    print(f"  {'Policy':<32s} {'CV mean':>10s} {'VAL':>10s} {'SACRED':>10s}")
    print("  " + "-" * 78)
    for name, res, sres in rows:
        m = res["mean"]
        v = res.get("val") or {}
        print(f"  {name:<32s} {m['competition_cost']:>10,.0f} "
              f"{v.get('competition_cost', float('nan')):>10,.0f} "
              f"{sres['competition_cost']:>10,.0f}")


if __name__ == "__main__":
    main()
