"""Final evaluation on the SACRED competition weeks (1-8).

Runs our CV-winning policy (ML Point cov=2 x1.05) through the real simulator.
Single-shot. Logs result to experiments.csv with policy name suffixed "[SACRED]".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.cv_harness import log_experiment  # noqa: E402
from policies import MLPointPolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


def main():
    candidates = [
        ("ML Point cov=2 x1.05 [SACRED]", dict(coverage_weeks=2, multiplier=1.05, safety_units=0.0)),
        ("ML Point cov=2 x1.10 [SACRED]", dict(coverage_weeks=2, multiplier=1.10, safety_units=0.0)),
        ("ML Point cov=2 [SACRED]",       dict(coverage_weeks=2, multiplier=1.00, safety_units=0.0)),
        ("ML Point cov=3 [SACRED]",       dict(coverage_weeks=3, multiplier=1.00, safety_units=0.0)),
    ]

    results = []
    for name, kwargs in candidates:
        print(f"\nRunning {name}")
        sim = InventorySimulator()
        policy = MLPointPolicy(master=sim.master, **kwargs)
        res = sim.run_simulation(policy)

        # Log to experiments.csv so it lives alongside CV results.
        fake_results_dict = {
            "mean": {
                "competition_cost": res["competition_cost"],
                "comp_holding": res["competition_holding"],
                "comp_shortage": res["competition_shortage"],
            },
            "val": None,
            "per_fold": None,
        }
        log_experiment(name, fake_results_dict, extra={"sacred": True})
        results.append((name, res))

        print(f"  Total Cost (wk 1-8):      {res['total_cost']:>10,.2f} EUR")
        print(f"  Setup Cost (wk 1-2):      {res['setup_cost']:>10,.2f} EUR")
        print(f"  Competition Cost (wk 3-8):{res['competition_cost']:>10,.2f} EUR")
        print(f"    Holding:                {res['competition_holding']:>10,.2f} EUR")
        print(f"    Shortage:               {res['competition_shortage']:>10,.2f} EUR")

    print("\n\n" + "=" * 70)
    print("  SACRED RESULTS")
    print("=" * 70)
    print(f"  {'Policy':<40s} {'Comp Cost':>12s}")
    print("  " + "-" * 58)
    refs = [
        ("Official benchmark target", 4334),
        ("Matias 2nd place", 3765),
        ("Matias replayed (validation)", 3737),
    ]
    for name, cost in refs:
        print(f"  {name:<40s} {cost:>12,}")
    print("  " + "-" * 58)
    for name, res in results:
        print(f"  {name:<40s} {res['competition_cost']:>12,.2f}")


if __name__ == "__main__":
    main()
