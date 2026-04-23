"""Seed-averaged sacred evaluation of the best policy."""
from __future__ import annotations

import numpy as np

from benchmark.cv_harness import build_extended_sales, log_experiment
from policies import MLPointPolicy
from simulation import InventorySimulator


def run(cov: int = 2, multiplier: float = 1.05, seeds=(42, 123, 456, 789, 1234),
        censoring_strategy: str = "mean_impute"):
    _, _, master = build_extended_sales()
    costs = []
    for seed in seeds:
        sim = InventorySimulator()
        policy = MLPointPolicy(
            coverage_weeks=cov, master=sim.master, multiplier=multiplier,
            random_state=seed, censoring_strategy=censoring_strategy,
        )
        res = sim.run_simulation(policy)
        costs.append(res["competition_cost"])
        print(f"  seed={seed:<5d} comp_cost={res['competition_cost']:,.2f}  "
              f"(h={res['competition_holding']:,.2f}, s={res['competition_shortage']:,.2f})")
    mean = float(np.mean(costs))
    std = float(np.std(costs, ddof=1))
    print(f"\n  MEAN = {mean:,.2f}  STD = {std:,.2f}  "
          f"[range {min(costs):,.2f} .. {max(costs):,.2f}]")
    log_experiment(
        f"ML Point cov={cov} x{multiplier} [{censoring_strategy}, SACRED, {len(seeds)}-seed mean]",
        {"mean": {"competition_cost": mean}, "val": None, "per_fold": None},
        extra={"sacred": True, "n_seeds": len(seeds), "seeds_std": std,
               "seed_range": f"{min(costs):.2f}..{max(costs):.2f}"},
    )
    return costs, mean, std
