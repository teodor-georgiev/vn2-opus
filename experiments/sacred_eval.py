"""Sacred evaluation of a policy (called by pipeline.py sacred).

Two modes:
  - Direct one-shot (unsafe): sacred without any gate.
  - Gated (recommended): runs CV+VAL first, checks coherence with baseline,
    then runs sacred only if VAL stays within tolerance.
"""
from __future__ import annotations

from benchmark.cv_harness import build_extended_sales, gated_sacred_eval, log_experiment
from policies import MLPointPolicy
from simulation import InventorySimulator


def run(cov: int = 2, multiplier: float = 1.05, censoring: str = "interpolate",
        random_state: int = 42, gated: bool = False, force_sacred: bool = False,
        val_tolerance: float = 0.10):
    """Direct mode (default) or gated mode.

    gated=True : run CV+VAL first, block sacred if VAL regresses > val_tolerance
                 vs baseline. Pass force_sacred=True to override.
    """
    if gated:
        fs, fis, master = build_extended_sales()

        def factory():
            return MLPointPolicy(
                coverage_weeks=cov, master=master, multiplier=multiplier,
                censoring_strategy=censoring, random_state=random_state,
            )
        return gated_sacred_eval(
            name=f"ML Point cov={cov} x{multiplier} [{censoring}]",
            policy_factory=factory,
            full_sales=fs, full_in_stock=fis, master=master,
            val_tolerance=val_tolerance, force_sacred=force_sacred,
            n_workers=3, coverage_weeks=cov,
        )

    # Direct one-shot sacred.
    sim = InventorySimulator()
    policy = MLPointPolicy(
        coverage_weeks=cov, master=sim.master, multiplier=multiplier,
        censoring_strategy=censoring, random_state=random_state,
    )
    res = sim.run_simulation(policy)
    name = f"ML Point cov={cov} x{multiplier} [{censoring}, SACRED]"
    print(f"\n{name}")
    print(f"  Total Cost (wk 1-8):      {res['total_cost']:>10,.2f} EUR")
    print(f"  Setup Cost (wk 1-2):      {res['setup_cost']:>10,.2f} EUR")
    print(f"  Competition Cost (wk 3-8):{res['competition_cost']:>10,.2f} EUR")
    print(f"    Holding:                {res['competition_holding']:>10,.2f} EUR")
    print(f"    Shortage:               {res['competition_shortage']:>10,.2f} EUR")
    log_experiment(
        name,
        {"mean": {"competition_cost": res["competition_cost"],
                  "comp_holding": res["competition_holding"],
                  "comp_shortage": res["competition_shortage"]},
         "val": None, "per_fold": None},
        extra={"sacred": True, "censoring": censoring, "seed": random_state},
    )
    return res
