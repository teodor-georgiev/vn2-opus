"""Generate per-round submission CSVs from our best policy (CostAware alpha=0.65).

Runs the sacred simulation, extracts each round's order, and writes one CSV per
round in the same format as `Week 0 - Submission Template.csv`:
    Header: Store,Product,0
    599 rows in Sales-CSV index order, integer ≥ 0 quantities.

Output: submissions/round_1.csv ... submissions/round_6.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

from policies import CostAwarePolicy, DiverseCostAwarePolicy  # noqa: E402
from run_lgb_share_sweep import DiverseLGBShareCostAwarePolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


def main():
    out_dir = Path("submissions")
    out_dir.mkdir(exist_ok=True)

    sim = InventorySimulator()
    policy = DiverseLGBShareCostAwarePolicy(
        lgb_share=0.90,
        alpha=0.5813428492464154,
        mult_low=1.1731901672544072,
        mult_high=1.1918987771673257,
        high_demand_quantile=0.75,
        w_ma=0.33800656001887974,
        backtest_window=26, safety_floor=0.5, rmse_horizons=1,
        censoring_strategy="mean_impute", random_state=42, master=sim.master,
        n_variants=5,
        categorical_features=True,
    )
    print("Running sacred sim with DiverseCostAware (5-model, Optuna-tuned)...")
    results = sim.run_simulation(policy)
    print(f"  competition_cost = {results['competition_cost']:,.2f} EUR")

    # The template's row order = the index of Week 0 sales CSV. Use that as canonical.
    template = pd.read_csv("Data/Week 0 - Submission Template.csv")
    canonical_index = list(zip(template["Store"], template["Product"]))

    for r, order in enumerate(sim.orders_placed, start=1):
        # `order` is a pd.Series indexed by (Store, Product) MultiIndex.
        # Reindex to canonical row order, fill any missing with 0, ensure int ≥ 0.
        ordered = order.reindex(canonical_index).fillna(0).clip(lower=0).round(0).astype(int)
        out = pd.DataFrame({
            "Store": [s for s, p in canonical_index],
            "Product": [p for s, p in canonical_index],
            "0": ordered.values,
        })
        path = out_dir / f"round_{r}.csv"
        out.to_csv(path, index=False)
        arrival_week = r + 2
        total_units = int(out["0"].sum())
        n_nonzero = int((out["0"] > 0).sum())
        print(f"  Round {r}  -> arrives Week {arrival_week}  -> "
              f"{path}  ({n_nonzero}/{len(out)} SKUs ordered, total {total_units:,} units)")

    print(f"\nSubmissions written to {out_dir.resolve()}")
    print("\nFor the order arriving in Week 8 (= Round 6), upload submissions/round_6.csv")


if __name__ == "__main__":
    main()
