"""Generate per-round submission CSVs from the canonical final VN2 policy.

The policy is loaded from `best_winner.json` so the upload files, documented
champion result, and GitHub Actions audit all use one frozen source of truth.

Output: submissions/round_1.csv ... submissions/round_6.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Reuse the audit-grade final runner so submission generation cannot drift from
# the documented champion config in best_winner.json.
from scripts.run_final_audit import load_policy_config, build_policy, write_submissions  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


def main():
    out_dir = Path("submissions")
    sim = InventorySimulator()
    cfg = load_policy_config("best_winner.json")
    policy = build_policy(cfg, sim.master)

    print("Running canonical final policy from best_winner.json...")
    print(f"  description: {cfg.get('description')}")
    print(f"  expected sacred_cost_seed42: {cfg.get('sacred_cost_seed42')}")

    results = sim.run_simulation(policy)
    print(f"  competition_cost = {results['competition_cost']:,.2f} EUR")
    print(f"  setup_cost       = {results['setup_cost']:,.2f} EUR")
    print(f"  total_cost       = {results['total_cost']:,.2f} EUR")

    write_submissions(sim, out_dir)

    for r, order in enumerate(sim.orders_placed, start=1):
        path = out_dir / f"round_{r}.csv"
        total_units = int(order.clip(lower=0).round(0).sum())
        n_nonzero = int((order > 0).sum())
        arrival_week = r + 2
        print(
            f"  Round {r} -> arrives Week {arrival_week} -> {path} "
            f"({n_nonzero}/{len(order)} SKUs ordered, total {total_units:,} units)"
        )

    print(f"\nSubmissions written to {out_dir.resolve()}")
    print("For the order arriving in Week 8 (= Round 6), upload submissions/round_6.csv")


if __name__ == "__main__":
    main()
