"""Audit core VN2 competition mechanics without fitting ML models.

This script is intentionally lightweight: it validates the simulator constants,
required data files, scoring window, lead-time behavior, and basic result shape.
It should run in CI before any expensive sacred/full-model run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from simulation import (
    DATA_DIR,
    HOLDING_COST,
    SHORTAGE_COST,
    LEAD_TIME,
    NUM_ROUNDS,
    TOTAL_WEEKS,
    InventorySimulator,
)

EXPECTED_ROWS = 599


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audit_constants() -> None:
    _assert(HOLDING_COST == 0.2, f"HOLDING_COST should be 0.2, got {HOLDING_COST}")
    _assert(SHORTAGE_COST == 1.0, f"SHORTAGE_COST should be 1.0, got {SHORTAGE_COST}")
    _assert(LEAD_TIME == 2, f"LEAD_TIME should be 2, got {LEAD_TIME}")
    _assert(NUM_ROUNDS == 6, f"NUM_ROUNDS should be 6, got {NUM_ROUNDS}")
    _assert(TOTAL_WEEKS == 8, f"TOTAL_WEEKS should be 8, got {TOTAL_WEEKS}")


def audit_required_data() -> None:
    required = [
        "Week 0 - 2024-04-08 - Sales.csv",
        "Week 0 - In Stock.csv",
        "Week 0 - 2024-04-08 - Initial State.csv",
        "Week 0 - Master.csv",
        "Week 0 - Submission Template.csv",
    ]
    for name in required:
        path = DATA_DIR / name
        _assert(path.exists(), f"Missing required data file: {path}")

    template = pd.read_csv(DATA_DIR / "Week 0 - Submission Template.csv")
    _assert(len(template) == EXPECTED_ROWS, f"Submission template should have {EXPECTED_ROWS} rows, got {len(template)}")
    _assert(list(template.columns[:2]) == ["Store", "Product"], "Submission template first columns must be Store,Product")

    week_files = sorted(DATA_DIR.glob("Week [1-8] - *Sales.csv"))
    _assert(len(week_files) >= TOTAL_WEEKS, f"Expected at least {TOTAL_WEEKS} actual-sales week files, got {len(week_files)}")


def zero_order_policy(sim: InventorySimulator, round_idx: int, sales_hist: pd.DataFrame) -> pd.Series:
    return pd.Series(0, index=sim.end_inventory.index)


def audit_simulator_run() -> None:
    sim = InventorySimulator()
    _assert(len(sim.initial_state) == EXPECTED_ROWS, f"Initial state should have {EXPECTED_ROWS} rows, got {len(sim.initial_state)}")
    _assert(sim.actual_sales.shape[1] == TOTAL_WEEKS, f"Actual sales should have {TOTAL_WEEKS} weeks, got {sim.actual_sales.shape[1]}")

    results = sim.run_simulation(zero_order_policy)
    weekly = results["weekly_log"]

    _assert(len(weekly) == TOTAL_WEEKS, f"Weekly log should have {TOTAL_WEEKS} rows, got {len(weekly)}")
    _assert(len(sim.orders_placed) == NUM_ROUNDS, f"Should place {NUM_ROUNDS} orders, got {len(sim.orders_placed)}")
    _assert(set(weekly["week"]) == set(range(TOTAL_WEEKS)), "Weekly log should cover week indexes 0..7")

    setup_expected = weekly.loc[weekly["week"] < LEAD_TIME, "total_cost"].sum()
    comp_expected = weekly.loc[weekly["week"] >= LEAD_TIME, "total_cost"].sum()
    total_expected = weekly["total_cost"].sum()

    _assert(abs(results["setup_cost"] - setup_expected) < 1e-9, "setup_cost does not match weeks 1-2")
    _assert(abs(results["competition_cost"] - comp_expected) < 1e-9, "competition_cost does not match weeks 3-8")
    _assert(abs(results["total_cost"] - total_expected) < 1e-9, "total_cost does not match weekly total")
    _assert(abs(results["total_cost"] - results["setup_cost"] - results["competition_cost"]) < 1e-9, "total != setup + competition")


def main() -> None:
    audit_constants()
    audit_required_data()
    audit_simulator_run()
    print("OK: competition mechanics audit passed")


if __name__ == "__main__":
    main()
