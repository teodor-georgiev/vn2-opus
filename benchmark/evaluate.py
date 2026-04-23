"""
Evaluate our policies against the official VN2 benchmark.

Runs the official benchmark code (standalone, as Nicolas published it)
and compares with our simulation results side by side.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from simulation import (
    InventorySimulator,
    benchmark_policy,
    HOLDING_COST,
    SHORTAGE_COST,
    LEAD_TIME,
    NUM_ROUNDS,
    TOTAL_WEEKS,
)
from forecaster import DemandForecaster

INDEX = ["Store", "Product"]
DATA_DIR = ROOT / "Data"


def run_official_benchmark():
    """Run the official benchmark exactly as Nicolas published it (single-shot, Week 0 only)."""
    sales = pd.read_csv(DATA_DIR / "Week 0 - 2024-04-08 - Sales.csv").set_index(INDEX)
    in_stock = pd.read_csv(DATA_DIR / "Week 0 - In Stock.csv").set_index(INDEX)
    state = pd.read_csv(DATA_DIR / "Week 0 - 2024-04-08 - Initial State.csv").set_index(INDEX)
    sales.columns = pd.to_datetime(sales.columns)
    in_stock.columns = pd.to_datetime(in_stock.columns)

    sales[~in_stock] = np.nan

    season = sales.mean().rename("Demand").to_frame()
    season["Week Number"] = season.index.isocalendar().week
    season = season.groupby("Week Number").mean()
    season = season / season.mean()

    sales_weeks = sales.columns.isocalendar().week
    sales_no_season = sales / (season.loc[sales_weeks.values]).values.reshape(-1)

    base_forecast = sales_no_season.iloc[:, -13:].mean(axis=1)

    f_periods = pd.date_range(
        start=sales.columns[-1], periods=10, inclusive="neither", freq="W-MON"
    )
    forecast = pd.DataFrame(
        data=base_forecast.values.reshape(-1, 1).repeat(len(f_periods), axis=1),
        columns=f_periods,
        index=sales.index,
    )
    forecast = forecast * (season.loc[f_periods.isocalendar().week.values]).values.reshape(-1)

    order_up_to = forecast.iloc[:, :4].sum(axis=1)
    net_inventory = state[["In Transit W+1", "In Transit W+2", "End Inventory"]].sum(axis=1)
    order = (order_up_to - net_inventory).clip(lower=0).round(0).astype(int)

    return order, forecast


def run_official_through_sim(official_order_round0):
    """Feed the official Round-0 order into our simulator, then use the same
    benchmark logic for subsequent rounds (since the official code only produces
    one order)."""
    sim = InventorySimulator()

    def official_policy(sim, round_idx, sales_hist):
        if round_idx == 0:
            return official_order_round0
        return benchmark_policy(sim, round_idx, sales_hist)

    return sim.run_simulation(official_policy)


def make_ml_policy(coverage_weeks=3, safety_factor=1.0):
    master = pd.read_csv(DATA_DIR / "Week 0 - Master.csv").set_index(INDEX)
    in_stock = pd.read_csv(DATA_DIR / "Week 0 - In Stock.csv").set_index(INDEX)
    in_stock.columns = pd.to_datetime(in_stock.columns)

    def ml_policy(sim, round_idx, sales_hist):
        forecaster = DemandForecaster(master=master)
        forecaster.fit(sales_hist, in_stock)
        forecast = forecaster.predict(horizon=coverage_weeks)
        demand_over_horizon = forecast.sum(axis=1) * safety_factor
        net_inv = sim.get_net_inventory_position()
        return (demand_over_horizon - net_inv).clip(lower=0).round(0).astype(int)

    return ml_policy


def make_winner_policy(coverage_weeks=3, phi=0.1, z=0.97):
    master = pd.read_csv(DATA_DIR / "Week 0 - Master.csv").set_index(INDEX)
    in_stock = pd.read_csv(DATA_DIR / "Week 0 - In Stock.csv").set_index(INDEX)
    in_stock.columns = pd.to_datetime(in_stock.columns)

    def winner_policy(sim, round_idx, sales_hist):
        forecaster = DemandForecaster(master=master)
        forecaster.fit(sales_hist, in_stock)
        forecast = forecaster.predict(horizon=coverage_weeks)
        demand_over_horizon = forecast.sum(axis=1)
        safety_stock = z * phi * np.sqrt(demand_over_horizon.clip(lower=0))
        order_up_to = demand_over_horizon + safety_stock
        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)

    return winner_policy


def format_results_table(results_dict):
    rows = []
    for name, res in results_dict.items():
        rows.append({
            "Policy": name,
            "Net Cost": res["competition_cost"],
            "Net Holding": res["competition_holding"],
            "Net Shortage": res["competition_shortage"],
            "Setup Cost": res["setup_cost"],
            "Total Cost": res["total_cost"],
        })
    df = pd.DataFrame(rows).set_index("Policy")
    return df


def print_comparison(results_dict, reference_key="Official Benchmark"):
    df = format_results_table(results_dict)
    ref_cost = df.loc[reference_key, "Net Cost"]

    print("\n" + "=" * 80)
    print("  VN2 BENCHMARK EVALUATION")
    print("=" * 80)

    print(f"\n  {'Policy':<30s} {'Net Cost':>10s} {'Holding':>10s} {'Shortage':>10s} {'vs Official':>12s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for name, row in df.iterrows():
        delta_pct = (ref_cost - row["Net Cost"]) / ref_cost * 100
        marker = " <-- ref" if name == reference_key else ""
        print(
            f"  {name:<30s} {row['Net Cost']:>10,.1f} {row['Net Holding']:>10,.1f} "
            f"{row['Net Shortage']:>10,.1f} {delta_pct:>+11.1f}%{marker}"
        )

    print(f"\n  Note: 'Net Cost' excludes the first {LEAD_TIME} setup weeks (uncontrollable).")
    print(f"  Holding cost = {HOLDING_COST}/unit/week, Shortage cost = {SHORTAGE_COST}/unit/week")

    print("\n\n  WEEKLY BREAKDOWN — Official Benchmark")
    print("  " + "-" * 70)
    ref_log = results_dict[reference_key]["weekly_log"]
    for _, row in ref_log.iterrows():
        tag = " [setup]" if row["week"] < LEAD_TIME else ""
        print(
            f"    Week {int(row['week'])} | demand={int(row['demand']):5d}  "
            f"sales={int(row['sales']):5d}  missed={int(row['missed_sales']):4d}  "
            f"end_inv={int(row['end_inventory']):5d}  cost={row['total_cost']:8.1f}{tag}"
        )

    return df


if __name__ == "__main__":
    results = {}

    print("1/4  Running official benchmark (Nicolas's code)...")
    official_order, _ = run_official_benchmark()
    official_results = run_official_through_sim(official_order)
    results["Official Benchmark"] = official_results

    print("2/4  Running our benchmark replica...")
    sim = InventorySimulator()
    our_bench = sim.run_simulation(benchmark_policy)
    results["Our Benchmark Replica"] = our_bench

    print("3/4  Running ML point forecast (3wk coverage)...")
    sim = InventorySimulator()
    ml_results = sim.run_simulation(make_ml_policy(coverage_weeks=3))
    results["ML Point (3wk)"] = ml_results

    print("4/4  Running winner approach (phi=0.1)...")
    sim = InventorySimulator()
    winner_results = sim.run_simulation(make_winner_policy(coverage_weeks=3, phi=0.1))
    results["Winner (phi=0.1)"] = winner_results

    df = print_comparison(results)

    out_path = ROOT / "benchmark" / "results.csv"
    df.to_csv(out_path)
    print(f"\n  Results saved to {out_path}")
