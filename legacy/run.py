"""
VN2 Inventory Planning — Full Pipeline
Forecasting (mlforecast + LightGBM/CatBoost) + Newsvendor Inventory Optimization
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from simulation import InventorySimulator, benchmark_policy, HOLDING_COST, SHORTAGE_COST, LEAD_TIME
from forecaster import DemandForecaster

os.chdir(Path(__file__).parent)

INDEX = ["Store", "Product"]
DATA_DIR = Path("Data")

CRITICAL_RATIO = SHORTAGE_COST / (SHORTAGE_COST + HOLDING_COST)  # 0.833


def make_ml_policy(coverage_weeks=3, safety_factor=1.0):
    """Create an ML-based order policy using mlforecast.

    coverage_weeks: how many weeks of forecast demand to cover
    safety_factor: multiplier on forecast for safety stock (1.0 = point forecast)
    """
    forecaster = None
    master = pd.read_csv(DATA_DIR / "Week 0 - Master.csv").set_index(INDEX)
    in_stock = pd.read_csv(DATA_DIR / "Week 0 - In Stock.csv").set_index(INDEX)
    in_stock.columns = pd.to_datetime(in_stock.columns)

    def ml_policy(sim, round_idx, sales_hist):
        nonlocal forecaster

        forecaster = DemandForecaster(master=master)
        forecaster.fit(sales_hist, in_stock)
        forecast = forecaster.predict(horizon=coverage_weeks)

        demand_over_horizon = forecast.sum(axis=1) * safety_factor
        net_inv = sim.get_net_inventory_position()
        order = (demand_over_horizon - net_inv).clip(lower=0).round(0).astype(int)
        return order

    return ml_policy


def make_ml_quantile_policy(coverage_weeks=3, quantile=None):
    """Order policy targeting a specific demand quantile (newsvendor optimal)."""
    if quantile is None:
        quantile = CRITICAL_RATIO

    master = pd.read_csv(DATA_DIR / "Week 0 - Master.csv").set_index(INDEX)
    in_stock = pd.read_csv(DATA_DIR / "Week 0 - In Stock.csv").set_index(INDEX)
    in_stock.columns = pd.to_datetime(in_stock.columns)

    def ml_quantile_policy(sim, round_idx, sales_hist):
        forecaster = DemandForecaster(master=master)
        forecaster.fit(sales_hist, in_stock)
        forecast = forecaster.predict_quantile(horizon=coverage_weeks, quantile=quantile)

        demand_over_horizon = forecast.sum(axis=1)
        net_inv = sim.get_net_inventory_position()
        order = (demand_over_horizon - net_inv).clip(lower=0).round(0).astype(int)
        return order

    return ml_quantile_policy


def make_winner_policy(coverage_weeks=3, phi=0.1, z=0.97):
    """Winner's approach: forecast + demand-dependent safety stock.

    Order = Forecast_3wk + z * phi * sqrt(Forecast_3wk) - Net_Inventory
    sigma = phi * sqrt(forecast) — Poisson-like variance assumption
    """
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
        order = (order_up_to - net_inv).clip(lower=0).round(0).astype(int)
        return order

    return winner_policy


def run_and_report(name, policy_fn):
    sim = InventorySimulator()
    results = sim.run_simulation(policy_fn)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Competition Cost (wk 3-8): {results['competition_cost']:>8,.2f} €")
    print(f"    Holding:  {results['competition_holding']:>8,.2f} €")
    print(f"    Shortage: {results['competition_shortage']:>8,.2f} €")
    print(f"  (Setup cost: {results['setup_cost']:,.2f} €)")
    print(f"\n  Weekly breakdown:")
    log = results["weekly_log"]
    for _, row in log.iterrows():
        tag = " [setup]" if row["week"] < LEAD_TIME else ""
        print(f"    Week {int(row['week']):d}: "
              f"demand={int(row['demand']):5d}  sales={int(row['sales']):5d}  "
              f"missed={int(row['missed_sales']):4d}  "
              f"end_inv={int(row['end_inventory']):5d}  "
              f"cost={row['total_cost']:8.1f} €{tag}")
    return results


def optimize_phi(phi_values=None):
    """Grid search over phi to find optimal safety stock scaling."""
    if phi_values is None:
        phi_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]

    results = {}
    for phi in phi_values:
        sim = InventorySimulator()
        policy = make_winner_policy(coverage_weeks=3, phi=phi)
        res = sim.run_simulation(policy)
        results[phi] = res["competition_cost"]
        print(f"  phi={phi:.2f} -> competition_cost={res['competition_cost']:,.2f} € "
              f"(holding={res['competition_holding']:,.2f}, shortage={res['competition_shortage']:,.2f})")

    best_phi = min(results, key=results.get)
    print(f"\n  Best phi={best_phi:.2f} -> {results[best_phi]:,.2f} €")
    return best_phi, results


if __name__ == "__main__":
    print("Running benchmark...")
    bench = run_and_report("BENCHMARK (Seasonal MA + 4wk coverage)", benchmark_policy)

    print("\n\nRunning ML model (point forecast, 3wk coverage)...")
    ml_point = run_and_report(
        "ML POINT FORECAST (3wk coverage)",
        make_ml_policy(coverage_weeks=3, safety_factor=1.0),
    )

    print("\n\nRunning winner's approach (phi=0.1, z=0.97)...")
    ml_winner = run_and_report(
        "WINNER APPROACH (phi=0.1, z=0.97)",
        make_winner_policy(coverage_weeks=3, phi=0.1),
    )

    print("\n\nOptimizing phi...")
    best_phi, phi_results = optimize_phi()

    print("\n\nRunning with optimal phi...")
    ml_best = run_and_report(
        f"OPTIMIZED (phi={best_phi:.2f})",
        make_winner_policy(coverage_weeks=3, phi=best_phi),
    )

    BENCHMARK_TARGET = 4334.0

    print("\n\n" + "=" * 60)
    print("  FINAL SUMMARY (Competition Cost = weeks 3-8, leaderboard-comparable)")
    print("=" * 60)
    print(f"  {'Policy':<25s} {'Comp Cost':>10s} {'vs Benchmark':>12s}")
    print(f"  {'-'*25:<25s} {'-'*10:>10s} {'-'*12:>12s}")
    print(f"  {'Competition Benchmark':<25s} {BENCHMARK_TARGET:>10,.2f} €")
    for name, res in [
        ("Our Benchmark", bench),
        ("ML Point", ml_point),
        ("Winner (phi=0.1)", ml_winner),
        (f"Optimized (phi={best_phi:.2f})", ml_best),
    ]:
        cost = res["competition_cost"]
        imp = (BENCHMARK_TARGET - cost) / BENCHMARK_TARGET * 100
        print(f"  {name:<25s} {cost:>10,.2f} € {imp:>+11.1f}%")
