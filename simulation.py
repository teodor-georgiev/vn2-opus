import pandas as pd
import numpy as np
from pathlib import Path

INDEX = ["Store", "Product"]
DATA_DIR = Path("Data")

HOLDING_COST = 0.2
SHORTAGE_COST = 1.0
LEAD_TIME = 2
NUM_ROUNDS = 6
TOTAL_WEEKS = NUM_ROUNDS + LEAD_TIME  # 8 weeks of actual sales needed


def load_initial_data():
    sales = pd.read_csv(DATA_DIR / "Week 0 - 2024-04-08 - Sales.csv").set_index(INDEX)
    in_stock = pd.read_csv(DATA_DIR / "Week 0 - In Stock.csv").set_index(INDEX)
    state = pd.read_csv(DATA_DIR / "Week 0 - 2024-04-08 - Initial State.csv").set_index(INDEX)
    master = pd.read_csv(DATA_DIR / "Week 0 - Master.csv").set_index(INDEX)

    sales.columns = pd.to_datetime(sales.columns)
    in_stock.columns = pd.to_datetime(in_stock.columns)

    return sales, in_stock, state, master


def load_all_actual_sales():
    """Load all weekly sales files and return the full actual sales for weeks 1-8."""
    all_sales = {}
    for week in range(1, TOTAL_WEEKS + 1):
        fname = list(DATA_DIR.glob(f"Week {week} - *Sales.csv"))
        if fname:
            df = pd.read_csv(fname[0]).set_index(INDEX)
            df.columns = pd.to_datetime(df.columns)
            last_col = df.columns[-1]
            all_sales[last_col] = df[last_col]
    return pd.DataFrame(all_sales)


class InventorySimulator:
    """Simulates inventory over 8 weeks given order decisions for rounds 1-6.

    Default constructor loads the real competition data from Data/. For CV on
    arbitrary historical 8-week windows, use the `from_window` classmethod or
    the module-level `run_window(...)` helper.
    """

    def __init__(
        self,
        sales_hist: pd.DataFrame | None = None,
        in_stock: pd.DataFrame | None = None,
        initial_state: pd.DataFrame | None = None,
        master: pd.DataFrame | None = None,
        actual_sales: pd.DataFrame | None = None,
    ):
        if sales_hist is None:
            # Default: load real competition data.
            self.sales_hist, self.in_stock, self.initial_state, self.master = load_initial_data()
            self.actual_sales = load_all_actual_sales()
        else:
            self.sales_hist = sales_hist
            self.in_stock = in_stock
            self.initial_state = initial_state
            self.master = master
            self.actual_sales = actual_sales

        self.end_inventory = self.initial_state["End Inventory"].copy()
        self.in_transit_w1 = self.initial_state["In Transit W+1"].copy()
        self.in_transit_w2 = self.initial_state["In Transit W+2"].copy()

        self.weekly_log = []
        self.orders_placed = []

    def reset(self):
        self.end_inventory = self.initial_state["End Inventory"].copy()
        self.in_transit_w1 = self.initial_state["In Transit W+1"].copy()
        self.in_transit_w2 = self.initial_state["In Transit W+2"].copy()
        self.weekly_log = []
        self.orders_placed = []

    def get_inventory_state(self):
        return pd.DataFrame({
            "End Inventory": self.end_inventory,
            "In Transit W+1": self.in_transit_w1,
            "In Transit W+2": self.in_transit_w2,
        })

    def get_net_inventory_position(self):
        return self.end_inventory + self.in_transit_w1 + self.in_transit_w2

    def get_sales_history(self, week_idx):
        """Return sales history available at the start of a given round (0-indexed)."""
        if week_idx == 0:
            return self.sales_hist.copy()
        cols = list(self.sales_hist.columns)
        for w in range(1, week_idx + 1):
            actual_cols = self.actual_sales.columns
            if w - 1 < len(actual_cols):
                col = actual_cols[w - 1]
                cols.append(col)
        all_cols = sorted(set(cols) | set(self.actual_sales.columns[:week_idx]))
        result = self.sales_hist.copy()
        for col in self.actual_sales.columns[:week_idx]:
            if col not in result.columns:
                result[col] = self.actual_sales[col]
        return result[sorted(result.columns)]

    def simulate_week(self, week_idx, actual_demand):
        """
        Simulate one week:
        1. Receive in-transit W+1 delivery
        2. Fulfill demand (or record shortage)
        3. Advance pipeline
        """
        start_inventory = self.end_inventory + self.in_transit_w1
        sales = np.minimum(start_inventory, actual_demand)
        missed_sales = actual_demand - sales
        end_inv = start_inventory - sales

        holding = end_inv * HOLDING_COST
        shortage = missed_sales * SHORTAGE_COST

        self.weekly_log.append({
            "week": week_idx,
            "start_inventory": start_inventory.sum(),
            "demand": actual_demand.sum(),
            "sales": sales.sum(),
            "missed_sales": missed_sales.sum(),
            "end_inventory": end_inv.sum(),
            "holding_cost": holding.sum(),
            "shortage_cost": shortage.sum(),
            "total_cost": holding.sum() + shortage.sum(),
        })

        self.end_inventory = end_inv
        self.in_transit_w1 = self.in_transit_w2.copy()
        self.in_transit_w2 = pd.Series(0, index=self.end_inventory.index)

    def place_order(self, order):
        """Place an order that will arrive in 2 weeks (sets in_transit_w2)."""
        order = order.clip(lower=0).round(0).astype(int)
        self.in_transit_w2 = order
        self.orders_placed.append(order.copy())

    def run_simulation(self, order_policy_fn):
        """
        Run full 6-round simulation.

        order_policy_fn(sim, round_idx) -> pd.Series of order quantities
        """
        self.reset()
        actual_cols = sorted(self.actual_sales.columns)

        for round_idx in range(TOTAL_WEEKS):
            actual_demand = self.actual_sales[actual_cols[round_idx]]
            self.simulate_week(round_idx, actual_demand)

            if round_idx < NUM_ROUNDS:
                sales_hist = self.get_sales_history(round_idx + 1)
                order = order_policy_fn(self, round_idx, sales_hist)
                self.place_order(order)

        return self.get_results()

    def get_results(self):
        log_df = pd.DataFrame(self.weekly_log)
        total_holding = log_df["holding_cost"].sum()
        total_shortage = log_df["shortage_cost"].sum()
        total_cost = total_holding + total_shortage

        # Competition scoring: weeks 1-2 (idx 0-1) are setup — identical for every
        # competitor because of the 2-week lead time (order placed in round 1 only
        # affects week 3). Score only indices 2..7 = competition weeks 3-8.
        competition = log_df[log_df["week"] >= LEAD_TIME]
        comp_holding = competition["holding_cost"].sum()
        comp_shortage = competition["shortage_cost"].sum()
        competition_cost = comp_holding + comp_shortage

        setup = log_df[log_df["week"] < LEAD_TIME]
        setup_cost = setup["holding_cost"].sum() + setup["shortage_cost"].sum()

        return {
            "total_cost": total_cost,
            "total_holding": total_holding,
            "total_shortage": total_shortage,
            "setup_cost": setup_cost,
            "competition_cost": competition_cost,
            "competition_holding": comp_holding,
            "competition_shortage": comp_shortage,
            "weekly_log": log_df,
        }


def run_window(
    full_sales: pd.DataFrame,
    window_start: int,
    full_in_stock: pd.DataFrame | None = None,
    master: pd.DataFrame | None = None,
    n_weeks: int = 8,
    policy_fn=None,
    initial_state: pd.DataFrame | None = None,
):
    """Score a policy over an arbitrary N-week window of historical sales.

    Parameters
    ----------
    full_sales : (Store, Product) x date wide DataFrame. Must include at least
        `window_start + n_weeks` date columns and the `window_start` columns
        before them for history.
    window_start : int
        Index (0-based) of the first column to use as simulated demand.
    full_in_stock : optional, same shape as full_sales.
    master : optional, master metadata passed through to sim.
    n_weeks : simulation horizon (default 8, matching competition).
    policy_fn : (sim, round_idx, sales_hist) -> pd.Series
    initial_state : DataFrame with columns End Inventory, In Transit W+1,
        In Transit W+2. Defaults to zeros (standard CV practice; first 2 weeks
        are "setup" anyway and excluded from competition_cost).

    Returns
    -------
    Result dict from sim.get_results(). `competition_cost` covers weeks
    window_start+2 .. window_start+n_weeks-1 (i.e., ignoring first LEAD_TIME
    setup weeks of the window).
    """
    assert policy_fn is not None, "policy_fn is required"
    sales_cols = sorted(full_sales.columns)
    assert window_start + n_weeks <= len(sales_cols), (
        f"Need {window_start + n_weeks} columns, got {len(sales_cols)}"
    )

    hist_cols = sales_cols[:window_start]
    demand_cols = sales_cols[window_start:window_start + n_weeks]

    sales_hist = full_sales[hist_cols].copy()
    in_stock_hist = full_in_stock[hist_cols].copy() if full_in_stock is not None else None
    actual_sales = full_sales[demand_cols].copy()

    if initial_state is None:
        idx = full_sales.index
        initial_state = pd.DataFrame(
            {
                "End Inventory": pd.Series(0.0, index=idx),
                "In Transit W+1": pd.Series(0.0, index=idx),
                "In Transit W+2": pd.Series(0.0, index=idx),
            }
        )

    sim = InventorySimulator(
        sales_hist=sales_hist,
        in_stock=in_stock_hist,
        initial_state=initial_state,
        master=master,
        actual_sales=actual_sales,
    )
    # run_simulation uses TOTAL_WEEKS (8). Our helper also defaults to 8, so OK.
    # For n_weeks != 8 we would need a loop here; keeping it simple for now.
    if n_weeks != TOTAL_WEEKS:
        raise NotImplementedError(f"run_window currently supports n_weeks={TOTAL_WEEKS} only")
    return sim.run_simulation(policy_fn)


def benchmark_policy(sim, round_idx, sales_hist):
    """Replicates the official benchmark: 13-week seasonal MA + 4-week coverage."""
    in_stock_full = sim.in_stock.copy()
    sales_clean = sales_hist.copy()
    matching_cols = sales_clean.columns[sales_clean.columns.isin(in_stock_full.columns)]
    for col in matching_cols:
        sales_clean.loc[~in_stock_full[col], col] = np.nan

    season = sales_clean.mean().rename("Demand").to_frame()
    season["Week Number"] = season.index.isocalendar().week.values
    season = season.groupby("Week Number")["Demand"].mean().to_frame()
    season = season / season.mean()

    sales_weeks = sales_clean.columns.isocalendar().week
    sales_no_season = sales_clean / season.loc[sales_weeks.values, "Demand"].values.reshape(-1)

    base_forecast = sales_no_season.iloc[:, -13:].mean(axis=1)

    f_periods = pd.date_range(
        start=sales_hist.columns[-1], periods=9, inclusive="neither", freq="W-MON"
    )
    forecast = pd.DataFrame(
        data=base_forecast.values.reshape(-1, 1).repeat(len(f_periods), axis=1),
        columns=f_periods,
        index=sales_hist.index,
    )
    season_factors = season.loc[f_periods.isocalendar().week.values, "Demand"].values.reshape(-1)
    forecast = forecast * season_factors

    order_up_to = forecast.iloc[:, :4].sum(axis=1)
    net_inv = sim.get_net_inventory_position()
    order = (order_up_to - net_inv).clip(lower=0).round(0).astype(int)
    return order


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)

    sim = InventorySimulator()
    results = sim.run_simulation(benchmark_policy)

    print("=== BENCHMARK SIMULATION RESULTS ===")
    print(f"Total Cost (all 8 weeks):       {results['total_cost']:,.2f} €")
    print(f"  Holding:                      {results['total_holding']:,.2f} €")
    print(f"  Shortage:                     {results['total_shortage']:,.2f} €")
    print(f"Setup Cost (wk 1-2, fixed):     {results['setup_cost']:,.2f} €")
    print(f"Competition Cost (wk 3-8):      {results['competition_cost']:,.2f} €  <-- leaderboard-comparable")
    print(f"  Holding:                      {results['competition_holding']:,.2f} €")
    print(f"  Shortage:                     {results['competition_shortage']:,.2f} €")
    print()
    print("Weekly breakdown:")
    print(results["weekly_log"].to_string(index=False))
