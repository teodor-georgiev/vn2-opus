"""
Bridge: wrap Matias Alvo's trained DRL policy as an order_policy_fn for our
InventorySimulator in simulation.py.

Approach
--------
Matias's env calls the policy BEFORE consuming the current period's demand, while
our simulate_week() consumes demand FIRST and then calls the policy. To keep our
simulator unchanged and still invoke Matias's model with the state distribution
it was trained on, the bridge maintains its OWN Matias-style observation that it
steps forward in parallel with our sim:

  - At round 0 (first policy call), initial Matias state[0] = start-of-week-1
    on-hand = Toni's (end_inv_0 + W1_0); state[1] = W2_0.
    Run model -> action_0. Return as Toni round-0 order.
  - At round r>0: update the Matias observation using the demand that just
    happened in Toni's sim (last column of sales_hist) and the last action we
    returned. Then run model -> action_r.

Lead-time 2 alignment: action_t is first consumed at Matias period t+2 == Toni
week t+3 == arrival week for the round-t order. Matches our competition setup.

This bridge does NOT train the model; use references/matias_alvo/main_run.py to
train first (set save_model=True in data_driven_net.yml and save the weights).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

MATIAS_DIR = Path(__file__).parent / "references" / "matias_alvo"
if str(MATIAS_DIR) not in sys.path:
    sys.path.insert(0, str(MATIAS_DIR))

# Now safe to import Matias's modules.
from data_handling import Scenario  # noqa: E402
from neural_networks import NeuralNetworkCreator  # noqa: E402


class MatiasPolicy:
    """Callable: (sim, round_idx, sales_hist) -> pd.Series of order quantities.

    Steps its own Matias-style observation forward per call using the last
    observed demand column.
    """

    def __init__(
        self,
        model,
        scenario,
        config_setting: dict,
        initial_state_df: pd.DataFrame,
        sales_hist_wide: pd.DataFrame,
        stock_hist_wide: pd.DataFrame,
        date_features: pd.DataFrame,
        product_features: pd.DataFrame,
        device: str = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.scenario = scenario
        self.obs_params = config_setting["observation_params"]
        self.store_params = config_setting["store_params"]

        # Feature name lists
        self.time_feature_names = self.obs_params.get("time_features", []) or []
        self.pf_names = self.obs_params.get("product_features", {}).get("features", [])
        self.past_demand_n = self.obs_params["demand"]["past_periods"]
        self.past_instock_n = self.obs_params.get("instock", {}).get("past_periods", 0)

        # Align index order: use initial_state_df index for all 599 rows.
        idx = initial_state_df.index  # (Store, Product) multi-index
        n = len(idx)

        # --- Build past_demands tensor [n, 1, past_demand_n] from sales_hist_wide. ---
        # Use the LAST past_demand_n columns, in date order.
        sales_cols = sorted(sales_hist_wide.columns)
        past_cols = sales_cols[-self.past_demand_n:]
        past_demand_arr = sales_hist_wide.loc[idx, past_cols].to_numpy(dtype=np.float32)
        self.past_demands = torch.tensor(past_demand_arr, device=device).view(n, 1, -1)

        # --- Build past_instocks tensor [n, 1, past_instock_n]. ---
        if self.past_instock_n > 0:
            stk_cols = sorted(stock_hist_wide.columns)
            stk_past_cols = stk_cols[-self.past_instock_n:]
            stk_arr = stock_hist_wide.loc[idx, stk_past_cols].to_numpy(dtype=np.float32)
            self.past_instocks = torch.tensor(stk_arr, device=device).view(n, 1, -1)
        else:
            self.past_instocks = None

        # --- Product features [n, len(pf_names)] ---
        pf_cols = [c for c in self.pf_names if c in product_features.columns]
        pf_indexed = product_features.set_index(["Store", "Product"], drop=False)
        pf_arr = pf_indexed.loc[idx, pf_cols].to_numpy(dtype=np.float32)
        self.product_features = torch.tensor(pf_arr, device=device)  # [n, k]

        # --- Time features: a dict of {name: current_scalar_value}, updated each step. ---
        # date_features has one row per date; we track the current date.
        df = date_features.copy()
        df["date"] = pd.to_datetime(df["date"])
        # Extend date_features forward so competition weeks are covered.
        last_known = df["date"].max()
        future = pd.date_range(last_known + pd.Timedelta(days=7), periods=12, freq="W-MON")
        if len(future) > 0:
            ext_rows = []
            for d in future:
                row = {"date": d}
                row["day_of_week"] = d.dayofweek
                row["year"] = d.year
                row["day_of_month"] = d.day
                # Days from (next) Christmas.
                xmas = pd.Timestamp(year=d.year, month=12, day=25)
                if d > xmas:
                    xmas = pd.Timestamp(year=d.year + 1, month=12, day=25)
                row["days_from_christmas"] = (xmas - d).days
                for m in range(1, 13):
                    row[f"month_{m}"] = 1 if d.month == m else 0
                ext_rows.append(row)
            df = pd.concat([df, pd.DataFrame(ext_rows)], ignore_index=True)
        self.date_features = df.set_index("date")
        # initial_date = last sales_hist date + 1 week (= first competition week).
        last_hist_date = pd.Timestamp(sales_cols[-1])
        self.current_date = last_hist_date + pd.Timedelta(days=7)

        # --- Initial store_inventories [n, 1, 2] = [on_hand_entering_wk1, arrival_wk2]. ---
        end_inv = initial_state_df["End Inventory"].astype(np.float32).to_numpy()
        w1 = initial_state_df["In Transit W+1"].astype(np.float32).to_numpy()
        w2 = initial_state_df["In Transit W+2"].astype(np.float32).to_numpy()
        on_hand = end_inv + w1  # start inventory for week 1
        in_transit_1 = w2  # arrives at week 2
        store_inv = np.stack([on_hand, in_transit_1], axis=1).reshape(n, 1, 2)
        self.store_inventories = torch.tensor(store_inv, dtype=torch.float32, device=device)

        # Static per-sample features.
        self.holding_costs = torch.full((n, 1), self.store_params["holding_cost"]["value"], device=device)
        self.underage_costs = torch.full((n, 1), self.store_params["underage_cost"]["value"], device=device)
        self.lead_times = torch.full((n, 1), float(self.store_params["lead_time"]["value"]), device=device)

        self._last_action: torch.Tensor | None = None
        self._round_idx_seen: int = -1
        self._toni_index = idx

    def _build_observation(self) -> dict:
        """Assemble the dict that DataDrivenNet.forward() expects."""
        obs = {
            "store_inventories": self.store_inventories,
            "past_demands": self.past_demands,
            "holding_costs": self.holding_costs,
            "underage_costs": self.underage_costs,
            "lead_times": self.lead_times,
            "product_features": self.product_features,
        }
        if self.past_instocks is not None:
            obs["past_instocks"] = self.past_instocks
        # Time features for the CURRENT period (scalar broadcast over batch).
        n = self.store_inventories.shape[0]
        if self.current_date in self.date_features.index:
            row = self.date_features.loc[self.current_date]
            for name in self.time_feature_names:
                if name in row:
                    obs[name] = torch.full((n, 1), float(row[name]), device=self.device)
        return obs

    def _step_matias_state(self, demand: torch.Tensor):
        """Advance store_inventories by one Matias period, using the action we returned last call.

        Equivalent to what env.step(action=self._last_action) would do for one store, lost-demand.
        """
        on_hand = self.store_inventories[:, :, 0]
        post = torch.clip(on_hand - demand, min=0.0)
        new_on_hand = post + self.store_inventories[:, :, 1]
        new_in_transit = self._last_action  # placed last call, arrives next period as on_hand
        self.store_inventories = torch.stack([new_on_hand, new_in_transit], dim=2)

    def __call__(self, sim, round_idx: int, sales_hist: pd.DataFrame) -> pd.Series:
        # Skip duplicate calls for the same round (shouldn't happen, but guard).
        if round_idx == self._round_idx_seen:
            # Rebuild and return.
            pass
        elif round_idx > 0:
            # Advance state using the demand that JUST happened in Toni's sim.
            sales_cols = sorted(sales_hist.columns)
            latest_col = sales_cols[-1]
            demand_arr = sales_hist.loc[self._toni_index, latest_col].to_numpy(dtype=np.float32)
            demand_t = torch.tensor(demand_arr, device=self.device).view(-1, 1)
            self._step_matias_state(demand_t)

            # Shift past_demands left, append latest.
            self.past_demands = torch.cat(
                [self.past_demands[:, :, 1:], demand_t.unsqueeze(2)], dim=2
            )
            if self.past_instocks is not None:
                # Assume in-stock = (demand was fully met). Best effort.
                # Toni doesn't track this during sim, so fall back to "1.0" (fully in stock).
                instock_t = torch.ones_like(demand_t)
                self.past_instocks = torch.cat(
                    [self.past_instocks[:, :, 1:], instock_t.unsqueeze(2)], dim=2
                )
            # Advance date.
            self.current_date = self.current_date + pd.Timedelta(days=7)

        self._round_idx_seen = round_idx

        # Run model.
        obs = self._build_observation()
        with torch.no_grad():
            out = self.model(obs)
        action = out["stores"]  # [n, n_stores, 1] or [n, n_stores]
        if action.dim() == 3:
            action = action[:, :, 0]
        # Snap to non-negative integers (VN2 submission requirement).
        action = torch.clip(action, min=0.0)
        self._last_action = action

        arr = action.squeeze(-1).cpu().numpy() if action.dim() > 1 else action.cpu().numpy()
        # action shape should be [n, 1] -> squeeze to [n]
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return pd.Series(np.rint(arr).astype(int), index=self._toni_index)


def load_matias_model(
    model_checkpoint: Path,
    config_setting_path: Path = MATIAS_DIR / "config_files" / "settings" / "vn2_round_1.yml",
    config_hyperparams_path: Path = MATIAS_DIR / "config_files" / "policies_and_hyperparams" / "data_driven_net.yml",
    device: str = "cpu",
):
    """Load a trained Matias model checkpoint. Returns (model, scenario, config_setting)."""
    with open(config_setting_path) as f:
        cfg = yaml.safe_load(f)
    with open(config_hyperparams_path) as f:
        hp = yaml.safe_load(f)
    obs_params = defaultdict(lambda: None, cfg["observation_params"])

    # Build a minimal Scenario just to instantiate the model architecture.
    # Use train periods so dimensions match what was trained.
    import os
    cwd = os.getcwd()
    os.chdir(MATIAS_DIR)  # Matias's Scenario reads files with relative paths
    try:
        scenario = Scenario(
            periods=None,
            problem_params=cfg["problem_params"],
            store_params=cfg["store_params"],
            warehouse_params=cfg["warehouse_params"],
            echelon_params=cfg["echelon_params"],
            num_samples=cfg["params_by_dataset"]["train"]["n_samples"],
            observation_params=obs_params,
            seeds=cfg["seeds"],
        )
        model = NeuralNetworkCreator().create_neural_network(scenario, hp["nn_params"], device=device)
        ckpt = torch.load(model_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "warehouse_upper_bound" in ckpt:
            model.warehouse_upper_bound = ckpt["warehouse_upper_bound"]
    finally:
        os.chdir(cwd)
    return model, scenario, cfg


def build_matias_policy_from_toni_data(
    model,
    scenario,
    config_setting: dict,
    toni_data_dir: Path = Path("Data"),
    device: str = "cpu",
) -> MatiasPolicy:
    """Construct a MatiasPolicy bound to Toni's simulator.

    Loads Toni's Week 0 files + historical sales as the context the policy will use
    for its first decision, plus Matias's date_features and product_features metadata.
    """
    # Toni's history (wide DataFrame, Store x Product x date)
    sales = pd.read_csv(toni_data_dir / "Week 0 - 2024-04-08 - Sales.csv").set_index(["Store", "Product"])
    sales.columns = pd.to_datetime(sales.columns)

    # in_stock: True/False -> 1.0/0.0
    in_stock = pd.read_csv(toni_data_dir / "Week 0 - In Stock.csv").set_index(["Store", "Product"])
    in_stock.columns = pd.to_datetime(in_stock.columns)
    in_stock_f = in_stock.astype(float)

    initial_state = pd.read_csv(toni_data_dir / "Week 0 - 2024-04-08 - Initial State.csv").set_index(["Store", "Product"])

    date_features = pd.read_csv(MATIAS_DIR / "vn2_processed_data" / "new_data" / "date_features.csv")
    date_features["date"] = pd.to_datetime(date_features["date"])
    product_features = pd.read_csv(MATIAS_DIR / "vn2_processed_data" / "new_data" / "product_features.csv")

    return MatiasPolicy(
        model=model,
        scenario=scenario,
        config_setting=config_setting,
        initial_state_df=initial_state,
        sales_hist_wide=sales,
        stock_hist_wide=in_stock_f,
        date_features=date_features,
        product_features=product_features,
        device=device,
    )
