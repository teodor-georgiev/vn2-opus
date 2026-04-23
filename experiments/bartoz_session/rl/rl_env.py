"""Gym-like environment for VN2 Stage 2b RL.

Wraps `simulation.InventorySimulator` for episode-based RL training. An "episode"
is an 8-week (TOTAL_WEEKS) window:
  - sales_hist comes from columns [: window_start] of the full pre-sacred sales
  - actual_demand for the 8 simulated weeks = columns [window_start : window_start + 8]

A pretrained forecaster (`policies.EnsemblePolicy`) is fit ONCE per episode (at the
window_start cut) and provides d1, d2 forecasts for each of 6 ordering rounds.

Per-SKU observation at round r (after simulate_week(r) and before placing order):
    [I0 / scale, T1 / scale, d1 / scale, d2 / scale, sigma / scale, log1p_scale]
where scale = backtest-RMSE-style sigma estimate (per SKU), and log1p_scale lets
the policy condition on demand magnitude.

Per-SKU action: (mult, buffer) with `mult ∈ [0, 2]` and `buffer ≥ 0`.
The agent's policy returns these PRE-squashed; the env applies tanh+1 / softplus.
Order = round(max(max(d2 - E1, 0) * mult + buffer * scale, 0)).

Reward at step r (after simulate_week(r)) = -(holding + shortage), summed across SKUs
and divided by N_SKU for stable magnitudes. Only competition weeks contribute.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecaster import DemandForecaster  # noqa: E402
from policies import EnsemblePolicy, _seasonal_ma_forecast  # noqa: E402
from simulation import (  # noqa: E402
    HOLDING_COST, LEAD_TIME, NUM_ROUNDS, SHORTAGE_COST, TOTAL_WEEKS,
    InventorySimulator,
)


@dataclass
class EnvConfig:
    backtest_window: int = 26
    safety_floor: float = 0.5
    rmse_horizons: int = 1
    ensemble_kwargs: dict | None = None
    # Reward scale (we divide raw -(H+S) sum by this to keep gradients stable).
    reward_scale: float = 100.0


class VN2Episode:
    """One 8-week episode given a window_start. Stateless after build()."""

    def __init__(
        self,
        full_sales: pd.DataFrame,
        full_in_stock: pd.DataFrame | None,
        master: pd.DataFrame | None,
        window_start: int,
        cfg: EnvConfig,
    ):
        self.cfg = cfg
        cols = sorted(full_sales.columns)
        assert window_start + TOTAL_WEEKS <= len(cols), (
            f"need {window_start + TOTAL_WEEKS}, got {len(cols)}"
        )
        self.window_start = window_start
        hist_cols = cols[:window_start]
        demand_cols = cols[window_start: window_start + TOTAL_WEEKS]

        self.sales_hist = full_sales[hist_cols].copy()
        self.in_stock_hist = full_in_stock[hist_cols].copy() if full_in_stock is not None else None
        self.actual_sales = full_sales[demand_cols].copy()
        self.master = master
        self.idx = full_sales.index
        self.n_sku = len(self.idx)

        # 1) Fit pretrained forecaster ONCE (frozen during the episode).
        ens_kwargs = cfg.ensemble_kwargs or {
            "coverage_weeks": 3, "w_ma": 0.25,
            "censoring_strategy": "mean_impute", "random_state": 42,
        }
        ens_kwargs.setdefault("coverage_weeks", 3)
        self._ensemble = EnsemblePolicy(**ens_kwargs)
        self._ensemble._fit(self.sales_hist, self.in_stock_hist)

        horizon = NUM_ROUNDS + LEAD_TIME + 3
        if self._ensemble.w_lgb_share is None:
            self._ensemble._all_ml_forecasts = self._ensemble._forecaster.predict(horizon=horizon)
        else:
            per_model = self._ensemble._forecaster.predict_models(horizon=horizon)
            self._ensemble._all_lgb_forecasts = per_model["lgbm"]
            self._ensemble._all_cb_forecasts = per_model["catboost"]
            self._ensemble._all_ml_forecasts = (
                self._ensemble.w_lgb_share * self._ensemble._all_lgb_forecasts
                + (1 - self._ensemble.w_lgb_share) * self._ensemble._all_cb_forecasts
            )
        self._ensemble._all_ma_forecasts = _seasonal_ma_forecast(
            self.sales_hist, self.in_stock_hist, horizon=horizon
        )

        # 2) Per-SKU sigma (RMSE-based) for observation scaling and order safety.
        W = cfg.backtest_window
        train = self.sales_hist.iloc[:, :-W]
        holdout = self.sales_hist.iloc[:, -W:]
        bt_is = self.in_stock_hist.iloc[:, :-W] if self.in_stock_hist is not None else None
        bt = DemandForecaster(
            master=self.master,
            random_state=self._ensemble.random_state,
            censoring_strategy=self._ensemble.censoring_strategy,
        )
        bt.fit(train, bt_is)
        bt_preds = bt.predict(horizon=W)
        n = min(bt_preds.shape[1], holdout.shape[1], cfg.rmse_horizons)
        resid = (
            holdout.iloc[:, :n].values
            - bt_preds.iloc[:, :n].reindex(self.idx).values
        )
        rmse = np.sqrt(np.nanmean(resid ** 2, axis=1))
        self.sigma = pd.Series(rmse, index=self.idx).fillna(cfg.safety_floor).clip(lower=cfg.safety_floor)
        self.scale = self.sigma.copy()  # used for obs normalization and order buffer

    def _get_d12(self, round_idx: int) -> tuple[np.ndarray, np.ndarray]:
        ens = self._ensemble
        ml = ens._all_ml_forecasts
        w = ens._current_w_ma(ml.index)
        out = []
        for off in (round_idx, round_idx + 1):
            d_ml = ml.iloc[:, off]
            d_ma = ens._all_ma_forecasts.iloc[:, off].reindex(ml.index).fillna(0)
            d = w * d_ma + (1 - w) * d_ml
            out.append(d.astype(float).values)
        return out[0], out[1]

    def reset(self) -> dict:
        idx = self.idx
        self.sim = InventorySimulator(
            sales_hist=self.sales_hist, in_stock=self.in_stock_hist,
            initial_state=pd.DataFrame({
                "End Inventory": pd.Series(0.0, index=idx),
                "In Transit W+1": pd.Series(0.0, index=idx),
                "In Transit W+2": pd.Series(0.0, index=idx),
            }),
            master=self.master, actual_sales=self.actual_sales,
        )
        self.sim.reset()
        self.round_idx = 0
        self.actual_cols = sorted(self.sim.actual_sales.columns)
        # Run the first simulate_week(0) to align with policy timing.
        self.sim.simulate_week(0, self.sim.actual_sales[self.actual_cols[0]])
        return self._observe(0)

    def _observe(self, round_idx: int) -> dict:
        I0 = self.sim.end_inventory.values.astype(np.float32)
        T1 = self.sim.in_transit_w1.values.astype(np.float32)
        d1, d2 = self._get_d12(round_idx)
        sigma = self.sigma.values.astype(np.float32)
        scale = self.scale.values.astype(np.float32)
        # Per-SKU observation: 6 features.
        obs = np.stack([
            I0 / scale,
            T1 / scale,
            d1.astype(np.float32) / scale,
            d2.astype(np.float32) / scale,
            sigma / scale,
            np.log1p(scale),
        ], axis=-1)  # [N, 6]
        return {
            "obs": obs,
            "d1": d1.astype(np.float32),
            "d2": d2.astype(np.float32),
            "I0": I0, "T1": T1, "scale": scale,
        }

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """action: [N, 2] of (mult_raw, buffer_raw). Apply transforms inside."""
        mult_raw = action[:, 0]
        buffer_raw = action[:, 1]
        # Squashing: mult ∈ [0, 2] via tanh+1 ; buffer ≥ 0 via softplus.
        mult = np.tanh(mult_raw) + 1.0
        buffer_units = np.log1p(np.exp(np.clip(buffer_raw, -20, 20)))  # softplus

        I0 = self.sim.end_inventory.values
        T1 = self.sim.in_transit_w1.values
        d1, d2 = self._get_d12(self.round_idx)
        E1 = np.maximum(I0 + T1 - d1, 0.0)
        base = np.maximum(d2 - E1, 0.0)
        scale = self.scale.values
        order = np.maximum(np.round(base * mult + buffer_units * scale), 0.0).astype(int)

        order_series = pd.Series(order, index=self.idx)
        self.sim.place_order(order_series)

        # Move to next round: simulate_week(r+1)
        next_round = self.round_idx + 1
        if next_round < TOTAL_WEEKS:
            self.sim.simulate_week(next_round, self.sim.actual_sales[self.actual_cols[next_round]])

        # Reward = -(holding + shortage) for the WEEK we just simulated.
        # Only count weeks ≥ LEAD_TIME (competition window).
        if next_round < TOTAL_WEEKS and next_round >= LEAD_TIME:
            wk_log = self.sim.weekly_log[-1]
            raw_cost = wk_log["holding_cost"] + wk_log["shortage_cost"]
            reward = -float(raw_cost) / max(self.cfg.reward_scale, 1.0)
        else:
            reward = 0.0

        self.round_idx = next_round
        done = self.round_idx >= NUM_ROUNDS  # 6 ordering rounds
        info = {"round": self.round_idx, "raw_cost_step": -reward * self.cfg.reward_scale if next_round >= LEAD_TIME else 0.0}

        next_obs = self._observe(self.round_idx) if not done else None
        # Episode termination: still need to simulate weeks 7..TOTAL_WEEKS-1 for cost accounting.
        if done:
            # Continue simulating remaining weeks (no more orders to place).
            cumulative_cost = 0.0
            while self.round_idx + 1 < TOTAL_WEEKS:
                next_round = self.round_idx + 1
                self.sim.simulate_week(next_round, self.sim.actual_sales[self.actual_cols[next_round]])
                wk = self.sim.weekly_log[-1]
                if next_round >= LEAD_TIME:
                    cumulative_cost += wk["holding_cost"] + wk["shortage_cost"]
                self.round_idx = next_round
            reward += -float(cumulative_cost) / max(self.cfg.reward_scale, 1.0)
            info["raw_cost_step"] = info.get("raw_cost_step", 0.0) - reward * self.cfg.reward_scale
            info["episode_total_cost"] = float(self.sim.get_results()["competition_cost"])
        return next_obs, reward, done, info


def make_episodes(
    full_sales: pd.DataFrame,
    full_in_stock: pd.DataFrame | None,
    master: pd.DataFrame | None,
    window_starts: list[int],
    cfg: EnvConfig | None = None,
) -> list[VN2Episode]:
    """Pre-build episodes (each does the expensive forecaster fit ONCE)."""
    cfg = cfg or EnvConfig()
    eps = []
    for ws in window_starts:
        print(f"  building episode at window_start={ws}...")
        eps.append(VN2Episode(full_sales, full_in_stock, master, ws, cfg))
    return eps
