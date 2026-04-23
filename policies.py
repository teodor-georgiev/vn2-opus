"""Order policies for VN2. Each policy exposes its last forecast (sum over
coverage_weeks) as `last_forecast` so the CV harness can score forecast
accuracy independently of the cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster import DemandForecaster, DiverseDemandForecaster, QuantileDemandForecaster, QuantileDiverseDemandForecaster, StatsForecasterEnsemble
from simulation import LEAD_TIME


def _seasonal_ma_forecast(sales_hist: pd.DataFrame, in_stock: pd.DataFrame | None,
                          horizon: int = 8) -> pd.DataFrame:
    """Replicate the organizer's 13-week seasonal-MA forecast used in benchmark_policy."""
    sales_clean = sales_hist.copy()
    if in_stock is not None:
        matching = sales_clean.columns[sales_clean.columns.isin(in_stock.columns)]
        for c in matching:
            sales_clean.loc[~in_stock[c], c] = np.nan

    season = sales_clean.mean().rename("Demand").to_frame()
    season["Week Number"] = season.index.isocalendar().week.values
    season = season.groupby("Week Number")["Demand"].mean().to_frame()
    season = season / season.mean()
    sales_weeks = sales_clean.columns.isocalendar().week
    sales_no_season = sales_clean / season.loc[sales_weeks.values, "Demand"].values.reshape(-1)
    base = sales_no_season.iloc[:, -13:].mean(axis=1)
    fut = pd.date_range(sales_hist.columns[-1], periods=horizon + 1, inclusive="neither", freq="W-MON")
    fc = pd.DataFrame(base.values.reshape(-1, 1).repeat(len(fut), axis=1), columns=fut, index=sales_hist.index)
    season_factors = season.loc[fut.isocalendar().week.values, "Demand"].values.reshape(-1)
    return fc * season_factors


class SeasonalBenchmarkPolicy:
    """Organizer's benchmark: seasonal MA forecast + 4-week order-up-to coverage."""

    def __init__(self, coverage_weeks: int = 4):
        self.coverage_weeks = coverage_weeks
        self.last_forecast: pd.Series | None = None

    def __call__(self, sim, round_idx, sales_hist):
        fc = _seasonal_ma_forecast(sales_hist, sim.in_stock, horizon=self.coverage_weeks)
        order_up_to = fc.iloc[:, : self.coverage_weeks].sum(axis=1)
        # Expose sum-of-next-coverage-weeks forecast for accuracy scoring.
        self.last_forecast = order_up_to.copy()
        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)


class MLPointPolicy:
    """Point-forecast LGB+CB via mlforecast with optional bias correction.

    order_up_to = multiplier * sum(mu_hat) + safety_units
    Fits once per episode (round_idx=0) and reuses direct multi-step forecasts.
    """

    def __init__(self, coverage_weeks: int = 3, master=None,
                 multiplier: float = 1.0, safety_units: float = 0.0,
                 random_state: int = 42, lgb_params: dict | None = None,
                 cb_params: dict | None = None,
                 coverage_per_sku: "pd.Series | None" = None,
                 censoring_strategy: str = "interpolate",
                 direct_forecast: bool = False):
        self.coverage_weeks = coverage_weeks
        self.master = master
        self.multiplier = multiplier
        self.safety_units = safety_units
        self.random_state = random_state
        self.lgb_params = lgb_params
        self.cb_params = cb_params
        self.coverage_per_sku = coverage_per_sku
        self.censoring_strategy = censoring_strategy
        # If True, use mlforecast's direct multi-horizon strategy (max_horizon
        # = NUM_ROUNDS + coverage + LEAD_TIME). Trains a separate estimator per
        # horizon using features at time t only; no recursive feedback.
        self.direct_forecast = direct_forecast
        self.last_forecast: pd.Series | None = None
        self._forecaster: DemandForecaster | None = None
        self._all_forecasts: pd.DataFrame | None = None

    def _fit(self, sales_hist, in_stock):
        max_cov = int(self.coverage_per_sku.max()) if self.coverage_per_sku is not None else self.coverage_weeks
        horizon = 6 + max_cov + LEAD_TIME
        self._forecaster = DemandForecaster(
            master=self.master, random_state=self.random_state,
            lgb_params=self.lgb_params, cb_params=self.cb_params,
            censoring_strategy=self.censoring_strategy,
            max_horizon=horizon if self.direct_forecast else None,
        )
        self._forecaster.fit(sales_hist, in_stock)

    def __call__(self, sim, round_idx, sales_hist):
        # For CV fairness, re-fit at every round using only data known up to now
        # would be ideal but expensive. Fit once at round 0 on the full history
        # available at that point; subsequent rounds reuse the model (as Matias
        # and most participants did — one-shot training per competition window).
        if round_idx == 0 or self._forecaster is None:
            self._fit(sales_hist, sim.in_stock)
            # Generate forecasts for NUM_ROUNDS + max(coverage) + LEAD_TIME weeks.
            max_cov = int(self.coverage_per_sku.max()) if self.coverage_per_sku is not None else self.coverage_weeks
            horizon = 6 + max_cov + LEAD_TIME
            self._all_forecasts = self._forecaster.predict(horizon=horizon)

        # Compute mu_sum (optionally per-SKU coverage).
        if self.coverage_per_sku is None:
            start = round_idx + LEAD_TIME
            end = start + self.coverage_weeks
            mu_sum = self._all_forecasts.iloc[:, start:end].sum(axis=1)
        else:
            cov_series = self.coverage_per_sku.reindex(self._all_forecasts.index).fillna(
                self.coverage_weeks
            ).astype(int)
            mu_sum = pd.Series(0.0, index=self._all_forecasts.index)
            for c in sorted(cov_series.unique()):
                mask = cov_series == c
                start = round_idx + LEAD_TIME
                end = start + int(c)
                mu_sum.loc[mask] = self._all_forecasts.loc[mask].iloc[:, start:end].sum(axis=1)

        self.last_forecast = mu_sum.copy()
        order_up_to = (self.multiplier * mu_sum + self.safety_units).clip(lower=0)

        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)


def _inv_normal_cdf(p: float) -> float:
    """Cheap approximation of standard-normal inverse CDF."""
    # Acklam's approximation (good to ~4e-4 absolute).
    import math
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.32806815528572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return ((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q) / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


class GaussianSafetyPolicy:
    """Point forecast + gaussian safety stock.

    order_up_to = sum(mu_t) + z * sigma * sqrt(coverage)
    where z = Phi^{-1}(alpha), sigma is per-SKU residual/historical std.

    Principled for iid-ish weekly demand under asymmetric shortage/holding
    costs (critical ratio = shortage / (shortage + holding) = 0.833).
    """

    def __init__(self, coverage_weeks: int = 3, alpha: float = 0.833,
                 sigma_source: str = "residuals", master=None):
        self.coverage_weeks = coverage_weeks
        self.alpha = alpha
        self.sigma_source = sigma_source  # "residuals" (from in-sample) or "history" (raw std of sales)
        self.master = master
        self.last_forecast: pd.Series | None = None
        self.last_quantile_forecast: pd.Series | None = None
        self._forecaster: DemandForecaster | None = None
        self._all_forecasts: pd.DataFrame | None = None
        self._sigma: pd.Series | None = None
        self._z = _inv_normal_cdf(alpha)

    def _fit(self, sales_hist, in_stock):
        self._forecaster = DemandForecaster(master=self.master)
        self._forecaster.fit(sales_hist, in_stock)
        # Estimate per-SKU sigma.
        if self.sigma_source == "residuals":
            # In-sample residual std on the last N weeks using the fitted model's
            # cross-validation isn't available cheaply; approximate via rolling
            # std of sales over last 26 weeks.
            recent = sales_hist.iloc[:, -26:]
            self._sigma = recent.std(axis=1).fillna(0.0)
        else:
            self._sigma = sales_hist.std(axis=1).fillna(0.0)

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._forecaster is None:
            self._fit(sales_hist, sim.in_stock)
            horizon = 6 + self.coverage_weeks + LEAD_TIME
            self._all_forecasts = self._forecaster.predict(horizon=horizon)

        start = round_idx + LEAD_TIME
        end = start + self.coverage_weeks
        mu_sum = self._all_forecasts.iloc[:, start:end].sum(axis=1)
        sigma = self._sigma.reindex(mu_sum.index).fillna(0.0)
        safety = self._z * sigma * np.sqrt(self.coverage_weeks)
        order_up_to = (mu_sum + safety).clip(lower=0)

        self.last_forecast = mu_sum.copy()  # point forecast sum (for MAE)
        self.last_quantile_forecast = order_up_to.copy()  # for pinball loss

        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)


class EnsemblePolicy:
    """Weighted ensemble of SeasonalMA + ML point forecast.

    order_up_to_r = mult_r * (w_ma_i * ma_sum + (1 - w_ma_i) * ml_sum) + safety_units

    Parameters
    ----------
    w_ma : float or pd.Series
        Global weight (float) or per-SKU weight (Series indexed by (Store, Product)).
    per_round_multiplier : list[float] | None
        If given, overrides `multiplier` with a per-round value indexed by round_idx.
    learn_w_ma_by_backtest : bool
        If True, estimate w_ma per SKU on training data by fitting both MA and ML
        on the first ~90% and comparing MAE on the last ~10%. The `w_ma` argument is
        then ignored except as a fallback when backtest is degenerate.
    """

    def __init__(self, coverage_weeks: int = 2, w_ma: "float | pd.Series" = 0.5,
                 multiplier: float = 1.05, safety_units: float = 0.0,
                 master=None, random_state: int = 42,
                 censoring_strategy: str = "mean_impute",
                 per_round_multiplier: "list[float] | None" = None,
                 learn_w_ma_by_backtest: bool = False,
                 backtest_window: int = 13,
                 w_lgb_share: "float | None" = None,
                 direct_forecast: bool = False,
                 extended_features: bool = False,
                 per_series_scaling: bool = False,
                 recency_decay: float | None = None,
                 categorical_features: bool = False,
                 intermittency_features: bool = False,
                 w_stats: float = 0.0,
                 demand_cluster_k: int | None = None,
                 hierarchical_features: bool = False):
        self.coverage_weeks = coverage_weeks
        self.w_ma = w_ma
        self.multiplier = multiplier
        self.safety_units = safety_units
        self.master = master
        self.random_state = random_state
        self.censoring_strategy = censoring_strategy
        self.per_round_multiplier = per_round_multiplier
        self.learn_w_ma_by_backtest = learn_w_ma_by_backtest
        self.backtest_window = backtest_window
        # If set, use this LGB/CB blend instead of the default 50/50 inside
        # DemandForecaster. w_lgb_share in [0,1]: w_lgb_share for LGB, 1-share for CB.
        self.w_lgb_share = w_lgb_share
        # Direct multi-horizon strategy: one estimator per horizon, no recursion.
        self.direct_forecast = direct_forecast
        # Extended feature set (seasonal lags 51/53, windows 3/5, Fourier).
        self.extended_features = extended_features
        # Per-series LocalStandardScaler target transform.
        self.per_series_scaling = per_series_scaling
        # Observation weighting: per-row sample_weight = decay ** age_blocks.
        self.recency_decay = recency_decay
        # Mark static features as pandas Categorical + CatBoost cat_features.
        self.categorical_features = categorical_features
        # Numba-JIT intermittency lag_transforms.
        self.intermittency_features = intermittency_features
        # Weight for Nixtla statsforecast ensemble member (AutoTheta + SNaive).
        # Final blend: w_ma*MA + w_stats*Stats + (1 - w_ma - w_stats) * ML.
        self.w_stats = w_stats
        # K-means demand-pattern cluster added as a static feature to ML.
        self.demand_cluster_k = demand_cluster_k
        # M5-style hierarchical (Department/Division/Store/global) rolling lags.
        self.hierarchical_features = hierarchical_features
        self._stats_forecaster: StatsForecasterEnsemble | None = None
        self._all_stats_forecasts: pd.DataFrame | None = None
        self.last_forecast: pd.Series | None = None
        self._forecaster: DemandForecaster | None = None
        self._all_ml_forecasts: pd.DataFrame | None = None
        self._all_lgb_forecasts: pd.DataFrame | None = None
        self._all_cb_forecasts: pd.DataFrame | None = None
        self._all_ma_forecasts: pd.DataFrame | None = None
        self._w_ma_per_sku: pd.Series | None = None

    def _fit(self, sales_hist, in_stock):
        horizon = 6 + self.coverage_weeks + LEAD_TIME
        self._forecaster = DemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy,
            lgb_params=getattr(self, "lgb_params", None),
            cb_params=getattr(self, "cb_params", None),
            max_horizon=horizon if self.direct_forecast else None,
            extended_features=self.extended_features,
            per_series_scaling=self.per_series_scaling,
            recency_decay=self.recency_decay,
            categorical_features=self.categorical_features,
            intermittency_features=self.intermittency_features,
            demand_cluster_k=self.demand_cluster_k,
            hierarchical_features=self.hierarchical_features,
        )
        self._forecaster.fit(sales_hist, in_stock)

        if self.learn_w_ma_by_backtest:
            W = self.backtest_window
            train_hist = sales_hist.iloc[:, :-W]
            holdout = sales_hist.iloc[:, -W:]
            # Pure MA backtest.
            ma_bt = _seasonal_ma_forecast(train_hist,
                                          in_stock.iloc[:, :-W] if in_stock is not None else None,
                                          horizon=W)
            # Pure ML backtest.
            ml_bt_forecaster = DemandForecaster(
                master=self.master, random_state=self.random_state,
                censoring_strategy=self.censoring_strategy,
            )
            ml_bt_forecaster.fit(train_hist, in_stock.iloc[:, :-W] if in_stock is not None else None)
            ml_bt = ml_bt_forecaster.predict(horizon=W)
            # Align columns with holdout.
            n_cols = min(ma_bt.shape[1], ml_bt.shape[1], holdout.shape[1])
            ma_vals = ma_bt.iloc[:, :n_cols].reindex(sales_hist.index).values
            ml_vals = ml_bt.iloc[:, :n_cols].reindex(sales_hist.index).values
            true_vals = holdout.iloc[:, :n_cols].values
            mae_ma = np.nanmean(np.abs(true_vals - ma_vals), axis=1)
            mae_ml = np.nanmean(np.abs(true_vals - ml_vals), axis=1)
            # Inverse-MAE weighting, clipped.
            inv_ma = 1.0 / np.clip(mae_ma, 0.1, None)
            inv_ml = 1.0 / np.clip(mae_ml, 0.1, None)
            w = inv_ma / (inv_ma + inv_ml)
            self._w_ma_per_sku = pd.Series(w, index=sales_hist.index).clip(0.0, 1.0)

    def _current_w_ma(self, index) -> pd.Series:
        if self.learn_w_ma_by_backtest and self._w_ma_per_sku is not None:
            return self._w_ma_per_sku.reindex(index).fillna(0.5)
        if isinstance(self.w_ma, pd.Series):
            return self.w_ma.reindex(index).fillna(0.5)
        return pd.Series(float(self.w_ma), index=index)

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._forecaster is None:
            self._fit(sales_hist, sim.in_stock)
            horizon = 6 + self.coverage_weeks + LEAD_TIME
            if self.w_lgb_share is None:
                self._all_ml_forecasts = self._forecaster.predict(horizon=horizon)
            else:
                # Get LGB and CB separately; blend with w_lgb_share.
                per_model = self._forecaster.predict_models(horizon=horizon)
                self._all_lgb_forecasts = per_model["lgbm"]
                self._all_cb_forecasts = per_model["catboost"]
                self._all_ml_forecasts = (
                    self.w_lgb_share * self._all_lgb_forecasts
                    + (1 - self.w_lgb_share) * self._all_cb_forecasts
                )
            self._all_ma_forecasts = _seasonal_ma_forecast(sales_hist, sim.in_stock, horizon=horizon)
            # Optional Nixtla statsforecast ensemble member.
            if self.w_stats > 0:
                self._stats_forecaster = StatsForecasterEnsemble(
                    censoring_strategy=self.censoring_strategy,
                )
                self._stats_forecaster.fit(sales_hist, sim.in_stock)
                self._all_stats_forecasts = self._stats_forecaster.predict(horizon=horizon)

        start = round_idx + LEAD_TIME
        end = start + self.coverage_weeks
        ml_sum = self._all_ml_forecasts.iloc[:, start:end].sum(axis=1)
        ma_sum = self._all_ma_forecasts.iloc[:, start:end].sum(axis=1).reindex(ml_sum.index).fillna(0)
        w = self._current_w_ma(ml_sum.index)

        # Three-way blend when stats is active:
        #   mu_sum = w_ma*MA + w_stats*Stats + (1 - w_ma - w_stats) * ML
        if self.w_stats > 0 and self._all_stats_forecasts is not None:
            stats_sum = (self._all_stats_forecasts.iloc[:, start:end]
                         .sum(axis=1).reindex(ml_sum.index).fillna(0))
            residual_w = (1 - w - self.w_stats).clip(lower=0)
            mu_sum = w * ma_sum + self.w_stats * stats_sum + residual_w * ml_sum
        else:
            mu_sum = w * ma_sum + (1 - w) * ml_sum
        self.last_forecast = mu_sum.copy()

        mult = (self.per_round_multiplier[round_idx]
                if self.per_round_multiplier is not None
                and round_idx < len(self.per_round_multiplier)
                else self.multiplier)
        order_up_to = (mult * mu_sum + self.safety_units).clip(lower=0)
        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)


class RMSESafetyPolicy:
    """ML Point forecast + k * RMSE safety from backtest residuals.

    Backtest-RMSE is measured per SKU on in-sample 1-step predictions (last K
    observations of the training data). This captures forecast error magnitude,
    not raw demand std (which we saw overshoots).

    order_up_to = sum(mu_hat) + k * RMSE * sqrt(coverage_weeks)
    """

    def __init__(self, coverage_weeks: int = 2, k: float = 0.5,
                 multiplier: float = 1.0, backtest_window: int = 26,
                 master=None, random_state: int = 42,
                 censoring_strategy: str = "mean_impute"):
        self.coverage_weeks = coverage_weeks
        self.k = k
        self.multiplier = multiplier
        self.backtest_window = backtest_window
        self.master = master
        self.random_state = random_state
        self.censoring_strategy = censoring_strategy
        self.last_forecast: pd.Series | None = None
        self.last_quantile_forecast: pd.Series | None = None
        self._forecaster: DemandForecaster | None = None
        self._all_forecasts: pd.DataFrame | None = None
        self._rmse: pd.Series | None = None

    def _fit(self, sales_hist, in_stock):
        # Backtest: fit on sales_hist[:-W], predict W weeks, measure residuals per SKU.
        W = self.backtest_window
        train_hist = sales_hist.iloc[:, :-W]
        holdout = sales_hist.iloc[:, -W:]
        bt_forecaster = DemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy,
        )
        bt_forecaster.fit(train_hist, in_stock.iloc[:, :-W] if in_stock is not None else None)
        preds_bt = bt_forecaster.predict(horizon=W)
        # Align and compute RMSE.
        common_cols = preds_bt.columns[: min(len(preds_bt.columns), holdout.shape[1])]
        if len(common_cols) == 0:
            self._rmse = pd.Series(1.0, index=sales_hist.index)
        else:
            resid = holdout.iloc[:, : len(common_cols)].values - preds_bt[common_cols].reindex(sales_hist.index).values
            self._rmse = pd.Series(np.sqrt(np.nanmean(resid ** 2, axis=1)), index=sales_hist.index).fillna(0)

        # Now fit the full model for forward predictions.
        self._forecaster = DemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy,
        )
        self._forecaster.fit(sales_hist, in_stock)

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._forecaster is None:
            self._fit(sales_hist, sim.in_stock)
            horizon = 6 + self.coverage_weeks + LEAD_TIME
            self._all_forecasts = self._forecaster.predict(horizon=horizon)

        start = round_idx + LEAD_TIME
        end = start + self.coverage_weeks
        mu_sum = self._all_forecasts.iloc[:, start:end].sum(axis=1)
        self.last_forecast = mu_sum.copy()
        safety = self.k * self._rmse.reindex(mu_sum.index).fillna(0) * np.sqrt(self.coverage_weeks)
        order_up_to = (self.multiplier * mu_sum + safety).clip(lower=0)
        self.last_quantile_forecast = order_up_to.copy()
        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)


class TrajectoryPolicy:
    """Trajectory-based order optimization.

    For each SKU independently, at round r, enumerate candidate order quantities
    Q in [0 .. Qmax] and simulate the forward 8-week trajectory assuming demand
    follows the forecasted mean (+ gaussian noise from backtest RMSE). Pick the Q
    minimizing expected total (holding + shortage) cost over the remaining horizon.

    This matches what Carlo (DeepAR) and Ruben (quantile PMF) did at top of VN2
    leaderboard.

    Simplifications vs full PMF enumeration:
      - Demand sampled as Normal(mu_hat, RMSE) rather than full distribution.
      - Per-SKU independent optimization (no cross-SKU interaction).
      - Finite search grid for Q.
    """

    HOLDING_COST = 0.2
    SHORTAGE_COST = 1.0
    LEAD_TIME = 2

    def __init__(self, coverage_weeks: int = 2, n_samples: int = 30,
                 q_grid_step: int = 1, multiplier: float = 1.0, master=None,
                 random_state: int = 42, censoring_strategy: str = "mean_impute"):
        self.coverage_weeks = coverage_weeks
        self.n_samples = n_samples
        self.q_grid_step = q_grid_step
        self.multiplier = multiplier
        self.master = master
        self.random_state = random_state
        self.censoring_strategy = censoring_strategy
        self.last_forecast: pd.Series | None = None
        self._forecaster: DemandForecaster | None = None
        self._all_forecasts: pd.DataFrame | None = None
        self._rmse: pd.Series | None = None

    def _fit(self, sales_hist, in_stock):
        # Same backtest RMSE estimation as RMSESafetyPolicy.
        W = 26
        bt = DemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy,
        )
        bt.fit(sales_hist.iloc[:, :-W], in_stock.iloc[:, :-W] if in_stock is not None else None)
        preds_bt = bt.predict(horizon=W)
        holdout = sales_hist.iloc[:, -W:]
        common_cols = preds_bt.columns[: min(len(preds_bt.columns), holdout.shape[1])]
        if len(common_cols) == 0:
            self._rmse = pd.Series(1.0, index=sales_hist.index)
        else:
            resid = holdout.iloc[:, : len(common_cols)].values - preds_bt[common_cols].reindex(sales_hist.index).values
            self._rmse = pd.Series(np.sqrt(np.nanmean(resid ** 2, axis=1)), index=sales_hist.index).fillna(0.5)

        self._forecaster = DemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy,
        )
        self._forecaster.fit(sales_hist, in_stock)

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._forecaster is None:
            self._fit(sales_hist, sim.in_stock)
            horizon = 6 + self.coverage_weeks + self.LEAD_TIME + 4  # lookahead buffer
            self._all_forecasts = self._forecaster.predict(horizon=horizon)

        # Current state per SKU.
        end_inv = sim.end_inventory.values
        w1 = sim.in_transit_w1.values
        w2 = sim.in_transit_w2.values  # currently 0 (cleared in sim), about to be set

        n = len(end_inv)
        # Forecast array over the next H weeks (H = horizon).
        fc = self._all_forecasts.reindex(sim.end_inventory.index).values  # [n, H]
        rmse = self._rmse.reindex(sim.end_inventory.index).fillna(0.5).values  # [n]

        # We decide Q for round `round_idx`. This Q arrives at period t+2 (weeks-from-now).
        # We simulate future weeks from now: we still have w1 arrival at t+1, then Q at t+2,
        # and then subsequent orders at t+3..t+final_round are unknown; assume follow-up
        # policy will match mu_hat demand perfectly (zero-holding lookahead).
        # Simulate a horizon of H_sim weeks ahead from now.
        H_sim = 6
        rng = np.random.default_rng(self.random_state + round_idx)

        best_q = np.zeros(n, dtype=int)
        self.last_forecast = pd.Series(fc[:, :self.coverage_weeks].sum(axis=1),
                                       index=sim.end_inventory.index)

        # Candidate order quantities per SKU: 0 .. ceil(mu_sum*2)
        mu_sum_cov = fc[:, :self.coverage_weeks].sum(axis=1)
        q_max = np.ceil(mu_sum_cov * 2 + 3 * rmse).astype(int)
        q_max = np.clip(q_max, 0, 200)

        # Vectorized per-SKU: loop only over SKUs; candidates × time is numpy.
        for i in range(n):
            qm = int(q_max[i])
            if qm == 0:
                best_q[i] = 0
                continue
            candidates = np.arange(0, qm + 1, self.q_grid_step, dtype=np.float32)  # [C]
            C = len(candidates)
            sigma = max(float(rmse[i]), 0.5)
            mu = fc[i, :H_sim].astype(np.float32)  # [H]
            # Sample S demand trajectories once: [S, H]
            samples = rng.normal(loc=mu, scale=sigma, size=(self.n_samples, H_sim)).astype(np.float32)
            np.clip(samples, 0, None, out=samples)

            # State: on_hand[C, S] per (candidate, sample). Evolves over time.
            on_hand = np.full((C, self.n_samples), float(end_inv[i]), dtype=np.float32)
            total_cost = np.zeros(C, dtype=np.float32)
            for t in range(H_sim):
                if t == 0:
                    on_hand += float(w1[i])
                elif t == 1:
                    on_hand += candidates[:, None]  # broadcast order arrival
                else:
                    on_hand += float(mu[t])  # assume future policy matches mean
                d = samples[:, t][None, :]  # [1, S]
                sales = np.minimum(on_hand, d)
                end_t = on_hand - sales
                missed = d - sales
                total_cost += missed.mean(axis=1) * self.SHORTAGE_COST \
                              + end_t.mean(axis=1) * self.HOLDING_COST
                on_hand = end_t  # propagate stochastic state

            best_idx = int(total_cost.argmin())
            best_q[i] = int(candidates[best_idx])

        order_series = pd.Series(best_q, index=sim.end_inventory.index)
        return (order_series * self.multiplier).clip(lower=0).round(0).astype(int)


class ConsensusPolicy:
    """Bayesian model averaging: average orders from K sub-policies per SKU per round.

    Each sub-policy is a fully configured callable (typically several EnsemblePolicy
    instances with different hyperparams/weights). At each round, all K are called,
    their integer orders are averaged (with optional weights), then rounded.

    Rationale: averaging cancels trial-to-trial noise. Picking a single
    best-by-CV config exposes us to CV-noise exploitation; consensus is
    structurally more robust.
    """

    def __init__(self, sub_policies: list, weights: "list[float] | None" = None):
        if not sub_policies:
            raise ValueError("ConsensusPolicy needs at least one sub-policy")
        self.sub_policies = sub_policies
        self.weights = weights if weights is not None else [1.0] * len(sub_policies)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        self.last_forecast: "pd.Series | None" = None

    def __call__(self, sim, round_idx, sales_hist):
        orders = []
        for p in self.sub_policies:
            o = p(sim, round_idx, sales_hist)
            orders.append(o.astype(float))
        # Weighted average per SKU.
        avg = pd.Series(0.0, index=orders[0].index)
        for w, o in zip(self.weights, orders):
            avg = avg + w * o.reindex(avg.index).fillna(0)
        # Expose the mean of sub-policy forecasts as last_forecast (best-effort).
        fcs = [getattr(p, "last_forecast", None) for p in self.sub_policies]
        fcs = [f for f in fcs if f is not None]
        if fcs:
            mean_fc = pd.DataFrame(fcs).mean(axis=0)
            self.last_forecast = mean_fc
        return avg.round(0).clip(lower=0).astype(int)


class ConformalEnsemblePolicy:
    """Ensemble (MA + ML) point forecaster + conformal per-SKU additive offset.

    Split-Conformal idea:
      - Hold out the last K weeks of training as calibration.
      - Fit the ensemble on the first (T-K) weeks.
      - Recursively forecast K weeks ahead on held-out, compute per-SKU residuals
        on all 2-week cumulative sums.
      - Per-SKU empirical alpha_cp quantile of residuals -> additive offset.
      - Re-fit ensemble on FULL data for inference; add the offset at order time.

    order_up_to_i = multiplier * ensemble_point_sum_i + q_{alpha_cp}(residuals_i)
    """

    def __init__(self, coverage_weeks: int = 2, w_ma: float = 0.25,
                 alpha_cp: float = 0.70, multiplier: float = 1.0,
                 backtest_window: int = 26, master=None, random_state: int = 42,
                 censoring_strategy: str = "mean_impute",
                 min_offset: float = 0.0):
        self.coverage_weeks = coverage_weeks
        self.w_ma = w_ma
        self.alpha_cp = alpha_cp
        self.multiplier = multiplier
        self.backtest_window = backtest_window
        self.master = master
        self.random_state = random_state
        self.censoring_strategy = censoring_strategy
        self.min_offset = min_offset
        self.last_forecast: "pd.Series | None" = None
        self.last_quantile_forecast: "pd.Series | None" = None
        self._ml_forecasts: "pd.DataFrame | None" = None
        self._ma_forecasts: "pd.DataFrame | None" = None
        self._offsets: "pd.Series | None" = None

    def _calibrate(self, sales_hist, in_stock):
        K = self.backtest_window
        cov = self.coverage_weeks
        train = sales_hist.iloc[:, :-K]
        is_train = in_stock.iloc[:, :-K] if in_stock is not None else None
        calib = sales_hist.iloc[:, -K:]

        # Fit ensemble on train.
        ml = DemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy,
        )
        ml.fit(train, is_train)
        ml_calib = ml.predict(horizon=K)
        ma_calib = _seasonal_ma_forecast(train, is_train, horizon=K)
        # Align columns.
        n_cols = min(ml_calib.shape[1], ma_calib.shape[1], calib.shape[1])
        ens = (self.w_ma * ma_calib.iloc[:, :n_cols].reindex(calib.index).fillna(0).values
               + (1 - self.w_ma) * ml_calib.iloc[:, :n_cols].reindex(calib.index).fillna(0).values)
        actual = calib.iloc[:, :n_cols].values

        # Per-SKU residuals on cov-week sliding-window sums.
        res_list = []
        for t in range(n_cols - cov + 1):
            actual_sum = actual[:, t:t + cov].sum(axis=1)
            pred_sum = ens[:, t:t + cov].sum(axis=1)
            res_list.append(actual_sum - pred_sum)
        residuals = np.column_stack(res_list)  # [n_sku, K-cov+1]

        # Per-SKU alpha_cp quantile (positive => under-forecast => need positive safety).
        self._offsets = pd.Series(
            np.quantile(residuals, self.alpha_cp, axis=1),
            index=sales_hist.index,
        ).clip(lower=self.min_offset)

    def _fit_full(self, sales_hist, in_stock):
        ml = DemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy,
        )
        ml.fit(sales_hist, in_stock)
        horizon = 6 + self.coverage_weeks + LEAD_TIME
        self._ml_forecasts = ml.predict(horizon=horizon)
        self._ma_forecasts = _seasonal_ma_forecast(sales_hist, in_stock, horizon=horizon)

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._offsets is None:
            self._calibrate(sales_hist, sim.in_stock)
            self._fit_full(sales_hist, sim.in_stock)

        start = round_idx + LEAD_TIME
        end = start + self.coverage_weeks
        ml_sum = self._ml_forecasts.iloc[:, start:end].sum(axis=1)
        ma_sum = self._ma_forecasts.iloc[:, start:end].sum(axis=1).reindex(ml_sum.index).fillna(0)
        mu_sum = self.w_ma * ma_sum + (1 - self.w_ma) * ml_sum
        offsets = self._offsets.reindex(mu_sum.index).fillna(0)

        order_up_to = (self.multiplier * mu_sum + offsets).clip(lower=0)
        self.last_forecast = mu_sum.copy()
        self.last_quantile_forecast = order_up_to.copy()

        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)


class CumulativeQuantilePolicy:
    """Order-up-to = cumulative-demand quantile forecast at alpha.

    Uses CumulativeQuantileForecaster. The model outputs Q_alpha(sum of the
    coverage-week window that our order will cover). Order = max(0, Q - net_inv).

    This is the approach Matias Alvo used: LGBM predicting quantiles of cumulative
    demand for various horizons (we use one horizon = the order-coverage window).
    """

    def __init__(self, coverage: int = 2, alpha: float = 0.65,
                 multiplier: float = 1.0, random_state: int = 42,
                 censoring_strategy: str = "mean_impute",
                 ensemble: bool = True):
        self.coverage = coverage
        self.alpha = alpha
        self.multiplier = multiplier
        self.random_state = random_state
        self.censoring_strategy = censoring_strategy
        self.ensemble = ensemble
        self.last_forecast: "pd.Series | None" = None
        self.last_quantile_forecast: "pd.Series | None" = None
        self._forecaster = None  # CumulativeQuantileForecaster
        self._sales_hist_ref: "pd.DataFrame | None" = None
        self._in_stock_ref: "pd.DataFrame | None" = None

    def _fit(self, sales_hist, in_stock):
        from cumulative_quantile import CumulativeQuantileForecaster
        self._forecaster = CumulativeQuantileForecaster(
            alpha=self.alpha, coverage=self.coverage,
            lead_offset=LEAD_TIME + 1,
            censoring_strategy=self.censoring_strategy,
            random_state=self.random_state,
            ensemble=self.ensemble,
        )
        self._forecaster.fit(sales_hist, in_stock)
        self._sales_hist_ref = sales_hist
        self._in_stock_ref = in_stock

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._forecaster is None:
            self._fit(sales_hist, sim.in_stock)

        # At round r, our order covers weeks (t_now + LEAD + 1) .. (t_now + LEAD + cov).
        # But the model was trained with targets summing over lead_offset..lead_offset+cov-1
        # from each training time t. At inference, we simply predict from the most recent
        # observation index in sales_hist (the last known week).
        t_now = sales_hist.shape[1] - 1
        # For rounds > 0, the "sales_hist" seen by the policy already includes previously
        # simulated weeks via sim.get_sales_history(); that's what we want.
        q = self._forecaster.predict(sales_hist, sim.in_stock, t_now)
        self.last_forecast = q.copy()
        self.last_quantile_forecast = q.copy()

        order_up_to = (self.multiplier * q).clip(lower=0)
        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)


class QuantilePolicy:
    """Order-up-to policy using a true-quantile forecaster at alpha."""

    def __init__(self, coverage_weeks: int = 3, alpha: float = 0.833, master=None):
        self.coverage_weeks = coverage_weeks
        self.alpha = alpha
        self.master = master
        self.last_forecast: pd.Series | None = None
        self.last_quantile_forecast: pd.Series | None = None
        self._forecaster: QuantileDemandForecaster | None = None
        self._all_forecasts: pd.DataFrame | None = None

    def _fit(self, sales_hist, in_stock):
        self._forecaster = QuantileDemandForecaster(master=self.master, alpha=self.alpha)
        self._forecaster.fit(sales_hist, in_stock)

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._forecaster is None:
            self._fit(sales_hist, sim.in_stock)
            horizon = 6 + self.coverage_weeks + LEAD_TIME
            self._all_forecasts = self._forecaster.predict(horizon=horizon)

        start = round_idx + LEAD_TIME
        end = start + self.coverage_weeks
        order_up_to = self._all_forecasts.iloc[:, start:end].sum(axis=1)
        # Report the same number as both "point" forecast for MAE purposes and
        # as the quantile forecast (it IS a quantile prediction already).
        self.last_forecast = order_up_to.copy()
        self.last_quantile_forecast = order_up_to.copy()

        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)


class TCNPolicy:
    """TCN+FiLM+Fourier forecaster (Bartosz-style) plugged into cov-based order-up-to.

    Trains once at round 0 on sales_hist. At each round, re-runs H-step inference
    using the latest context window (sales_hist grows with simulated actuals).

    order_up_to = multiplier * sum(mu_hat[LEAD_TIME : LEAD_TIME+coverage]) + safety_units

    where mu_hat is the H-step point forecast from the trained TCN (scaled back
    to raw units). `coverage_weeks` must satisfy LEAD_TIME + coverage <= H.
    """

    def __init__(self, coverage_weeks: int = 2, multiplier: float = 1.0,
                 safety_units: float = 0.0, cfg=None, verbose: bool = False):
        from tcn_forecaster import TCNConfig
        self.coverage_weeks = coverage_weeks
        self.multiplier = multiplier
        self.safety_units = safety_units
        self.cfg = cfg if cfg is not None else TCNConfig()
        self.verbose = verbose
        self.last_forecast: "pd.Series | None" = None
        self._trained: "dict | None" = None
        # Sanity: coverage must fit in horizon.
        assert LEAD_TIME + self.coverage_weeks <= self.cfg.horizon, (
            f"coverage={self.coverage_weeks} + LEAD_TIME={LEAD_TIME} exceeds H={self.cfg.horizon}"
        )

    def _fit(self, sales_hist, in_stock):
        from tcn_forecaster import train_tcn
        self._trained = train_tcn(sales_hist, in_stock, cfg=self.cfg, verbose=self.verbose)

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._trained is None:
            self._fit(sales_hist, sim.in_stock)

        from tcn_forecaster import predict_tcn
        preds = predict_tcn(self._trained, sales_hist, sim.in_stock)  # [N, H]

        # Order placed at round r arrives at horizon LEAD_TIME+1 (1-indexed),
        # i.e. preds column index LEAD_TIME (0-indexed). Sum coverage_weeks.
        start = LEAD_TIME           # horizon index where the arrival week sits
        end = start + self.coverage_weeks
        mu_sum = preds.iloc[:, start:end].sum(axis=1)
        self.last_forecast = mu_sum.copy()

        order_up_to = (self.multiplier * mu_sum + self.safety_units).clip(lower=0)
        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)


# --------------------------------------------------------------------------- #
# Stage 2 — Bartosz's cost-aware ordering policy                              #
# --------------------------------------------------------------------------- #
class CostAwarePolicy:
    """Stage 2: inventory-projection + critical-ratio-quantile ordering.

    Adapted from Bartosz's deck (which assumes LT=3: T1, T2 both pipeline) to
    the VN2 problem's LT=2 (only T1 pre-existing; our new order becomes T2 and
    arrives 2 weeks later). For LT=2, the analogous state-projection rule is:

        E1 = max( I0 + T1 - d1, 0 )        # projected end-of-wk1 inventory
        Q_alpha(d2) = d2 + z_alpha * sigma_d2   # critical-ratio quantile on arrival week
        order = max( Q_alpha(d2) - E1, 0 )

    where d1 = forecast for week r+1, d2 = forecast for week r+2 (arrival week).

    alpha=0.833 is the critical ratio (shortage / (shortage + holding) = 1/1.2).
    sigma_d3 is estimated per-SKU from backtest RMSE on the last `backtest_window` weeks.

    This policy is forecaster-agnostic: it uses an EnsemblePolicy internally for
    point forecasts d1..d3, but swaps the "cov × multiplier" ordering rule for
    the state-projection rule above. Our best ensemble's forecaster is reused
    directly via `ensemble_cfg`.

    Parameters
    ----------
    alpha : float
        Critical ratio for the Q_alpha target; default 0.833 (= 1/1.2).
    backtest_window : int
        Weeks held out to estimate per-SKU sigma_d3 via RMSE; default 26.
    safety_floor : float
        Lower bound on sigma_d3 per SKU (prevents zero-safety for flat series).
    multiplier : float
        Additional scalar on the Q_alpha target (default 1.0; use to bias orders).
    ensemble_cfg : dict | None
        kwargs forwarded to the internal EnsemblePolicy. If None, uses defaults
        (coverage_weeks=3, w_ma=0.25, censoring='mean_impute', random_state=42).
    """

    def __init__(self, alpha: float = 0.833, backtest_window: int = 26,
                 safety_floor: float = 0.5, multiplier: float = 1.0,
                 rmse_horizons: int = 1,
                 per_round_multiplier: "list[float] | None" = None,
                 multiplier_per_sku: "pd.Series | None" = None,
                 ensemble_cfg: dict | None = None):
        self.alpha = alpha
        self.backtest_window = backtest_window
        self.safety_floor = safety_floor
        self.multiplier = multiplier
        # Optional per-round override: list of length 6 (one mult per round 0..5).
        # If provided, overrides `multiplier` at each round; rounds beyond list use `multiplier`.
        self.per_round_multiplier = per_round_multiplier
        # Optional per-SKU override: Series indexed by (Store, Product) of multipliers.
        # Multiplied with `multiplier` (and per_round_multiplier if both provided).
        self.multiplier_per_sku = multiplier_per_sku
        # Use only the first `rmse_horizons` columns of the backtest residuals
        # for per-SKU sigma. Default 1 = 1-step-ahead RMSE (matches short-horizon
        # forecast noise). Setting to backtest_window ≈ mixed-horizon RMSE.
        self.rmse_horizons = max(1, int(rmse_horizons))
        # Force coverage_weeks=3 so the internal ensemble's forecast matrix
        # supports indexing d1, d2, d3 at each round.
        default_cfg = {
            "coverage_weeks": 3,
            "w_ma": 0.25,
            "multiplier": 1.0,
            "censoring_strategy": "mean_impute",
            "random_state": 42,
        }
        if ensemble_cfg:
            default_cfg.update(ensemble_cfg)
        # Ensure coverage >= 3 for the 3-step forecast matrix.
        default_cfg["coverage_weeks"] = max(3, int(default_cfg.get("coverage_weeks", 3)))
        self._ensemble_cfg = default_cfg
        self._ensemble: EnsemblePolicy | None = None
        self._rmse: pd.Series | None = None
        self._z_alpha = _inv_normal_cdf(alpha)
        self.last_forecast: pd.Series | None = None
        self.last_quantile_forecast: pd.Series | None = None

    def _fit(self, sales_hist, in_stock):
        # 1) Fit internal ensemble forecaster.
        self._ensemble = EnsemblePolicy(**self._ensemble_cfg)
        self._ensemble._fit(sales_hist, in_stock)

        # 2) Populate the forecast matrices the ensemble would compute in its own __call__.
        horizon = 6 + LEAD_TIME + 3  # d1..d3 indices for rounds 0..5
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
            sales_hist, in_stock, horizon=horizon
        )
        if self._ensemble.w_stats > 0:
            self._ensemble._stats_forecaster = StatsForecasterEnsemble(
                censoring_strategy=self._ensemble.censoring_strategy,
            )
            self._ensemble._stats_forecaster.fit(sales_hist, in_stock)
            self._ensemble._all_stats_forecasts = self._ensemble._stats_forecaster.predict(
                horizon=horizon
            )

        # 3) Estimate per-SKU sigma_d3 via holdout RMSE.
        W = self.backtest_window
        train = sales_hist.iloc[:, :-W]
        holdout = sales_hist.iloc[:, -W:]
        bt_is = in_stock.iloc[:, :-W] if in_stock is not None else None
        bt = DemandForecaster(
            master=self._ensemble.master,
            random_state=self._ensemble.random_state,
            censoring_strategy=self._ensemble.censoring_strategy,
        )
        bt.fit(train, bt_is)
        bt_preds = bt.predict(horizon=W)
        n = min(bt_preds.shape[1], holdout.shape[1], self.rmse_horizons)
        resid = (
            holdout.iloc[:, :n].values
            - bt_preds.iloc[:, :n].reindex(sales_hist.index).values
        )
        # Use only first `rmse_horizons` steps for sigma estimation; averaging
        # over 26 mixed-horizon weeks over-inflates sigma for short-horizon orders.
        rmse = np.sqrt(np.nanmean(resid ** 2, axis=1))
        self._rmse = pd.Series(rmse, index=sales_hist.index).fillna(self.safety_floor)
        self._rmse = self._rmse.clip(lower=self.safety_floor)

    def _get_d12(self, round_idx: int) -> tuple[pd.Series, pd.Series]:
        """Return point forecasts (d1, d2) for weeks r+1 (next) and r+2 (arrival)."""
        ens = self._ensemble
        ml = ens._all_ml_forecasts
        w = ens._current_w_ma(ml.index)
        out = []
        for h_offset in (round_idx, round_idx + 1):
            d_ml = ml.iloc[:, h_offset]
            d_ma = ens._all_ma_forecasts.iloc[:, h_offset].reindex(ml.index).fillna(0)
            if ens.w_stats > 0 and ens._all_stats_forecasts is not None:
                d_st = ens._all_stats_forecasts.iloc[:, h_offset].reindex(ml.index).fillna(0)
                residual_w = (1 - w - ens.w_stats).clip(lower=0)
                d = w * d_ma + ens.w_stats * d_st + residual_w * d_ml
            else:
                d = w * d_ma + (1 - w) * d_ml
            out.append(d.astype(float))
        return out[0], out[1]

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._ensemble is None:
            self._fit(sales_hist, sim.in_stock)

        d1, d2 = self._get_d12(round_idx)

        I0 = sim.end_inventory.astype(float)
        T1 = sim.in_transit_w1.astype(float)
        # sim.in_transit_w2 is 0 at policy call time (cleared in simulate_week before
        # this; we're about to set it via place_order).

        # Project end-of-(r+1) inventory.
        E1 = (I0 + T1 - d1).clip(lower=0)

        sigma = self._rmse.reindex(d2.index).fillna(self.safety_floor)
        round_mult = (
            self.per_round_multiplier[round_idx]
            if (self.per_round_multiplier is not None
                and round_idx < len(self.per_round_multiplier))
            else self.multiplier
        )
        if self.multiplier_per_sku is not None:
            sku_mult = self.multiplier_per_sku.reindex(d2.index).fillna(1.0)
            mult = round_mult * sku_mult
        else:
            mult = round_mult
        target = mult * (d2 + self._z_alpha * sigma)
        target = target.clip(lower=0)

        order = (target - E1).clip(lower=0).round(0).astype(int)

        # Expose diagnostics for the CV harness.
        self.last_forecast = d2.copy()
        self.last_quantile_forecast = target.copy()
        return order


class ConformalCostAwarePolicy:
    """CostAware Stage-2 with mlforecast conformal prediction quantiles.

    Drop-in replacement for CostAwarePolicy that uses mlforecast's built-in
    conformal prediction intervals to get the alpha-quantile of d2 DIRECTLY,
    instead of approximating it as `d2 + z_alpha * sigma_d2` under a Normal assumption.

    Mechanics:
      1. Fit DemandForecaster with prediction_intervals=PredictionIntervals(...)
         which performs internal cross-validation to compute conformal residuals.
      2. At inference, call predict_quantile_conformal(alpha) to get the alpha-percentile
         demand forecast per SKU per horizon (no Normal assumption).
      3. Same E1 projection rule as CostAware:
            E1 = max(I0 + T1 - d1, 0)
            order = max(target_d2 - E1, 0)
         where target_d2 = predict_quantile_conformal(alpha).
    """

    def __init__(self, alpha: float = 0.65, multiplier: float = 1.0,
                 prediction_intervals_n_windows: int = 4,
                 prediction_intervals_h: int = 11,  # must be >= predict horizon
                 ensemble_cfg: dict | None = None):
        self.alpha = alpha
        self.multiplier = multiplier
        self.pi_n_windows = prediction_intervals_n_windows
        self.pi_h = prediction_intervals_h
        default_cfg = {
            "censoring_strategy": "mean_impute",
            "random_state": 42,
            # MA blending of point forecast — used for d1 (next-week projection)
            # since we don't need a quantile for d1.
            "w_ma": 0.25,
        }
        if ensemble_cfg:
            default_cfg.update(ensemble_cfg)
        self._cfg = default_cfg
        self._forecaster: DemandForecaster | None = None
        self._d1_blend_weights: pd.Series | None = None
        self._all_d1_point: pd.DataFrame | None = None
        self._all_d2_quantile: pd.DataFrame | None = None
        self.last_forecast: pd.Series | None = None
        self.last_quantile_forecast: pd.Series | None = None

    def _fit(self, sales_hist, in_stock):
        # 1) Single DemandForecaster with conformal PI enabled.
        self._forecaster = DemandForecaster(
            censoring_strategy=self._cfg["censoring_strategy"],
            random_state=self._cfg["random_state"],
            prediction_intervals_n_windows=self.pi_n_windows,
            prediction_intervals_h=self.pi_h,
        )
        self._forecaster.fit(sales_hist, in_stock)

        # 2) Pre-compute all needed forecasts.
        horizon = 6 + LEAD_TIME + 3
        self._all_d1_point = self._forecaster.predict(horizon=horizon)
        self._all_d2_quantile = self._forecaster.predict_quantile_conformal(
            horizon=horizon, alpha=self.alpha
        )

        # 3) MA blend for d1 (point forecast, no quantile needed for projection).
        ma = _seasonal_ma_forecast(sales_hist, in_stock, horizon=horizon)
        common = self._all_d1_point.columns[: min(len(self._all_d1_point.columns), len(ma.columns))]
        w = self._cfg.get("w_ma", 0.25)
        self._all_d1_point = (
            (1 - w) * self._all_d1_point[common]
            + w * ma[common].reindex(self._all_d1_point.index).fillna(0)
        )
        # Same blend on the quantile forecast so it's compatible scale-wise.
        # We blend the quantile's *median* with MA, then re-add the conformal width.
        # Simpler: just leave quantile as-is. (Conformal already captures distribution.)

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._forecaster is None:
            self._fit(sales_hist, sim.in_stock)

        d1 = self._all_d1_point.iloc[:, round_idx].astype(float)
        d2 = self._all_d2_quantile.iloc[:, round_idx + 1].astype(float)  # ALPHA quantile

        I0 = sim.end_inventory.astype(float)
        T1 = sim.in_transit_w1.astype(float)
        E1 = (I0 + T1 - d1).clip(lower=0)
        target = (self.multiplier * d2).clip(lower=0)
        order = (target - E1).clip(lower=0).round(0).astype(int)

        self.last_forecast = d2.copy()
        self.last_quantile_forecast = target.copy()
        return order


class DiverseCostAwarePolicy:
    """CostAware ordering + 5-model diverse ensemble (equal-weight) + seasonal-MA blend.

    Base models (via DiverseDemandForecaster):
      - lgbm_default, lgbm_shallow, lgbm_tweedie, catboost_default, catboost_deep
    Ensemble = equal-weight mean of the 5 models
    Blend with Seasonal MA at w_ma (default 0.25)

    Rationale: fixed equal weights are robust to CV noise (no overfit risk from
    learning weights). Diversity via Tweedie + shallow/deep gives uncorrelated
    errors that cancel under averaging.
    """

    def __init__(self, alpha: float = 0.65, backtest_window: int = 26,
                 safety_floor: float = 0.5, multiplier: float = 1.0,
                 rmse_horizons: int = 1,
                 w_ma: float = 0.25,
                 censoring_strategy: str = "mean_impute",
                 random_state: int = 42,
                 master=None,
                 n_variants: int = 5,
                 # Optional per-demand-bin multipliers. If both set, overrides `multiplier`.
                 mult_high: "float | None" = None,
                 mult_low:  "float | None" = None,
                 high_demand_quantile: float = 0.75,
                 bin_basis: str = "sku",  # "sku" (per-SKU mean) or "store" (per-store aggregate)
                 recency_decay: "float | None" = None):
        self.alpha = alpha
        self.backtest_window = backtest_window
        self.safety_floor = safety_floor
        self.multiplier = multiplier
        self.rmse_horizons = max(1, int(rmse_horizons))
        self.w_ma = w_ma
        self.censoring_strategy = censoring_strategy
        self.random_state = random_state
        self.master = master
        self.n_variants = n_variants
        self.mult_high = mult_high
        self.mult_low  = mult_low
        self.high_demand_quantile = high_demand_quantile
        assert bin_basis in ("sku", "store"), f"bin_basis must be 'sku' or 'store', got {bin_basis!r}"
        self.bin_basis = bin_basis
        self.recency_decay = recency_decay
        self._sku_multiplier: pd.Series | None = None
        self._forecaster: DiverseDemandForecaster | None = None
        self._all_ml_forecasts: pd.DataFrame | None = None
        self._all_ma_forecasts: pd.DataFrame | None = None
        self._rmse: pd.Series | None = None
        self._z_alpha = _inv_normal_cdf(alpha)
        self.last_forecast: pd.Series | None = None
        self.last_quantile_forecast: pd.Series | None = None

    def _fit(self, sales_hist, in_stock):
        def _log(m): print(f"    [diverse] {m}", flush=True)
        # 0) Per-SKU (or per-store-aggregated) multiplier series if high/low overrides are provided.
        if self.mult_high is not None or self.mult_low is not None:
            if self.bin_basis == "sku":
                # Split SKUs individually by their own mean weekly demand.
                sku_metric = sales_hist.mean(axis=1)
            else:  # "store": split by store's total mean demand (sum across its products)
                per_store = sales_hist.groupby(level=0).sum().mean(axis=1)  # store -> mean
                # Broadcast store-level metric back onto each (Store, Product) row.
                sku_metric = pd.Series(
                    per_store.reindex(sales_hist.index.get_level_values(0)).values,
                    index=sales_hist.index,
                )
            thr = sku_metric.quantile(self.high_demand_quantile)
            high_mask = sku_metric >= thr
            mult_l = self.mult_low  if self.mult_low  is not None else self.multiplier
            mult_h = self.mult_high if self.mult_high is not None else self.multiplier
            self._sku_multiplier = pd.Series(mult_l, index=sales_hist.index)
            self._sku_multiplier.loc[high_mask] = mult_h
            _log(f"per-{self.bin_basis} mult: high(>=q{self.high_demand_quantile})={mult_h:.3f} "
                 f"low={mult_l:.3f}  n_high={int(high_mask.sum())}/{len(sales_hist.index)}")
        else:
            self._sku_multiplier = None

        # 1) Diverse forecaster — 5 base models trained on full history.
        self._forecaster = DiverseDemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy, n_variants=self.n_variants,
            recency_decay=self.recency_decay,
        )
        _log("fitting 5-model diverse forecaster...")
        self._forecaster.fit(sales_hist, in_stock)
        _log("fit done, predicting...")
        horizon = 6 + LEAD_TIME + 3
        per_model = self._forecaster.predict_models(horizon=horizon)
        model_names = list(per_model.keys())
        _log(f"got {len(model_names)} models: {model_names}")
        # 2) Equal-weight mean across models
        arrs = [per_model[n].reindex(sales_hist.index).values for n in model_names]
        ml_mean = np.mean(np.stack(arrs, axis=0), axis=0)
        ml_df = pd.DataFrame(ml_mean, index=sales_hist.index,
                              columns=per_model[model_names[0]].columns).clip(lower=0)
        # 3) Seasonal MA
        ma_df = _seasonal_ma_forecast(sales_hist, in_stock, horizon=horizon)
        # 4) Blend w_ma·MA + (1 - w_ma)·ML_mean
        n_cols = min(ml_df.shape[1], ma_df.shape[1])
        ma_aligned = ma_df.iloc[:, :n_cols].reindex(ml_df.index).fillna(0)
        blended = (self.w_ma * ma_aligned
                    + (1 - self.w_ma) * ml_df.iloc[:, :n_cols]).clip(lower=0)
        self._all_ml_forecasts = blended
        self._all_ma_forecasts = ma_df

        # 5) Per-SKU sigma via backtest RMSE (same as CostAwarePolicy).
        W = self.backtest_window
        train = sales_hist.iloc[:, :-W]
        holdout = sales_hist.iloc[:, -W:]
        bt_is = in_stock.iloc[:, :-W] if in_stock is not None else None
        _log("backtest for sigma (re-fitting 5 models on train-W)...")
        bt = DiverseDemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy, n_variants=self.n_variants,
            recency_decay=self.recency_decay,
        )
        bt.fit(train, bt_is)
        bt_per = bt.predict_models(horizon=W)
        bt_arrs = [bt_per[n].reindex(sales_hist.index).values for n in bt_per]
        bt_mean = np.mean(np.stack(bt_arrs, axis=0), axis=0)
        n = min(bt_mean.shape[1], holdout.shape[1], self.rmse_horizons)
        resid = (holdout.iloc[:, :n].values - bt_mean[:, :n])
        rmse = np.sqrt(np.nanmean(resid ** 2, axis=1))
        self._rmse = pd.Series(rmse, index=sales_hist.index).fillna(self.safety_floor).clip(lower=self.safety_floor)
        _log("fit complete.")

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._forecaster is None:
            self._fit(sales_hist, sim.in_stock)

        d1 = self._all_ml_forecasts.iloc[:, round_idx]
        d2 = self._all_ml_forecasts.iloc[:, round_idx + 1]

        I0 = sim.end_inventory.astype(float)
        T1 = sim.in_transit_w1.astype(float)
        E1 = (I0 + T1 - d1).clip(lower=0)

        sigma = self._rmse.reindex(d2.index).fillna(self.safety_floor)
        if self._sku_multiplier is not None:
            mult = self._sku_multiplier.reindex(d2.index).fillna(self.multiplier)
        else:
            mult = self.multiplier
        target = mult * (d2 + self._z_alpha * sigma)
        target = target.clip(lower=0)

        order = (target - E1).clip(lower=0).round(0).astype(int)
        self.last_forecast = d2.copy()
        self.last_quantile_forecast = target.copy()
        return order


class QuantileCostAwarePolicy:
    """Cost-aware ordering using DIRECTLY-TRAINED quantile forecasts (pinball loss).

    Instead of `target = d2 + z_alpha * sigma` (Normal approximation), we train an
    ensemble with pinball loss at alpha=0.833 so its output IS the conditional
    alpha-quantile of demand. No sigma estimation needed.

    Policy flow (LT=2):
      d1 = blended point forecast  (for inventory projection)
      Q_alpha(d2) = mean of 5 quantile-trained models
      E1 = max(I0 + T1 - d1, 0)
      order = max(multiplier * Q_alpha(d2) - E1, 0)

    This is the "cost-aware training" approach: the loss function the base
    learners minimize is asymmetric in the same 5:1 shortage:holding ratio as
    the cost function, so the forecast already encodes the cost structure.
    """

    def __init__(self, target_quantile: float = 0.833,
                 multiplier: float = 1.0,
                 w_ma: float = 0.329,
                 point_n_variants: int = 5,
                 censoring_strategy: str = "mean_impute",
                 random_state: int = 42,
                 master=None):
        self.target_quantile = target_quantile
        self.multiplier = multiplier
        self.w_ma = w_ma
        self.point_n_variants = point_n_variants
        self.censoring_strategy = censoring_strategy
        self.random_state = random_state
        self.master = master
        self._point_fc: pd.DataFrame | None = None   # blended point forecasts (for d1)
        self._quant_fc: pd.DataFrame | None = None   # mean of quantile forecasts (Q_alpha)
        self.last_forecast: pd.Series | None = None
        self.last_quantile_forecast: pd.Series | None = None

    def _fit(self, sales_hist, in_stock):
        def _log(m): print(f"    [qcap] {m}", flush=True)
        horizon = 6 + LEAD_TIME + 3

        # 1) POINT forecaster for d1 (inventory projection) — reuse DiverseDemandForecaster.
        _log(f"fitting point forecaster ({self.point_n_variants} models)...")
        point_fcst = DiverseDemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy, n_variants=self.point_n_variants,
        )
        point_fcst.fit(sales_hist, in_stock)
        point_per_model = point_fcst.predict_models(horizon=horizon)
        pm = [point_per_model[k].reindex(sales_hist.index).values for k in point_per_model]
        point_ml_mean = np.mean(np.stack(pm, axis=0), axis=0)
        point_ml_df = pd.DataFrame(point_ml_mean, index=sales_hist.index,
                                    columns=point_per_model[list(point_per_model)[0]].columns).clip(lower=0)
        ma_df = _seasonal_ma_forecast(sales_hist, in_stock, horizon=horizon)
        ma_aligned = ma_df.iloc[:, :point_ml_df.shape[1]].reindex(point_ml_df.index).fillna(0)
        self._point_fc = (self.w_ma * ma_aligned
                          + (1 - self.w_ma) * point_ml_df).clip(lower=0)
        _log("point forecaster done")

        # 2) QUANTILE forecaster (5 models, pinball loss at target_quantile) for Q_alpha(d2).
        _log(f"fitting quantile forecaster (5 models @ alpha={self.target_quantile:.3f})...")
        quant_fcst = QuantileDiverseDemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy,
            target_quantile=self.target_quantile,
        )
        quant_fcst.fit(sales_hist, in_stock)
        quant_per_model = quant_fcst.predict_models(horizon=horizon)
        qm = [quant_per_model[k].reindex(sales_hist.index).values for k in quant_per_model]
        quant_mean = np.mean(np.stack(qm, axis=0), axis=0)
        self._quant_fc = pd.DataFrame(
            quant_mean, index=sales_hist.index,
            columns=quant_per_model[list(quant_per_model)[0]].columns,
        ).clip(lower=0)
        _log("quantile forecaster done.")

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._point_fc is None:
            self._fit(sales_hist, sim.in_stock)

        d1    = self._point_fc.iloc[:, round_idx]
        q_d2  = self._quant_fc.iloc[:, round_idx + 1]

        I0 = sim.end_inventory.astype(float)
        T1 = sim.in_transit_w1.astype(float)
        E1 = (I0 + T1 - d1).clip(lower=0)

        target = (self.multiplier * q_d2).clip(lower=0)
        order = (target - E1).clip(lower=0).round(0).astype(int)

        self.last_forecast = q_d2.copy()
        self.last_quantile_forecast = target.copy()
        return order


class StackedCostAwarePolicy:
    """CostAware ordering + Ridge-stacked forecast.

    Instead of fixed weights (0.25 MA + 0.75 ML(=0.5 LGB + 0.5 CB)), learn the
    blend weights by backtest:
      - Hold out last `backtest_window` weeks
      - Fit LGB, CB on the rest; compute MA forecast for the held-out window
      - Build (n_sku × n_weeks) rows of (lgb_pred, cb_pred, ma_pred) and actual
      - Fit a sklearn regressor (default: non-negative Ridge with intercept) to map
        base-preds -> actual demand
      - At inference: generate all 3 base preds, combine via the fitted meta-learner

    The ordering layer is identical to CostAwarePolicy:
      E1 = max(I0 + T1 - d1, 0)
      target = d2 + z_alpha * sigma_d2
      order = max(target - E1, 0)

    Parameters
    ----------
    meta_regressor : "ridge_nn" (default) | "ridge" | "linreg"
        Meta-learner choice. "ridge_nn" = non-negative Ridge (interpretable weights).
    """

    def __init__(self, alpha: float = 0.65, backtest_window: int = 26,
                 safety_floor: float = 0.5, multiplier: float = 1.0,
                 rmse_horizons: int = 1,
                 meta_regressor: str = "ridge_nn",
                 ridge_alpha: float = 1.0,
                 ensemble_cfg: dict | None = None):
        self.alpha = alpha
        self.backtest_window = backtest_window
        self.safety_floor = safety_floor
        self.multiplier = multiplier
        self.rmse_horizons = max(1, int(rmse_horizons))
        self.meta_regressor = meta_regressor
        self.ridge_alpha = ridge_alpha
        default_cfg = {
            "coverage_weeks": 3,
            "w_ma": 0.0,        # unused; we do our own blend
            "censoring_strategy": "mean_impute",
            "random_state": 42,
        }
        if ensemble_cfg:
            default_cfg.update(ensemble_cfg)
        default_cfg["coverage_weeks"] = max(3, int(default_cfg.get("coverage_weeks", 3)))
        self._ensemble_cfg = default_cfg
        self._ensemble: EnsemblePolicy | None = None
        self._meta = None
        self._meta_coefs: dict | None = None
        self._rmse: pd.Series | None = None
        self._z_alpha = _inv_normal_cdf(alpha)
        self.last_forecast: pd.Series | None = None
        self.last_quantile_forecast: pd.Series | None = None

    def _fit_meta_learner(self, lgb_bt, cb_bt, ma_bt, actual):
        """Fit the meta regressor on backtest triplets."""
        def _log(m): print(f"    [stacked-meta] {m}", flush=True)
        _log(f"enter: lgb {lgb_bt.shape} cb {cb_bt.shape} ma {ma_bt.shape} act {actual.shape}")
        n_cols = min(lgb_bt.shape[1], cb_bt.shape[1], ma_bt.shape[1], actual.shape[1])
        idx = lgb_bt.index
        lgb_v = lgb_bt.iloc[:, :n_cols].reindex(idx).values.flatten()
        cb_v  = cb_bt.iloc[:, :n_cols].reindex(idx).values.flatten()
        ma_v  = ma_bt.iloc[:, :n_cols].reindex(idx).fillna(0).values.flatten()
        y     = actual.iloc[:, :n_cols].reindex(idx).values.flatten()
        _log(f"flattened n_cols={n_cols}, lgb_v {lgb_v.shape}")
        X = np.stack([lgb_v, cb_v, ma_v], axis=-1)

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask].astype(np.float64)
        y = y[mask].astype(np.float64)
        _log(f"post-mask X {X.shape} y {y.shape}  mean_y={y.mean():.2f}")

        # Append intercept column.
        X_with_int = np.column_stack([X, np.ones(len(X))])
        if self.meta_regressor == "ridge_nn":
            # Non-negative Ridge via coordinate descent using only elementwise
            # ops + reductions (no matmul/LAPACK to avoid MKL crash on Win).
            a = float(self.ridge_alpha)
            cols = [X[:, i] - X[:, i].mean() for i in range(X.shape[1])]
            yc = y - y.mean()
            col_sq = [float((c * c).sum() + a) for c in cols]
            # Gram matrix G[i,j] = cols[i].cols[j], via elementwise * + sum
            G = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    G[i, j] = float((cols[i] * cols[j]).sum())
            by = np.array([float((cols[i] * yc).sum()) for i in range(3)])
            w = np.zeros(3)
            for _it in range(500):
                w_prev = w.copy()
                for i in range(3):
                    # residual inner-product = by[i] - sum_j (G[i,j] * w[j]) + G[i,i]*w[i]
                    g_row = G[i]
                    partial = float(by[i] - (g_row[0]*w[0] + g_row[1]*w[1] + g_row[2]*w[2]) + G[i, i]*w[i])
                    w[i] = max(0.0, partial / col_sq[i])
                if float(np.abs(w - w_prev).max()) < 1e-8:
                    break
            intercept = y.mean() - sum(float(X[:, i].mean()) * float(w[i]) for i in range(3))
            coefs = np.concatenate([w, [intercept]])
        elif self.meta_regressor == "ridge":
            # Ridge via pure-Python Cramer's rule on a 3x3 system. Avoids numpy
            # linalg (LAPACK) which crashes after LightGBM loads on Windows
            # due to MKL/OpenMP conflict.
            a = float(self.ridge_alpha)
            Xc = X - X.mean(axis=0, keepdims=True)
            yc = y - y.mean()
            # Build 3x3 normal equations: A = Xc'Xc + a*I, b = Xc'yc.
            # Use elementwise sums to avoid BLAS.
            A = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    A[i, j] = float((Xc[:, i] * Xc[:, j]).sum())
                A[i, i] += a
            b = np.array([float((Xc[:, i] * yc).sum()) for i in range(3)])
            # Cramer's rule for 3x3 (no LAPACK).
            def _det3(M):
                return (M[0,0] * (M[1,1]*M[2,2] - M[1,2]*M[2,1])
                        - M[0,1] * (M[1,0]*M[2,2] - M[1,2]*M[2,0])
                        + M[0,2] * (M[1,0]*M[2,1] - M[1,1]*M[2,0]))
            detA = _det3(A)
            w = np.zeros(3)
            for k in range(3):
                Ak = A.copy()
                Ak[:, k] = b
                w[k] = _det3(Ak) / detA
            intercept = y.mean() - float(X.mean(axis=0) @ w)
            coefs = np.concatenate([w, [intercept]])
        else:  # "linreg": ordinary least squares
            coefs, *_ = np.linalg.lstsq(X_with_int, y, rcond=None)
        self._meta = coefs  # ndarray of [w_lgb, w_cb, w_ma, intercept]
        self._meta_coefs = {
            "lgb": float(coefs[0]), "cb": float(coefs[1]),
            "ma": float(coefs[2]), "intercept": float(coefs[3]),
            "n_samples": int(len(y)),
        }

    def _combine(self, lgb_fc, cb_fc, ma_fc) -> pd.DataFrame:
        """Apply meta-learner to per-model forecasts. All three [n_sku, h] DataFrames."""
        n_cols = min(lgb_fc.shape[1], cb_fc.shape[1], ma_fc.shape[1])
        idx = lgb_fc.index
        combined = np.zeros((len(idx), n_cols), dtype=float)
        w_lgb, w_cb, w_ma, intercept = self._meta
        for h in range(n_cols):
            combined[:, h] = (
                w_lgb * lgb_fc.iloc[:, h].reindex(idx).fillna(0).values
                + w_cb * cb_fc.iloc[:, h].reindex(idx).fillna(0).values
                + w_ma * ma_fc.iloc[:, h].reindex(idx).fillna(0).values
                + intercept
            )
        return pd.DataFrame(combined, index=idx,
                             columns=lgb_fc.columns[:n_cols]).clip(lower=0)

    def _fit(self, sales_hist, in_stock):
        import sys as _sys
        def _log(msg): print(f"    [stacked] {msg}", flush=True)
        _log(f"_fit start: sales_hist shape={sales_hist.shape}")
        # 1) Fit the internal EnsemblePolicy to get the LGB+CB models fitted on full data.
        self._ensemble = EnsemblePolicy(**self._ensemble_cfg)
        self._ensemble._fit(sales_hist, in_stock)
        _log("ensemble._fit done")

        # 2) Backtest: hold out last W weeks, fit a second forecaster on the rest,
        #    produce per-model predictions + MA for the holdout.
        W = self.backtest_window
        train = sales_hist.iloc[:, :-W]
        holdout = sales_hist.iloc[:, -W:]
        bt_is = in_stock.iloc[:, :-W] if in_stock is not None else None
        bt = DemandForecaster(
            master=self._ensemble.master,
            random_state=self._ensemble.random_state,
            censoring_strategy=self._ensemble.censoring_strategy,
        )
        bt.fit(train, bt_is)
        _log("backtest forecaster fit")
        bt_models = bt.predict_models(horizon=W)
        lgb_bt = bt_models["lgbm"]
        cb_bt  = bt_models["catboost"]
        ma_bt  = _seasonal_ma_forecast(train, bt_is, horizon=W)
        _log(f"backtest preds: lgb {lgb_bt.shape}, cb {cb_bt.shape}, ma {ma_bt.shape}")
        # Trigger any pending lazy column computations (pandas can segfault with LGB
        # predictions + tricky reindex on Windows). Materialize to dense arrays now.
        lgb_bt = lgb_bt.astype(np.float64).copy()
        cb_bt  = cb_bt.astype(np.float64).copy()
        ma_bt  = ma_bt.astype(np.float64).copy()
        _log("materialized backtest frames")

        # Align column labels to match holdout columns
        holdout.columns = holdout.columns  # already aligned
        lgb_bt.columns = holdout.columns[:lgb_bt.shape[1]]
        cb_bt.columns = holdout.columns[:cb_bt.shape[1]]
        ma_bt.columns = holdout.columns[:ma_bt.shape[1]]

        # 3) Fit meta-learner
        _log("calling _fit_meta_learner...")
        self._fit_meta_learner(lgb_bt, cb_bt, ma_bt, holdout)
        _log(f"meta coefs: {self._meta_coefs}")

        # 4) Generate the FULL forecast matrix (rounds 0..5 will index it).
        horizon = 6 + LEAD_TIME + 3
        full_models = self._ensemble._forecaster.predict_models(horizon=horizon)
        lgb_fc = full_models["lgbm"]
        cb_fc  = full_models["catboost"]
        ma_fc  = _seasonal_ma_forecast(sales_hist, in_stock, horizon=horizon)
        ma_fc.columns = lgb_fc.columns[:ma_fc.shape[1]]
        self._ensemble._all_ml_forecasts = self._combine(lgb_fc, cb_fc, ma_fc)
        self._ensemble._all_ma_forecasts = ma_fc  # for diagnostics
        # Override ensemble's blend weight: we already produced the final blend.
        self._ensemble.w_ma = 0.0

        # 5) Per-SKU sigma for safety stock (same as CostAwarePolicy).
        resid = (
            holdout.iloc[:, :self.rmse_horizons].values
            - self._ensemble._all_ml_forecasts.iloc[:, :self.rmse_horizons].reindex(sales_hist.index).values
        )
        rmse = np.sqrt(np.nanmean(resid ** 2, axis=1))
        self._rmse = pd.Series(rmse, index=sales_hist.index).fillna(self.safety_floor)
        self._rmse = self._rmse.clip(lower=self.safety_floor)

    def _get_d12(self, round_idx: int) -> tuple[pd.Series, pd.Series]:
        fc = self._ensemble._all_ml_forecasts
        return fc.iloc[:, round_idx], fc.iloc[:, round_idx + 1]

    def __call__(self, sim, round_idx, sales_hist):
        if round_idx == 0 or self._ensemble is None:
            self._fit(sales_hist, sim.in_stock)
            if self._meta_coefs is not None:
                c = self._meta_coefs
                print(f"    [stacked] meta coefs: lgb={c['lgb']:.3f} cb={c['cb']:.3f} "
                      f"ma={c['ma']:.3f} int={c['intercept']:+.3f} n={c['n_samples']}",
                      flush=True)

        d1, d2 = self._get_d12(round_idx)
        I0 = sim.end_inventory.astype(float)
        T1 = sim.in_transit_w1.astype(float)
        E1 = (I0 + T1 - d1).clip(lower=0)

        sigma = self._rmse.reindex(d2.index).fillna(self.safety_floor)
        target = self.multiplier * (d2 + self._z_alpha * sigma)
        target = target.clip(lower=0)

        order = (target - E1).clip(lower=0).round(0).astype(int)
        self.last_forecast = d2.copy()
        self.last_quantile_forecast = target.copy()
        return order


class OrderEnsemble:
    """Average orders from N policies, weighted.

    Each inner policy must return a Series of order quantities indexed by
    (Store, Product). Output = round(sum(w_i * order_i)) clipped to ≥ 0.

    `last_forecast` is exposed as the weighted-mean of inner policies'
    `last_forecast` (if all expose it) so the CV harness can still score
    forecast accuracy.
    """

    def __init__(self, policies: list, weights: "list[float] | None" = None):
        assert len(policies) > 0
        self.policies = policies
        n = len(policies)
        if weights is None:
            self.weights = [1.0 / n] * n
        else:
            assert len(weights) == n
            s = float(sum(weights))
            assert s > 0
            self.weights = [w / s for w in weights]
        self.last_forecast: pd.Series | None = None
        self.last_quantile_forecast: pd.Series | None = None

    def __call__(self, sim, round_idx, sales_hist):
        orders: list[pd.Series] = []
        forecasts: list[pd.Series] = []
        quants: list[pd.Series] = []
        for p in self.policies:
            o = p(sim, round_idx, sales_hist)
            orders.append(o.astype(float))
            if getattr(p, "last_forecast", None) is not None:
                forecasts.append(p.last_forecast.astype(float))
            if getattr(p, "last_quantile_forecast", None) is not None:
                quants.append(p.last_quantile_forecast.astype(float))

        idx = orders[0].index
        avg = pd.Series(0.0, index=idx)
        for w, o in zip(self.weights, orders):
            avg = avg.add(w * o.reindex(idx).fillna(0), fill_value=0)
        out = avg.round(0).clip(lower=0).astype(int)

        if forecasts:
            fc = pd.Series(0.0, index=idx)
            wsum = 0.0
            for w, f in zip(self.weights[: len(forecasts)], forecasts):
                fc = fc.add(w * f.reindex(idx).fillna(0), fill_value=0)
                wsum += w
            self.last_forecast = (fc / wsum) if wsum > 0 else fc
        if quants:
            qf = pd.Series(0.0, index=idx)
            wsum = 0.0
            for w, q in zip(self.weights[: len(quants)], quants):
                qf = qf.add(w * q.reindex(idx).fillna(0), fill_value=0)
                wsum += w
            self.last_quantile_forecast = (qf / wsum) if wsum > 0 else qf

        return out
