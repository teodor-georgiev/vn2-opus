"""Cumulative-horizon quantile forecaster (Matias Alvo approach).

Instead of predicting per-period demand quantiles and summing them (statistically
incorrect — sum of quantiles != quantile of sum), train LGBM/CatBoost directly
on the CUMULATIVE sum over a future window as the target, with quantile loss.

For VN2 with LEAD_TIME=2 and coverage H, the target at time t is:
    y_t = sum(sales[t+LEAD+1 .. t+LEAD+H])
i.e., demand in the window our order-placed-now will cover (arrives at t+LEAD+1).

The model output is directly Q_alpha( sum of H-week future demand | features_t ).
Used as order-up-to level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb

from forecaster import prepare_static_features


DEFAULT_LAGS = (1, 2, 3, 4, 8, 13, 26, 52)
DEFAULT_ROLLING_WINDOWS = (4, 8, 13)
DEFAULT_SEASONAL_LAG = 52


def _apply_censoring(sales_wide, in_stock_wide, strategy: str):
    """Return censoring-adjusted sales DataFrame (wide)."""
    if strategy == "zero" or in_stock_wide is None:
        return sales_wide.copy()
    s = sales_wide.copy()
    matching = s.columns[s.columns.isin(in_stock_wide.columns)]
    mask = pd.DataFrame(False, index=s.index, columns=s.columns)
    for c in matching:
        mask.loc[:, c] = ~in_stock_wide[c]

    if strategy == "interpolate":
        for c in matching:
            s.loc[mask[c], c] = np.nan
        s = s.interpolate(axis=1, limit_direction="both").fillna(0)
    elif strategy == "mean_impute":
        uncens = s.mask(mask)
        per_mean = uncens.mean(axis=1)
        for c in matching:
            s.loc[mask[c], c] = per_mean[mask[c]]
        s = s.fillna(0)
    else:
        raise ValueError(f"Unknown censoring: {strategy}")
    return s


def _build_features(
    sales_clean: pd.DataFrame,
    times: list[int],  # indices into sales_clean.columns
    lags=DEFAULT_LAGS,
    rolling_windows=DEFAULT_ROLLING_WINDOWS,
    seasonal_lag: int = DEFAULT_SEASONAL_LAG,
    static: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a feature matrix: one row per (SKU, t).

    Columns: lag_1..lag_L, roll_mean_W1..roll_mean_WK, seasonal_lag, wk_of_year, static*.
    sales_clean is expected to be non-censored (cleaned via _apply_censoring).
    """
    cols = list(sales_clean.columns)
    n_skus = len(sales_clean.index)
    rows = []
    for t in times:
        feat = pd.DataFrame(index=sales_clean.index)
        # Lags.
        for L in lags:
            if t - L >= 0:
                feat[f"lag_{L}"] = sales_clean.iloc[:, t - L].values
            else:
                feat[f"lag_{L}"] = 0.0
        # Rolling means of recent windows (ending at t-1).
        for W in rolling_windows:
            start = max(0, t - W)
            if start < t:
                feat[f"rm_{W}"] = sales_clean.iloc[:, start:t].mean(axis=1).values
            else:
                feat[f"rm_{W}"] = 0.0
        # Seasonal anchor (mean of same week in prior year(s)).
        if t - seasonal_lag >= 0:
            feat[f"sl_{seasonal_lag}"] = sales_clean.iloc[:, t - seasonal_lag].values
        else:
            feat[f"sl_{seasonal_lag}"] = 0.0
        # Week of year (cyclic-ish — integer is fine for tree models).
        feat["wk_of_year"] = pd.Timestamp(cols[t]).isocalendar().week
        feat["month"] = pd.Timestamp(cols[t]).month
        feat["t"] = t
        feat["sku_id"] = range(n_skus)  # row index
        rows.append(feat)
    X = pd.concat(rows, ignore_index=False)
    # Merge static features if provided.
    if static is not None:
        # static has unique_id = Store_Product; we have (Store, Product) as index.
        pass  # skip merge for simplicity; tree can learn per-SKU via sku_id
    return X


class CumulativeQuantileForecaster:
    """Direct quantile forecaster of H-week cumulative demand.

    At inference time, `predict(sales_hist_wide, t_now)` returns one quantile
    value per SKU: estimated quantile of demand sum over coverage future weeks.
    """

    def __init__(
        self,
        alpha: float = 0.833,
        coverage: int = 2,
        lead_offset: int = 3,  # first covered week = t + lead_offset
        censoring_strategy: str = "mean_impute",
        random_state: int = 42,
        lgb_params: dict | None = None,
        cb_params: dict | None = None,
        ensemble: bool = True,
    ):
        self.alpha = alpha
        self.coverage = coverage
        self.lead_offset = lead_offset
        self.censoring_strategy = censoring_strategy
        self.random_state = random_state
        self.lgb_params = lgb_params or {}
        self.cb_params = cb_params or {}
        self.ensemble = ensemble
        self.lgb_model = None
        self.cb_model = None
        self.feature_cols: list[str] | None = None

    def _prep_training(self, sales_wide, in_stock_wide):
        """Build (X, y) from all valid (SKU, t) training points."""
        clean = _apply_censoring(sales_wide, in_stock_wide, self.censoring_strategy)
        n_cols = clean.shape[1]
        earliest_t = max(DEFAULT_LAGS + (DEFAULT_SEASONAL_LAG,))
        latest_t = n_cols - self.lead_offset - self.coverage  # ensure target is defined
        times = list(range(earliest_t, latest_t))
        X = _build_features(clean, times)
        # Target per row: cumulative sum over future window.
        target_rows = []
        idx = clean.index
        for t in times:
            slc = clean.iloc[:, t + self.lead_offset : t + self.lead_offset + self.coverage].sum(axis=1)
            target_rows.append(slc.values)
        y = np.concatenate(target_rows)
        return X.reset_index(drop=True), y

    def fit(self, sales_wide, in_stock_wide=None):
        X, y = self._prep_training(sales_wide, in_stock_wide)
        self.feature_cols = [c for c in X.columns if c not in ()]
        # Drop NaN rows (shouldn't exist after _apply_censoring + fillna(0)).
        mask = np.isfinite(y)
        X = X.loc[mask]
        y = y[mask]

        lgb_defaults = dict(
            objective="quantile", alpha=self.alpha,
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            verbosity=-1, n_jobs=-1, random_state=self.random_state,
        )
        lgb_defaults.update(self.lgb_params)
        self.lgb_model = lgb.LGBMRegressor(**lgb_defaults)
        self.lgb_model.fit(X[self.feature_cols], y)

        if self.ensemble:
            cb_defaults = dict(
                loss_function=f"Quantile:alpha={self.alpha}",
                iterations=300, learning_rate=0.05, depth=6,
                verbose=0, thread_count=-1, random_seed=self.random_state,
            )
            cb_defaults.update(self.cb_params)
            self.cb_model = cb.CatBoostRegressor(**cb_defaults)
            self.cb_model.fit(X[self.feature_cols], y)
        return self

    def predict(self, sales_wide: pd.DataFrame, in_stock_wide: pd.DataFrame | None,
                t_now: int) -> pd.Series:
        """Predict quantile of cumulative H-week demand for each SKU at time t_now."""
        clean = _apply_censoring(sales_wide, in_stock_wide, self.censoring_strategy)
        X = _build_features(clean, [t_now])
        preds = self.lgb_model.predict(X[self.feature_cols])
        if self.ensemble and self.cb_model is not None:
            cb_preds = self.cb_model.predict(X[self.feature_cols])
            preds = 0.5 * (preds + cb_preds)
        return pd.Series(np.clip(preds, 0, None), index=sales_wide.index)
