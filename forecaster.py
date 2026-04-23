import pandas as pd
import numpy as np
from pathlib import Path
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd, SeasonalRollingMean
try:
    from mlforecast.lag_transforms import ExponentiallyWeightedMean
    _HAS_EWM = True
except Exception:
    _HAS_EWM = False
try:
    from mlforecast.target_transforms import LocalStandardScaler, BaseTargetTransform
    _HAS_LOCAL_SCALER = True
except Exception:
    _HAS_LOCAL_SCALER = False
    BaseTargetTransform = object
import lightgbm as lgb
import catboost as cb
from numba import njit


@njit(cache=True)
def _time_since_last_nonzero(x):
    """Weeks since last non-zero value, per position. NaN treated as 0."""
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    counter = 0.0
    for i in range(n):
        v = x[i]
        if not np.isnan(v) and v > 0.0:
            counter = 0.0
        else:
            counter += 1.0
        out[i] = counter
    return out


@njit(cache=True)
def _rolling_nonzero_rate_12(x):
    """Fraction of non-zero observations over trailing 12-week window."""
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = i - 11 if i >= 11 else 0
        cnt = 0
        nz = 0
        for j in range(start, i + 1):
            v = x[j]
            if not np.isnan(v):
                cnt += 1
                if v > 0.0:
                    nz += 1
        out[i] = (nz / cnt) if cnt > 0 else np.nan
    return out


@njit(cache=True)
def _rolling_nonzero_rate_26(x):
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = i - 25 if i >= 25 else 0
        cnt = 0
        nz = 0
        for j in range(start, i + 1):
            v = x[j]
            if not np.isnan(v):
                cnt += 1
                if v > 0.0:
                    nz += 1
        out[i] = (nz / cnt) if cnt > 0 else np.nan
    return out


@njit(cache=True)
def _rolling_mad_zscore_13(x):
    """Robust z-score of current value vs 13-week rolling median and MAD."""
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = i - 12 if i >= 12 else 0
        cnt = 0
        for j in range(start, i + 1):
            if not np.isnan(x[j]):
                cnt += 1
        if cnt < 3:
            out[i] = np.nan
            continue
        buf = np.empty(cnt, dtype=np.float64)
        k = 0
        for j in range(start, i + 1):
            if not np.isnan(x[j]):
                buf[k] = x[j]
                k += 1
        sorted_buf = np.sort(buf)
        mid = cnt // 2
        median = sorted_buf[mid] if cnt % 2 else 0.5 * (sorted_buf[mid - 1] + sorted_buf[mid])
        dev = np.empty(cnt, dtype=np.float64)
        for j in range(cnt):
            dev[j] = abs(buf[j] - median)
        sorted_dev = np.sort(dev)
        mad = sorted_dev[mid] if cnt % 2 else 0.5 * (sorted_dev[mid - 1] + sorted_dev[mid])
        if mad < 0.01:
            mad = 0.01
        out[i] = (x[i] - median) / mad if not np.isnan(x[i]) else np.nan
    return out


class CatBoostWithAutoCats(cb.CatBoostRegressor):
    """CatBoostRegressor that passes pandas Categorical columns as cat_features
    at fit time. Avoids the sklearn-clone issue that arises when cat_features
    is set via the constructor.
    """

    def fit(self, X, y=None, **kwargs):
        if kwargs.get("cat_features") is None and hasattr(X, "columns"):
            cat_cols = [
                c for c in X.columns
                if pd.api.types.is_categorical_dtype(X[c])
            ]
            if cat_cols:
                kwargs["cat_features"] = cat_cols
        return super().fit(X, y, **kwargs)


class AnnualDivisorScaler(BaseTargetTransform):
    """Paper's per-series scaling: divide target by max(W * mean(last W non-null y), 1).

    scale_factor_i = max(window * mean_non_missing(y_i, t-window:t), min_factor)

    Division-only (non-shifting; zeros stay zeros). One scale factor per series.
    If the tail has fewer than `min_obs` observations, falls back to expanding mean.
    """

    def __init__(self, window: int = 53, min_obs: int = 45, min_factor: float = 1.0):
        self.window = window
        self.min_obs = min_obs
        self.min_factor = min_factor
        self.scale_: dict = {}

    def fit_transform(self, df):
        self.scale_ = {}
        out = df.copy()
        for uid, grp in df.groupby(self.id_col, sort=False):
            y = grp.sort_values(self.time_col)[self.target_col].to_numpy(dtype=float)
            tail = y[-self.window:]
            tail_nz = tail[~np.isnan(tail)]
            if len(tail_nz) >= self.min_obs:
                mean = float(tail_nz.mean())
            else:
                y_valid = y[~np.isnan(y)]
                mean = float(y_valid.mean()) if len(y_valid) > 0 else 0.0
            self.scale_[uid] = max(self.window * mean, self.min_factor)
        out[self.target_col] = out[self.target_col] / out[self.id_col].map(self.scale_)
        return out

    def inverse_transform(self, df):
        out = df.copy()
        scales = out[self.id_col].map(self.scale_)
        for col in out.columns:
            if col in (self.id_col, self.time_col):
                continue
            out[col] = out[col] * scales
        return out

INDEX = ["Store", "Product"]
DATA_DIR = Path("Data")


def wide_to_long(sales_wide, in_stock_wide=None, censoring_strategy: str = "interpolate"):
    """Convert wide sales DataFrame to long format for mlforecast.

    censoring_strategy ∈ {"interpolate", "mean_impute", "seasonal_impute", "zero"}:
      interpolate      — set out-of-stock sales to NaN, linearly interpolate across time.
      mean_impute      — replace out-of-stock with SKU's non-censored mean.
      seasonal_impute  — replace with same-week-of-year average (non-censored).
      zero             — leave sales as observed (stockout weeks contribute raw 0).
    """
    sales_clean = sales_wide.copy()
    if in_stock_wide is not None and censoring_strategy != "zero":
        matching_cols = sales_clean.columns[sales_clean.columns.isin(in_stock_wide.columns)]
        mask_cens = pd.DataFrame(False, index=sales_clean.index, columns=sales_clean.columns)
        for col in matching_cols:
            mask_cens.loc[:, col] = ~in_stock_wide[col]
        if censoring_strategy == "interpolate":
            for col in matching_cols:
                sales_clean.loc[mask_cens[col], col] = np.nan
            sales_clean = sales_clean.interpolate(axis=1, limit_direction="both")
            sales_clean = sales_clean.fillna(0)
        elif censoring_strategy == "mean_impute":
            # Per-SKU mean of non-censored weeks.
            uncens = sales_clean.mask(mask_cens)
            per_sku_mean = uncens.mean(axis=1)
            for col in matching_cols:
                sales_clean.loc[mask_cens[col], col] = per_sku_mean[mask_cens[col]]
            sales_clean = sales_clean.fillna(0)
        elif censoring_strategy == "seasonal_impute":
            # For each (SKU, censored week), use avg of same ISO week number in other years.
            week_of_year = pd.Index(pd.to_datetime(sales_clean.columns)).isocalendar().week
            uncens = sales_clean.mask(mask_cens)
            for wk in sorted(set(week_of_year)):
                cols_this_wk = [c for c, w in zip(sales_clean.columns, week_of_year) if w == wk]
                if not cols_this_wk:
                    continue
                week_mean = uncens[cols_this_wk].mean(axis=1)
                for col in cols_this_wk:
                    if col not in matching_cols:
                        continue
                    sales_clean.loc[mask_cens[col], col] = week_mean[mask_cens[col]]
            sales_clean = sales_clean.fillna(0)
        else:
            raise ValueError(f"Unknown censoring_strategy: {censoring_strategy}")

    long = sales_clean.stack(dropna=False).reset_index()
    long.columns = ["Store", "Product", "ds", "y"]
    long["unique_id"] = long["Store"].astype(str) + "_" + long["Product"].astype(str)
    long["ds"] = pd.to_datetime(long["ds"])
    long = long[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return long


STATIC_CATEGORICAL_COLS = [
    "ProductGroup", "Division", "Department", "DepartmentGroup",
    "StoreFormat", "Format",
]


def compute_demand_cluster(sales_wide, in_stock_wide=None, k: int = 8,
                            random_state: int = 42):
    """Quantile-based demand cluster per (Store, Product).

    We bucket SKUs into k groups using two orthogonal axes:
      - `log_mean` demand level → row index in k-bucket grid
      - `nonzero_rate` (intermittency) → column index
    then encode as a single integer label. Simple, fast, no sklearn
    (avoids a Windows DLL/OpenMP crash seen with KMeans in this env).

    Returns pd.Series indexed like sales_wide.index.
    """
    y = sales_wide.copy()
    if in_stock_wide is not None:
        mask = ~in_stock_wide.reindex_like(y).fillna(True)
        y = y.mask(mask)
    nz = (y > 0).astype(float)
    log_mean = np.log1p(y.mean(axis=1).fillna(0))
    nonzero_rate = nz.mean(axis=1).fillna(0)

    # Split k roughly into a √k × √k grid (min 2 each axis).
    import math
    rows = max(2, int(round(math.sqrt(k))))
    cols = max(2, int(math.ceil(k / rows)))

    def _qcut_safe(s, q):
        # pd.qcut can fail with duplicate quantile edges. Fall back to rank-based.
        try:
            return pd.qcut(s, q=q, labels=False, duplicates="drop")
        except Exception:
            return pd.cut(s.rank(method="average"), bins=q, labels=False)

    row_label = _qcut_safe(log_mean, rows).fillna(0).astype(int)
    col_label = _qcut_safe(nonzero_rate, cols).fillna(0).astype(int)
    labels = row_label * cols + col_label
    return pd.Series(labels.values, index=sales_wide.index, name="demand_cluster")


def prepare_static_features(master, categorical: bool = False):
    """Convert master data into static features keyed by unique_id.

    If categorical=True, cast to pandas Categorical so LightGBM autodetects
    them as categorical and CatBoost (via cat_features=) treats them properly.
    """
    static = master.reset_index()
    static["unique_id"] = static["Store"].astype(str) + "_" + static["Product"].astype(str)
    static = static[["unique_id"] + STATIC_CATEGORICAL_COLS]
    for col in STATIC_CATEGORICAL_COLS:
        static[col] = static[col].astype(int)
        if categorical:
            static[col] = static[col].astype("category")
    return static


class DemandForecaster:
    """Forecaster using Nixtla mlforecast with LightGBM and CatBoost."""

    def __init__(self, master=None, random_state: int = 42, lgb_params: dict | None = None,
                 cb_params: dict | None = None, censoring_strategy: str = "interpolate",
                 max_horizon: int | None = None,
                 extended_features: bool = False,
                 per_series_scaling: bool = False,
                 recency_decay: float | None = None,
                 recency_block_weeks: int = 53,
                 categorical_features: bool = False,
                 intermittency_features: bool = False,
                 demand_cluster_k: int | None = None,
                 hierarchical_features: bool = False,
                 prediction_intervals_n_windows: int | None = None,
                 prediction_intervals_h: int = 3):
        self.master = master
        self.categorical_features = categorical_features
        # Intermittency features (numba @njit lag_transforms): time_since_spike,
        # rolling nonzero rates, MAD-based robust z-score.
        self.intermittency_features = intermittency_features
        # K-means demand-pattern cluster to inject as static feature.
        self.demand_cluster_k = demand_cluster_k
        # Hierarchical / M5-style group-level rolling features
        # (RollingMean with groupby=[Department|Store|...], plus global_=True).
        self.hierarchical_features = hierarchical_features
        self.static_features = (
            prepare_static_features(master, categorical=categorical_features)
            if master is not None else None
        )
        self.model = None
        self.random_state = random_state
        self.lgb_params = lgb_params or {}
        # mlforecast conformal-prediction-intervals settings (optional).
        # If n_windows is set, fit() will pass PredictionIntervals(n_windows, h)
        # so that predict_quantile() can return per-SKU per-horizon quantiles.
        self.prediction_intervals_n_windows = prediction_intervals_n_windows
        self.prediction_intervals_h = prediction_intervals_h
        self.cb_params = cb_params or {}
        self.censoring_strategy = censoring_strategy
        # Direct multi-horizon: when set, MLForecast trains `max_horizon`
        # separate estimators, one per horizon, using features at time t only.
        # None == recursive (default): one estimator, forecasts fed back.
        self.max_horizon = max_horizon
        # Extended features: seasonal lags 51/53, windows 3/5, Fourier harmonics.
        self.extended_features = extended_features
        # Per-series target scaling (LocalStandardScaler). Normalizes scale
        # heterogeneity so the global model learns patterns, not levels.
        self.per_series_scaling = per_series_scaling
        # Recency weighting: None disables. Otherwise each row's sample_weight
        # = recency_decay ** (age_in_blocks). Paper uses 0.5 with 53-week blocks.
        self.recency_decay = recency_decay
        self.recency_block_weeks = recency_block_weeks

    def build_model(self):
        lgb_defaults = dict(
            n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8, verbosity=-1, n_jobs=-1,
            random_state=self.random_state,
        )
        lgb_defaults.update(self.lgb_params)
        lgb_model = lgb.LGBMRegressor(**lgb_defaults)

        cb_defaults = dict(
            iterations=200, learning_rate=0.05, depth=6, verbose=0, thread_count=-1,
            random_seed=self.random_state,
        )
        cb_defaults.update(self.cb_params)
        # When categorical features are in play, use the wrapper that passes
        # cat_features at fit time (avoids CatBoost's clone-incompatibility
        # of constructor cat_features). LightGBM auto-detects via dtype=category.
        cb_class = CatBoostWithAutoCats if self.categorical_features else cb.CatBoostRegressor
        cb_model = cb_class(**cb_defaults)

        if self.extended_features:
            # Paper's high-importance features (Table 1 of VN2 paper):
            #   week_of_year (top driver), fourier_sin_2/3, seasonal lags 51/52/53.
            # Plus EWM (spans 5, 10) and extra rolling std windows (3, 5, 8).
            import numpy as np
            lags = [1, 2, 3, 4, 8, 13, 26, 51, 52, 53]
            transforms_at_1 = [
                RollingMean(window_size=3),
                RollingMean(window_size=5),
                RollingMean(window_size=8),
                RollingMean(window_size=13),
                RollingStd(window_size=3),
                RollingStd(window_size=5),
                RollingStd(window_size=8),
                RollingStd(window_size=13),
            ]
            if _HAS_EWM:
                transforms_at_1.append(ExponentiallyWeightedMean(alpha=2 / (5 + 1)))   # span 5
                transforms_at_1.append(ExponentiallyWeightedMean(alpha=2 / (10 + 1)))  # span 10
            lag_transforms = {
                1: transforms_at_1,
                52: [RollingMean(window_size=3)],
            }

            def _mk_fourier(order: int, kind: str):
                # Callable takes datetime index, returns series of same length.
                def f(dates):
                    woy = dates.isocalendar().week.astype(int).values
                    arg = 2 * np.pi * order * woy / 52.0
                    return np.sin(arg) if kind == "sin" else np.cos(arg)
                f.__name__ = f"fourier_{kind}_{order}"
                return f

            date_features = ["week", "month"] + [
                _mk_fourier(k, kind) for k in (1, 2, 3) for kind in ("sin", "cos")
            ]
        else:
            lags = [1, 2, 3, 4, 8, 13, 26, 52]
            base_at_1 = [
                RollingMean(window_size=4),
                RollingMean(window_size=8),
                RollingMean(window_size=13),
                RollingStd(window_size=4),
                RollingStd(window_size=13),
            ]
            lag_transforms = {1: base_at_1, 52: [RollingMean(window_size=3)]}
            date_features = ["week", "month"]

        # Numba-JIT intermittency features (paper Table 1 high-importance).
        if self.intermittency_features:
            lag_transforms.setdefault(1, []).extend([
                _time_since_last_nonzero,
                _rolling_nonzero_rate_12,
                _rolling_nonzero_rate_26,
                _rolling_mad_zscore_13,
            ])

        # M5-style hierarchical rolling features via mlforecast's native
        # RollingMean(..., groupby=[...]) / global_=True. Each group-level
        # transform uses lag=1 as the base, so features represent "mean of
        # the group's past weeks" per current (SKU, t) row.
        if self.hierarchical_features:
            lag_transforms.setdefault(1, []).extend([
                RollingMean(window_size=4, groupby=["Department"]),
                RollingMean(window_size=13, groupby=["Department"]),
                RollingMean(window_size=4, groupby=["Division"]),
                RollingMean(window_size=4, groupby=["ProductGroup"]),
                RollingMean(window_size=4, groupby=["StoreFormat"]),
                RollingMean(window_size=4, global_=True),
                RollingMean(window_size=13, global_=True),
            ])

        # Allow subclasses / callers to override the set of base learners.
        default_models = {"lgbm": lgb_model, "catboost": cb_model}
        models = getattr(self, "_override_models", None) or default_models
        mlf_kwargs = dict(
            models=models,
            freq="W-MON",
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
            num_threads=4,
        )
        if self.per_series_scaling and _HAS_LOCAL_SCALER:
            if self.per_series_scaling == "annual":
                # Paper's exact recipe (53-wk rolling mean × 53, clipped ≥1).
                mlf_kwargs["target_transforms"] = [AnnualDivisorScaler()]
            else:
                # Default: LocalStandardScaler (per-series mean-subtract and /σ).
                mlf_kwargs["target_transforms"] = [LocalStandardScaler()]
        self.model = MLForecast(**mlf_kwargs)
        return self.model

    def fit(self, sales_wide, in_stock_wide=None):
        """Fit the model on historical sales data."""
        long_df = wide_to_long(sales_wide, in_stock_wide,
                               censoring_strategy=self.censoring_strategy)
        long_df = long_df.dropna(subset=["y"])

        if self.model is None:
            self.build_model()

        # Optionally compute demand-pattern clusters from the training window
        # and attach as an additional static feature. No future info leaks.
        static_feats = self.static_features
        if self.demand_cluster_k is not None:
            cluster = compute_demand_cluster(
                sales_wide, in_stock_wide=in_stock_wide,
                k=int(self.demand_cluster_k),
                random_state=self.random_state,
            ).reset_index()
            cluster["unique_id"] = cluster["Store"].astype(str) + "_" + cluster["Product"].astype(str)
            cluster = cluster[["unique_id", "demand_cluster"]]
            if self.categorical_features:
                cluster["demand_cluster"] = cluster["demand_cluster"].astype("category")
            if static_feats is None:
                static_feats = cluster
            else:
                static_feats = static_feats.merge(cluster, on="unique_id", how="left")

        static_cols = None
        if static_feats is not None:
            long_df = long_df.merge(static_feats, on="unique_id", how="left")
            static_cols = [c for c in static_feats.columns if c != "unique_id"]

        fit_kwargs = {"static_features": static_cols}
        # Direct multi-horizon strategy: one estimator per horizon (no recursion).
        if self.max_horizon is not None:
            fit_kwargs["max_horizon"] = int(self.max_horizon)

        # Recency-based observation weighting. Per-row weight = decay ** age_blocks,
        # where age_blocks = weeks-from-series-end // block_weeks (paper: 53-wk blocks).
        # Enable mlforecast conformal prediction intervals if requested.
        if self.prediction_intervals_n_windows is not None:
            from mlforecast.utils import PredictionIntervals
            fit_kwargs["prediction_intervals"] = PredictionIntervals(
                n_windows=int(self.prediction_intervals_n_windows),
                h=int(self.prediction_intervals_h),
                method="conformal_distribution",
            )

        if self.recency_decay is not None:
            import numpy as np
            last_ds = long_df.groupby("unique_id")["ds"].transform("max")
            age_days = (last_ds - long_df["ds"]).dt.days
            age_blocks = (age_days // 7 // int(self.recency_block_weeks)).astype(int)
            long_df = long_df.copy()
            long_df["sample_weight"] = np.power(float(self.recency_decay), age_blocks)
            fit_kwargs["weight_col"] = "sample_weight"

        self.model.fit(long_df, **fit_kwargs)
        return self

    def predict(self, horizon=3):
        """Generate point forecasts for the next `horizon` weeks.

        Returns wide DataFrame indexed by (Store, Product) with date columns.
        """
        preds = self.model.predict(horizon)
        preds["ensemble"] = (preds["lgbm"] + preds["catboost"]) / 2
        preds[["Store", "Product"]] = preds["unique_id"].str.split("_", expand=True).astype(int)
        wide = preds.pivot_table(index=["Store", "Product"], columns="ds", values="ensemble")
        wide = wide.clip(lower=0)
        return wide

    def predict_quantile_conformal(self, horizon: int = 3, alpha: float = 0.65,
                                    model_name: str = "ensemble") -> pd.DataFrame:
        """Return alpha-quantile predictions per (Store, Product) per horizon.

        Requires the model was fit with prediction_intervals enabled.
        Uses mlforecast's built-in conformal prediction via `level` parameter.

        alpha=0.65 → 65th percentile (L=30 in mlforecast level convention).
        alpha=0.50 → median (just the point forecast).
        alpha < 0.5 → returns the lower bound at corresponding level.
        """
        if abs(alpha - 0.5) < 1e-6:
            return self.predict(horizon=horizon)
        L = abs(alpha - 0.5) * 200  # alpha=0.65 → 30 ; alpha=0.833 → 66.6
        preds = self.model.predict(horizon, level=[L])
        # Output cols include e.g. lgbm-lo-30, lgbm-hi-30, catboost-lo-30, catboost-hi-30
        bound = "hi" if alpha > 0.5 else "lo"
        if model_name == "ensemble":
            lgb_col = f"lgbm-{bound}-{int(round(L))}"
            cb_col = f"catboost-{bound}-{int(round(L))}"
            # Fall back to .0 suffix if int rounding mismatches mlforecast format
            for c in (lgb_col, cb_col):
                if c not in preds.columns:
                    # try with float-like formatting
                    matches = [col for col in preds.columns
                               if col.startswith(f"{c.rsplit('-', 2)[0]}-{bound}-")
                               and abs(float(col.rsplit('-', 1)[1]) - L) < 0.5]
                    if matches:
                        if "lgbm" in c: lgb_col = matches[0]
                        else: cb_col = matches[0]
            preds["q"] = (preds[lgb_col] + preds[cb_col]) / 2
        else:
            col = f"{model_name}-{bound}-{int(round(L))}"
            preds["q"] = preds[col]
        preds[["Store", "Product"]] = preds["unique_id"].str.split("_", expand=True).astype(int)
        wide = preds.pivot_table(index=["Store", "Product"], columns="ds", values="q")
        wide = wide.clip(lower=0)
        return wide

    def predict_models(self, horizon=3):
        """Return raw per-model predictions as wide DataFrames."""
        preds = self.model.predict(horizon)
        preds[["Store", "Product"]] = preds["unique_id"].str.split("_", expand=True).astype(int)
        # Build one wide DataFrame per model column present in preds.
        model_names = [c for c in preds.columns
                       if c not in {"unique_id", "ds", "Store", "Product"}]
        result = {}
        for name in model_names:
            w = preds.pivot_table(index=["Store", "Product"], columns="ds", values=name)
            result[name] = w.clip(lower=0)
        return result


class QuantileDiverseDemandForecaster(DemandForecaster):
    """5-model ensemble where EVERY model is trained with pinball (quantile) loss at a target alpha.

    Output is the conditional alpha-quantile of demand directly — no Normal-σ
    approximation needed in the ordering policy.

    Models (all pinball loss at target_quantile):
      - lgbm_q_default     (num_leaves=31)
      - lgbm_q_shallow     (num_leaves=7, max_depth=3)
      - lgbm_q_deep        (num_leaves=127, max_depth=10)
      - catboost_q_default (depth=6, Quantile loss)
      - catboost_q_deep    (depth=10, Quantile loss)

    Use with `QuantileCostAwarePolicy` which uses the output directly as Q_alpha(d2).
    """

    N_MODELS = ["lgbm_q_default", "lgbm_q_shallow", "lgbm_q_deep",
                "catboost_q_default", "catboost_q_deep"]

    def __init__(self, target_quantile: float = 0.833, **kwargs):
        super().__init__(**kwargs)
        self.target_quantile = float(target_quantile)

    def build_model(self):
        rs = self.random_state
        q = self.target_quantile
        lgb_base = dict(
            n_estimators=200, learning_rate=0.05, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8, verbosity=-1,
            n_jobs=-1, random_state=rs,
            objective="quantile", alpha=q,
        )
        lgb_q_default = lgb.LGBMRegressor(**lgb_base, num_leaves=31)
        lgb_q_shallow = lgb.LGBMRegressor(**{**lgb_base,
                                              "num_leaves": 7, "max_depth": 3,
                                              "n_estimators": 300, "learning_rate": 0.04,
                                              "min_child_samples": 20})
        lgb_q_deep = lgb.LGBMRegressor(**{**lgb_base,
                                           "num_leaves": 127, "max_depth": 10,
                                           "n_estimators": 150, "learning_rate": 0.04})

        cb_base = dict(iterations=200, learning_rate=0.05, verbose=0,
                        thread_count=-1, random_seed=rs,
                        loss_function=f"Quantile:alpha={q}")
        cb_class = CatBoostWithAutoCats if self.categorical_features else cb.CatBoostRegressor
        cb_q_default = cb_class(**cb_base, depth=6)
        cb_q_deep = cb_class(**{**cb_base, "iterations": 150, "learning_rate": 0.04, "depth": 10})

        self._override_models = {
            "lgbm_q_default":     lgb_q_default,
            "lgbm_q_shallow":     lgb_q_shallow,
            "lgbm_q_deep":        lgb_q_deep,
            "catboost_q_default": cb_q_default,
            "catboost_q_deep":    cb_q_deep,
        }
        return super().build_model()


class DiverseDemandForecaster(DemandForecaster):
    """DemandForecaster with N diverse base models inside a single MLForecast.

    Default: 5 models (lgbm_default, lgbm_shallow, lgbm_tweedie,
    catboost_default, catboost_deep). Pass n_variants=9 to expand to 9 models
    covering more objectives (Poisson, quantile, MAE) and depths.

    Shares lags / date features / target transforms with the parent, so
    feature engineering runs once across all N models.
    """

    N_MODELS_5 = ["lgbm_default", "lgbm_shallow", "lgbm_tweedie",
                   "catboost_default", "catboost_deep"]
    N_MODELS_9 = N_MODELS_5 + ["lgbm_deep", "lgbm_poisson",
                                "lgbm_quantile_median", "catboost_mae"]

    def __init__(self, n_variants: int = 5, **kwargs):
        super().__init__(**kwargs)
        assert n_variants in (5, 9), "n_variants must be 5 or 9"
        self.n_variants = n_variants

    def build_model(self):
        rs = self.random_state
        lgb_base = dict(
            n_estimators=200, learning_rate=0.05, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8, verbosity=-1,
            n_jobs=-1, random_state=rs,
        )
        # Core 5
        lgb_default = lgb.LGBMRegressor(**lgb_base, num_leaves=31)
        lgb_shallow = lgb.LGBMRegressor(**{**lgb_base,
                                            "num_leaves": 7, "max_depth": 3,
                                            "n_estimators": 300, "learning_rate": 0.04,
                                            "min_child_samples": 20})
        lgb_tweedie = lgb.LGBMRegressor(**{**lgb_base,
                                            "num_leaves": 31,
                                            "objective": "tweedie",
                                            "tweedie_variance_power": 1.5})

        cb_base = dict(iterations=200, learning_rate=0.05, verbose=0,
                        thread_count=-1, random_seed=rs)
        cb_class = CatBoostWithAutoCats if self.categorical_features else cb.CatBoostRegressor
        cb_default = cb_class(**cb_base, depth=6)
        cb_deep = cb_class(**{**cb_base, "iterations": 150, "learning_rate": 0.04, "depth": 10})

        models = {
            "lgbm_default":    lgb_default,
            "lgbm_shallow":    lgb_shallow,
            "lgbm_tweedie":    lgb_tweedie,
            "catboost_default": cb_default,
            "catboost_deep":   cb_deep,
        }

        # Extra 4 for n_variants=9
        if self.n_variants == 9:
            lgb_deep = lgb.LGBMRegressor(**{**lgb_base,
                                             "num_leaves": 127, "max_depth": 10,
                                             "n_estimators": 150, "learning_rate": 0.04})
            lgb_poisson = lgb.LGBMRegressor(**{**lgb_base,
                                                "num_leaves": 31,
                                                "objective": "poisson"})
            lgb_quantile_med = lgb.LGBMRegressor(**{**lgb_base,
                                                     "num_leaves": 31,
                                                     "objective": "quantile",
                                                     "alpha": 0.5})
            cb_mae = cb_class(**{**cb_base, "depth": 6, "loss_function": "MAE"})
            models.update({
                "lgbm_deep":             lgb_deep,
                "lgbm_poisson":          lgb_poisson,
                "lgbm_quantile_median":  lgb_quantile_med,
                "catboost_mae":          cb_mae,
            })

        self._override_models = models
        return super().build_model()

    def predict_quantile(self, horizon=3, quantile=0.833):
        """Forecast at a given quantile using model spread as uncertainty proxy."""
        models = self.predict_models(horizon)
        mean = (models["lgbm"] + models["catboost"]) / 2
        spread = (models["lgbm"] - models["catboost"]).abs()

        z = {0.833: 0.97, 0.9: 1.28, 0.95: 1.65, 0.99: 2.33}.get(round(quantile, 3), 1.0)
        upper = mean + z * spread
        return upper.clip(lower=0)


class StatsForecasterEnsemble:
    """Fast statistical ensemble: AutoTheta + SeasonalNaive via Nixtla statsforecast.

    Both are cheap and complementary:
      - AutoTheta: decomposition-based, strong for short horizons
      - SeasonalNaive: y_{t+h} = y_{t+h-season}, bias-free seasonality

    Predict returns a wide DataFrame (SKU x future_dates) with the mean of both.
    """

    def __init__(self, freq: str = "W-MON", season_length: int = 52,
                 censoring_strategy: str = "mean_impute"):
        self.freq = freq
        self.season_length = season_length
        self.censoring_strategy = censoring_strategy
        self._sf = None
        self._fitted_long: pd.DataFrame | None = None

    def fit(self, sales_wide, in_stock_wide=None):
        from statsforecast import StatsForecast
        from statsforecast.models import AutoTheta, SeasonalNaive
        long_df = wide_to_long(sales_wide, in_stock_wide,
                               censoring_strategy=self.censoring_strategy)
        long_df = long_df.dropna(subset=["y"])
        self._fitted_long = long_df
        models = [
            AutoTheta(season_length=self.season_length),
            SeasonalNaive(season_length=self.season_length),
        ]
        # n_jobs=1 for Windows-compat (parallel via multiprocessing can crash on Win).
        self._sf = StatsForecast(models=models, freq=self.freq, n_jobs=1, verbose=False)
        self._sf.fit(long_df)
        return self

    def predict(self, horizon: int = 3) -> pd.DataFrame:
        preds = self._sf.predict(h=horizon)
        preds = preds.reset_index(drop=True) if "unique_id" in preds.columns else preds.reset_index()
        # Keep only the named model columns (exclude any accidental 'index').
        KNOWN_MODELS = {"AutoTheta", "SeasonalNaive", "AutoETS", "HoltWinters",
                        "AutoARIMA", "Theta", "SeasonalExponentialSmoothingOptimized"}
        model_cols = [c for c in preds.columns if c in KNOWN_MODELS]
        if not model_cols:
            raise RuntimeError(f"No model output columns found in {list(preds.columns)}")
        preds["stats_ensemble"] = preds[model_cols].mean(axis=1)
        preds[["Store", "Product"]] = preds["unique_id"].str.split("_", expand=True).astype(int)
        wide = preds.pivot_table(
            index=["Store", "Product"], columns="ds", values="stats_ensemble"
        )
        return wide.clip(lower=0)


class QuantileDemandForecaster:
    """True quantile forecaster: LightGBM + CatBoost trained with quantile loss.

    Both models use `alpha` as the quantile target (shared). The final quantile
    forecast is the mean of the two models' outputs (ensemble).
    """

    def __init__(self, master=None, alpha: float = 0.833):
        self.alpha = alpha
        self.master = master
        self.static_features = prepare_static_features(master) if master is not None else None
        self.model = None

    def build_model(self):
        lgb_q = lgb.LGBMRegressor(
            objective="quantile",
            alpha=self.alpha,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
            n_jobs=-1,
        )
        cb_q = cb.CatBoostRegressor(
            loss_function=f"Quantile:alpha={self.alpha}",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            thread_count=-1,
        )
        self.model = MLForecast(
            models={"lgbm_q": lgb_q, "catboost_q": cb_q},
            freq="W-MON",
            lags=[1, 2, 3, 4, 8, 13, 26, 52],
            lag_transforms={
                1: [
                    RollingMean(window_size=4),
                    RollingMean(window_size=8),
                    RollingMean(window_size=13),
                    RollingStd(window_size=4),
                    RollingStd(window_size=13),
                ],
                52: [RollingMean(window_size=3)],
            },
            date_features=["week", "month"],
            num_threads=4,
        )
        return self.model

    def fit(self, sales_wide, in_stock_wide=None):
        long_df = wide_to_long(sales_wide, in_stock_wide)
        long_df = long_df.dropna(subset=["y"])
        if self.model is None:
            self.build_model()
        static_cols = None
        if self.static_features is not None:
            long_df = long_df.merge(self.static_features, on="unique_id", how="left")
            static_cols = [c for c in self.static_features.columns if c != "unique_id"]
        self.model.fit(long_df, static_features=static_cols)
        return self

    def predict(self, horizon=3) -> pd.DataFrame:
        """Return wide DataFrame of ensemble quantile forecasts, indexed by (Store, Product)."""
        preds = self.model.predict(horizon)
        preds["ensemble"] = (preds["lgbm_q"] + preds["catboost_q"]) / 2
        preds[["Store", "Product"]] = preds["unique_id"].str.split("_", expand=True).astype(int)
        wide = preds.pivot_table(index=["Store", "Product"], columns="ds", values="ensemble")
        return wide.clip(lower=0)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)

    sales = pd.read_csv(DATA_DIR / "Week 0 - 2024-04-08 - Sales.csv").set_index(INDEX)
    in_stock = pd.read_csv(DATA_DIR / "Week 0 - In Stock.csv").set_index(INDEX)
    master = pd.read_csv(DATA_DIR / "Week 0 - Master.csv").set_index(INDEX)
    sales.columns = pd.to_datetime(sales.columns)
    in_stock.columns = pd.to_datetime(in_stock.columns)

    forecaster = DemandForecaster(master=master)
    forecaster.fit(sales, in_stock)
    preds = forecaster.predict(horizon=3)
    print("Forecast shape:", preds.shape)
    print(preds.head())
    print("\nForecast sum per period:")
    print(preds.sum())
