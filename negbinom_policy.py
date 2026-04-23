"""Per-SKU Negative Binomial newsvendor policy.

At each round, fit a NegBinom to each SKU's historical COVERAGE-week rolling
sums of demand, then order at the alpha-quantile minus net inventory.

NegBinom(r, p) with mean mu and variance mu + mu^2/r captures overdispersion
common in retail demand (especially intermittent/slow movers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from simulation import LEAD_TIME


def _fit_negbinom_moment(y: np.ndarray) -> tuple[float, float]:
    """Method-of-moments for NegBinom; returns (r, p)."""
    y = y[~np.isnan(y)]
    if len(y) < 3 or y.sum() == 0:
        return (1.0, 0.5)  # degenerate fallback
    mean = y.mean()
    var = y.var(ddof=1)
    if var <= mean:
        # underdispersed (rare) — use Poisson-ish fallback
        return (max(mean * 5, 0.5), 0.5)
    r = mean ** 2 / max(var - mean, 1e-3)
    p = r / (r + mean)
    return (max(r, 0.1), min(max(p, 0.01), 0.99))


class NegBinomPolicy:
    """Order-up-to = NegBinom quantile over coverage-week demand sum (per SKU)."""

    def __init__(self, coverage_weeks: int = 2, alpha: float = 0.70,
                 multiplier: float = 1.0, min_history: int = 26):
        self.coverage_weeks = coverage_weeks
        self.alpha = alpha
        self.multiplier = multiplier
        self.min_history = min_history
        self.last_forecast: pd.Series | None = None
        self.last_quantile_forecast: pd.Series | None = None
        self._fit_cache: dict = {}

    def __call__(self, sim, round_idx, sales_hist):
        # Build rolling coverage-week sums: for each SKU, construct training
        # series y_t = sum(sales[t-cov+1 .. t]). Use that as the target variable
        # we need a quantile of.
        sales = sales_hist.copy()
        cov = self.coverage_weeks
        if sales.shape[1] < self.min_history + cov:
            # Fallback: order 1 unit for each SKU
            order = pd.Series(1, index=sales.index)
            self.last_forecast = pd.Series(1.0, index=sales.index)
            self.last_quantile_forecast = order.astype(float).copy()
            return order.astype(int)

        # Rolling cov-week sum for each SKU.
        rolling = sales.rolling(window=cov, axis=1, min_periods=cov).sum()
        # Drop leading NaN columns.
        rolling = rolling.iloc[:, cov - 1:]

        # Fit per SKU on last K observations (use all available).
        mu_hat = pd.Series(0.0, index=sales.index)
        quantiles = pd.Series(0.0, index=sales.index)
        for idx, row in rolling.iterrows():
            y = row.to_numpy(dtype=float)
            y = y[~np.isnan(y)]
            if len(y) < 3 or y.sum() == 0:
                mu_hat.loc[idx] = 0.0
                quantiles.loc[idx] = 1.0
                continue
            r, p = _fit_negbinom_moment(y)
            mu = (1 - p) * r / p  # NegBinom(r, p) mean
            mu_hat.loc[idx] = mu
            # scipy NegBinom: stats.nbinom(n=r, p=p).
            q = stats.nbinom.ppf(self.alpha, n=r, p=p)
            quantiles.loc[idx] = q if np.isfinite(q) else mu * 1.5

        order_up_to = (self.multiplier * quantiles).clip(lower=0)
        self.last_forecast = mu_hat.copy()
        self.last_quantile_forecast = order_up_to.copy()

        net_inv = sim.get_net_inventory_position()
        return (order_up_to - net_inv).clip(lower=0).round(0).astype(int)
