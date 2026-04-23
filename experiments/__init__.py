"""VN2 experiment registry and shared utilities.

Each experiment is a top-level `run()` function in a submodule. The registry
below maps a name (usable on the CLI) to that function.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root on sys.path when experiments are imported.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


def _lazy(module_name: str, func_name: str = "run"):
    """Lazy-loader so CLI `list` works without importing heavy modules."""
    def _wrap(*args, **kwargs):
        import importlib
        mod = importlib.import_module(f"experiments.{module_name}")
        return getattr(mod, func_name)(*args, **kwargs)
    _wrap.__name__ = f"{module_name}.{func_name}"
    return _wrap


REGISTRY: dict[str, dict] = {
    "coverage_sweep": {
        "description": "ML Point across coverage_weeks in {1,2,3,4}",
        "fn": _lazy("sweeps", "run_coverage_sweep"),
    },
    "multiplier_sweep": {
        "description": "ML Point cov=2 x multiplier in {0.90..1.20}",
        "fn": _lazy("sweeps", "run_multiplier_sweep"),
    },
    "per_sku_sweep": {
        "description": "Per-SKU coverage routing (volume / Syntetos rules)",
        "fn": _lazy("sweeps", "run_per_sku_sweep"),
    },
    "censoring": {
        "description": "Censoring strategies: interpolate / mean_impute / seasonal / zero",
        "fn": _lazy("censoring", "run"),
    },
    "seed_avg": {
        "description": "Seed-averaged sacred eval of best policy (5 seeds)",
        "fn": _lazy("seed_avg", "run"),
    },
    "optuna": {
        "description": "Optuna hyperparameter search on ML Point (20 trials)",
        "fn": _lazy("optuna_tune", "run"),
    },
    "optuna_ensemble": {
        "description": "Optuna hyperparameter search on Ensemble (20 trials)",
        "fn": _lazy("optuna_ensemble", "run"),
    },
    "optuna_weights": {
        "description": "Optuna on ENSEMBLE WEIGHTS (w_ma, LGB/CB share, mult, safety)",
        "fn": _lazy("optuna_weights", "run"),
    },
    "optuna_joint": {
        "description": "Joint Optuna: LGB+CB hyperparams AND weights (16-dim, seeded)",
        "fn": _lazy("optuna_joint", "run"),
    },
    "negbinom": {
        "description": "NegBinom newsvendor grid (cov × alpha)",
        "fn": _lazy("negbinom", "run"),
    },
    "ensemble": {
        "description": "Weighted ensemble: Seasonal MA + LGB + CB",
        "fn": _lazy("ensemble", "run"),
    },
    "ensemble_plus": {
        "description": "Ensemble with per-SKU w_ma + per-round multiplier",
        "fn": _lazy("ensemble_plus", "run"),
    },
    "rmse_safety": {
        "description": "k*RMSE safety stock with backtest residuals",
        "fn": _lazy("rmse_safety", "run"),
    },
    "trajectory": {
        "description": "Trajectory-simulation order optimization",
        "fn": _lazy("trajectory", "run"),
    },
    "cum_quantile": {
        "description": "Cumulative-demand quantile (LGBM+CB quantile on N-wk sum)",
        "fn": _lazy("cum_quantile", "run"),
    },
    "conformal": {
        "description": "Ensemble + conformal per-SKU additive offset (split CP on backtest)",
        "fn": _lazy("conformal", "run"),
    },
    "sacred": {
        "description": "One-shot sacred evaluation of a policy spec",
        "fn": _lazy("sacred_eval", "run"),
    },
    "report": {
        "description": "Pretty-print benchmark/experiments.csv (top results)",
        "fn": _lazy("report", "run"),
    },
}
