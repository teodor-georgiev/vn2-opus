"""Leakage-safe VN2 8-week WAPE benchmark.

This is an experiment runner, not part of the production/competition policy.
It selects a forecast blend only on pre-competition rolling 8-week windows,
then evaluates the locked blend once on the eight competition weeks.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


DATA_PATH = Path("Data/Week 8 - 2024-06-03 - Sales.csv")
OUT_PATH = Path("artifacts/vn2_8w_wape_results.json")
HORIZON = 8
CV_STARTS = [85, 93, 101, 109, 117, 125, 133, 141]
VAL_START = 149
SACRED_START = 157
RANDOM_STATE = 42


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = float(np.abs(actual).sum())
    if denom <= 0:
        return float("nan")
    return float(np.abs(actual - pred).sum() / denom)


def load_sales() -> tuple[pd.DataFrame, np.ndarray, list[pd.Timestamp]]:
    raw = pd.read_csv(DATA_PATH)
    raw = raw.set_index(["Store", "Product"]).sort_index()
    dates = [pd.Timestamp(c) for c in raw.columns]
    y = raw.to_numpy(dtype=np.float32)
    assert y.shape[0] == 599, y.shape
    assert len(dates) == y.shape[1]
    assert dates[SACRED_START] == pd.Timestamp("2024-04-15"), dates[SACRED_START]
    assert dates[SACRED_START + HORIZON - 1] == pd.Timestamp("2024-06-03")
    return raw, y, dates


def recent_mean(y: np.ndarray, origin: int, window: int, horizon: int = HORIZON) -> np.ndarray:
    mean = np.nanmean(y[:, origin - window + 1 : origin + 1], axis=1)
    return np.repeat(mean[:, None], horizon, axis=1)


def seasonal52(y: np.ndarray, target_start: int, horizon: int = HORIZON) -> np.ndarray:
    cols = [target_start + h - 52 for h in range(horizon)]
    return y[:, cols].astype(np.float64)


def seasonal_two_year(y: np.ndarray, target_start: int, horizon: int = HORIZON) -> np.ndarray:
    one = seasonal52(y, target_start, horizon)
    cols2 = [target_start + h - 104 for h in range(horizon)]
    if min(cols2) < 0:
        return one
    two = y[:, cols2].astype(np.float64)
    return 0.70 * one + 0.30 * two


def _features_for_origin(
    y: np.ndarray,
    stores: np.ndarray,
    products: np.ndarray,
    dates: list[pd.Timestamp],
    origin: int,
    horizon: int,
) -> pd.DataFrame:
    n = y.shape[0]
    target_idx = origin + horizon
    target_date = dates[target_idx]

    def lag(k: int) -> np.ndarray:
        idx = origin - k + 1
        if idx < 0:
            return np.full(n, np.nan, dtype=np.float32)
        return y[:, idx]

    def roll_mean(window: int) -> np.ndarray:
        lo = max(0, origin - window + 1)
        return np.nanmean(y[:, lo : origin + 1], axis=1)

    def roll_std(window: int) -> np.ndarray:
        lo = max(0, origin - window + 1)
        return np.nanstd(y[:, lo : origin + 1], axis=1)

    def nonzero_rate(window: int) -> np.ndarray:
        lo = max(0, origin - window + 1)
        return np.mean(y[:, lo : origin + 1] > 0, axis=1)

    seasonal_idx = target_idx - 52
    seasonal_104_idx = target_idx - 104
    seasonal_52 = y[:, seasonal_idx] if seasonal_idx >= 0 else np.full(n, np.nan)
    seasonal_104 = y[:, seasonal_104_idx] if seasonal_104_idx >= 0 else np.full(n, np.nan)

    woy = int(target_date.isocalendar().week)
    angle = 2.0 * np.pi * woy / 52.0

    return pd.DataFrame(
        {
            "store": stores,
            "product": products,
            "horizon": np.full(n, horizon, dtype=np.int16),
            "week": np.full(n, woy, dtype=np.int16),
            "month": np.full(n, target_date.month, dtype=np.int8),
            "woy_sin": np.full(n, np.sin(angle), dtype=np.float32),
            "woy_cos": np.full(n, np.cos(angle), dtype=np.float32),
            "lag1": lag(1),
            "lag2": lag(2),
            "lag3": lag(3),
            "lag4": lag(4),
            "lag8": lag(8),
            "lag13": lag(13),
            "lag26": lag(26),
            "mean4": roll_mean(4),
            "mean8": roll_mean(8),
            "mean13": roll_mean(13),
            "mean26": roll_mean(26),
            "std4": roll_std(4),
            "std13": roll_std(13),
            "std26": roll_std(26),
            "nz8": nonzero_rate(8),
            "nz13": nonzero_rate(13),
            "nz26": nonzero_rate(26),
            "seasonal52": seasonal_52,
            "seasonal104": seasonal_104,
        }
    )


def make_training(
    y: np.ndarray,
    stores: np.ndarray,
    products: np.ndarray,
    dates: list[pd.Timestamp],
    eval_origin: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    # Use only examples whose targets are already observed at eval_origin.
    # Keep the most recent ~84 origin weeks to control runtime while retaining
    # at least one full annual cycle of training examples.
    min_origin = max(52, eval_origin - 84)
    frames: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    for origin in range(min_origin, eval_origin):
        max_h = min(HORIZON, eval_origin - origin)
        for h in range(1, max_h + 1):
            frames.append(_features_for_origin(y, stores, products, dates, origin, h))
            targets.append(y[:, origin + h].astype(np.float64))
    X = pd.concat(frames, ignore_index=True)
    target = np.concatenate(targets)
    X["store"] = X["store"].astype("category")
    X["product"] = X["product"].astype("category")
    return X, target


def fit_lgb(
    y: np.ndarray,
    stores: np.ndarray,
    products: np.ndarray,
    dates: list[pd.Timestamp],
    eval_origin: int,
) -> lgb.LGBMRegressor:
    X, target = make_training(y, stores, products, dates, eval_origin)
    model = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=260,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, target, categorical_feature=["store", "product"])
    return model


def predict_lgb(
    model: lgb.LGBMRegressor,
    y: np.ndarray,
    stores: np.ndarray,
    products: np.ndarray,
    dates: list[pd.Timestamp],
    origin: int,
) -> np.ndarray:
    cols = []
    for h in range(1, HORIZON + 1):
        X = _features_for_origin(y, stores, products, dates, origin, h)
        X["store"] = X["store"].astype("category")
        X["product"] = X["product"].astype("category")
        cols.append(np.clip(model.predict(X), 0.0, None))
    return np.column_stack(cols)


def evaluate_origin(
    y: np.ndarray,
    stores: np.ndarray,
    products: np.ndarray,
    dates: list[pd.Timestamp],
    target_start: int,
) -> dict[str, object]:
    origin = target_start - 1
    actual = y[:, target_start : target_start + HORIZON].astype(np.float64)
    model = fit_lgb(y, stores, products, dates, origin)
    lgb_pred = predict_lgb(model, y, stores, products, dates, origin)
    recent13 = recent_mean(y, origin, 13)
    seasonal = seasonal52(y, target_start)
    seasonal2 = seasonal_two_year(y, target_start)

    return {
        "start": target_start,
        "start_date": dates[target_start].strftime("%Y-%m-%d"),
        "actual": actual,
        "predictions": {
            "lgb_l1": lgb_pred,
            "recent13": recent13,
            "seasonal52": seasonal,
            "seasonal2y": seasonal2,
        },
    }


def pooled_wape(rows: list[dict[str, object]], method: str) -> float:
    actual = np.concatenate([row["actual"].ravel() for row in rows])
    pred = np.concatenate([row["predictions"][method].ravel() for row in rows])
    return wape(actual, pred)


def blended_prediction(row: dict[str, object], weights: tuple[float, float, float]) -> np.ndarray:
    wlgb, wseasonal, wrecent = weights
    p = row["predictions"]
    return (
        wlgb * p["lgb_l1"]
        + wseasonal * p["seasonal52"]
        + wrecent * p["recent13"]
    )


def choose_blend(cv_rows: list[dict[str, object]], val_row: dict[str, object]) -> dict[str, object]:
    candidates = []
    # Simplex grid at 0.05 increments. Tune only on CV.
    grid = np.arange(0.0, 1.0001, 0.05)
    cv_actual = np.concatenate([row["actual"].ravel() for row in cv_rows])
    val_actual = val_row["actual"].ravel()
    for wlgb in grid:
        for wseasonal in grid:
            wrecent = 1.0 - wlgb - wseasonal
            if wrecent < -1e-9:
                continue
            wrecent = max(0.0, float(wrecent))
            weights = (float(wlgb), float(wseasonal), wrecent)
            cv_pred = np.concatenate([blended_prediction(row, weights).ravel() for row in cv_rows])
            val_pred = blended_prediction(val_row, weights).ravel()
            candidates.append(
                {
                    "weights": weights,
                    "cv_wape": wape(cv_actual, cv_pred),
                    "val_wape": wape(val_actual, val_pred),
                }
            )
    candidates.sort(key=lambda x: x["cv_wape"])
    cv_best = candidates[0]

    # Validation gate: among blends within 0.5 percentage point of best CV,
    # choose the one with the best VAL WAPE. This keeps sacred untouched.
    threshold = cv_best["cv_wape"] + 0.005
    robust = [c for c in candidates if c["cv_wape"] <= threshold]
    robust.sort(key=lambda x: (x["val_wape"], x["cv_wape"]))
    chosen = robust[0]
    return {"cv_best": cv_best, "chosen": chosen, "top10": candidates[:10]}


def result_row(row: dict[str, object], weights: tuple[float, float, float]) -> dict[str, object]:
    actual = row["actual"]
    methods = {
        name: wape(actual, pred) for name, pred in row["predictions"].items()
    }
    blend = blended_prediction(row, weights)
    methods["blend"] = wape(actual, blend)
    return {
        "start": row["start"],
        "start_date": row["start_date"],
        "wape": methods,
    }


def main() -> None:
    sales, y, dates = load_sales()
    stores = sales.index.get_level_values("Store").to_numpy(dtype=np.int32)
    products = sales.index.get_level_values("Product").to_numpy(dtype=np.int32)

    cv_rows = []
    for start in CV_STARTS:
        print(f"Fitting CV start={start} date={dates[start].date()}...", flush=True)
        cv_rows.append(evaluate_origin(y, stores, products, dates, start))

    print(f"Fitting VAL start={VAL_START} date={dates[VAL_START].date()}...", flush=True)
    val_row = evaluate_origin(y, stores, products, dates, VAL_START)

    selection = choose_blend(cv_rows, val_row)
    weights = tuple(selection["chosen"]["weights"])
    print("Chosen weights (lgb, seasonal52, recent13):", weights, flush=True)

    cv_summary = {
        method: pooled_wape(cv_rows, method)
        for method in ["lgb_l1", "recent13", "seasonal52", "seasonal2y"]
    }
    cv_actual = np.concatenate([row["actual"].ravel() for row in cv_rows])
    cv_blend = np.concatenate([blended_prediction(row, weights).ravel() for row in cv_rows])
    cv_summary["blend"] = wape(cv_actual, cv_blend)

    val_summary = result_row(val_row, weights)["wape"]

    # Lock the blend before touching the competition window, fit using data only
    # through 2024-04-08, then score 2024-04-15 .. 2024-06-03 once.
    print(f"Fitting locked SACRED model from origin={SACRED_START - 1}...", flush=True)
    sacred_row = evaluate_origin(y, stores, products, dates, SACRED_START)
    sacred_result = result_row(sacred_row, weights)
    sacred_blend = blended_prediction(sacred_row, weights)
    sacred_actual = sacred_row["actual"]
    horizon_wape = [
        wape(sacred_actual[:, h], sacred_blend[:, h]) for h in range(HORIZON)
    ]

    result = {
        "metric": "WAPE = sum(abs(actual-forecast)) / sum(abs(actual))",
        "n_series": int(y.shape[0]),
        "history_start": dates[0].strftime("%Y-%m-%d"),
        "competition_start": dates[SACRED_START].strftime("%Y-%m-%d"),
        "competition_end": dates[SACRED_START + HORIZON - 1].strftime("%Y-%m-%d"),
        "selection_protocol": {
            "cv_starts": CV_STARTS,
            "val_start": VAL_START,
            "sacred_start": SACRED_START,
            "blend_grid_step": 0.05,
            "validation_gate_cv_tolerance": 0.005,
        },
        "blend_selection": selection,
        "cv_pooled_wape": cv_summary,
        "val_wape": val_summary,
        "folds": [result_row(row, weights) for row in cv_rows],
        "competition_wape": sacred_result["wape"],
        "competition_blend_horizon_wape": horizon_wape,
        "chosen_weights": {
            "lgb_l1": weights[0],
            "seasonal52": weights[1],
            "recent13": weights[2],
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== VN2 8-WEEK WAPE ===")
    print("CV pooled:", {k: f"{100*v:.2f}%" for k, v in cv_summary.items()})
    print("VAL:", {k: f"{100*v:.2f}%" for k, v in val_summary.items()})
    print("COMPETITION:", {k: f"{100*v:.2f}%" for k, v in sacred_result["wape"].items()})
    print("Blend horizon WAPE:", [f"{100*v:.2f}%" for v in horizon_wape])
    print("Results:", OUT_PATH)


if __name__ == "__main__":
    main()
