"""Coverage / multiplier / safety / per-SKU sweeps for ML Point."""
from __future__ import annotations

import pandas as pd

from policies import MLPointPolicy
from benchmark.cv_harness import build_extended_sales
from experiments._shared import print_summary_table, run_and_log, sacred_eval_and_log


def _mk(master, cov=2, multiplier=1.0, safety_units=0.0, coverage_per_sku=None):
    def f():
        return MLPointPolicy(
            coverage_weeks=cov, master=master,
            multiplier=multiplier, safety_units=safety_units,
            coverage_per_sku=coverage_per_sku,
        )
    return f


def run_coverage_sweep():
    fs, fis, master = build_extended_sales()
    rows = []
    for cov in (1, 2, 3, 4):
        name = f"ML Point cov={cov}"
        res = run_and_log(name, _mk(master, cov=cov), fs, fis, master, coverage_weeks=cov)
        rows.append((name, res, None))
    print_summary_table(rows)


def run_multiplier_sweep():
    fs, fis, master = build_extended_sales()
    rows = []
    for cov in (2,):
        for mult in (0.95, 1.00, 1.05, 1.10, 1.15, 1.20):
            name = f"ML Point cov={cov} x{mult}"
            res = run_and_log(name, _mk(master, cov=cov, multiplier=mult), fs, fis, master, coverage_weeks=cov)
            rows.append((name, res, None))
    print_summary_table(rows)


def _classify_skus(full_sales: pd.DataFrame, history_end: int = 149) -> pd.DataFrame:
    hist = full_sales.iloc[:, :history_end].fillna(0.0)
    st = pd.DataFrame(index=full_sales.index)
    st["mean"] = hist.mean(axis=1)
    st["std"] = hist.std(axis=1)
    st["cv2"] = (st["std"] / st["mean"].clip(lower=0.01)) ** 2
    nz = (hist > 0).astype(int)
    nz_counts = nz.sum(axis=1).clip(lower=1)
    st["adi"] = hist.shape[1] / nz_counts
    return st


def _rule_volume(st: pd.DataFrame) -> pd.Series:
    c = pd.Series(2, index=st.index, dtype=int)
    c[st["mean"] >= 3] = 3
    c[st["mean"] < 1] = 1
    return c


def _rule_syntetos(st: pd.DataFrame) -> pd.Series:
    c = pd.Series(2, index=st.index, dtype=int)
    c[(st["adi"] < 1.32) & (st["cv2"] < 0.49)] = 3
    c[(st["adi"] >= 1.32) & (st["cv2"] >= 0.49)] = 1
    return c


def run_per_sku_sweep():
    fs, fis, master = build_extended_sales()
    st = _classify_skus(fs)
    experiments = [
        ("Per-SKU [volume] x1.00", _rule_volume(st), 1.00),
        ("Per-SKU [volume] x1.10", _rule_volume(st), 1.10),
        ("Per-SKU [syntetos] x1.00", _rule_syntetos(st), 1.00),
        ("Per-SKU [syntetos] x1.10", _rule_syntetos(st), 1.10),
    ]
    rows = []
    for name, cov_series, mult in experiments:
        factory = _mk(master, multiplier=mult, coverage_per_sku=cov_series)
        res = run_and_log(name, factory, fs, fis, master, coverage_weeks=2)
        sres = sacred_eval_and_log(name, factory)
        rows.append((name, res, sres))
    print_summary_table(rows)
