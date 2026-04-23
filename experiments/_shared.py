"""Shared helpers used by experiment modules."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from benchmark.cv_harness import (
    build_extended_sales,
    format_summary,
    log_experiment,
    score_policy_on_folds,
)
from simulation import InventorySimulator


def get_n_workers(default: int = 3) -> int:
    return int(os.environ.get("CV_WORKERS", str(default)))


def run_and_log(name: str, factory, full_sales, full_in_stock, master,
                coverage_weeks: int = 2, alpha=None, include_val=True,
                n_workers: int | None = None, extra=None):
    """Run one policy through CV and log to experiments.csv. Returns result dict."""
    nw = n_workers if n_workers is not None else get_n_workers()
    print(f"\nRunning: {name}")
    res = score_policy_on_folds(
        policy_factory=factory,
        full_sales=full_sales, full_in_stock=full_in_stock, master=master,
        coverage_weeks=coverage_weeks, alpha=alpha,
        n_workers=nw, include_val=include_val,
    )
    log_experiment(name, res, extra=extra)
    print(format_summary(res, name))
    return res


def sacred_eval_and_log(name: str, factory, extra: dict | None = None) -> dict:
    """Run a policy ONCE on sacred weeks 1-8 and log."""
    sim = InventorySimulator()
    sres = sim.run_simulation(factory())
    print(f"  SACRED: {sres['competition_cost']:,.2f}  "
          f"(h={sres['competition_holding']:,.2f}, s={sres['competition_shortage']:,.2f})")
    extra = (extra or {}).copy()
    extra["sacred"] = True
    log_experiment(
        f"{name} [SACRED]",
        {"mean": {"competition_cost": sres["competition_cost"],
                  "comp_holding": sres["competition_holding"],
                  "comp_shortage": sres["competition_shortage"]},
         "val": None, "per_fold": None},
        extra=extra,
    )
    return sres


def print_summary_table(rows: list[tuple[str, dict, dict | None]]):
    """rows: [(name, cv_result, sacred_result_or_None)]."""
    print("\n" + "=" * 80)
    print(f"  {'Policy':<52s} {'CV mean':>8s} {'VAL':>8s} {'SACRED':>8s}")
    print("  " + "-" * 78)
    for name, res, sres in rows:
        m = res["mean"]
        v = res.get("val") or {}
        sac = f"{sres['competition_cost']:,.0f}" if sres else ""
        print(f"  {name:<52s} {m['competition_cost']:>8,.0f} "
              f"{v.get('competition_cost', float('nan')):>8,.0f} {sac:>8s}")
