"""Run the canonical final VN2 policy from best_winner.json.

This script is the audit-grade entrypoint for CI because it uses the same frozen
policy metadata that documents the champion result, then optionally writes all
six competition submission files and validates their format.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_lgb_share_sweep import DiverseLGBShareCostAwarePolicy
from simulation import InventorySimulator

EXPECTED_ROWS = 599


def load_policy_config(path: str | Path = "best_winner.json") -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing policy config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("policy_class") != "DiverseLGBShareCostAwarePolicy":
        raise ValueError(f"Unsupported policy_class in {cfg_path}: {cfg.get('policy_class')}")
    return cfg


def build_policy(cfg: dict[str, Any], master) -> DiverseLGBShareCostAwarePolicy:
    params = dict(cfg["policy_params"])
    params["master"] = master
    return DiverseLGBShareCostAwarePolicy(**params)


def write_submissions(sim: InventorySimulator, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    template = pd.read_csv("Data/Week 0 - Submission Template.csv")
    if len(template) != EXPECTED_ROWS:
        raise AssertionError(f"Template should have {EXPECTED_ROWS} rows, got {len(template)}")
    canonical_index = list(zip(template["Store"], template["Product"]))

    if len(sim.orders_placed) != 6:
        raise AssertionError(f"Expected 6 placed orders, got {len(sim.orders_placed)}")

    for r, order in enumerate(sim.orders_placed, start=1):
        ordered = order.reindex(canonical_index).fillna(0).clip(lower=0).round(0).astype(int)
        out = pd.DataFrame({
            "Store": [s for s, _p in canonical_index],
            "Product": [p for _s, p in canonical_index],
            "0": ordered.values,
        })
        out.to_csv(out_dir / f"round_{r}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="best_winner.json")
    ap.add_argument("--write-submissions", action="store_true")
    ap.add_argument("--submission-dir", default="submissions")
    ap.add_argument("--results-json", default="audit_artifacts/final_results.json")
    args = ap.parse_args()

    cfg = load_policy_config(args.config)
    sim = InventorySimulator()
    policy = build_policy(cfg, sim.master)

    print("=== VN2 final policy audit ===")
    print(f"description: {cfg.get('description')}")
    print(f"policy_class: {cfg.get('policy_class')}")
    print(f"expected sacred_cost_seed42: {cfg.get('sacred_cost_seed42')}")
    print(f"selection: {cfg.get('selection')}")

    results = sim.run_simulation(policy)
    print("\n--- Result ---")
    print(f"competition_cost  = {results['competition_cost']:,.2f} EUR")
    print(f"competition_holding = {results['competition_holding']:,.2f}")
    print(f"competition_shortage = {results['competition_shortage']:,.2f}")
    print(f"setup_cost = {results['setup_cost']:,.2f}")
    print(f"total_cost = {results['total_cost']:,.2f}")

    out = {
        "policy_config": cfg,
        "actual": {
            "competition_cost": float(results["competition_cost"]),
            "competition_holding": float(results["competition_holding"]),
            "competition_shortage": float(results["competition_shortage"]),
            "setup_cost": float(results["setup_cost"]),
            "total_cost": float(results["total_cost"]),
        },
        "matches_expected_seed42": (
            abs(float(results["competition_cost"]) - float(cfg.get("sacred_cost_seed42", "nan"))) < 1e-6
        ),
    }
    results_path = Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {results_path}")

    if args.write_submissions:
        write_submissions(sim, Path(args.submission_dir))
        print(f"Wrote submissions to {args.submission_dir}")


if __name__ == "__main__":
    main()
