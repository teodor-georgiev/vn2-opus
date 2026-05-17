"""Strict audit for VN2 order timing / information availability.

Official VN2 flow:
- Round 1 order uses only historical Week 0 data.
- Week 1 data is revealed after Round 1.
- Round r+1 may use weeks 1..r actuals, not week r+1 actuals.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from simulation import InventorySimulator, NUM_ROUNDS  # noqa: E402


def main() -> None:
    sim = InventorySimulator()
    hidden_cols = set(sim.actual_sales.columns)
    seen = []

    def probe_policy(sim, round_idx, sales_hist):
        leaked = [c for c in sales_hist.columns if c in hidden_cols]
        seen.append((round_idx, len(leaked), leaked))
        return pd.Series(0, index=sim.end_inventory.index)

    sim.run_simulation(probe_policy)

    failures = []
    for round_idx, n_hidden_seen, leaked in seen[:NUM_ROUNDS]:
        # Before placing round r+1, only weeks 1..r should be known.
        expected = round_idx
        if n_hidden_seen != expected:
            failures.append(
                f"round_idx={round_idx}: policy saw {n_hidden_seen} actual future weeks; "
                f"expected {expected}. Last leaked={leaked[-1] if leaked else None}"
            )

    if failures:
        raise AssertionError("Information-timing audit failed:\n" + "\n".join(failures))

    print("OK: order timing / information availability audit passed")


if __name__ == "__main__":
    main()
