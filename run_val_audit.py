"""Find the experiment with the best VAL cost; compare to our champion."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

CSV = ROOT / "benchmark" / "experiments.csv"

records = []
with open(CSV, encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for fields in reader:
        if len(fields) < 4:
            continue
        ts = fields[0]
        policy = fields[1]
        # SCHEMA DETECT
        # Old schema: timestamp, policy, cv_mean_cost, val_cost, fc_mae, fc_bias  (6 cols)
        # New schema: ..., cv_mean_cost(2), cv_holding(3), cv_shortage(4), val_cost(5), val_holding(6), val_shortage(7), fc_mae(8), ...
        try:
            cv_mean = float(fields[2])
        except Exception:
            continue

        val_cost = None
        if len(fields) <= 6:
            try: val_cost = float(fields[3])
            except: pass
        else:
            # New schema — val_cost at position 5
            try: val_cost = float(fields[5])
            except: pass

        if val_cost is None or val_cost <= 0:
            continue

        # Skip placeholder one-shot sacred rows (no real VAL — they have val_cost from a fake bucket)
        is_sacred = "SACRED" in policy
        records.append({
            "timestamp": ts,
            "policy": policy,
            "cv_mean": cv_mean,
            "val_cost": val_cost,
            "is_sacred": is_sacred,
        })

df = pd.DataFrame(records)
df = df.drop_duplicates("policy", keep="last")
# Drop rows where val_cost looks bogus (e.g., > 30000 or < 100)
df = df[(df["val_cost"] >= 100) & (df["val_cost"] <= 30000)]

print(f"Total experiments with real VAL: {len(df)}\n")

print("=== TOP 20 by VAL cost (lowest = best) ===")
top = df.sort_values("val_cost").head(20)
print(top[["policy", "cv_mean", "val_cost"]].to_string(
    index=False,
    formatters={"cv_mean": "{:,.1f}".format, "val_cost": "{:,.1f}".format},
))

print("\n=== Where our champion sits ===")
champ = df[df["policy"] == "cap2_a0.65_m1.0"]
if len(champ):
    rank = (df.sort_values("val_cost").reset_index(drop=True)
              .index[df.sort_values("val_cost").reset_index(drop=True)["policy"] == "cap2_a0.65_m1.0"][0]) + 1
    print(f"  cap2_a0.65_m1.0 (sacred 3,350): VAL = {champ['val_cost'].iloc[0]:,.1f}, "
          f"rank #{rank} of {len(df)} on VAL")

# Also show top-5 only among 'comparable' (cap2 / oe / Ensemble) policy families
print("\n=== Top 10 strictly within session families (cap2_/oe_) + best Ensemble ===")
session = df[df["policy"].str.startswith(("cap2_", "oe_"))]
if not session.empty:
    print(session.sort_values("val_cost").head(10)
          [["policy", "cv_mean", "val_cost"]].to_string(
              index=False,
              formatters={"cv_mean": "{:,.1f}".format, "val_cost": "{:,.1f}".format},
          ))
