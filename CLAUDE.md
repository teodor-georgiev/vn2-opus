# VN2 Inventory Planning Challenge

## Scenario
Monday, April 15th, 2024, early morning. You are the inventory planner of a
retail chain. Every week before the store opens, you place an order to the
central warehouse; it arrives in two weeks. You know current stock and
incoming deliveries, and have ~3 years of historical weekly sales + stock
availability. You place orders over 6 consecutive weeks.

## Costs (minimize total)
- Holding: **0.2€** per unit of end-of-week inventory (no holding on in-transit).
- Shortage: **1€** per unit of lost sales (no backorders).
- Excess inventory carries forward with no waste, loss, or obsolescence.
- Critical ratio: `1 / (1 + 0.2) = 0.833`.

## Mechanics
- **Review period**: one order per week, placed at the end of the week before
  the next starts.
- **Lead time**: 2 weeks. Order placed at end of week X arrives at the start
  of week X+3 (available for week X+3 sales).
- **Supply**: infinite capacity, stable lead time.
- **6 rounds** of orders (rounds 1–6). Simulation continues 2 more weeks after
  round 6 so the final order (round 6) can arrive in week 8.
- If a participant submits no order in a round, the last submission is reused.

## All participants share
- Same initial inventory + in-transit state.
- Same revealed sales sequence (real anonymized company data).

## Submission format
- CSV, `,` separator, 599 rows (same Store×Product index order as sales).
- No negatives, no missing. Only the latest submission counts per round.

## Evaluation protocol — IMPORTANT

Three-tier split to avoid test-set leakage:

1. **CV folds** (weeks 0–148): `benchmark/cv_harness.py` runs 4 pseudo-competition windows. Use for all tuning and policy comparison.
2. **VAL holdout** (weeks 149–156): a single 8-week window just before competition. Use as tiebreaker between CV-equivalent candidates. Run each policy once.
3. **SACRED** (weeks 157–164, the real competition): `run_sacred.py` / `simulation.InventorySimulator`. **Touch only once per policy variant.**

**Selection rule:** A number from sacred only counts if the policy was chosen on CV (and VAL as tiebreak) BEFORE running on sacred. Running many variants on sacred and reporting the best is cherry-picking the test set. Treat sacred as reporting, never as selection signal.

Our current honest best: **cov=2 ×1.05 = 3,786 EUR on sacred** (tied with ×1.10 on CV at 3,754; ×1.05 better on VAL at 2,990 vs 3,064). `×1.10 sacred = 3,707` exists but was selected by peeking at sacred — reporting it as "our result" is the bias trap we're avoiding.

## Scoring window — IMPORTANT
**We report `competition_cost` = weeks 3–8 (indices 2..7 in the simulator).**

Why: with a 2-week lead time, the first order any participant can place
(round 1) only arrives at the start of week 3. Weeks 1 and 2 are determined
entirely by the shared initial state and are identical for every competitor —
ignore them for ranking. The simulator's `setup_cost` field captures them
separately.

- `total_cost` = all 8 weeks (includes fixed setup weeks 1–2).
- `competition_cost` = weeks 3–8 (what the leaderboard reflects).
- `setup_cost` = weeks 1–2 (constant across all participants).

Use `competition_cost` when comparing policies or reporting scores. The
official benchmark target is **4,334€** on this basis.

## Environment
- Python: `d:\miniconda\envs\forecasting_2026\python.exe`
  (mlforecast 1.0.31, LightGBM 4.6, CatBoost 1.2.10)

## Key files
- `simulation.py` — inventory state machine + `benchmark_policy` (Seasonal MA
  + 4-week coverage).
- `forecaster.py` — mlforecast pipeline (LGB + CatBoost).
- `run.py` — pipeline runner, phi grid search.
- `benchmark/evaluate.py` — side-by-side policy comparison.
- `Data/` — weekly sales, in-stock, initial state, master CSVs.
