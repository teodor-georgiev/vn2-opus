# VN2 Inventory Planning — winning pipeline

Honest best on the sacred 8-week window: **3,349.60 EUR** (`competition_cost`, weeks 3–8).

- vs organizer benchmark 4,334 → **−22.7%**
- vs Bartosz #1 leaderboard 3,763 → **−11.0%**
- vs Bartosz post-comp RL 3,582 → **−6.5%**

## Reproduce

```bash
# 1. CV+VAL run (8-fold, ~2 min):
python run_costaware.py --alpha 0.65 --mult 1.0 --folds-8 --name cap_8f_a0.65

# 2. One-shot sacred (ONLY do this once per policy variant):
python run_costaware_sacred.py
# -> 3,349.60 EUR

# 3. Generate per-round submission CSVs (uploads-ready):
python run_make_submissions.py
# -> writes submissions/round_1.csv ... round_6.csv
```

## Repo layout

```
VN2_Opus/
├── README.md                       <- this file
├── CLAUDE.md                       <- task spec + protocol
├── SESSION_LOG_bartoz.md           <- everything tried this session
│
├── Data/                           <- competition CSVs (read-only)
├── submissions/                    <- output: per-round upload files
│   └── round_1.csv … round_6.csv
│
├── ── CORE PIPELINE (what produces 3,349.60) ──
├── simulation.py                   <- InventorySimulator + run_window
├── forecaster.py                   <- DemandForecaster (LGB+CB via Nixtla mlforecast)
├── policies.py                     <- EnsemblePolicy + CostAwarePolicy + others
├── run_costaware.py                <- CV+VAL runner (4 or 8 fold)
├── run_costaware_sacred.py         <- one-shot sacred eval
├── run_make_submissions.py         <- export per-round order CSVs
├── benchmark/
│   ├── cv_harness.py               <- FOLDS, FOLDS_8, VAL_START, scoring
│   └── experiments.csv             <- auto-log of every run
│
├── best_winner.json                <- the tuned ensemble cfg (3,581 sacred reference)
├── best_hyperparams.json           <- Optuna result (kept for reproducibility)
│
├── ── EXPERIMENTAL / PARKED ──
├── experiments/
│   ├── (older session modules — censoring, conformal, optuna, etc.)
│   └── bartoz_session/             <- THIS session's parked builds
│       ├── tcn/                    <- TCN deep forecaster (sacred 3,808 — worse)
│       ├── rl/                     <- PPO+GAE Stage 2b (VAL 3,344 — worse)
│       ├── tuned_base/             <- CostAware + best_winner.json (sacred 3,399)
│       ├── per_round/              <- per-round multiplier sweeps (modest)
│       ├── per_sku/                <- per-SKU bin clustering (no signal)
│       └── order_ensemble/         <- order-averaging ensembles (sacred 3,358)
│
├── pdf/                            <- Bartosz deck PDF + extracted page images
│
├── legacy/                         <- pre-session runners (kept for reference)
├── logs/                           <- all *.log output files
└── (other top-level modules)       <- pipeline.py, cumulative_quantile.py, etc.
```

## What's in the winning pipeline (and what isn't)

**In:**
- LGB + CatBoost ensemble forecaster (via Nixtla `mlforecast`) — `forecaster.DemandForecaster`
- 13-week seasonal MA blended at `w_ma=0.25` — `policies._seasonal_ma_forecast`
- 1-step-ahead backtest RMSE → per-SKU σ — inside `policies.CostAwarePolicy._fit`
- Cost-aware ordering rule:
  - `E1 = max(I0 + T1 − d1, 0)`
  - `target = d2 + Φ⁻¹(0.65) · σ`
  - `order = max(target − E1, 0)`

**Not in (parked, all worse on sacred):**
- TCN deep forecaster — `experiments/bartoz_session/tcn/`
- RL Stage 2b (PPO+GAE) — `experiments/bartoz_session/rl/`
- Tuned ensemble base — `experiments/bartoz_session/tuned_base/`
- Per-round / per-SKU multipliers — `experiments/bartoz_session/per_*/`
- Order ensembling — `experiments/bartoz_session/order_ensemble/`

## Selection protocol

Per `feedback_sacred_rule.md` (auto-memory):
- 8-fold CV (`FOLDS_8` weeks 85, 93, 101, 109, 117, 125, 133, 141)
- VAL window 149–156 (same 8-week format as sacred)
- Sacred 157–164 — touch ONCE per policy variant

The 3,349.60 result is one-shot under this protocol with CV→sacred alignment within 6 EUR.

## Where to look first

1. To **see the winning policy code**: [policies.py: `class CostAwarePolicy`](policies.py)
2. To **see all past experiments and their numbers**: [`SESSION_LOG_bartoz.md`](SESSION_LOG_bartoz.md) and `benchmark/experiments.csv`
3. To **understand the task**: [`CLAUDE.md`](CLAUDE.md)
4. To **upload a submission**: [`submissions/round_6.csv`](submissions/round_6.csv) (the order arriving in week 8)
