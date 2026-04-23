# Session log — Bartosz rebuild + Stage 2 build (2026-04-21 → 22)

This doc consolidates everything tried in this session. Source of truth for raw numbers
is `benchmark/experiments.csv`; this is the human-readable summary.

## Headline

**Honest best sacred score: 3,255.20 EUR**
(DiverseCostAwarePolicy α=0.600, mult=1.138, w_ma=0.329, 5-model ensemble, Optuna-tuned)

vs prior session-best 3,581 (LGB+CB Ensemble + cov × 1.05 ordering): **−326 EUR (−9.1%)**
vs Bartosz #1 leaderboard 3,763: **−508 EUR (−13.5%)**
vs benchmark 4,334: **−1,079 EUR (−24.9%)**

## Session trajectory

| Date | Variant | Sacred |
|---|---|---|
| Session start | LGB+CB Ensemble cov×1.05 | 3,581 |
| 2026-04-22 | CostAware α=0.65 (2-model + MA blend) | 3,349.60 |
| 2026-04-23 | Diverse 5-model α=0.65 | 3,326.60 |
| 2026-04-23 | Diverse 5-model α=0.70 | 3,311.00 |
| **2026-04-23** | **Diverse 5-model Optuna-tuned (α=0.600, mult=1.138, w_ma=0.329)** | **3,255.20** |

## What we built (and didn't)

| Component | File | Status |
|---|---|---|
| Stage-1 deep forecaster (TCN+FiLM+Fourier) | `tcn_forecaster.py` | ✅ built, validated against deck images, **PARKED** (worse than ensemble) |
| Stage-2a deterministic cost-aware ordering | `policies.py: CostAwarePolicy` | ✅ built, **WINNING POLICY** |
| Stage-2b RL ordering (PPO+GAE+Node/Chain graph) | — | ❌ NOT built (4–6 day effort, not justified) |

## Key design decisions / bugs we caught

- **Bartosz's E1/E2 formulation assumes LT=3** (T1 + T2 both pre-existing pipeline).
  VN2 has LT=2 (only T1). We adapted to one projection step:
  `E1 = max(I0+T1−d1, 0); order = max(Q_α(d2) − E1, 0)`.
- **Backtest RMSE inflates over 26 mixed horizons** — fixed by using only 1-step-ahead
  RMSE (`rmse_horizons=1`).
- **Softplus on the TCN decoder hurts** (3,808 → 4,193). Removed it; v2 still slightly
  worse than v1, so v1 is the parked TCN.
- **Selection rule is now 8-fold CV + VAL (sacred-format)** per user instruction.
  See `memory/feedback_sacred_rule.md`.

## Experiments run this session (chronological)

### TCN forecaster (Stage 1)

| Variant | CV mean (8-fold unless noted) | VAL | Sacred | Notes |
|---|---|---|---|---|
| `tcn_cov2_m1.0_h5_hd64` | 3,960 (cv0 only) | — | — | smoke test |
| `tcn_cov2_m1.10_fast` | 3,674 (4-fold) | 2,851 | — | first full CV+VAL |
| `tcn_cov2_m1.05` | 3,703 (4-fold) | 2,857 | — | sweep |
| `tcn_cov2_m1.15` | 3,671 (4-fold) | 2,912 | — | sweep |
| **`tcn_cov2_m1.10_SACRED`** | — | — | **3,808** | one-shot, **WORSE than ensemble** |
| `tcn_v2_cov2_m1.10` | 4,193 | 3,189 | — | added augs + new seasonal head + softplus + tanh-FiLM (faithful to deck) |
| `tcn_v2_noaug_cov2_m1.10` | 4,212 | 3,209 | — | augs OFF — augs aren't the cause |
| `tcn_v2_nosoftplus_cov2_m1.10` | 3,787 | 2,869 | — | softplus IS the culprit; v1 still wins |

Conclusion: TCN parked. Honest WAPE 46.4% ≈ Ensemble WAPE 46.2%.

### CostAware (Stage 2a) — default Ensemble base

Sweeps at LT=2-corrected rule (α=0.5..0.833 on 4-fold CV first):

| α | 4-fold CV | VAL |
|---|---|---|
| 0.55 | 3,435 | 2,566 |
| 0.60 | 3,380 | 2,570 |
| 0.625 | 3,363 | 2,585 |
| **0.65** | **3,356** | 2,593 |
| 0.70 | 3,365 | 2,639 |
| 0.75 | 3,409 | 2,687 |

CV winner: α=0.65. **Sacred one-shot: 3,349.60** ✓ (CV→sacred alignment within 6 EUR).

### CostAware on 8-fold CV (per user's updated rule)

| α | 8-fold CV | VAL | comment |
|---|---|---|---|
| 0.55 | 4,468 | 2,566 | high CV due to harder early folds |
| 0.60 | 4,391 | 2,570 | |
| 0.625 | 4,362 | 2,585 | |
| 0.65 | 4,339 | 2,593 | (this is the sacred-3,350 variant on 8-fold) |
| **0.70** | **4,319** | 2,639 | new 8-fold CV winner |
| 0.75 | 4,323 | 2,687 | |

### CostAware + tuned Ensemble base (best_winner.json config)

α sweep:
| α | CV | VAL |
|---|---|---|
| 0.60 | 4,380 | 2,586 |
| 0.65 | 4,328 | 2,586 |
| **0.70** | **4,296** | 2,611 |

Mult sweep at α=0.70:
| mult | CV | VAL |
|---|---|---|
| 0.95 | 4,352 | 2,616 |
| 1.00 | 4,296 | 2,611 |
| 1.05 | 4,259 | 2,629 |
| 1.10 | 4,230 | 2,651 |
| 1.15 | 4,222 | 2,699 |
| **1.20** | **4,219** | 2,745 |

Selected α=0.70 m=1.05 (best CV-VAL trade-off). **Sacred: 3,399** — 50 EUR worse than 3,350.
CV→sacred didn't transfer; tuned-base Optuna may overfit CV folds.

### Per-round multiplier (only round-5 swept so far)

α=0.65, all rounds=1.0 except round-5:
| r5 mult | CV | VAL |
|---|---|---|
| 0.7 | 4,523 | 2,692 |
| 0.8 | 4,448 | 2,649 |
| 0.9 | 4,389 | 2,613 |
| 1.0 | 4,339 | 2,593 |
| 1.1 | 4,305 | 2,588 |
| 1.2 | 4,278 | 2,589 |
| 1.3 | 4,268 | 2,601 |
| **1.4** | **4,262** | 2,615 |
| 1.5 | 4,264 | 2,645 |

R5-mult=1.4 CV winner (~77 EUR CV improvement over 1.0). Hypothesis was "lower mult to fix
overstock at w8" — wrong; we were under-ordering. Not yet sacred-tested.

### Per-SKU bins (current work)

Diagnostic on cv0 (`run_sku_cluster_diagnostic.py`):
- **Heavy Pareto**: top 100 SKUs (17%) = 53% of cost
- **Q4_high (top 25% by demand) = 54% of cost** — over-stocking (H/S = 0.69 vs optimal ~0.20)
- **Intermittent (66% of products) = 47% of cost** — under-stocking (H/S = 0.34)

Intermittent-bin mult sweep currently running.

## Files created this session

| File | Purpose |
|---|---|
| `tcn_forecaster.py` | TCN model + training + inference (parked) |
| `run_tcn_cv.py` | TCN CV runner |
| `run_tcn_sacred.py` | TCN sacred one-shot |
| `policies.py: CostAwarePolicy` | Stage 2 cost-aware ordering policy |
| `run_costaware.py` | CostAware CV+VAL runner (4-fold or 8-fold) |
| `run_costaware_sacred.py` | CostAware sacred one-shot |
| `run_costaware_tuned.py` | CostAware + best_winner.json base, sweep α/mult |
| `run_costaware_tuned_sacred.py` | sacred one-shot with tuned base |
| `run_cap_diagnostic.py` | Per-round cost matrix |
| `run_cap_per_round.py` | Per-round mult sweep (round-5 done) |
| `run_sku_cluster_diagnostic.py` | Per-SKU bin cost analysis |
| `run_cap_per_bin.py` | Per-bin mult sweep (intermittent in progress) |
| `pdf/Bartoz_deep.pdf` → `pdf/pages/p*.png` | Deck pages exported for image validation |

All runs append to `benchmark/experiments.csv`. Filter by policy name prefix:
- `tcn_*` for TCN variants
- `cap2_*` for CostAware variants

## Next planned

1. Finish intermittent-bin mult sweep
2. Sweep smooth bin (over-stocker, expect mult < 1.0)
3. Sweep Q4_high bin (over-stocker)
4. Combine winners → one sacred shot if CV+VAL both improve over 3,350
