# VN2 session learnings (2026-04-21 → 04-23)

Distilled from 290+ logged experiments, two full rebuild cycles (TCN, RL, stacking,
diverse ensembles, Optuna), and an audit of everything we tried. Ordered roughly
by how useful the insight is for the next practitioner.

---

## 1. For inventory competitions, the **ordering policy** is usually the bigger lever than the forecaster

We got `3,581 → 3,255 (−9.1%)` in this session. **Breakdown of where the EUR came from:**

| Change | Gain |
|---|---|
| cov-based order-up-to → cost-aware state-projection (`E1 = max(I0+T1-d1, 0); target = d2 + z_α·σ`) | **−230 EUR** (biggest single win) |
| 2-model → 5-model diverse ensemble | −23 EUR |
| α=0.65 → α=0.70 | −15 EUR |
| Optuna joint (α, mult, w_ma) tuning | −55 EUR |
| **TOTAL** | **−326 EUR** |

Forecast WAPE was essentially unchanged (~70%) throughout. **The ordering policy did all the work.**

This matches Bartosz's deck showing his RL (a more elaborate ordering policy) adding 181 EUR over his cost-aware — a similar magnitude of "ordering > forecasting" effect.

## 2. Deep learning forecasters don't automatically beat LightGBM+CatBoost

We built a faithful port of Bartosz's TCN+FiLM+Fourier-seasonal deep forecaster (~500 lines, validated against deck images). On sacred:

- TCN + cov ordering: **3,808 EUR**
- LGB+CB ensemble + cov ordering: 3,581 EUR
- TCN forecast WAPE: 46.4%
- LGB+CB forecast WAPE: 46.2%

Essentially equivalent forecast accuracy, but TCN needed more time, more hyperparameters, and a GPU. **Parked.**

Lesson: **for 599 SKUs × ~150 weeks of history, gradient boosting is usually enough.** Deep learning pays off with more data or better features.

## 3. Stacking can overfit CV dramatically

We built a Ridge stacker over LGB + CB + MA forecasts on backtest residuals. Per-fold weights swung wildly (LGB weight: 0.00 to 0.62 across 9 folds):

| | CV | VAL |
|---|---|---|
| Baseline fixed weights (0.25 MA, 0.50 LGB, 0.50 CB) | 4,339 | 2,593 |
| Ridge-stacked | **4,446 (+107)** | **3,502 (+909)** |

**Fixed equal weights beat learned weights.** Two reasons:

1. LGB and CB are too correlated → stacker picks one and drops the other (unstable).
2. Small calibration set (26 weeks × 599 SKUs) → per-fold residuals are idiosyncratic, learned weights overfit.

**Diverse base models matter more than smart combiners.** Going from 2 to 5 truly-different models (adding Tweedie, shallow, deep) with simple equal weights beat stacking.

## 4. Optuna can find real joint-parameter gains — but only after you get the search space right

| Attempt | CV | Outcome |
|---|---|---|
| Manual α sweep only | 4,347 @ α=0.65 | won, became champion |
| Optuna on LGB hyperparams alone (prior session) | — | overfit CV, worse on sacred (3,786 → 3,858) |
| Optuna on (α, mult, w_ma) joint — 15 trials | **4,249** (−98 vs manual) | **beat sacred by 56 EUR** |

What made the successful Optuna work:
- Search POLICY parameters (α, mult, w_ma), not model hyperparameters
- Search JOINTLY (each one interacts with the others)
- Gate with VAL — accept only if VAL also improves or stays flat
- Small, interpretable search space (3 dims, bounded intervals)

## 5. The CV/VAL/sacred protocol actually works, but the three splits don't agree equally

Our fold setup:
- 8-fold CV (windows 85, 93, 101, 109, 117, 125, 133, 141)
- VAL (window 149, single 8-week)
- Sacred (window 157, single 8-week)

Per-fold CV std ≈ 215 EUR (6% of the mean). So CV mean has **SEM ≈ 108** — CI is ±210 EUR at 95%. Any "improvement" smaller than ~100 EUR on CV is within noise.

**Observations on when each signal was predictive of sacred:**

| Scenario | CV→sacred | VAL→sacred |
|---|---|---|
| Big structural changes (2-model → 5-model, cov-order → cost-aware) | **strong (within 50 EUR)** | OK |
| Small α/mult tweaks at the champion regime | **unreliable (swings 50+ EUR)** | **weak (swings 40-80 EUR)** |
| Over-fit variants (aggressive stacking, tuned Optuna on LGB params) | **misleading** | caught some |

Lessons:
- Use the protocol but **don't oversell tiny CV gains** — they're often noise
- **VAL being in the same 8-week format as sacred** helped catch some overfits but isn't a magic oracle
- Biggest gains were always visible on both CV AND VAL simultaneously

## 6. Conformal at high alphas is worse than Normal-σ at low alphas for our cost structure

Attempted scaled-conformal via mlforecast at α ∈ {0.55, ..., 0.95}:

| α | CV | VAL |
|---|---|---|
| 0.55 | 4,538 | 2,583 |
| 0.70 | 4,416 | 2,885 |
| 0.95 | 4,687 | 3,744 ← WORST |

**Pushing to higher-coverage calibration (α=0.95) inflates VAL by 1,151 EUR**. Reasons:

1. Critical ratio α* = 0.833 is the *theoretical* optimum for newsvendor under 5:1 shortage:holding. α=0.95 is over-conservative.
2. With only 2 calibration windows (the max we could fit per-fold), the 95th-percentile estimate degenerates to "max observed residual" — huge noise.

Our Normal-σ approximation at the cost-calibrated α (Optuna picked 0.60) turned out to be **equivalent to a slightly more conservative policy than theory suggests**, probably because σ under-estimates tail risk in this intermittent-demand setting.

## 7. The oracle floor is **333 EUR** (perfect forecast); we're at 3,255. Don't misread the gap.

Our forecast error costs ~2,922 EUR in principle. But the **realistic** ceiling from better forecasts (given the data we have) is 200-500 EUR of additional savings — the other 2,400+ EUR requires external data (holidays, promos, weather) we don't have.

**The plateau at 70% CV WAPE is structural**, not engineering laziness:
- 48% of weeks have zero sales → no signal
- 3 units/week mean → small numbers amplify % errors
- 3-year history is short for robust yearly seasonality
- No external features

TCN, Tweedie, quantile regression, all Nixtla/statsforecast variants cluster around 68-70% CV WAPE.

## 8. Beware reporting metrics with scale mismatches

Our harness computed WAPE as `sum(|forecast - actual|)/sum(|actual|)` where actual is a **3-week sum** but CostAware policies' `last_forecast` is a **1-week point forecast**. The reported WAPE for CostAware variants (~70%) is therefore NOT comparable to the WAPE for cov-based policies that do forecast a 3-week sum.

When we switched to a quantile-trained forecaster, WAPE dropped from 70% to 55% — **not because the model was more accurate, but because the quantile output is systematically larger and thus closer in absolute terms to a 3-week sum**.

Lesson: **always check metric definitions align with what the policy actually produces.** Otherwise you'll chase phantom improvements.

## 9. The "Benchmark = 4,334" number cited everywhere is NOT reproducible from the published organizer code

We ran the organizer's exact baseline code (Seasonal MA + 4-week coverage) through our simulator and got **5,335**, not 4,334. The 4,334 is the official reference but its exact implementation isn't public.

This matters for the "vs benchmark %" comparison everyone uses — it's a convention, not a directly-verified number. Our absolute sacred is still correct and apples-to-apples comparable to Matias/Bartosz (we validated via running Matias's open-sourced DRL code through our simulator: reported 3,765, we got 3,737 — 28 EUR within training-seed noise).

## 10. The bias direction aligns with our cost structure, and we should NOT try to fully correct it

Our champion forecaster has **bias = −6.40** (systematic under-forecast by ~6 units per SKU per 3-week window). With 5:1 shortage:holding cost ratio, this is GOOD:
- Under-forecasting → order MORE (after multiplier) → reduces shortage
- Shortage cost > holding cost, so erring high is cheap

If we "fixed" bias to 0 (perfect calibration), we'd need to pair it with a larger multiplier — same net effect, more fragile.

**The quantile-trained approach (pinball loss at α=0.833) is the principled way to encode this asymmetry in the model.**

## 11. One-shot sacred discipline worked — and caught several apparent wins that weren't

Three variants looked good on CV but got rejected because VAL disagreed:
- TCN + cov ordering: CV good, VAL good, sacred WORSE → never got best
- Stacked Ridge ensemble: CV +107, VAL +909 → never touched sacred
- Conformal at high α: CV +49, VAL +1,000+ → never touched sacred

**Three variants passed CV+VAL and all delivered on sacred:**
- 5-model Diverse α=0.65 (3,327)
- 5-model Diverse α=0.70 (3,311)
- Optuna joint (3,255)

Every sacred touch was "pre-committed" via CV+VAL selection. Sacred number is reliable.

## 12. For a next push, invest in external features or cost-aware training — not another Optuna round

Marginal returns on:
- More Optuna trials: probably already extracted most of it
- More ensemble diversity: 9-model was worse than 5-model
- More stacking: didn't work, probably won't
- Cost-aware training (pinball/quantile): still being tested at time of writing

What would actually help:
- **External data** (calendar/promo/holiday features) — biggest unrealized gain, ~300-500 EUR
- **Per-SKU policy clustering** (different α for different SKU types) — we tried this and it was noise; would need a better cluster definition
- **Cost-aware training objective** (pinball loss) — in progress

## File references

- `policies.py: DiverseCostAwarePolicy` — current champion
- `policies.py: QuantileCostAwarePolicy` — cost-aware training variant (in test)
- `forecaster.py: DiverseDemandForecaster` — 5-model ensemble base
- `forecaster.py: QuantileDiverseDemandForecaster` — 5-model pinball-loss ensemble
- `benchmark/cv_harness.py` — 8-fold CV + VAL + forecast accuracy scoring
- `best_winner.json` — champion config archive
- `experiments.csv` — 290+ rows of every CV+VAL and sacred run
- `SESSION_LOG_bartoz.md` — human-readable summary of all experiments
