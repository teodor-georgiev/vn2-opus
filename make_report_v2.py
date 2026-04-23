"""HTML report v2 — built from CostAware (current champion, sacred 3,349.60).

Adds:
- Updated all charts using current champion data
- NEW: total weekly demand across the FULL pre-sacred history (weeks 0..156)
  to visualize seasonality and put sacred weeks in context.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

OUT = ROOT / "report.html"


def load_experiments() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "benchmark" / "experiments.csv",
                     on_bad_lines="skip", engine="python")
    df["is_sacred"] = df["policy"].str.contains("SACRED", regex=True, na=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def run_costaware_with_capture():
    """Run current champion (DiverseCostAware, Optuna-tuned) on sacred; capture forecasts/orders/actuals."""
    from simulation import InventorySimulator, LEAD_TIME
    from policies import DiverseCostAwarePolicy

    sim = InventorySimulator()

    forecasts, orders, target_actuals = [], [], []

    class Captured(DiverseCostAwarePolicy):
        def __call__(self, sim_arg, round_idx, sales_hist):
            order = super().__call__(sim_arg, round_idx, sales_hist)
            if self.last_forecast is not None:
                forecasts.append(self.last_forecast.copy())
            orders.append(order.copy())
            actual_cols = sorted(sim_arg.actual_sales.columns)
            arrive = round_idx + LEAD_TIME
            if arrive < len(actual_cols):
                target_actuals.append(sim_arg.actual_sales[actual_cols[arrive]].copy())
            return order

    policy = Captured(
        alpha=0.599816047538945,
        multiplier=1.1376785766024788,
        w_ma=0.3293972738151323,
        backtest_window=26, safety_floor=0.5, rmse_horizons=1,
        censoring_strategy="mean_impute", random_state=42, master=sim.master,
        n_variants=5,
    )
    res = sim.run_simulation(policy)

    rounds = []
    for i, (fc, ac) in enumerate(zip(forecasts, target_actuals)):
        fc = fc.reindex(ac.index)
        diff = fc - ac
        rounds.append({
            "round": i + 1,
            "fc_sum": float(fc.sum()),
            "actual_sum": float(ac.sum()),
            "mae": float(diff.abs().mean()),
            "wape": float(diff.abs().sum() / ac.abs().sum()) if ac.abs().sum() > 0 else float("nan"),
            "bias": float(diff.mean()),
            "order_sum": int(orders[i].sum()),
        })
    return res, pd.DataFrame(rounds)


def historical_demand():
    """Total weekly demand across all SKUs + model predictions for CV/VAL/SACRED."""
    from benchmark.cv_harness import build_extended_sales, FOLDS_8, VAL_START
    from policies import EnsemblePolicy, _seasonal_ma_forecast
    from simulation import LEAD_TIME

    full_sales, full_in_stock, master = build_extended_sales()
    weekly_total = full_sales.sum(axis=0).sort_index()
    n_weeks = len(weekly_total)
    dates = [str(c.date()) if hasattr(c, "date") else str(c) for c in weekly_total.index]

    # For each window, fit ensemble at that cutoff and predict the next 8 weeks
    # totals (sum across SKUs), aligning with the historical date axis.
    SACRED_START = 157
    HORIZON = 8

    def fit_predict(ws: int) -> list[float]:
        sales_hist = full_sales.iloc[:, :ws]
        in_stock_hist = full_in_stock.iloc[:, :ws]
        ens = EnsemblePolicy(coverage_weeks=3, w_ma=0.25,
                              censoring_strategy="mean_impute", random_state=42)
        ens._fit(sales_hist, in_stock_hist)
        ml = ens._forecaster.predict(horizon=HORIZON)
        ma = _seasonal_ma_forecast(sales_hist, in_stock_hist, horizon=HORIZON)
        w = 0.25
        # Align columns
        common = ml.columns[: min(len(ml.columns), len(ma.columns))]
        blend = w * ma[common].reindex(ml.index).fillna(0) + (1 - w) * ml[common]
        return blend.sum(axis=0).tolist()  # 8 weekly totals

    cv_starts = list(FOLDS_8.values())            # 8 starts
    pred_lines = {}                                # name -> array of length n_weeks (NaN elsewhere)

    print("  fitting per-window predictions (10 windows)...")
    # CV folds: combine into one "CV predictions" line (concatenated, with gaps)
    cv_arr = [None] * n_weeks
    for ws in cv_starts:
        print(f"    CV fold ws={ws}...")
        preds = fit_predict(ws)
        for h, v in enumerate(preds):
            idx = ws + h
            if idx < n_weeks:
                cv_arr[idx] = float(v)
    pred_lines["cv"] = cv_arr

    print(f"    VAL ws={VAL_START}...")
    val_preds = fit_predict(VAL_START)
    val_arr = [None] * n_weeks
    for h, v in enumerate(val_preds):
        idx = VAL_START + h
        if idx < n_weeks:
            val_arr[idx] = float(v)
    pred_lines["val"] = val_arr

    print(f"    SACRED ws={SACRED_START}...")
    sac_preds = fit_predict(SACRED_START)
    sac_arr = [None] * n_weeks
    for h, v in enumerate(sac_preds):
        idx = SACRED_START + h
        if idx < n_weeks:
            sac_arr[idx] = float(v)
    pred_lines["sacred"] = sac_arr

    return {
        "dates": dates,
        "totals": weekly_total.values.astype(float).tolist(),
        "sacred_start_idx": SACRED_START,
        "val_start_idx": VAL_START,
        "preds_cv": pred_lines["cv"],
        "preds_val": pred_lines["val"],
        "preds_sacred": pred_lines["sacred"],
    }


def table_html(df: pd.DataFrame, top_n: int = 60) -> str:
    keep = ["timestamp", "policy", "cv_mean_cost", "val_cost",
            "cv_mean_holding", "cv_mean_shortage", "fc_mae", "fc_bias",
            "is_sacred"]
    keep = [c for c in keep if c in df.columns]
    d = df[keep].copy()
    # sort: sacred first then by timestamp desc
    d = d.sort_values(["is_sacred", "timestamp"], ascending=[False, False]).head(top_n)
    d["timestamp"] = d["timestamp"].dt.strftime("%m-%d %H:%M")
    for c in d.select_dtypes("number").columns:
        d[c] = d[c].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "")
    d["is_sacred"] = d["is_sacred"].map({True: "Y", False: ""})
    rows = []
    for _, r in d.iterrows():
        tds = "".join(f"<td>{r[c]}</td>" for c in d.columns)
        sacred_class = ' class="sacred"' if r["is_sacred"] == "Y" else ""
        rows.append(f"<tr{sacred_class}>{tds}</tr>")
    hdr = "".join(f"<th onclick=\"sortTable({i})\">{c}</th>"
                   for i, c in enumerate(d.columns))
    return f"""
<table id="exp" class="tbl">
  <thead><tr>{hdr}</tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
<p class="muted">Showing top {len(d)} rows by sacred-flag then date.</p>"""


def main():
    print("Running CostAware on sacred (capturing forecasts)...")
    res, rounds_df = run_costaware_with_capture()

    print("Computing historical total demand...")
    hist = historical_demand()

    print("Loading experiments...")
    exp = load_experiments()

    weekly = res["weekly_log"]
    week_labels = [f"wk{int(w)+1}" for w in weekly["week"]]
    data = {
        "weeks":         week_labels,
        "demand":        weekly["demand"].tolist(),
        "sales":         weekly["sales"].tolist(),
        "missed_sales":  weekly["missed_sales"].tolist(),
        "start_inv":     weekly["start_inventory"].tolist(),
        "end_inv":       weekly["end_inventory"].tolist(),
        "holding":       weekly["holding_cost"].tolist(),
        "shortage":      weekly["shortage_cost"].tolist(),
        "cum_total":     (weekly["holding_cost"] + weekly["shortage_cost"]).cumsum().tolist(),
        "rounds":        rounds_df["round"].tolist(),
        "fc_sum":        rounds_df["fc_sum"].tolist(),
        "actual_sum":    rounds_df["actual_sum"].tolist(),
        "mae":           rounds_df["mae"].tolist(),
        "wape":          (rounds_df["wape"] * 100).tolist(),
        "bias":          rounds_df["bias"].tolist(),
        "order_sum":     rounds_df["order_sum"].tolist(),
        "history":       hist,
    }
    data_json = json.dumps(data)

    print("Building HTML...")
    tbl = table_html(exp)

    total_comp = res["competition_cost"]
    total_total = res["total_cost"]
    setup = res["setup_cost"]
    h = res["competition_holding"]
    s = res["competition_shortage"]
    overall_mae = float(rounds_df["mae"].mean())
    overall_wape = float(rounds_df["wape"].mean() * 100)
    overall_bias = float(rounds_df["bias"].mean())

    cfg_json = json.dumps({
        "policy": "DiverseCostAwarePolicy (Optuna-tuned)",
        "alpha": 0.599816047538945,
        "multiplier": 1.1376785766024788,
        "w_ma": 0.3293972738151323,
        "backtest_window": 26,
        "safety_floor": 0.5,
        "rmse_horizons": 1,
        "n_variants": 5,
        "ensemble_models": [
            "lgbm_default", "lgbm_shallow", "lgbm_tweedie",
            "catboost_default", "catboost_deep",
        ],
    }, indent=2)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>VN2 Report (CostAware champion)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
<style>
  body {{ font: 13px/1.4 system-ui, Segoe UI, sans-serif; margin: 20px;
          background: #fafafa; color: #222; }}
  h1, h2 {{ margin-top: 1.1em; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  .card {{ background: white; padding: 16px; border-radius: 6px;
           box-shadow: 0 1px 2px rgba(0,0,0,.08); margin: 12px 0; }}
  .big {{ font-size: 22px; font-weight: 600; color: #1B5E20; }}
  .kpi {{ display: inline-block; padding: 8px 14px; margin: 6px 8px 0 0;
           background: #f0f0f0; border-radius: 4px; }}
  .kpi .v {{ font-size: 16px; font-weight: 600; }}
  .kpi .l {{ font-size: 11px; color: #666; text-transform: uppercase; }}
  canvas {{ max-height: 330px; }}
  table.tbl {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  table.tbl th, table.tbl td {{ padding: 4px 8px; border-bottom: 1px solid #eee;
                                  text-align: right; }}
  table.tbl th {{ background: #f3f3f3; cursor: pointer; user-select: none;
                   text-align: left; position: sticky; top: 0; }}
  table.tbl th:first-child, table.tbl td:first-child {{ text-align: left; }}
  table.tbl td:nth-child(2) {{ text-align: left; font-family: monospace;
                                 font-size: 11px; max-width: 460px;
                                 overflow: hidden; text-overflow: ellipsis;
                                 white-space: nowrap; }}
  tr.sacred {{ background: #FFFDE7; }}
  tr:hover {{ background: #EEF5FF; }}
  .muted {{ color: #888; }}
  pre {{ background: #f5f5f5; padding: 10px; border-radius: 4px;
          overflow-x: auto; font-size: 11px; }}
</style>
</head><body>

<h1>VN2 Inventory Challenge — Report (CostAware champion)</h1>

<div class="card">
  <div><b>Champion:</b> DiverseCostAwarePolicy (5-model ensemble, Optuna-tuned: alpha=0.600, mult=1.138, w_ma=0.329).
        Selected on 8-fold CV via 15-trial Optuna (TPE), one-shot sacred.</div>
  <div class="big" style="margin-top:6px;">SACRED Competition Cost: EUR {total_comp:,.2f}</div>
  <div>
    <span class="kpi"><div class="l">Total 8wk</div><div class="v">EUR {total_total:,.0f}</div></span>
    <span class="kpi"><div class="l">Setup (fixed)</div><div class="v">EUR {setup:,.0f}</div></span>
    <span class="kpi"><div class="l">Holding</div><div class="v">EUR {h:,.0f}</div></span>
    <span class="kpi"><div class="l">Shortage</div><div class="v">EUR {s:,.0f}</div></span>
    <span class="kpi"><div class="l">vs Benchmark 4,334</div><div class="v">{(total_comp-4334)/4334*100:+.1f}%</div></span>
    <span class="kpi"><div class="l">vs Bartosz #1 3,763</div><div class="v">{(total_comp-3763)/3763*100:+.1f}%</div></span>
    <span class="kpi"><div class="l">vs Oracle 333</div><div class="v">{total_comp/333.2:.1f}x</div></span>
  </div>
</div>

<h2>Forecast accuracy</h2>
<div class="grid">
  <div class="card">
    <h2>Forecast accuracy per round</h2>
    <div><canvas id="chart_accuracy"></canvas></div>
  </div>
  <div class="card">
    <h2>Forecast vs actual (per round, summed across SKUs)</h2>
    <div><canvas id="chart_fcVSactual"></canvas></div>
  </div>
</div>
<div>
  <span class="kpi"><div class="l">Avg MAE per SKU</div><div class="v">{overall_mae:.2f}</div></span>
  <span class="kpi"><div class="l">Avg WAPE</div><div class="v">{overall_wape:.1f}%</div></span>
  <span class="kpi"><div class="l">Avg bias</div><div class="v">{overall_bias:+.2f}</div></span>
</div>

<h2>NEW: Total weekly demand — full history</h2>
<div class="card">
  <h2>Total demand summed across all 599 SKUs (~3 years history)</h2>
  <canvas id="chart_history" style="max-height: 380px;"></canvas>
  <p class="muted">Yellow shaded = VAL (weeks 149-156). Red shaded = SACRED (weeks 157-164).
       <b>Dashed purple dots</b> = predictions from each of the 8 CV folds (each 8-week ahead).
       <b>Solid orange dots</b> = VAL prediction. <b>Solid dark-red dots</b> = SACRED prediction.</p>
</div>

<div class="grid">
  <div class="card">
    <h2>Sacred-window weekly demand (zoom)</h2>
    <canvas id="chart_demand"></canvas>
  </div>
  <div class="card">
    <h2>Inventory evolution (sacred)</h2>
    <canvas id="chart_inv"></canvas>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Weekly cost breakdown (sacred)</h2>
    <canvas id="chart_cost"></canvas>
  </div>
  <div class="card">
    <h2>Cumulative cost (sacred)</h2>
    <canvas id="chart_cum"></canvas>
  </div>
</div>

<div class="card">
  <h2>Champion config</h2>
  <pre>{cfg_json}</pre>
</div>

<div class="card">
  <h2>All experiments (sacred + recent)</h2>
  {tbl}
</div>

<script>
const D = {data_json};

const opts = {{
  responsive: true,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ legend: {{ position: 'top' }} }},
}};

// NEW: Total weekly demand history with CV/VAL/SACRED predictions overlaid.
{{
  const dates = D.history.dates;
  const totals = D.history.totals;
  const sStart = D.history.sacred_start_idx;
  const vStart = D.history.val_start_idx;
  // Highlight bands for VAL and SACRED windows
  const valArr = totals.map((v,i) => (i >= vStart && i < sStart) ? v : null);
  const sacArr = totals.map((v,i) => (i >= sStart) ? v : null);
  new Chart(document.getElementById("chart_history"), {{
    type: 'line',
    data: {{
      labels: dates,
      datasets: [
        {{ label: 'Actual total demand', data: totals,
           borderColor: '#1976D2', borderWidth: 1.5, pointRadius: 0,
           tension: 0.1, fill: false }},
        {{ label: 'VAL window (actual)', data: valArr,
           borderColor: '#FFB300', backgroundColor: '#FFF59D55',
           borderWidth: 2, pointRadius: 2, fill: true }},
        {{ label: 'SACRED window (actual)', data: sacArr,
           borderColor: '#D32F2F', backgroundColor: '#FFCDD255',
           borderWidth: 2, pointRadius: 2, fill: true }},
        {{ label: 'CV-fold predictions (8 folds, 8wk each)',
           data: D.history.preds_cv,
           borderColor: '#7E57C2', backgroundColor: 'transparent',
           borderWidth: 2, pointRadius: 1.5, pointBackgroundColor: '#7E57C2',
           tension: 0, fill: false, spanGaps: false, borderDash: [4,3] }},
        {{ label: 'VAL prediction', data: D.history.preds_val,
           borderColor: '#FF9800', backgroundColor: 'transparent',
           borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: '#FF9800',
           tension: 0, fill: false, spanGaps: false }},
        {{ label: 'SACRED prediction', data: D.history.preds_sacred,
           borderColor: '#B71C1C', backgroundColor: 'transparent',
           borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: '#B71C1C',
           tension: 0, fill: false, spanGaps: false }},
      ],
    }},
    options: {{ ...opts, scales: {{ x: {{ ticks: {{ maxTicksLimit: 16 }} }} }} }},
  }});
}}

// 1. Sacred week demand
new Chart(document.getElementById("chart_demand"), {{
  type: 'bar',
  data: {{
    labels: D.weeks,
    datasets: [
      {{ label: 'Demand', data: D.demand, backgroundColor: '#90CAF9' }},
      {{ type: 'line', label: 'Fulfilled', data: D.sales,
         borderColor: '#1976D2', borderWidth: 2, tension: 0.2, fill: false }},
      {{ type: 'line', label: 'Missed', data: D.missed_sales,
         borderColor: '#F44336', borderWidth: 2, tension: 0.2, fill: false, borderDash: [4,4] }},
    ],
  }},
  options: opts,
}});

// 2. Inventory
new Chart(document.getElementById("chart_inv"), {{
  type: 'line',
  data: {{
    labels: D.weeks,
    datasets: [
      {{ label: 'Start Inventory', data: D.start_inv,
         borderColor: '#43A047', backgroundColor: '#43A04733', fill: true, tension: 0.2 }},
      {{ label: 'End Inventory', data: D.end_inv,
         borderColor: '#2E7D32', borderWidth: 2, tension: 0.2, fill: false }},
    ],
  }},
  options: opts,
}});

// 3. Cost breakdown
new Chart(document.getElementById("chart_cost"), {{
  type: 'bar',
  data: {{
    labels: D.weeks,
    datasets: [
      {{ label: 'Holding', data: D.holding, backgroundColor: '#66BB6A' }},
      {{ label: 'Shortage', data: D.shortage, backgroundColor: '#EF5350' }},
    ],
  }},
  options: {{ ...opts, scales: {{ x: {{ stacked: true }}, y: {{ stacked: true }} }} }},
}});

// 4. Cumulative
new Chart(document.getElementById("chart_cum"), {{
  type: 'line',
  data: {{
    labels: D.weeks,
    datasets: [{{ label: 'Cumulative total cost', data: D.cum_total,
                   borderColor: '#7B1FA2', backgroundColor: '#7B1FA222',
                   fill: true, tension: 0.1 }}],
  }},
  options: opts,
}});

// 5. Forecast accuracy per round
new Chart(document.getElementById("chart_accuracy"), {{
  type: 'bar',
  data: {{
    labels: D.rounds.map(r => 'R' + r),
    datasets: [
      {{ label: 'WAPE %', data: D.wape, backgroundColor: '#5C6BC0', yAxisID: 'y1' }},
      {{ type: 'line', label: 'Bias (per SKU)', data: D.bias,
         borderColor: '#EF6C00', borderWidth: 2, tension: 0.2, fill: false, yAxisID: 'y2' }},
    ],
  }},
  options: {{
    ...opts,
    scales: {{
      y1: {{ position: 'left', beginAtZero: true, title: {{ display: true, text: 'WAPE %' }} }},
      y2: {{ position: 'right', grid: {{ display: false }}, title: {{ display: true, text: 'Bias' }} }},
    }},
  }},
}});

// 6. Forecast vs actual sums
new Chart(document.getElementById("chart_fcVSactual"), {{
  type: 'bar',
  data: {{
    labels: D.rounds.map(r => 'R' + r),
    datasets: [
      {{ label: 'Forecast (sum)', data: D.fc_sum, backgroundColor: '#42A5F5' }},
      {{ label: 'Actual (sum)', data: D.actual_sum, backgroundColor: '#26A69A' }},
      {{ label: 'Order placed (sum)', data: D.order_sum, backgroundColor: '#AB47BC' }},
    ],
  }},
  options: opts,
}});

function sortTable(col) {{
  const tbody = document.querySelector('#exp tbody');
  const rows = Array.from(tbody.rows);
  const asc = !tbody.dataset.lastSort || tbody.dataset.lastSort != col + 'asc';
  rows.sort((a, b) => {{
    let av = a.cells[col].textContent.replace(/,/g, '');
    let bv = b.cells[col].textContent.replace(/,/g, '');
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) {{ av = an; bv = bn; }}
    return asc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  }});
  rows.forEach(r => tbody.appendChild(r));
  tbody.dataset.lastSort = col + (asc ? 'asc' : 'desc');
}}
</script>
</body></html>
"""

    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT.resolve()}")
    print(f"  Sacred competition_cost = EUR {total_comp:,.2f}")
    print(f"  Open in browser: {OUT.as_uri()}")


if __name__ == "__main__":
    main()
