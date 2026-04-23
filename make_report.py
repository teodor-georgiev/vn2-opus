"""HTML report using Chart.js: total demand, costs, forecasting error, experiments table."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
OUT = ROOT / "report.html"


def load_experiments() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "benchmark" / "experiments.csv",
                     on_bad_lines="skip", engine="python")
    df["is_sacred"] = df["policy"].str.contains(r"\[SACRED", regex=True, na=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def run_champion_with_forecasts():
    """Run champion on sacred; capture per-round forecasts, orders, and actuals."""
    from simulation import InventorySimulator, LEAD_TIME
    from policies import EnsemblePolicy

    sim = InventorySimulator()
    winner = json.loads((ROOT / "best_winner.json").read_text())
    cfg = winner["config"]

    forecasts, orders, target_actuals = [], [], []

    class Captured(EnsemblePolicy):
        def __call__(self, sim_arg, round_idx, sales_hist):
            order = super().__call__(sim_arg, round_idx, sales_hist)
            # last_forecast = 2-week sum forecast per SKU (what this order will cover)
            if self.last_forecast is not None:
                forecasts.append(self.last_forecast.copy())
            orders.append(order.copy())
            # Actual 2-week-sum demand for the covered window (t+LEAD+1 .. t+LEAD+cov)
            actual_cols = sorted(sim_arg.actual_sales.columns)
            start = round_idx + LEAD_TIME
            end = start + self.coverage_weeks
            if end <= len(actual_cols):
                actual = sim_arg.actual_sales[actual_cols[start:end]].sum(axis=1)
                target_actuals.append(actual.copy())
            return order

    policy = Captured(
        coverage_weeks=cfg.get("coverage_weeks", 2),
        w_ma=cfg.get("w_ma", 0.2875),
        multiplier=cfg.get("multiplier", 1.0731),
        safety_units=cfg.get("safety_units", 0.5962),
        w_lgb_share=cfg.get("w_lgb_share", 0.6971),
        master=sim.master,
        censoring_strategy=cfg.get("censoring_strategy", "mean_impute"),
        per_series_scaling=cfg.get("per_series_scaling", False),
        demand_cluster_k=cfg.get("demand_cluster_k", None),
    )
    res = sim.run_simulation(policy)

    # Per-round forecast accuracy.
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
    return res, winner, pd.DataFrame(rounds)


def table_html(df: pd.DataFrame) -> str:
    keep = ["timestamp", "policy", "cv_mean_cost", "val_cost",
            "cv_mean_holding", "cv_mean_shortage", "fc_mae", "fc_bias",
            "is_sacred"]
    keep = [c for c in keep if c in df.columns]
    d = df[keep].copy()
    d["timestamp"] = d["timestamp"].dt.strftime("%m-%d %H:%M")
    for c in d.select_dtypes("number").columns:
        d[c] = d[c].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "")
    d["is_sacred"] = d["is_sacred"].map({True: "✓", False: ""})
    rows = []
    for _, r in d.iterrows():
        tds = "".join(f"<td>{r[c]}</td>" for c in d.columns)
        sacred_class = ' class="sacred"' if r["is_sacred"] == "✓" else ""
        rows.append(f"<tr{sacred_class}>{tds}</tr>")
    hdr = "".join(f"<th onclick=\"sortTable({i})\">{c}</th>"
                   for i, c in enumerate(d.columns))
    return f"""
<table id="exp" class="tbl">
  <thead><tr>{hdr}</tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
<p class="muted">{len(df)} rows. Click any column header to sort. Sacred rows highlighted.</p>"""


def main():
    print("Running champion on sacred (capturing forecasts)...")
    res, winner, rounds_df = run_champion_with_forecasts()
    weekly = res["weekly_log"]

    print("Loading experiments...")
    exp = load_experiments()

    # Prep chart data (JSON-ready).
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
    }
    data_json = json.dumps(data)

    print("Building HTML...")
    tbl = table_html(exp)
    cfg_json = json.dumps(winner["config"], indent=2)

    # Precomputed headline numbers.
    total_comp = res["competition_cost"]
    total_total = res["total_cost"]
    setup = res["setup_cost"]
    h = res["competition_holding"]
    s = res["competition_shortage"]
    overall_mae = float(rounds_df["mae"].mean())
    overall_wape = float(rounds_df["wape"].mean() * 100)
    overall_bias = float(rounds_df["bias"].mean())

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>VN2 Report</title>
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

<h1>VN2 Inventory Challenge — Report</h1>

<div class="card">
  <div><b>Honest submission:</b> {winner['description']}</div>
  <div class="big" style="margin-top:6px;">SACRED Competition Cost: € {total_comp:,.2f}</div>
  <div>
    <span class="kpi"><div class="l">Total 8wk</div><div class="v">€ {total_total:,.0f}</div></span>
    <span class="kpi"><div class="l">Setup (fixed)</div><div class="v">€ {setup:,.0f}</div></span>
    <span class="kpi"><div class="l">Holding</div><div class="v">€ {h:,.0f}</div></span>
    <span class="kpi"><div class="l">Shortage</div><div class="v">€ {s:,.0f}</div></span>
    <span class="kpi"><div class="l">vs Benchmark 4,334</div><div class="v">−{(4334-total_comp)/4334*100:.1f}%</div></span>
    <span class="kpi"><div class="l">vs Matias 3,765</div><div class="v">{(total_comp-3765)/3765*100:+.1f}%</div></span>
  </div>
</div>

<div class="card">
  <h2>Forecast accuracy per round</h2>
  <div>
    <span class="kpi"><div class="l">Avg MAE per SKU</div><div class="v">{overall_mae:.2f}</div></span>
    <span class="kpi"><div class="l">Avg WAPE</div><div class="v">{overall_wape:.1f}%</div></span>
    <span class="kpi"><div class="l">Avg bias</div><div class="v">{overall_bias:+.2f}</div></span>
  </div>
  <div class="grid" style="margin-top: 8px;">
    <div><canvas id="chart_accuracy"></canvas></div>
    <div><canvas id="chart_fcVSactual"></canvas></div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Total weekly demand</h2>
    <canvas id="chart_demand"></canvas>
  </div>
  <div class="card">
    <h2>Inventory evolution</h2>
    <canvas id="chart_inv"></canvas>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Weekly cost breakdown</h2>
    <canvas id="chart_cost"></canvas>
  </div>
  <div class="card">
    <h2>Cumulative cost</h2>
    <canvas id="chart_cum"></canvas>
  </div>
</div>

<div class="card">
  <h2>Winner config</h2>
  <pre>{cfg_json}</pre>
</div>

<div class="card">
  <h2>All experiments</h2>
  {tbl}
</div>

<script>
const D = {data_json};

const opts = {{
  responsive: true,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ legend: {{ position: 'top' }} }},
}};

// 1. Demand
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
    labels: D.rounds.map(r => "r" + r),
    datasets: [
      {{ label: 'MAE per SKU', data: D.mae, yAxisID: 'y', backgroundColor: '#FFA726' }},
      {{ label: 'WAPE %',       data: D.wape, yAxisID: 'y1', type: 'line',
         borderColor: '#6D4C41', borderWidth: 2, fill: false, tension: 0.2 }},
      {{ label: 'Bias',         data: D.bias, yAxisID: 'y', type: 'line',
         borderColor: '#0277BD', borderWidth: 2, fill: false, tension: 0.2 }},
    ],
  }},
  options: {{
    ...opts,
    scales: {{
      y:  {{ position: 'left',  title: {{ display: true, text: 'MAE / Bias' }} }},
      y1: {{ position: 'right', grid: {{ drawOnChartArea: false }},
             title: {{ display: true, text: 'WAPE %' }} }},
    }},
  }},
}});

// 6. Forecast sum vs actual sum per round
new Chart(document.getElementById("chart_fcVSactual"), {{
  type: 'line',
  data: {{
    labels: D.rounds.map(r => "r" + r),
    datasets: [
      {{ label: 'Forecast sum (2-wk)', data: D.fc_sum, borderColor: '#1976D2',
         backgroundColor: '#1976D222', fill: true, tension: 0.2 }},
      {{ label: 'Actual sum (2-wk)',   data: D.actual_sum, borderColor: '#E64A19',
         borderWidth: 2, fill: false, tension: 0.2 }},
      {{ type: 'bar', label: 'Order quantity', data: D.order_sum,
         backgroundColor: '#9E9E9E55' }},
    ],
  }},
  options: opts,
}});

// Table sort
function sortTable(col) {{
  const t = document.getElementById("exp");
  const tb = t.tBodies[0];
  const rows = Array.from(tb.rows);
  const asc = !(t.dataset.sortCol == col && t.dataset.sortDir == "asc");
  t.dataset.sortCol = col; t.dataset.sortDir = asc ? "asc" : "desc";
  rows.sort((a, b) => {{
    const va = a.cells[col].textContent.replace(/,/g, "").trim();
    const vb = b.cells[col].textContent.replace(/,/g, "").trim();
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tb.appendChild(r));
}}
</script>

<p class="muted">Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} · {len(exp)} experiments</p>
</body></html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
