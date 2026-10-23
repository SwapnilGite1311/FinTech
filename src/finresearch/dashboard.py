"""Generate a self-contained HTML dashboard for historical simulated trading."""

from __future__ import annotations

from html import escape
from pathlib import Path
import json

from .backtest import Bar


def _payload(bars: list[Bar], replay: dict, capital: float, cost_bps: float) -> str:
    fee = cost_bps / 10_000
    benchmark_shares = int(capital / (bars[0].close * (1 + fee)))
    benchmark_cash = capital - benchmark_shares * bars[0].close * (1 + fee)
    benchmark = [round(benchmark_cash + benchmark_shares * b.close * (1 - fee), 2) for b in bars]
    equity = [point["equity"] for point in replay["equity_curve"]]
    series = {
        "dates": [bar.date.isoformat() for bar in bars],
        "prices": [round(bar.close, 4) for bar in bars],
        "equity": equity,
        "benchmark": benchmark,
        "orders": replay["simulated_orders"],
    }
    return json.dumps(series, separators=(",", ":")).replace("<", "\\u003c")


def generate_dashboard(
    bars: list[Bar], replay: dict, output_path: Path, title: str,
    data_label: str, data_note: str, cost_bps: float,
) -> Path:
    if not bars:
        raise ValueError("Cannot create a dashboard without price bars")
    payload = _payload(bars, replay, replay["starting_capital"], cost_bps)
    start_value = replay["starting_capital"]
    end_value = replay["ending_equity"]
    ret = replay["net_return_pct"]
    direction = "positive" if ret > 0 else "negative" if ret < 0 else "neutral"
    orders = replay["simulated_orders"]
    buys = sum(order["side"] == "BUY" for order in orders)
    sells = sum(order["side"] == "SELL" for order in orders)
    max_drawdown = max((float(point["drawdown_pct"]) for point in replay["equity_curve"]), default=0.0)
    label = escape(data_label)
    note = escape(data_note)
    safe_title = escape(title)
    if orders:
        empty_or_status = f"{buys} simulated buys · {sells} simulated sells · No broker connection"
    else:
        empty_or_status = "No simulated fills for these prices and cash limits. No broker connection."

    page = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Finresearch · __TITLE__</title>
  <style>
    :root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:#172b4d;background:#f4f7fb;font-synthesis:none;text-rendering:optimizeLegibility;--ink:#172b4d;--muted:#62738a;--line:#e1e8f0;--teal:#0b8f82;--red:#df665f;--blue:#3867d6;--surface:#fff;--wash:#edf2f8}
    *{box-sizing:border-box}body{margin:0}.shell{max-width:1440px;margin:auto;padding:30px 34px 48px}.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:26px}.brand{display:flex;gap:12px;align-items:center;font-weight:750;letter-spacing:-.02em}.brand-mark{width:36px;height:36px;border-radius:11px;background:#163e63;color:#fff;display:grid;place-items:center;font-size:17px}.brand small{display:block;color:var(--muted);font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;margin-top:3px}.live{display:inline-flex;align-items:center;gap:8px;background:#e7f5f1;border:1px solid #c7e7de;border-radius:999px;padding:8px 12px;color:#08705f;font-size:12px;font-weight:700}.live i{height:7px;width:7px;background:#11a586;border-radius:99px}.heading{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:18px}.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);font-weight:750}.heading h1{font-size:28px;letter-spacing:-.04em;margin:6px 0 7px;color:var(--ink)}.sub{color:var(--muted);font-size:13px}.badge{white-space:nowrap;background:#edf3fa;border:1px solid #dce6f0;color:#3e5974;padding:8px 11px;border-radius:8px;font-size:11px;font-weight:700}.notice{border:1px solid #e9d8a5;background:#fff8e7;color:#705715;border-radius:10px;padding:12px 15px;font-size:12px;line-height:1.55;margin-bottom:18px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin-bottom:16px}.metric,.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:0 2px 8px #183b5c05}.metric{padding:17px 18px}.metric .label{font-size:11px;color:var(--muted);font-weight:650}.metric .value{font-size:24px;font-weight:760;letter-spacing:-.04em;margin-top:9px}.metric .hint{font-size:11px;color:var(--muted);margin-top:6px}.positive{color:#08705f}.negative{color:#b84343}.neutral{color:#36516f}.panel{padding:19px 20px;margin-bottom:15px}.panel-head{display:flex;justify-content:space-between;align-items:start;gap:12px;margin-bottom:10px}.panel h2{font-size:15px;margin:0;letter-spacing:-.02em}.panel p{font-size:11px;color:var(--muted);margin:5px 0 0}.legend{display:flex;gap:14px;align-items:center;color:var(--muted);font-size:11px;flex-wrap:wrap}.legend span{display:inline-flex;align-items:center;gap:6px}.swatch{width:9px;height:9px;border-radius:50%;display:inline-block}.chart{width:100%;height:auto;display:block;overflow:visible}.chart text{font-family:inherit;font-size:10px;fill:#718198}.chart .gridline{stroke:#e8edf3;stroke-width:1}.chart .price-line{fill:none;stroke:var(--blue);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}.chart .equity-line{fill:none;stroke:var(--teal);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}.chart .benchmark-line{fill:none;stroke:#94a2b4;stroke-width:1.8;stroke-dasharray:5 5}.chart .buy{fill:#e5f5ef;stroke:#0b8f82;stroke-width:2}.chart .sell{fill:#fff0ed;stroke:#df665f;stroke-width:2}.chart .marker-label{font-size:9px;font-weight:800;fill:#172b4d}.chart .baseline{stroke:#bdc9d7;stroke-width:1;stroke-dasharray:4 4}.axis-row{display:flex;justify-content:space-between;color:var(--muted);font-size:10px;padding:0 15px 0 52px;margin-top:-5px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}th{text-align:left;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;padding:11px 9px;border-bottom:1px solid var(--line)}td{padding:12px 9px;border-bottom:1px solid #edf1f5;color:#314761}tr:last-child td{border-bottom:0}.side{font-weight:800}.side.buy{color:#08705f}.side.sell{color:#b84343}.empty{padding:24px;text-align:center;color:var(--muted);font-size:12px;background:#f8fafc;border-radius:8px}.footer{font-size:11px;color:var(--muted);display:flex;justify-content:space-between;gap:14px;padding:10px 2px}.footer strong{color:#40566f}@media(max-width:800px){.shell{padding:20px 16px 34px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.heading{align-items:start;flex-direction:column}.heading h1{font-size:24px}.panel{padding:15px 12px}.panel-head{flex-direction:column}.topbar{margin-bottom:20px}}@media(max-width:420px){.metric{padding:13px}.metric .value{font-size:20px}.grid{gap:9px}.footer{flex-direction:column}}
  </style>
</head>
<body>
<main class="shell">
  <div class="topbar"><div class="brand"><div class="brand-mark">F</div><div>Finresearch<small>Paper lab · local report</small></div></div><div class="live"><i></i> SIMULATED ONLY</div></div>
  <section class="heading"><div><div class="eyebrow">Historical strategy replay</div><h1>__TITLE__</h1><div class="sub">__START__ — __END__ · SMA __FAST__ / __SLOW__ · __CAPITAL__ starting cash</div></div><span class="badge">__DATA_LABEL__</span></section>
  <div class="notice"><strong>Simulation only.</strong> This report replays historical prices; it did not place broker orders. Historical results do not predict future returns. __DATA_NOTE__</div>
  <section class="grid" aria-label="Performance summary">
    <div class="metric"><div class="label">Ending equity</div><div class="value __DIRECTION__">₹__END_VALUE__</div><div class="hint">Started with ₹__START_VALUE__</div></div>
    <div class="metric"><div class="label">Net return</div><div class="value __DIRECTION__">__RETURN__%</div><div class="hint">After estimated costs</div></div>
    <div class="metric"><div class="label">Maximum drawdown</div><div class="value">__DRAWDOWN__%</div><div class="hint">Peak-to-trough decline</div></div>
    <div class="metric"><div class="label">Simulated fills</div><div class="value">__ORDERS__</div><div class="hint">__ORDER_STATUS__</div></div>
  </section>
  <section class="panel"><div class="panel-head"><div><h2>Price and simulated orders</h2><p>Signals are decided after a daily close and simulated at the next session open.</p></div><div class="legend"><span><i class="swatch" style="background:#3867d6"></i>Adjusted close</span><span><i class="swatch" style="background:#0b8f82"></i>Buy</span><span><i class="swatch" style="background:#df665f"></i>Sell</span></div></div><svg id="price-chart" class="chart" viewBox="0 0 1000 300" role="img" aria-label="Historical adjusted price chart with simulated buy and sell markers"></svg><div class="axis-row"><span>__START__</span><span>__END__</span></div></section>
  <section class="panel"><div class="panel-head"><div><h2>Portfolio value</h2><p>Simulated portfolio after estimated exit costs, compared with whole-share buy-and-hold.</p></div><div class="legend"><span><i class="swatch" style="background:#0b8f82"></i>Strategy equity</span><span><i class="swatch" style="background:#94a2b4"></i>Buy-and-hold</span></div></div><svg id="equity-chart" class="chart" viewBox="0 0 1000 250" role="img" aria-label="Portfolio equity compared with buy and hold"></svg><div class="axis-row"><span>__START__</span><span>__END__</span></div></section>
  <section class="panel"><div class="panel-head"><div><h2>Simulated order log</h2><p>Whole shares only · estimated __COST__ bps cost per side</p></div></div><div id="orders-table"></div></section>
  <div class="footer"><span><strong>Data:</strong> __DATA_LABEL__ · __START__ to __END__</span><span>This local page makes no network requests.</span></div>
</main>
<script>
const data=__DATA__;
function chart(id,values,kind,orders=[]){
  const svg=document.getElementById(id),box=svg.viewBox.baseVal,W=box.width,H=box.height,m={l:58,r:18,t:18,b:25},pw=W-m.l-m.r,ph=H-m.t-m.b;
  const finite=values.filter(Number.isFinite),lo=Math.min(...finite),hi=Math.max(...finite),pad=(hi-lo)*.09||1,y0=lo-pad,y1=hi+pad;
  const x=i=>m.l+(i/Math.max(values.length-1,1))*pw,y=v=>m.t+(1-(v-y0)/(y1-y0))*ph;
  let out='';for(let k=0;k<5;k++){const v=y0+(y1-y0)*k/4,yy=y(v);out+=`<line class="gridline" x1="${m.l}" x2="${W-m.r}" y1="${yy}" y2="${yy}"/><text x="${m.l-8}" y="${yy+3}" text-anchor="end">${kind==='price'?'₹'+Math.round(v):'₹'+Math.round(v)}</text>`}
  const path=values.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  if(kind==='equity'){const baseline=y(data.equity[0]);out+=`<line class="baseline" x1="${m.l}" x2="${W-m.r}" y1="${baseline}" y2="${baseline}"/>`;out+=`<path class="benchmark-line" d="${data.benchmark.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')}"/>`;out+=`<path class="equity-line" d="${path}"/>`}
  else{out+=`<path class="price-line" d="${path}"/>`;for(const order of orders){const i=data.dates.indexOf(order.date);if(i<0)continue;const cls=order.side==='BUY'?'buy':'sell';const cy=y(Number(order.simulated_price));const cx=x(i);out+=`<circle class="${cls}" cx="${cx}" cy="${cy}" r="7"><title>${order.side} ${order.shares} share(s) at ₹${Number(order.simulated_price).toFixed(2)} on ${order.date}</title></circle><text class="marker-label" x="${cx}" y="${cy-11}" text-anchor="middle">${order.side==='BUY'?'B':'S'}</text>`}}
  svg.innerHTML=out;
}
chart('price-chart',data.prices,'price',data.orders);chart('equity-chart',data.equity,'equity');
const host=document.getElementById('orders-table');
if(!data.orders.length){host.innerHTML='<div class="empty">No fills occurred with this dataset and cash limit. The equity line stays in cash. Try a suitable dataset and allocation to see simulated orders.</div>'}
else{host.innerHTML='<div class="table-wrap"><table><thead><tr><th>Date</th><th>Side</th><th>Shares</th><th>Simulated price</th><th>Estimated cost</th><th>Cash after</th><th>Reason</th></tr></thead><tbody>'+data.orders.slice().reverse().map(o=>`<tr><td>${o.date}</td><td class="side ${o.side.toLowerCase()}">${o.side}</td><td>${o.shares}</td><td>₹${Number(o.simulated_price).toFixed(2)}</td><td>₹${Number(o.estimated_cost).toFixed(2)}</td><td>₹${Number(o.cash_after).toFixed(2)}</td><td>${o.reason}</td></tr>`).join('')+'</tbody></table></div>'}
</script>
</body>
</html>'''
    substitutions = {
        "__TITLE__": safe_title,
        "__START__": bars[0].date.isoformat(),
        "__END__": bars[-1].date.isoformat(),
        "__FAST__": str(replay["fast_period"]),
        "__SLOW__": str(replay["slow_period"]),
        "__CAPITAL__": f"₹{start_value:,.2f}",
        "__START_VALUE__": f"{start_value:,.2f}",
        "__END_VALUE__": f"{end_value:,.2f}",
        "__RETURN__": f"{ret:+.2f}",
        "__DIRECTION__": direction,
        "__DRAWDOWN__": f"{max_drawdown:.2f}",
        "__ORDERS__": str(len(orders)),
        "__ORDER_STATUS__": escape(empty_or_status),
        "__DATA_LABEL__": label,
        "__DATA_NOTE__": note,
        "__COST__": f"{cost_bps:g}",
        "__DATA__": payload,
    }
    for token, value in substitutions.items():
        page = page.replace(token, value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path
