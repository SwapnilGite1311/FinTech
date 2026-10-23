"""Build a standalone, offline dashboard from model replay artifacts."""

from __future__ import annotations

from pathlib import Path
import csv
import json


def _safe_json(value: object) -> str:
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))


def _symbol_for_csv(csv_path: Path) -> str:
    metadata_path = csv_path.with_suffix(".source.json")
    if metadata_path.exists():
        try:
            return str(json.loads(metadata_path.read_text(encoding="utf-8")).get("symbol", "")).upper()
        except (OSError, json.JSONDecodeError):
            pass
    parts = csv_path.stem.split("_")
    return parts[1].upper() if len(parts) > 2 and parts[0] == "NSE" else csv_path.stem.upper()


def _read_bars(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {str(name).strip().lower(): name for name in (reader.fieldnames or [])}
        aliases = {
            "date": ("date", "timestamp"), "open": ("open", "open price"),
            "high": ("high", "high price"), "low": ("low", "low price"),
            "close": ("close", "close price", "adj close"),
        }
        resolved = {}
        for key, candidates in aliases.items():
            source = next((fields[candidate] for candidate in candidates if candidate in fields), None)
            if source is None:
                raise ValueError(f"{path.name} is missing a {key} column")
            resolved[key] = source
        result = []
        for row in reader:
            try:
                result.append({"date": str(row[resolved["date"]])[:10],
                               "open": float(row[resolved["open"]]),
                               "high": float(row[resolved["high"]]),
                               "low": float(row[resolved["low"]]),
                               "close": float(row[resolved["close"]])})
            except (TypeError, ValueError):
                continue
    return result


def build_dashboard(data_dir: Path, results_dir: Path, output: Path) -> Path:
    """Combine local adjusted prices and saved model replay artifacts into one HTML file."""
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    if not results_dir.is_dir():
        raise ValueError(f"Model results directory does not exist: {results_dir}")

    csv_by_symbol: dict[str, Path] = {}
    for csv_path in sorted(data_dir.glob("*_adjusted.csv")):
        symbol = _symbol_for_csv(csv_path)
        previous = csv_by_symbol.get(symbol)
        if previous is None or csv_path.stat().st_mtime > previous.stat().st_mtime:
            csv_by_symbol[symbol] = csv_path

    payload: dict[str, dict] = {}
    for symbol_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        report_path = symbol_dir / "model_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        symbol = str(report.get("symbol", symbol_dir.name)).upper()
        csv_path = csv_by_symbol.get(symbol)
        if csv_path is None:
            continue
        bars = _read_bars(csv_path)
        split_data = {}
        for split in ("validation", "test"):
            if split not in report:
                continue
            split_report = report[split]
            orders_path = symbol_dir / f"model_{split}_orders.json"
            order_output = json.loads(orders_path.read_text(encoding="utf-8")) if orders_path.exists() else {}
            start, end = split_report["start"], split_report["end"]
            candles = [bar for bar in bars if start <= bar["date"] <= end]
            split_data[split] = {
                "start": start,
                "end": end,
                "classification": split_report["classification"],
                "portfolio": split_report["portfolio"],
                "candles": candles,
                "equity": order_output.get("equity_curve", []),
                "orders": order_output.get("orders", []),
                "trades": order_output.get("trades", []),
            }
        if split_data:
            payload[symbol] = {"data_start": report.get("data_start"), "data_end": report.get("data_end"),
                               "horizon": report.get("horizon_sessions", 5), "splits": split_data}

    if not payload:
        raise ValueError("No matching per-symbol model reports and adjusted CSV files were found")

    signals_path = results_dir / "latest_signals.json"
    latest_signals = json.loads(signals_path.read_text(encoding="utf-8")) if signals_path.exists() else []
    data_json = _safe_json({"symbols": payload, "latest_signals": latest_signals})
    html = _HTML.replace("__DASHBOARD_DATA__", data_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>Fintech · Paper Replay</title>
<style>
:root{color-scheme:dark;--bg:#080b10;--panel:#10151d;--panel2:#0c1118;--line:#202a36;--muted:#8492a4;--text:#edf3fa;--green:#36d399;--red:#ff6878;--blue:#76a9ff;--amber:#f6c66b;--radius:12px;font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);min-height:100vh}.shell{max-width:1440px;margin:auto;padding:28px 30px 52px}.top{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:22px}.brand{display:flex;align-items:center;gap:12px}.mark{display:grid;place-items:center;width:38px;height:38px;border:1px solid #2c3b4d;border-radius:10px;color:var(--green);font-size:18px;background:#0d1719}.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#95a4b6}.title{font-size:19px;font-weight:650;letter-spacing:-.025em}.topright{display:flex;align-items:center;gap:10px}.status{display:inline-flex;gap:7px;align-items:center;border:1px solid #47382a;border-radius:99px;padding:7px 11px;color:#e7c68b;background:#1b1711;font-size:11px;font-weight:600;letter-spacing:.03em}.status i{width:7px;height:7px;border-radius:50%;background:var(--amber)}.run{font:inherit;font-size:12px;font-weight:650;color:#07120e;background:var(--green);border:1px solid #65e6b0;border-radius:8px;padding:9px 13px;cursor:pointer;white-space:nowrap}.run:hover:not(:disabled){background:#72e9bb}.run:disabled{opacity:.55;cursor:not-allowed}select,.seg button{font:inherit;color:var(--text);background:#101720;border:1px solid var(--line);border-radius:8px;padding:9px 12px;outline:none}select:focus-visible,button:focus-visible{outline:2px solid var(--blue);outline-offset:2px}.runstatus{min-height:17px;text-align:right;color:var(--muted);font-size:10px}.controls{display:flex;gap:6px;align-items:center}.seg{display:flex;padding:3px;background:#0c1117;border:1px solid var(--line);border-radius:9px}.seg button{border:0;background:transparent;color:var(--muted);padding:6px 12px;cursor:pointer}.seg button.active{background:#202b39;color:var(--text);box-shadow:inset 0 0 0 1px #2c3a4a}.notice{display:flex;gap:10px;align-items:flex-start;padding:11px 14px;margin-bottom:18px;border:1px solid #403621;border-radius:10px;background:#17150f;color:#c7b888;font-size:12px}.notice b{color:#f0cf7e}.notice .icon{color:var(--amber);font-size:15px;line-height:1}.headline{display:flex;justify-content:space-between;align-items:end;gap:16px;margin:8px 0 16px}.headline h1{font-size:25px;letter-spacing:-.04em;margin:0 0 3px}.sub{color:var(--muted);font-size:12px}.period{color:#aab8c8;font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:12px}.metric{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 15px;min-height:80px}.metric .label{color:var(--muted);font-size:11px;margin-bottom:8px}.metric .value{font-size:20px;letter-spacing:-.035em;font-weight:640;font-variant-numeric:tabular-nums}.metric .hint{color:var(--muted);font-size:10px;margin-top:2px}.positive{color:var(--green)!important}.negative{color:var(--red)!important}.charts{display:grid;grid-template-columns:1.55fr 1fr;gap:12px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);min-width:0;overflow:hidden}.panelhead{height:54px;padding:0 16px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line)}.paneltitle{font-size:13px;font-weight:620}.paneldesc{font-size:10px;color:var(--muted);margin-top:2px}.legend{display:flex;align-items:center;gap:13px;font-size:10px;color:#aab7c6}.legend span{display:flex;align-items:center;gap:5px}.swatch{width:8px;height:8px;border-radius:2px;background:var(--green)}.swatch.red{background:var(--red)}.swatch.blue{background:var(--blue)}canvas{display:block;width:100%;height:300px;cursor:crosshair}.equity canvas{height:300px}.chartnote{padding:0 16px 13px;color:var(--muted);font-size:10px}.lower{display:grid;grid-template-columns:1.5fr 1fr;gap:12px;margin-top:12px}.tablewrap{overflow:auto;max-height:340px}table{border-collapse:collapse;width:100%;font-size:11px;white-space:nowrap}th{text-align:left;color:var(--muted);font-weight:500;padding:10px 13px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel2)}td{padding:9px 13px;border-bottom:1px solid #1a222c;color:#d3dce7;font-variant-numeric:tabular-nums}tr:last-child td{border-bottom:0}.side{font-size:10px;font-weight:700;letter-spacing:.04em}.side.buy{color:var(--green)}.side.sell{color:var(--red)}.empty{padding:25px;color:var(--muted);text-align:center;font-size:12px}.explain{padding:14px 16px;color:#a6b2c1;font-size:11px;line-height:1.6}.explain strong{color:#dbe6f1}.foot{margin-top:16px;color:#667487;font-size:10px;display:flex;justify-content:space-between;gap:12px}.tooltip{position:fixed;z-index:5;pointer-events:none;display:none;padding:8px 10px;border:1px solid #354355;border-radius:7px;background:#080c12ee;color:var(--text);font:10px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;box-shadow:0 8px 30px #0008}
@media(max-width:900px){.shell{padding:20px 16px 36px}.charts,.lower{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(3,1fr)}canvas{height:270px}.equity canvas{height:250px}}
@media(max-width:560px){.top{align-items:flex-start;flex-direction:column}.topright{width:100%;justify-content:space-between}.headline{align-items:flex-start;flex-direction:column}.headline h1{font-size:22px}.metrics{grid-template-columns:repeat(2,1fr)}.metric .value{font-size:18px}.panelhead{padding:0 12px}.legend{gap:8px}canvas{height:240px}.equity canvas{height:230px}.foot{flex-direction:column}}
</style></head>
<body><main class="shell">
<header class="top"><div class="brand"><div class="mark">⌁</div><div><div class="eyebrow">Fintech Research</div><div class="title">Paper Replay</div></div></div><div class="topright"><div class="status"><i></i> RESEARCH CYCLE · NO LIVE ORDERS</div><select id="symbol" aria-label="Choose stock"></select><button class="run" id="runModel" type="button">Run model · all stocks</button></div></header><div class="runstatus" id="runStatus" role="status" aria-live="polite"></div>
<div class="notice"><span class="icon">◈</span><div><b>Research replay, not live trading.</b> Orders shown here were simulated from historical daily data and filled at the next available open. The dashboard reads saved results only; it is outside the model and order path.</div></div>
<section class="headline"><div><h1><span id="headingSymbol">TCS</span> <span style="color:#6f7d8e;font-weight:450">/</span> <span id="headingSplit">Out-of-sample test</span></h1><div class="sub">Adjusted daily candles · model decisions · simulated account value</div></div><div class="controls"><div class="seg" role="group" aria-label="Evaluation period"><button data-split="validation">Validation</button><button data-split="test" class="active">Test</button></div><div class="period" id="period"></div></div></section>
<section class="panel" style="margin-bottom:12px"><div class="panelhead"><div><div class="paneltitle">Latest research signals · all stocks</div><div class="paneldesc">Scored from each local dataset; not broker orders</div></div><div class="sub" id="signalStatus">Run the model to refresh signals</div></div><div class="tablewrap"><table><thead><tr><th>Symbol</th><th>Data through</th><th>Signal</th><th>Probability</th><th>Last close</th></tr></thead><tbody id="signalsBody"></tbody></table><div class="empty" id="emptySignals">Start the local dashboard server, then click Run model to score the full symbol set.</div></div></section>
<section class="metrics" aria-label="Performance summary"><article class="metric"><div class="label">Model return</div><div id="modelReturn" class="value">—</div><div class="hint">after estimated costs</div></article><article class="metric"><div class="label">Buy &amp; hold</div><div id="benchmark" class="value">—</div><div class="hint">same period, whole shares</div></article><article class="metric"><div class="label">Simulated ending value</div><div id="ending" class="value">—</div><div id="capitalHint" class="hint">from paper capital</div></article><article class="metric"><div class="label">Balanced accuracy</div><div id="accuracy" class="value">—</div><div class="hint">direction classification</div></article><article class="metric"><div class="label">Max drawdown</div><div id="drawdown" class="value">—</div><div class="hint">peak-to-trough decline</div></article></section>
<section class="charts"><article class="panel"><div class="panelhead"><div><div class="paneltitle">Stock price</div><div class="paneldesc">Each candle shows open, high, low and close</div></div><div class="legend"><span><i class="swatch"></i>Up</span><span><i class="swatch red"></i>Down</span><span><i class="swatch blue"></i>Buy</span><span style="color:var(--red)">▼ Sell</span></div></div><canvas id="priceChart" aria-label="Historical stock candlestick chart with simulated buy and sell markers"></canvas><div class="chartnote">Marker price is the simulated fill. Hover over a candle to inspect its daily OHLC values.</div></article>
<article class="panel equity"><div class="panelhead"><div><div class="paneltitle">Simulated account value</div><div class="paneldesc">Cash + marked position value, after estimated costs</div></div><div class="legend"><span><i class="swatch blue"></i>Model equity</span></div></div><canvas id="equityChart" aria-label="Simulated account value over time"></canvas><div class="chartnote">This is portfolio equity, not the stock price. It starts with the scenario's paper capital.</div></article></section>
<section class="lower"><article class="panel"><div class="panelhead"><div><div class="paneltitle">Simulated fills</div><div class="paneldesc">Chronological orders from the selected period</div></div><div class="sub" id="tradeCount"></div></div><div class="tablewrap"><table><thead><tr><th>Date</th><th>Side</th><th>Shares</th><th>Fill price</th><th>Model probability</th><th>Reason</th></tr></thead><tbody id="ordersBody"></tbody></table><div class="empty" id="emptyOrders" hidden>No simulated fills in this period.</div></div></article><article class="panel"><div class="panelhead"><div><div class="paneltitle">How to read this view</div><div class="paneldesc">The two graphs answer different questions</div></div></div><div class="explain"><strong>Stock price:</strong> candles show the market's adjusted daily price range; blue markers are simulated buys, red markers are simulated sells.<br><br><strong>Account value:</strong> the line shows how the ₹10,000 research scenario changed while holding cash or shares. It is not a prediction of future value.<br><br><strong>Accuracy:</strong> balanced accuracy averages correct up/down detection so a majority class cannot inflate the score. Near 50% is close to chance.<br><br><strong>Limit:</strong> daily historical replay is not millisecond trading or broker paper trading. Past performance does not guarantee future results.</div></article></section>
<footer class="foot"><span>Local standalone report · data and charts are embedded · no external chart service</span><span id="sourceRange"></span></footer>
</main><div class="tooltip" id="tooltip"></div>
<script>
const APP_DATA=__DASHBOARD_DATA__,DATA=APP_DATA.symbols||{},SIGNALS=APP_DATA.latest_signals||[];
const palette={green:'#36d399',red:'#ff6878',blue:'#76a9ff',muted:'#8492a4',line:'#24303d',text:'#edf3fa'};
const money=n=>'₹'+Number(n||0).toLocaleString('en-IN',{maximumFractionDigits:0});
const pct=n=>`${Number(n||0)>=0?'+':''}${Number(n||0).toFixed(2)}%`;
const select=document.getElementById('symbol');
Object.keys(DATA).sort().forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;select.append(o)});
select.value=DATA.TCS?'TCS':Object.keys(DATA).sort()[0];let split='test';
function setText(id,v){document.getElementById(id).textContent=v}
function fmt(n,d=2){return Number(n).toLocaleString('en-IN',{minimumFractionDigits:d,maximumFractionDigits:d})}
function setupCanvas(canvas){const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.floor(rect.width*ratio));canvas.height=Math.max(1,Math.floor(rect.height*ratio));const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);return {ctx,w:rect.width,h:rect.height}}
function axes(ctx,w,h,min,max,labels){const L=62,R=15,T=16,B=27,plotW=w-L-R,plotH=h-T-B;ctx.font='10px ui-monospace,Consolas,monospace';ctx.lineWidth=1;ctx.strokeStyle=palette.line;ctx.fillStyle=palette.muted;ctx.textAlign='right';ctx.textBaseline='middle';for(let i=0;i<5;i++){const y=T+i*plotH/4;ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(w-R,y);ctx.stroke();const v=max-(max-min)*i/4;ctx.fillText(labels(v),L-8,y)}return {L,R,T,B,plotW,plotH,min,max,x:i=>L+(i+.5)*plotW, y:v=>T+(max-v)/(max-min||1)*plotH}}
function drawPrice(series){const c=document.getElementById('priceChart'),{ctx,w,h}=setupCanvas(c),bars=series.candles||[];ctx.clearRect(0,0,w,h);if(!bars.length){ctx.fillStyle=palette.muted;ctx.fillText('No price bars for this period',20,30);return}let lo=Math.min(...bars.map(b=>b.low)),hi=Math.max(...bars.map(b=>b.high)),pad=(hi-lo)*.08||1;lo-=pad;hi+=pad;const a=axes(ctx,w,h,lo,hi,v=>fmt(v));const step=a.plotW/bars.length,candleW=Math.max(2,Math.min(10,step*.58));bars.forEach((b,i)=>{const x=a.x(i),up=b.close>=b.open,color=up?palette.green:palette.red;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=Math.max(1,Math.min(2,step*.15));ctx.beginPath();ctx.moveTo(x,a.y(b.high));ctx.lineTo(x,a.y(b.low));ctx.stroke();const top=a.y(Math.max(b.open,b.close)),bottom=a.y(Math.min(b.open,b.close)),bodyH=Math.max(1,bottom-top);ctx.fillRect(x-candleW/2,top,candleW,bodyH)});const index=new Map(bars.map((b,i)=>[b.date,i]));(series.orders||[]).forEach(o=>{const i=index.get(o.date);if(i===undefined)return;const x=a.x(i),y=a.y(Number(o.price));ctx.beginPath();if(o.side==='BUY'){ctx.moveTo(x,y-10);ctx.lineTo(x-5,y-3);ctx.lineTo(x+5,y-3);ctx.closePath();ctx.fillStyle=palette.blue}else{ctx.moveTo(x,y+10);ctx.lineTo(x-5,y+3);ctx.lineTo(x+5,y+3);ctx.closePath();ctx.fillStyle=palette.red}ctx.fill();ctx.strokeStyle='#080b10';ctx.lineWidth=1;ctx.stroke()});ctx.fillStyle=palette.muted;ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.fillText(bars[0].date, a.L,h-7);ctx.textAlign='right';ctx.fillText(bars[bars.length-1].date,w-a.R,h-7);c._chart={bars,axes:a};}
function drawEquity(series){const c=document.getElementById('equityChart'),{ctx,w,h}=setupCanvas(c),points=series.equity||[];ctx.clearRect(0,0,w,h);if(!points.length){ctx.fillStyle=palette.muted;ctx.fillText('No account-value points',20,30);return}const values=points.map(p=>Number(p.equity)),capital=Number(series.portfolio.starting_capital)||values[0],lo=Math.min(...values,capital),hi=Math.max(...values,capital),pad=(hi-lo)*.16||Math.max(20,capital*.01);const a=axes(ctx,w,h,lo-pad,hi+pad,v=>money(v));const y0=a.y(capital);ctx.save();ctx.setLineDash([4,4]);ctx.strokeStyle='#526173';ctx.beginPath();ctx.moveTo(a.L,y0);ctx.lineTo(w-a.R,y0);ctx.stroke();ctx.restore();ctx.beginPath();points.forEach((p,i)=>{const x=a.x(i),y=a.y(Number(p.equity));i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=palette.blue;ctx.lineWidth=2;ctx.stroke();const gradient=ctx.createLinearGradient(0,a.T,0,h-a.B);gradient.addColorStop(0,'#76a9ff30');gradient.addColorStop(1,'#76a9ff00');ctx.lineTo(a.x(points.length-1),h-a.B);ctx.lineTo(a.x(0),h-a.B);ctx.closePath();ctx.fillStyle=gradient;ctx.fill();ctx.fillStyle=palette.muted;ctx.textAlign='left';ctx.fillText(points[0].date,a.L,h-7);ctx.textAlign='right';ctx.fillText(points.at(-1).date,w-a.R,h-7);ctx.textAlign='left';ctx.fillStyle='#98a6b7';ctx.fillText('— start capital',a.L+6,Math.max(a.T+10,y0-6))}
function renderSignals(){const body=document.getElementById('signalsBody');body.replaceChildren();document.getElementById('emptySignals').hidden=SIGNALS.length>0;setText('signalStatus',SIGNALS.length?`${SIGNALS.length} symbols · latest local data`:'Run the model to refresh signals');SIGNALS.forEach(s=>{const tr=document.createElement('tr'),values=[s.symbol,s.as_of_date,s.action,`${(s.probability_return_exceeds_estimated_costs*100).toFixed(1)}%`,money(s.last_close)];values.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===2)td.className='side '+(s.action==='BUY_IF_FLAT'?'buy':s.action==='SELL_IF_HOLDING'?'sell':'');tr.append(td)});body.append(tr)})}
function render(){const symbol=select.value,entry=DATA[symbol],s=entry.splits[split];setText('headingSymbol',symbol);setText('headingSplit',split==='test'?'Out-of-sample test':'Validation');setText('period',`${s.start} — ${s.end}`);setText('sourceRange',`Adjusted data: ${entry.data_start} — ${entry.data_end}`);setText('modelReturn',pct(s.portfolio.net_return_pct));document.getElementById('modelReturn').className='value '+(s.portfolio.net_return_pct>=0?'positive':'negative');setText('benchmark',pct(s.portfolio.buy_and_hold_return_pct));document.getElementById('benchmark').className='value '+(s.portfolio.buy_and_hold_return_pct>=0?'positive':'negative');setText('ending',money(s.portfolio.ending_equity));setText('capitalHint',`from ${money(s.portfolio.starting_capital)} scenario capital`);setText('accuracy',pct(s.classification.balanced_accuracy_pct));setText('drawdown',`${Number(s.portfolio.max_drawdown_pct||0).toFixed(2)}%`);setText('tradeCount',`${(s.orders||[]).length} simulated fills`);drawPrice(s);drawEquity(s);const body=document.getElementById('ordersBody');body.replaceChildren();const orders=[...(s.orders||[])].reverse();document.getElementById('emptyOrders').hidden=orders.length>0;orders.forEach(o=>{const tr=document.createElement('tr'),values=[o.date,o.side,String(o.shares),money(o.price),o.probability==null?'—':`${(o.probability*100).toFixed(1)}%`,o.reason];values.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===1)td.className='side '+(o.side==='BUY'?'buy':'sell');tr.append(td)});body.append(tr)})}
renderSignals();const runButton=document.getElementById('runModel'),runStatus=document.getElementById('runStatus');if(location.protocol==='file:'||!['127.0.0.1','localhost'].includes(location.hostname)){runButton.disabled=true;runButton.title='Start python -m finresearch dashboard-server and open its localhost URL to enable this button.'}runButton.addEventListener('click',async()=>{runButton.disabled=true;runStatus.textContent='Training and scoring every available symbol…';try{const response=await fetch('/api/run-model',{method:'POST'});const result=await response.json();if(!response.ok)throw new Error(result.error||'Model cycle failed');runStatus.textContent=`Completed ${result.signal_count} symbols. Reloading results…`;location.reload()}catch(error){runStatus.textContent=`Could not complete model run: ${error.message}`;runButton.disabled=false}});
document.querySelectorAll('[data-split]').forEach(b=>b.addEventListener('click',()=>{split=b.dataset.split;document.querySelectorAll('[data-split]').forEach(x=>x.classList.toggle('active',x===b));render()}));select.addEventListener('change',render);window.addEventListener('resize',render);
const tooltip=document.getElementById('tooltip'),priceCanvas=document.getElementById('priceChart');priceCanvas.addEventListener('pointermove',e=>{const c=priceCanvas._chart;if(!c)return;const r=priceCanvas.getBoundingClientRect(),x=e.clientX-r.left,a=c.axes,i=Math.max(0,Math.min(c.bars.length-1,Math.floor((x-a.L)/a.plotW*c.bars.length))),b=c.bars[i];if(!b)return;tooltip.innerHTML=`<b>${b.date}</b><br>Open ${fmt(b.open)} · High ${fmt(b.high)}<br>Low ${fmt(b.low)} · Close ${fmt(b.close)}`;tooltip.style.display='block';tooltip.style.left=Math.min(e.clientX+12,innerWidth-190)+'px';tooltip.style.top=Math.min(e.clientY+12,innerHeight-70)+'px'});priceCanvas.addEventListener('pointerleave',()=>tooltip.style.display='none');render();
</script></body></html>'''
