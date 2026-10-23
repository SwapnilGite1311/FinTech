# Fintech Trading Research

An educational research tool for testing trading ideas before risking money. The first version is a daily-bar backtester. It uses **synthetic data only when explicitly requested** and never places broker orders.

## Start

Requires Python 3.11+. Install the local package and its scikit-learn model dependency once:

```powershell
python -m pip install -e .
python -m finresearch --help
```

## Train the first prediction model

The first model is an interpretable logistic-regression baseline. It estimates whether the next five-session return, entering at the next open, will exceed the estimated round-trip costs. Inputs are daily returns, moving-average distances, volatility, RSI, and relative volume. It is not pretrained.

Train and evaluate it using a chronological 60/20/20 split:

```powershell
python -m finresearch train-model --csv .\data\market-data\NSE_TCS_2021-01-01_2025-12-31_adjusted.csv --symbol TCS --horizon 5 --capital 10000 --cost-bps 30 --allocation-pct 50 --output model-results
```

Labels whose five-session outcome crosses a split boundary are excluded. Scaling is fit only on training data, and rolling-origin checks use a five-session gap. A small set of probability thresholds and optional 3%/5% protective stops is screened across the training rolling-origin folds and the validation segment. A candidate must be profitable in every segment, have at least three validation trades, and keep drawdown within `--max-drawdown-pct` (default 5%) in every segment to be selected; if no candidate qualifies, the model stands aside. The stop uses daily OHLC estimates and can gap through its threshold. This validation policy can still overfit. Results include net P&L, drawdown, win rate, profit factor, largest trade loss, and a no-stop baseline for comparison. The 2025 test data in the current project has already been inspected and must be treated as an audit period, not a pristine final holdout. The separate saved `model.json` is refit on all available labeled history for exploratory scoring only.

Score the latest row in a CSV with the saved research model:

```powershell
python -m finresearch predict-model --csv .\data\market-data\NSE_TCS_2021-01-01_2025-12-31_adjusted.csv --model .\model-results\model.json --output .\model-signal.json
```

This prints `BUY_IF_FLAT`, `SELL_IF_HOLDING`, or `HOLD`; it only saves a research prediction and sends no order. The file above ends in 2025, so its latest-row prediction is historical and stale. Download a fresh adjusted CSV before interpreting any latest-row score. `--capital` is simulated research capital; default ₹10,000. The project does not connect to a broker or place orders.

The current TCS file is only a first prototype dataset. Its 2025 audit period was close to chance and did not establish a reliable edge. One stock across 2021–2025 is not enough to justify live trading. The current validation drawdown filter is a research selection rule, not a guarantee or hard live risk limit. Future data after 2025 is needed for a genuinely unseen final evaluation. Live capital remains capped at ₹500 if a broker feature is ever considered; no such feature exists now.

## Get real historical data without a broker account

The project can download NSE daily OHLCV from TejHQ without broker credentials or a trading deposit:

```powershell
python -m finresearch fetch-eod --symbol TCS --start 2021-01-01 --end 2026-10-20
```

The command saves a CSV and a `.source.json` provenance file under `data\market-data`. The downloaded values are **raw, not adjusted for splits or dividends**. A split can look like a huge loss in raw prices, so don't use this file to judge strategy returns. Use `fetch-adjusted` below for an adjusted series. Keep downloaded data local. The checked-in sample is synthetic only.

TejHQ documents daily OHLCV coverage from 2010 for NSE. See [TejHQ dataset documentation](https://huggingface.co/datasets/tejhq/indian-markets) and [API documentation](https://tejhq.dev/docs).

For split- and dividend-adjusted data, create a free key at [tejhq.dev/keys](https://tejhq.dev/keys), then set it for the current PowerShell window and download the adjusted file:

```powershell
$env:FINRESEARCH_TEJHQ_API_KEY = "paste-your-key-here"
python -m finresearch fetch-adjusted --symbol TCS --start 2021-01-01 --end 2025-12-31
```

The key is read from the environment and is not written to disk. The command joins the adjusted close to raw OHLCV by date and scales open/high/low by the same provider factor; volume remains raw. It saves source metadata next to the CSV, under the ignored local `data\market-data` folder. Check the provider and applicable exchange terms before using the feed for simulation or commercial purposes.

To build a small research basket, set the key in the same PowerShell window, fetch adjusted history for each symbol, then fit independent per-symbol models. The example symbols are a data-pipeline example, not stock recommendations:

```powershell
python -m finresearch fetch-adjusted-many --symbols TCS,INFY,RELIANCE,SBIN,ITC,HDFCBANK,ICICIBANK,LT,AXISBANK,HINDUNILVR --start 2021-01-01 --end 2025-12-31
python -m finresearch train-model --data-dir .\data\market-data --horizon 5 --capital 10000 --cost-bps 30 --allocation-pct 50 --output .\model-basket
```

The basket command keeps one file per symbol (choosing the file with the most rows if duplicate date ranges exist) and runs each symbol as a separate scenario using the requested capital, reporting results separately in `basket_summary.json`. The scenarios run independently, not simultaneously: each one may use up to ₹10,000, but they do not represent one combined ten-stock portfolio. This is historical replay, not broker paper trading. A 2021–2025 basket also has survivorship and sample-size limits; it cannot establish future profitability.

For a one-click research run across every adjusted symbol in the local data folder, start the local dashboard server:

```powershell
python -m finresearch dashboard-server --data-dir .\data\market-data --results-dir .\model-basket --output .\dashboard-paper-replay.html
```

Keep that PowerShell window open and browse to `http://127.0.0.1:8765`. Click **Run model · all stocks** to retrain/evaluate each model on its local adjusted CSV, score the latest available row, save the research-only signals, and refresh the charts. The button runs across all symbols, not just the currently selected chart. Use **Play historical replay** under the price graph to animate the saved test candles and reveal simulated buy/sell fills one session at a time. The **Paper portfolio · profit / loss** panel reports net account P&L after estimated costs for the selected stock and updates during replay. Each stock has a separate scenario capital amount; these are not a combined portfolio. This uses local data only; download updated prices first if the local files are stale. It does not connect to a broker, refresh market data, or place paper/live orders.

### Optional live price monitor (read-only)

The dashboard has a separate live price graph backed by Upstox's V3 market-data WebSocket. It does not place orders. The current daily model is deliberately **not applied** to the live ticks: it was trained on daily bars and needs a separate intraday dataset, model, and evaluation before its intraday scores are meaningful. To enable the chart, create an Upstox Analytics Token from the developer apps page and set it in the current PowerShell window (never paste it into source code or chat):

```powershell
$env:FINRESEARCH_UPSTOX_ANALYTICS_TOKEN = "paste-read-only-token-here"
# Optional: default instrument is NSE_EQ|INE467B01029 (TCS)
$env:FINRESEARCH_UPSTOX_INSTRUMENT_KEYS = "NSE_EQ|INE467B01029"
python -m pip install -e .
python -m finresearch dashboard-server --data-dir .\data\market-data --results-dir .\model-basket --output .\dashboard-paper-replay.html
```

The monitor needs a valid read-only token and network access; it only displays real feed data when the market is publishing ticks. It appends received ticks to the ignored local folder `data\live-capture` for later research. The token is kept in memory and is not written to logs or project files. Upstox documents Analytics Tokens as read-only, valid for one year, and usable with WebSocket feeds. Check current account and API terms directly with Upstox before setup. This feed is only a visualization/data-collection step, not a profitable strategy test or broker paper-trading account.

To export a static dashboard file without the run button enabled, use `model-dashboard`. It shows adjusted-price candles with simulated buy/sell fill markers, a separate account equity chart, model metrics, and the simulated order log.

The CSV input must contain date, open, high, low, close, and volume columns, with one row per trading session and prices in the same currency. The loader accepts the project's lowercase headers and NSE-style headers such as `Date`, `Open Price`, `High Price`, `Low Price`, `Close Price`, and `Total Traded Quantity`. Example invocation:

```powershell
python -m finresearch backtest --csv path\to\prices.csv --fast 5 --slow 20 --capital 10000 --cost-bps 15 --output results
```

To select moving-average periods using the oldest 60% of the adjusted data, inspect the following 20%, then report the final 20% as untouched test data:

```powershell
python -m finresearch walk-forward --csv .\data\market-data\NSE_TCS_2021-01-01_2025-12-31_adjusted.csv --capital 10000 --cost-bps 30
```

This is a chronological baseline, not machine learning and not evidence of future profit. The selected periods are chosen by training return, so trying many candidates can overfit. The untouched test segment and buy-and-hold comparison make that visible. The cost input is only a rough per-side estimate, not a complete Indian brokerage, tax, or exchange-fee calculation.

To exercise the simulated order workflow against the same historical CSV, with a 25% position cap and a 5% loss stop:

```powershell
python -m finresearch paper-replay --csv path\to\prices.csv --fast 5 --slow 20 --capital 10000 --cost-bps 15 --max-allocation-pct 25 --max-loss-pct 5 --output paper-results
```

To try the command without finding market data first, use `data\demo_prices.csv`. This file contains synthetic demo data and must not be used to judge a strategy's performance:

```powershell
python -m finresearch paper-replay --csv .\data\demo_prices.csv --capital 10000
```

This is **historical replay, not live paper trading**. It creates no broker connection or real orders. It writes `paper_summary.json`, `paper_orders.csv`, and `paper_equity.csv`.

The moving-average crossover strategy calculates its signal from each session's close and fills at the **next session's open** to avoid using future information. Cost basis points are charged on both entry and exit as a single configurable estimate. Outputs are `summary.json` and `trades.csv` in the chosen output folder.

## What this version does and does not model

- Long-only, one position at a time; no leverage or short selling.
- Daily OHLCV data and next-open fills. It is not a millisecond or intraday simulator.
- Estimated proportional costs only. It does not yet model taxes, exchange-specific charges, spread, market impact, partial fills, or broker/API failures.
- Results are historical simulations, not forecasts or promises of profit. Data quality and realistic costs matter.
- The paper replay loss stop halts new entries and schedules any open position to exit at the next available session open. If the CSV ends first, the position remains marked to market and the summary flags the pending exit.

## Roadmap

1. Confirm the provider's and exchange's terms cover our intended research and simulation use.
2. Compare simple strategies across multiple stocks and market periods, with realistic costs and untouched test periods.
3. Add a market-data adapter for simulated paper orders after the provider's terms explicitly allow that use.
4. Review current broker and exchange requirements before considering any live order integration.

Python is used for research iteration. C++ should be considered only if profiling later shows a measured performance bottleneck.
