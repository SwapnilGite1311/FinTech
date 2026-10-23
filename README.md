# Fintech Trading Research

An educational research tool for testing trading ideas before risking money. The first version is a daily-bar backtester. It uses **synthetic data only when explicitly requested** and never places broker orders.

## Start

Requires Python 3.11+. The app has no runtime dependencies. Install the local package once:

```powershell
python -m pip install -e .
python -m finresearch --help
```

## Get real historical data without a broker account

The project can download NSE daily OHLCV from TejHQ's keyless endpoint. This provides a concrete research dataset without broker credentials or a trading deposit:

```powershell
python -m finresearch fetch-eod --symbol TCS --start 2021-01-01 --end 2026-10-20
```

The command saves a CSV and a `.source.json` provenance file under `data\market-data`. The downloaded values are **raw, not adjusted for splits or dividends**. A split can look like a huge loss in raw prices, so don't use this file to judge strategy returns. TejHQ publishes an adjusted-price endpoint with a free key; the current downloader intentionally fetches only the keyless raw endpoint. Keep downloaded data local. The checked-in sample is synthetic only.

TejHQ currently documents daily OHLCV coverage from 2010 for NSE. Its free adjusted-price endpoint requires a free API key, which can be considered later if we need adjustment-aware backtests. See [TejHQ dataset documentation](https://huggingface.co/datasets/tejhq/indian-markets) and [API documentation](https://tejhq.dev/docs).

The CSV input must contain date, open, high, low, close, and volume columns, with one row per trading session and prices in the same currency. The loader accepts the project's lowercase headers and NSE-style headers such as `Date`, `Open Price`, `High Price`, `Low Price`, `Close Price`, and `Total Traded Quantity`. Example invocation:

```powershell
python -m finresearch backtest --csv path\to\prices.csv --fast 5 --slow 20 --capital 10000 --cost-bps 15 --output results
```

To select moving-average periods using the oldest 60% of the data, inspect the following 20%, then report the final 20% as untouched test data:

```powershell
python -m finresearch walk-forward --csv path\to\adjusted_prices.csv --capital 10000 --cost-bps 30
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

1. Obtain and use corporate-action-adjusted data under terms that permit research simulation.
2. Compare simple strategies across multiple stocks and market periods, with realistic costs and untouched test periods.
3. Add a market-data adapter for simulated paper orders after the provider's terms explicitly allow that use.
4. Review current broker and exchange requirements before considering any live order integration.

Python is used for research iteration. C++ should be considered only if profiling later shows a measured performance bottleneck.
