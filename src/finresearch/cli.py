from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .backtest import load_bars, run_backtest, run_walk_forward, write_results, write_walk_forward_result
from .paper import replay_paper, write_paper_results
from .data_source import fetch_eod_csv
from datetime import date


def main() -> int:
    parser = argparse.ArgumentParser(prog="finresearch", description="Educational trading strategy research")
    subparsers = parser.add_subparsers(dest="command", required=True)
    backtest = subparsers.add_parser("backtest", help="run a daily-bar moving-average crossover backtest")
    backtest.add_argument("--csv", required=True, type=Path, help="OHLCV CSV file")
    backtest.add_argument("--fast", type=int, default=5, help="fast moving-average period")
    backtest.add_argument("--slow", type=int, default=20, help="slow moving-average period")
    backtest.add_argument("--capital", type=float, default=10000.0, help="starting capital in the CSV price currency")
    backtest.add_argument("--cost-bps", type=float, default=15.0, help="estimated cost per side in basis points")
    backtest.add_argument("--output", type=Path, default=Path("results"), help="directory for summary.json and trades.csv")
    walk = subparsers.add_parser("walk-forward", help="select periods on earlier data and report later untouched periods")
    walk.add_argument("--csv", required=True, type=Path, help="OHLCV CSV file")
    walk.add_argument("--capital", type=float, default=10000.0, help="starting capital in the CSV price currency")
    walk.add_argument("--cost-bps", type=float, default=30.0, help="estimated total per-side cost in basis points")
    walk.add_argument("--fast-periods", default="3,5,10", help="comma-separated fast SMA candidates")
    walk.add_argument("--slow-periods", default="15,20,30,50", help="comma-separated slow SMA candidates")
    walk.add_argument("--train-pct", type=float, default=60.0, help="earliest-data share used to choose parameters")
    walk.add_argument("--validation-pct", type=float, default=20.0, help="middle-data share used for validation")
    walk.add_argument("--output", type=Path, default=Path("results"), help="directory for walk_forward_summary.json")
    paper = subparsers.add_parser("paper-replay", help="replay a historical CSV with simulated orders; no broker connection")
    paper.add_argument("--csv", required=True, type=Path, help="OHLCV CSV file")
    paper.add_argument("--fast", type=int, default=5, help="fast moving-average period")
    paper.add_argument("--slow", type=int, default=20, help="slow moving-average period")
    paper.add_argument("--capital", type=float, default=10000.0, help="starting capital in the CSV price currency")
    paper.add_argument("--cost-bps", type=float, default=15.0, help="estimated cost per side in basis points")
    paper.add_argument("--max-allocation-pct", type=float, default=25.0, help="maximum starting-capital allocation per position")
    paper.add_argument("--max-loss-pct", type=float, default=5.0, help="halt new entries after this loss from starting capital")
    paper.add_argument("--output", type=Path, default=Path("paper-results"), help="directory for simulated order and equity logs")
    fetch = subparsers.add_parser("fetch-eod", help="download real NSE daily OHLCV data from TejHQ (no broker login)")
    fetch.add_argument("--symbol", required=True, help="NSE symbol, for example TCS")
    fetch.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    fetch.add_argument("--end", required=True, help="last date, YYYY-MM-DD")
    fetch.add_argument("--output", type=Path, default=Path("data/market-data"), help="where to save CSV and source metadata")
    args = parser.parse_args()
    try:
        if args.command == "fetch-eod":
            output_path = fetch_eod_csv(args.symbol, date.fromisoformat(args.start),
                                        date.fromisoformat(args.end), args.output)
            print(f"Downloaded end-of-day data: {output_path.resolve()}")
            print("This is raw historical data; it is not adjusted for corporate actions.")
            return 0
        bars = load_bars(args.csv)
        if args.command == "backtest":
            result = run_backtest(bars, args.fast, args.slow, args.capital, args.cost_bps)
            write_results(result, args.output)
        elif args.command == "walk-forward":
            fast_periods = [int(value.strip()) for value in args.fast_periods.split(",") if value.strip()]
            slow_periods = [int(value.strip()) for value in args.slow_periods.split(",") if value.strip()]
            result = run_walk_forward(bars, args.capital, args.cost_bps, fast_periods, slow_periods,
                                      args.train_pct, args.validation_pct)
            write_walk_forward_result(result, args.output)
        else:
            result = replay_paper(bars, args.fast, args.slow, args.capital, args.cost_bps,
                                  args.max_allocation_pct, args.max_loss_pct)
            write_paper_results(result, args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "backtest":
        print(f"Backtest complete: {result['net_return_pct']:.2f}% net return, "
              f"{result['closed_trades']} closed trades, max drawdown {result['max_drawdown_pct']:.2f}%")
    elif args.command == "walk-forward":
        print(f"Walk-forward complete (historical only): selected SMA {result['selected_fast_period']}/{result['selected_slow_period']}; "
              f"untouched test return {result['segments']['test']['strategy_return_pct']:.2f}% vs "
              f"buy-and-hold {result['segments']['test']['buy_and_hold_return_pct']:.2f}%")
    else:
        print(f"Historical paper replay complete (simulated only): {result['net_return_pct']:.2f}% net return, "
              f"{len(result['simulated_orders'])} orders, halted={result['halted']}")
    print(f"Results: {args.output.resolve()}")
    return 0
