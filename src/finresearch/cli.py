from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .backtest import load_bars, run_backtest, run_walk_forward, write_results, write_walk_forward_result
from .paper import replay_paper, write_paper_results
from .data_source import fetch_eod_csv, fetch_adjusted_eod_csv
from .model import train_and_evaluate, write_model_outputs, predict_latest
from datetime import date
import os


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
    model = subparsers.add_parser("train-model", help="train and chronologically evaluate a small daily direction model")
    model.add_argument("--csv", action="append", type=Path, help="adjusted OHLCV CSV; repeat for a research basket")
    model.add_argument("--data-dir", type=Path, help="directory of *_adjusted.csv files; trains a separate model per symbol")
    model.add_argument("--symbol", help="symbol label for one CSV; otherwise read from source metadata")
    model.add_argument("--horizon", type=int, default=5, help="sessions from next-open entry to label exit")
    model.add_argument("--capital", type=float, default=10000.0, help="paper-replay capital (not live capital)")
    model.add_argument("--cost-bps", type=float, default=30.0, help="estimated one-way cost in basis points")
    model.add_argument("--allocation-pct", type=float, default=25.0, help="maximum starting-cash allocation per simulated position")
    model.add_argument("--probability-threshold", type=float, default=0.55, help="fixed probability threshold for simulated buys")
    model.add_argument("--output", type=Path, default=Path("model-results"), help="directory for model and evaluation artifacts")
    predict_model = subparsers.add_parser("predict-model", help="score the latest row with a saved research model; no order is sent")
    predict_model.add_argument("--csv", required=True, type=Path, help="adjusted OHLCV CSV for the model symbol")
    predict_model.add_argument("--model", required=True, type=Path, help="model.json created by train-model")
    predict_model.add_argument("--symbol", help="symbol label; inferred from the CSV source metadata when available")
    predict_model.add_argument("--output", type=Path, default=Path("model-signal.json"), help="where to save the research-only prediction")
    fetch = subparsers.add_parser("fetch-eod", help="download real NSE daily OHLCV data from TejHQ (no broker login)")
    fetch.add_argument("--symbol", required=True, help="NSE symbol, for example TCS")
    fetch.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    fetch.add_argument("--end", required=True, help="last date, YYYY-MM-DD")
    fetch.add_argument("--output", type=Path, default=Path("data/market-data"), help="where to save CSV and source metadata")
    fetch_adjusted = subparsers.add_parser("fetch-adjusted", help="download split/dividend-adjusted NSE daily OHLCV")
    fetch_adjusted.add_argument("--symbol", required=True, help="NSE symbol, for example TCS")
    fetch_adjusted.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    fetch_adjusted.add_argument("--end", required=True, help="last date, YYYY-MM-DD")
    fetch_adjusted.add_argument("--output", type=Path, default=Path("data/market-data"), help="where to save CSV and source metadata")
    fetch_batch = subparsers.add_parser("fetch-adjusted-many", help="download adjusted daily data for several symbols sequentially")
    fetch_batch.add_argument("--symbols", required=True, help="comma-separated NSE symbols, for example TCS,INFY,SBIN")
    fetch_batch.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    fetch_batch.add_argument("--end", required=True, help="last date, YYYY-MM-DD")
    fetch_batch.add_argument("--output", type=Path, default=Path("data/market-data"), help="where to save CSVs and source metadata")
    args = parser.parse_args()
    try:
        if args.command == "fetch-eod":
            output_path = fetch_eod_csv(args.symbol, date.fromisoformat(args.start),
                                        date.fromisoformat(args.end), args.output)
            print(f"Downloaded end-of-day data: {output_path.resolve()}")
            print("This is raw historical data; it is not adjusted for corporate actions.")
            return 0
        if args.command == "fetch-adjusted":
            output_path = fetch_adjusted_eod_csv(
                args.symbol, date.fromisoformat(args.start), date.fromisoformat(args.end),
                args.output, os.environ.get("FINRESEARCH_TEJHQ_API_KEY", ""))
            print(f"Downloaded adjusted end-of-day data: {output_path.resolve()}")
            print("Prices are adjusted; volume is raw. Source metadata was saved beside the CSV.")
            return 0
        if args.command == "fetch-adjusted-many":
            symbols = list(dict.fromkeys(value.strip().upper() for value in args.symbols.split(",") if value.strip()))
            if not symbols:
                raise ValueError("Provide at least one comma-separated NSE symbol")
            start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
            api_key = os.environ.get("FINRESEARCH_TEJHQ_API_KEY", "")
            failures = []
            for symbol in symbols:
                try:
                    output_path = fetch_adjusted_eod_csv(symbol, start, end, args.output, api_key)
                    print(f"Downloaded adjusted data for {symbol}: {output_path.resolve()}")
                except (OSError, ValueError) as exc:
                    failures.append(f"{symbol}: {exc}")
            if failures:
                print("Some symbols failed:", file=sys.stderr)
                for failure in failures:
                    print(f"  {failure}", file=sys.stderr)
                return 2
            print(f"Completed {len(symbols)} symbols. No market or broker orders were placed.")
            return 0
        if args.command == "train-model":
            if args.csv and args.data_dir:
                raise ValueError("Use --csv or --data-dir, not both")
            csv_files = args.csv or (sorted(args.data_dir.glob("*_adjusted.csv")) if args.data_dir else [])
            if not csv_files:
                raise ValueError("Provide --csv <file> or --data-dir <directory with *_adjusted.csv files>")
            if len(csv_files) > 1 and args.symbol:
                raise ValueError("Omit --symbol for multiple CSVs; symbols are read from source metadata")
            if args.data_dir:
                best_by_symbol = {}
                for csv_path in csv_files:
                    metadata_path = csv_path.with_suffix(".source.json")
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
                    pieces = csv_path.stem.split("_")
                    symbol_key = str(metadata.get("symbol", pieces[1] if len(pieces) > 2 and pieces[0] == "NSE" else csv_path.stem)).upper()
                    row_count = int(metadata.get("row_count", 0))
                    if symbol_key not in best_by_symbol or row_count > best_by_symbol[symbol_key][0]:
                        best_by_symbol[symbol_key] = (row_count, csv_path)
                csv_files = [entry[1] for _, entry in sorted(best_by_symbol.items())]
            results = []
            # Each symbol is an independent scenario run sequentially with the
            # requested capital. This is not a combined, simultaneous basket.
            per_symbol_capital = args.capital
            for csv_path in csv_files:
                metadata_path = csv_path.with_suffix(".source.json")
                if args.symbol:
                    symbol = args.symbol
                elif metadata_path.exists():
                    symbol = json.loads(metadata_path.read_text(encoding="utf-8")).get("symbol", csv_path.stem)
                else:
                    pieces = csv_path.stem.split("_")
                    symbol = pieces[1] if len(pieces) > 2 and pieces[0] == "NSE" else csv_path.stem
                bars = load_bars(csv_path)
                report, extras, _ = train_and_evaluate(
                    bars, symbol, args.horizon, per_symbol_capital, args.cost_bps,
                    args.allocation_pct, args.probability_threshold)
                output_dir = args.output / symbol if len(csv_files) > 1 else args.output
                write_model_outputs(report, extras, output_dir)
                results.append({"symbol": symbol, "file": str(csv_path),
                                "test_accuracy_pct": report["test"]["classification"]["accuracy_pct"],
                                "test_balanced_accuracy_pct": report["test"]["classification"]["balanced_accuracy_pct"],
                                "test_roc_auc": report["test"]["classification"]["roc_auc"],
                                "test_model_return_pct": report["test"]["portfolio"]["net_return_pct"],
                                "test_buy_and_hold_return_pct": report["test"]["portfolio"]["buy_and_hold_return_pct"],
                                "test_trade_count": report["test"]["portfolio"]["closed_trades"],
                                "artifacts": str(output_dir)})
                print(f"{symbol}: test accuracy {results[-1]['test_accuracy_pct']:.1f}%, "
                      f"model {results[-1]['test_model_return_pct']:.2f}% vs "
                      f"buy-and-hold {results[-1]['test_buy_and_hold_return_pct']:.2f}%")
            basket_summary = {
                "method": "independent per-symbol models with chronological holdouts; scenarios run separately, not concurrently",
                "capital_per_simulation": round(per_symbol_capital, 2),
                "symbols": results,
                "note": "Each symbol is simulated independently using the requested capital. These outcomes cannot be combined as a simultaneous portfolio, and no broker orders were sent.",
            }
            if len(csv_files) > 1:
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / "basket_summary.json").write_text(json.dumps(basket_summary, indent=2) + "\n", encoding="utf-8")
            print(f"Research artifacts: {args.output.resolve()}")
            return 0
        bars = load_bars(args.csv)
        if args.command == "predict-model":
            artifact = json.loads(args.model.read_text(encoding="utf-8"))
            symbol = args.symbol or artifact.get("symbol", "UNKNOWN")
            metadata_path = args.csv.with_suffix(".source.json")
            if args.symbol is None and metadata_path.exists():
                symbol = json.loads(metadata_path.read_text(encoding="utf-8")).get("symbol", symbol)
            signal = predict_latest(bars, symbol, args.model)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(signal, indent=2) + "\n", encoding="utf-8")
            print(f"Research-only prediction for {symbol} as of {signal['as_of_date']}: "
                  f"{signal['action']} ({signal['probability_return_exceeds_estimated_costs']:.1%} probability)")
            print(f"No broker order sent. Saved: {args.output.resolve()}")
            return 0
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
