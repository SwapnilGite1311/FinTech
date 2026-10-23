"""A deliberately small, long-only daily-bar backtester."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
import csv
import json
import math


@dataclass(frozen=True)
class Bar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float


def load_bars(path: Path) -> list[Bar]:
    aliases = {
        "date": {"date", "tradedate"},
        "open": {"open", "openprice"},
        "high": {"high", "highprice"},
        "low": {"low", "lowprice"},
        "close": {"close", "closeprice"},
        "volume": {"volume", "totaltradedquantity", "tradedquantity"},
    }
    bars: list[Bar] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV is missing its header row")
        normalized_headers = {
            "".join(char.lower() for char in name.strip() if char.isalnum()): name
            for name in reader.fieldnames if name
        }
        columns = {
            field: next((normalized_headers[alias] for alias in accepted if alias in normalized_headers), None)
            for field, accepted in aliases.items()
        }
        missing = [field for field, column in columns.items() if column is None]
        if missing:
            raise ValueError(
                "CSV needs date, open, high, low, close, and volume columns. "
                f"Missing: {', '.join(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                raw_date = row[columns["date"]].strip()
                try:
                    parsed_date = date.fromisoformat(raw_date)
                except ValueError:
                    parsed_date = datetime.strptime(raw_date, "%d-%b-%Y").date()

                def number(field: str) -> float:
                    # NSE CSVs may format quantities using Indian digit grouping.
                    return float(row[columns[field]].strip().replace(",", ""))

                bar = Bar(
                    date=parsed_date,
                    open=number("open"), high=number("high"),
                    low=number("low"), close=number("close"),
                    volume=number("volume"),
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid data on CSV row {row_number}: {exc}") from exc
            values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"Non-finite number on CSV row {row_number}")
            if min(bar.open, bar.high, bar.low, bar.close) <= 0 or bar.volume < 0:
                raise ValueError(f"Prices must be positive and volume nonnegative (row {row_number})")
            if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close):
                raise ValueError(f"Inconsistent OHLC values on CSV row {row_number}")
            if bars and bar.date <= bars[-1].date:
                raise ValueError("CSV dates must be strictly increasing with no duplicates")
            bars.append(bar)
    if not bars:
        raise ValueError("CSV contains no price rows")
    return bars


def run_backtest(bars: list[Bar], fast: int, slow: int, capital: float, cost_bps: float,
                 warmup_bars: list[Bar] | None = None) -> dict:
    if fast < 1 or slow <= fast:
        raise ValueError("Periods must satisfy 1 <= fast < slow")
    if capital <= 0 or not math.isfinite(capital):
        raise ValueError("Capital must be a positive finite number")
    if cost_bps < 0 or not math.isfinite(cost_bps) or cost_bps >= 10000:
        raise ValueError("Cost basis points must be finite and between 0 and 10000")

    history = warmup_bars or []
    if history and history[-1].date >= bars[0].date:
        raise ValueError("Warm-up bars must end before the evaluated bars begin")
    closes = [bar.close for bar in history] + [bar.close for bar in bars]
    cash = capital
    shares = 0
    entry_date = ""
    entry_price = 0.0
    entry_fee = 0.0
    trades: list[Trade] = []
    equity_curve: list[float] = []
    fee_rate = cost_bps / 10000

    # Signal computed at close[i-1] is filled at open[i].
    offset = len(history)
    for i, bar in enumerate(bars):
        global_i = offset + i
        if global_i > 0 and global_i >= slow:
            prior_closes = closes[:global_i]
            fast_sma = sum(prior_closes[-fast:]) / fast
            slow_sma = sum(prior_closes[-slow:]) / slow
            want_long = fast_sma > slow_sma
            if want_long and shares == 0:
                shares = int(cash / (bar.open * (1 + fee_rate)))
                if shares:
                    gross = shares * bar.open
                    entry_fee = gross * fee_rate
                    cash -= gross + entry_fee
                    entry_date, entry_price = bar.date.isoformat(), bar.open
            elif not want_long and shares:
                gross = shares * bar.open
                exit_fee = gross * fee_rate
                cash += gross - exit_fee
                pnl = (gross - exit_fee) - (shares * entry_price + entry_fee)
                trades.append(Trade(entry_date, bar.date.isoformat(), entry_price, bar.open,
                                    shares, pnl, pnl / (shares * entry_price + entry_fee) * 100))
                shares = 0
                entry_date = ""
                entry_price = entry_fee = 0.0
        # Mark the open position at estimated liquidation value so its exit cost
        # is reflected even when the test ends before a sell signal arrives.
        equity_curve.append(cash + shares * bar.close * (1 - fee_rate))

    final_equity = equity_curve[-1]
    peak = capital
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    daily_returns = [equity_curve[i] / equity_curve[i - 1] - 1
                     for i in range(1, len(equity_curve)) if equity_curve[i - 1] > 0]
    mean_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0.0
    variance = (sum((value - mean_return) ** 2 for value in daily_returns) / (len(daily_returns) - 1)
                if len(daily_returns) > 1 else 0.0)
    volatility = math.sqrt(variance) * math.sqrt(252)
    daily_std = math.sqrt(variance)
    elapsed_days = max((bars[-1].date - bars[0].date).days, 1)
    annualized_return = (final_equity / capital) ** (365.25 / elapsed_days) - 1 if final_equity > 0 else -1.0
    buy_hold_end = capital * bars[-1].close / bars[0].close * (1 - fee_rate) ** 2
    winners = [trade.pnl for trade in trades if trade.pnl > 0]
    losers = [trade.pnl for trade in trades if trade.pnl < 0]
    gross_profit, gross_loss = sum(winners), abs(sum(losers))
    return {
        "strategy": "moving_average_crossover",
        "fast_period": fast,
        "slow_period": slow,
        "start_date": bars[0].date.isoformat(),
        "end_date": bars[-1].date.isoformat(),
        "starting_capital": round(capital, 2),
        "ending_equity": round(final_equity, 2),
        "net_return_pct": round((final_equity / capital - 1) * 100, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "annualized_return_pct": round(annualized_return * 100, 4),
        "annualized_volatility_pct": round(volatility * 100, 4),
        "sharpe_ratio_zero_risk_free": round(mean_return / daily_std * math.sqrt(252), 4) if daily_std else 0.0,
        "buy_and_hold_return_pct": round((buy_hold_end / capital - 1) * 100, 4),
        "closed_trades": len(trades),
        "win_rate_pct": round(len(winners) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else (None if not gross_profit else "infinite"),
        "cost_bps_per_side": cost_bps,
        "open_position_shares": shares,
        "open_position_mark": round(shares * bars[-1].close * (1 - fee_rate), 2),
        "trades": [asdict(trade) for trade in trades],
        "equity_curve": [round(value, 2) for value in equity_curve],
    }


def run_walk_forward(bars: list[Bar], capital: float, cost_bps: float,
                     fast_periods: list[int], slow_periods: list[int],
                     train_pct: float = 60.0, validation_pct: float = 20.0) -> dict:
    """Choose moving-average periods on the earliest segment; report later untouched segments."""
    if not 0 < train_pct < 100 or not 0 < validation_pct < 100 or train_pct + validation_pct >= 100:
        raise ValueError("Train and validation percentages must be positive and leave a test segment")
    if len(bars) < 30:
        raise ValueError("Walk-forward analysis needs at least 30 daily bars")
    split_train = int(len(bars) * train_pct / 100)
    split_validation = int(len(bars) * (train_pct + validation_pct) / 100)
    train_bars, validation_bars, test_bars = bars[:split_train], bars[split_train:split_validation], bars[split_validation:]
    if not train_bars or not validation_bars or not test_bars:
        raise ValueError("Date range is too short for the requested chronological splits")
    candidates = []
    for fast in sorted(set(fast_periods)):
        for slow in sorted(set(slow_periods)):
            if fast >= 1 and slow > fast and slow < len(train_bars):
                result = run_backtest(train_bars, fast, slow, capital, cost_bps)
                candidates.append({"fast": fast, "slow": slow, "train_return_pct": result["net_return_pct"]})
    if not candidates:
        raise ValueError("No valid fast/slow period pairs fit the training segment")
    selected = max(candidates, key=lambda row: (row["train_return_pct"], -row["fast"], -row["slow"]))
    fast, slow = selected["fast"], selected["slow"]
    validation = run_backtest(validation_bars, fast, slow, capital, cost_bps, train_bars)
    test = run_backtest(test_bars, fast, slow, capital, cost_bps, bars[:split_validation])
    return {
        "method": "chronological_train_validation_test",
        "selection_rule": "highest net return on training segment; no retraining on validation or test",
        "selected_fast_period": fast,
        "selected_slow_period": slow,
        "candidate_count": len(candidates),
        "segments": {
            "train": {"start_date": train_bars[0].date.isoformat(), "end_date": train_bars[-1].date.isoformat(),
                      "bars": len(train_bars), "selected_strategy_return_pct": selected["train_return_pct"]},
            "validation": {"start_date": validation_bars[0].date.isoformat(), "end_date": validation_bars[-1].date.isoformat(),
                           "bars": len(validation_bars), "strategy_return_pct": validation["net_return_pct"],
                           "buy_and_hold_return_pct": validation["buy_and_hold_return_pct"]},
            "test": {"start_date": test_bars[0].date.isoformat(), "end_date": test_bars[-1].date.isoformat(),
                     "bars": len(test_bars), "strategy_return_pct": test["net_return_pct"],
                     "buy_and_hold_return_pct": test["buy_and_hold_return_pct"],
                     "max_drawdown_pct": test["max_drawdown_pct"], "closed_trades": test["closed_trades"]},
        },
        "warning": "A single historical split is not evidence of future profitability; results depend on data quality, corporate actions, and cost assumptions.",
    }


def write_results(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key not in {"trades", "equity_curve"}}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    trade_rows = result["trades"]
    with (output_dir / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(Trade.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trade_rows)


def write_walk_forward_result(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "walk_forward_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
