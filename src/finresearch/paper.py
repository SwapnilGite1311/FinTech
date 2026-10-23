"""Historical CSV replay with simulated orders only; never connects to a broker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math

from .backtest import Bar


@dataclass(frozen=True)
class PaperOrder:
    date: str
    side: str
    shares: int
    simulated_price: float
    estimated_cost: float
    cash_after: float
    position_after: int
    reason: str


def replay_paper(
    bars: list[Bar], fast: int, slow: int, capital: float,
    cost_bps: float, max_allocation_pct: float, max_loss_pct: float,
) -> dict:
    if fast < 1 or slow <= fast:
        raise ValueError("Periods must satisfy 1 <= fast < slow")
    if capital <= 0 or not math.isfinite(capital):
        raise ValueError("Capital must be a positive finite number")
    if cost_bps < 0 or not math.isfinite(cost_bps) or cost_bps >= 10000:
        raise ValueError("Cost basis points must be finite and between 0 and 10000")
    if not 0 < max_allocation_pct <= 100:
        raise ValueError("Maximum allocation must be greater than 0 and at most 100 percent")
    if not 0 < max_loss_pct < 100:
        raise ValueError("Maximum loss stop must be greater than 0 and less than 100 percent")

    cash = capital
    shares = 0
    fee_rate = cost_bps / 10000
    orders: list[PaperOrder] = []
    equity_curve: list[dict] = []
    pending_signal: bool | None = None
    risk_exit_pending = False
    halted = False
    halt_date: str | None = None
    closes = [bar.close for bar in bars]

    for i, bar in enumerate(bars):
        # A signal generated on the previous close is simulated at today's open.
        if risk_exit_pending and shares > 0:
            quantity = shares
            gross = quantity * bar.open
            fee = gross * fee_rate
            cash += gross - fee
            shares = 0
            orders.append(PaperOrder(bar.date.isoformat(), "SELL", quantity, bar.open,
                                     round(fee, 2), round(cash, 2), shares,
                                     "loss limit reached; simulated exit at next-session open"))
            risk_exit_pending = False
        elif pending_signal is True and shares == 0 and not halted:
            budget = min(cash, capital * max_allocation_pct / 100)
            quantity = int(budget / (bar.open * (1 + fee_rate)))
            if quantity > 0:
                gross = quantity * bar.open
                fee = gross * fee_rate
                cash -= gross + fee
                shares = quantity
                orders.append(PaperOrder(bar.date.isoformat(), "BUY", quantity, bar.open,
                                         round(fee, 2), round(cash, 2), shares,
                                         "fast SMA above slow SMA; next-session open"))
        elif pending_signal is False and shares > 0:
            quantity = shares
            gross = quantity * bar.open
            fee = gross * fee_rate
            cash += gross - fee
            shares = 0
            orders.append(PaperOrder(bar.date.isoformat(), "SELL", quantity, bar.open,
                                     round(fee, 2), round(cash, 2), shares,
                                     "fast SMA at or below slow SMA; next-session open"))
        pending_signal = None

        equity = cash + shares * bar.close * (1 - fee_rate)
        drawdown_pct = max(0.0, (capital - equity) / capital * 100)
        if not halted and drawdown_pct >= max_loss_pct:
            halted = True
            halt_date = bar.date.isoformat()
            # Cancel new entries. Existing exposure is retained and tracked; no
            # same-bar fill is invented. Any open position exits next session.
            pending_signal = None
            risk_exit_pending = shares > 0

        equity_curve.append({"date": bar.date.isoformat(), "equity": round(equity, 2),
                             "cash": round(cash, 2), "shares": shares,
                             "drawdown_pct": round(drawdown_pct, 4)})

        # Today's close determines an order for the following session only.
        if not halted and i + 1 >= slow:
            prior_closes = closes[:i + 1]
            fast_sma = sum(prior_closes[-fast:]) / fast
            slow_sma = sum(prior_closes[-slow:]) / slow
            pending_signal = fast_sma > slow_sma

    final_equity = equity_curve[-1]["equity"]
    return {
        "mode": "historical_csv_replay_simulated_orders",
        "broker_connected": False,
        "strategy": "moving_average_crossover",
        "fast_period": fast,
        "slow_period": slow,
        "start_date": bars[0].date.isoformat(),
        "end_date": bars[-1].date.isoformat(),
        "starting_capital": round(capital, 2),
        "ending_equity": round(final_equity, 2),
        "net_return_pct": round((final_equity / capital - 1) * 100, 4),
        "max_allocation_pct": max_allocation_pct,
        "max_loss_stop_pct": max_loss_pct,
        "halted": halted,
        "halt_date": halt_date,
        "risk_exit_pending_at_end": risk_exit_pending,
        "open_position_shares": shares,
        "estimated_cost_bps_per_side": cost_bps,
        "simulated_orders": [asdict(order) for order in orders],
        "equity_curve": equity_curve,
    }


def write_paper_results(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items()
               if key not in {"simulated_orders", "equity_curve"}}
    (output_dir / "paper_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "paper_orders.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(PaperOrder.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["simulated_orders"])
    with (output_dir / "paper_equity.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["date", "equity", "cash", "shares", "drawdown_pct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["equity_curve"])
