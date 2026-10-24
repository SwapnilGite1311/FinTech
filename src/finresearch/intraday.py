"""Minute-bar ML research with chronological, purged evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import json
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "return_1", "return_3", "return_5", "return_10", "return_20",
    "sma_gap_5_20", "volatility_20", "bar_range", "bar_body",
    "relative_volume_20", "session_minute",
]


@dataclass(frozen=True)
class MinuteBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_minute_bars(path: Path) -> list[MinuteBar]:
    aliases = {"timestamp": {"timestamp", "time", "datetime"}, "open": {"open"},
               "high": {"high"}, "low": {"low"}, "close": {"close"},
               "volume": {"volume"}}
    result: list[MinuteBar] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Minute CSV is missing its header row")
        normalized = {"".join(c.lower() for c in n if c.isalnum()): n for n in reader.fieldnames if n}
        columns = {name: next((normalized[a] for a in accepted if a in normalized), None)
                   for name, accepted in aliases.items()}
        missing = [name for name, column in columns.items() if column is None]
        if missing:
            raise ValueError(f"Minute CSV is missing columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(row[columns["timestamp"]].strip())
                values = {name: float(row[columns[name]].replace(",", ""))
                          for name in ("open", "high", "low", "close", "volume")}
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid minute CSV row {row_number}: {exc}") from exc
            if timestamp.tzinfo is None:
                raise ValueError(f"Minute timestamp on row {row_number} must include a timezone")
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f"Non-finite OHLCV value on row {row_number}")
            if min(values["open"], values["high"], values["low"], values["close"]) <= 0 or values["volume"] < 0:
                raise ValueError(f"Invalid price or volume on minute CSV row {row_number}")
            if values["high"] < max(values["open"], values["close"], values["low"]):
                raise ValueError(f"Invalid high price on minute CSV row {row_number}")
            if values["low"] > min(values["open"], values["close"]):
                raise ValueError(f"Invalid low price on minute CSV row {row_number}")
            bar = MinuteBar(timestamp, **values)
            if result and bar.timestamp <= result[-1].timestamp:
                raise ValueError("Minute timestamps must be strictly increasing without duplicates")
            result.append(bar)
    if not result:
        raise ValueError("Minute CSV has no bars")
    return result


def _continuous(bars: list[MinuteBar], start: int, end: int) -> bool:
    return (bars[start].timestamp.date() == bars[end].timestamp.date()
            and all((bars[i].timestamp - bars[i - 1].timestamp).total_seconds() <= 90
                    for i in range(start + 1, end + 1)))


def features_at(bars: list[MinuteBar], index: int) -> np.ndarray | None:
    """Return the trained model features using only bars through index."""
    if index < 20 or not _continuous(bars, index - 20, index):
        return None
    closes = np.array([b.close for b in bars[index - 20:index + 1]], dtype=float)
    volumes = np.array([b.volume for b in bars[index - 20:index]], dtype=float)
    bar = bars[index]
    prior_returns = np.diff(closes) / closes[:-1]
    mean_volume = float(np.mean(volumes))
    session_minute = bar.timestamp.hour * 60 + bar.timestamp.minute - (9 * 60 + 15)
    return np.asarray([
        closes[-1] / closes[-2] - 1,
        closes[-1] / closes[-4] - 1,
        closes[-1] / closes[-6] - 1,
        closes[-1] / closes[-11] - 1,
        closes[-1] / closes[-21] - 1,
        float(np.mean(closes[-5:]) / np.mean(closes[-20:]) - 1),
        float(np.std(prior_returns, ddof=1)),
        (bar.high - bar.low) / bar.close,
        (bar.close - bar.open) / bar.open,
        bar.volume / max(mean_volume, 1.0),
        float(session_minute),
    ], dtype=float)


def score_intraday_features(artifact: dict, features: np.ndarray) -> float:
    if artifact.get("model") != "minute_logistic_regression":
        raise ValueError("Unsupported intraday model artifact")
    mean = np.asarray(artifact["mean"], dtype=float)
    scale = np.asarray(artifact["scale"], dtype=float)
    coefficients = np.asarray(artifact["coefficients"], dtype=float)
    if not (len(mean) == len(scale) == len(coefficients) == len(features)):
        raise ValueError("Intraday model feature dimensions do not match")
    z = (features - mean) / np.where(scale == 0, 1.0, scale)
    logit = float(np.dot(z, coefficients) + artifact["intercept"])
    return 1 / (1 + math.exp(-max(-40, min(40, logit))))


def _make_samples(bars: list[MinuteBar], horizon: int, cost_bps: float):
    if horizon < 1:
        raise ValueError("Horizon must be at least one minute")
    samples = []
    for i in range(20, len(bars) - horizon - 1):
        exit_index = i + horizon + 1
        if not (_continuous(bars, i - 20, i) and _continuous(bars, i, exit_index)):
            continue
        features = features_at(bars, i)
        if features is None:
            continue
        gross_return = bars[exit_index].open / bars[i + 1].open - 1
        target = int(gross_return > 2 * cost_bps / 10000)
        samples.append((i, exit_index, features, target, gross_return))
    if len(samples) < 200:
        raise ValueError(f"Only {len(samples)} usable intraday samples; at least 200 are required")
    return samples


def _portfolio(bars, samples, scores, threshold, capital, allocation_pct, cost_bps,
               horizon, max_loss_pct):
    fee = cost_bps / 10000
    cash = capital
    shares = 0
    trades = []
    equity = []
    blocked_until = -1
    halted = False
    sample_map = {sample[0]: (sample, float(scores[j])) for j, sample in enumerate(samples)}
    test_start = samples[0][0]
    test_end = samples[-1][1]
    for i in range(test_start, test_end + 1):
        bar = bars[i]
        if shares and i == blocked_until:
            exit_price = bar.open
            net_proceeds = shares * exit_price * (1 - fee)
            pnl = net_proceeds - trades[-1]["cost_basis"]
            cash += net_proceeds
            trades[-1].update(exit_timestamp=bar.timestamp.isoformat(), exit_price=exit_price,
                              pnl=round(pnl, 2), return_pct=round(pnl / trades[-1]["cost_basis"] * 100, 4))
            shares = 0
        elif not shares and not halted and i in sample_map:
            sample, probability = sample_map[i]
            if probability >= threshold and i + 1 <= test_end:
                entry = bars[i + 1].open
                allocation = min(cash, capital * allocation_pct / 100)
                quantity = int(allocation / (entry * (1 + fee)))
                if quantity > 0:
                    cost_basis = quantity * entry * (1 + fee)
                    cash -= cost_basis
                    shares = quantity
                    blocked_until = sample[1]
                    trades.append({"entry_timestamp": bars[i + 1].timestamp.isoformat(),
                                   "entry_price": entry, "exit_timestamp": None,
                                   "exit_price": None, "shares": quantity,
                                   "cost_basis": round(cost_basis, 2), "pnl": None,
                                   "model_score": round(probability, 5)})
        mark = cash + shares * bar.close * (1 - fee)
        if not halted and capital - mark >= capital * max_loss_pct / 100:
            if shares:
                exit_price = bar.close
                net_proceeds = shares * exit_price * (1 - fee)
                pnl = net_proceeds - trades[-1]["cost_basis"]
                cash += net_proceeds
                trades[-1].update(exit_timestamp=bar.timestamp.isoformat(), exit_price=exit_price,
                                  pnl=round(pnl, 2), return_pct=round(pnl / trades[-1]["cost_basis"] * 100, 4),
                                  exit_reason="account loss guard")
                shares = 0
            halted = True
            mark = cash
        equity.append((bar.timestamp.isoformat(), mark))
    ending = equity[-1][1] if equity else capital
    peak = capital
    max_dd = 0.0
    for _, value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    completed = [trade for trade in trades if trade["pnl"] is not None]
    gross_profit = sum(max(t["pnl"], 0) for t in completed)
    gross_loss = abs(sum(min(t["pnl"], 0) for t in completed))
    first_entry = bars[test_start + 1].open
    buy_hold_quantity = int((capital * allocation_pct / 100) / (first_entry * (1 + fee)))
    buy_hold_cost = buy_hold_quantity * first_entry * (1 + fee)
    buy_hold_end = (capital - buy_hold_cost
                    + buy_hold_quantity * bars[test_end].close * (1 - fee))
    return {
        "starting_capital": round(capital, 2), "ending_equity": round(ending, 2),
        "net_pnl": round(ending - capital, 2),
        "net_return_pct": round((ending / capital - 1) * 100, 4),
        "max_drawdown_pct": round(max_dd * 100, 4), "closed_trades": len(completed),
        "win_rate_pct": round(sum(t["pnl"] > 0 for t in completed) / len(completed) * 100, 2) if completed else 0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else None,
        "buy_and_hold_return_pct": round((buy_hold_end / capital - 1) * 100, 4),
        "trades": trades, "equity_curve": equity, "halted_by_loss_guard": halted,
        "max_loss_pct": max_loss_pct,
        "cost_bps_per_side": cost_bps, "allocation_pct": allocation_pct,
        "holding_horizon_minutes": horizon,
    }


def train_intraday(bars: list[MinuteBar], symbol: str, horizon: int = 5,
                   capital: float = 10000, cost_bps: float = 30,
                   allocation_pct: float = 50, threshold: float = 0.55,
                   max_loss_pct: float = 5.0) -> tuple[dict, dict]:
    if capital <= 0 or not math.isfinite(capital):
        raise ValueError("Capital must be positive and finite")
    if cost_bps < 0 or cost_bps >= 10000 or not math.isfinite(cost_bps):
        raise ValueError("Cost bps must be finite and between 0 and 10000")
    if allocation_pct <= 0 or allocation_pct > 100:
        raise ValueError("Allocation percent must be greater than 0 and at most 100")
    if max_loss_pct <= 0 or max_loss_pct > 100:
        raise ValueError("Maximum loss percent must be greater than 0 and at most 100")
    if not 0.5 <= threshold < 1:
        raise ValueError("Probability threshold must be at least 0.5 and less than 1")
    samples = _make_samples(bars, horizon, cost_bps)
    n = len(samples)
    train_end = int(n * 0.6)
    validation_end = int(n * 0.8)
    purge = horizon + 2
    train = samples[:max(0, train_end - purge)]
    validation = samples[min(n, train_end + purge):max(train_end + purge, validation_end - purge)]
    test = samples[min(n, validation_end + purge):]
    if not train or not validation or len(test) < 50:
        raise ValueError("Not enough chronological samples after horizon purging; fetch more history")
    X_train = np.vstack([sample[2] for sample in train])
    y_train = np.array([sample[3] for sample in train], dtype=int)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels contain only one class; more varied history is needed")
    scaler = StandardScaler().fit(X_train)
    classifier = LogisticRegression(C=0.1, max_iter=1000, random_state=7)
    classifier.fit(scaler.transform(X_train), y_train)
    y_test = np.array([sample[3] for sample in test], dtype=int)
    X_test = np.vstack([sample[2] for sample in test])
    scores = classifier.predict_proba(scaler.transform(X_test))[:, 1]
    predicted = (scores >= 0.5).astype(int)
    validation_X = np.vstack([sample[2] for sample in validation])
    validation_y = np.array([sample[3] for sample in validation], dtype=int)
    validation_scores = classifier.predict_proba(scaler.transform(validation_X))[:, 1]
    validation_predicted = (validation_scores >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_test, scores))
    except ValueError:
        auc = None
    test_bars = bars
    portfolio = _portfolio(test_bars, test, scores, threshold, capital,
                           allocation_pct, cost_bps, horizon, max_loss_pct)
    report = {
        "model": "minute logistic regression", "symbol": symbol, "interval": "1 minute",
        "target": f"{horizon}-minute gross return exceeds estimated round-trip costs",
        "data_start": bars[0].timestamp.isoformat(), "data_end": bars[-1].timestamp.isoformat(),
        "sample_counts": {"all": n, "train": len(train), "validation": len(validation), "test": len(test)},
        "validation_classification": {
            "accuracy_pct": round(float(accuracy_score(validation_y, validation_predicted)) * 100, 2),
            "balanced_accuracy_pct": round(float(balanced_accuracy_score(validation_y, validation_predicted)) * 100, 2),
        },
        "test_classification": {
            "accuracy_pct": round(float(accuracy_score(y_test, predicted)) * 100, 2),
            "balanced_accuracy_pct": round(float(balanced_accuracy_score(y_test, predicted)) * 100, 2),
            "roc_auc": round(auc, 4) if auc is not None else None,
            "positive_label_pct": round(float(y_test.mean()) * 100, 2),
        },
        "test_portfolio": portfolio,
        "validation_note": "Chronological train/validation/test split with horizon purge; validation is reported but not used to tune the fixed score threshold.",
        "score_note": "Logistic outputs are uncalibrated model scores, not reliable real-world probabilities.",
        "limitations": ["Research simulation only; no broker orders.",
                        "Estimated costs omit market impact, spread, taxes, exchange and broker charges.",
                        "A positive historical test result is not evidence of future profitability."],
    }
    artifact = {
        "format_version": 1, "model": "minute_logistic_regression", "symbol": symbol,
        "interval": "1 minute", "horizon_minutes": horizon,
        "score_threshold": threshold, "max_loss_pct": max_loss_pct,
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
        "trained_through": bars[train[-1][0]].timestamp.isoformat(),
        "warning": "Research artifact. Not connected to the live paper account or a broker.",
    }
    return report, artifact


def write_intraday_outputs(report: dict, artifact: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "intraday_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "intraday_model.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    portfolio = report["test_portfolio"]
    trades = portfolio["trades"]
    fields = ["entry_timestamp", "exit_timestamp", "entry_price", "exit_price", "shares",
              "cost_basis", "pnl", "return_pct", "model_score", "exit_reason"]
    with (output_dir / "intraday_test_trades.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trade in trades:
            writer.writerow({field: trade.get(field) for field in fields})
    with (output_dir / "intraday_test_equity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "equity"])
        writer.writerows(portfolio["equity_curve"])
