"""Leakage-aware supervised direction model and cash-constrained evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import Bar, run_backtest


FEATURE_NAMES = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "close_vs_sma5", "close_vs_sma20", "close_vs_sma50",
    "volatility_20d", "rsi14_centered", "log_volume_vs_20d",
]
MIN_HISTORY = 50


@dataclass(frozen=True)
class Sample:
    index: int
    exit_index: int
    features: tuple[float, ...]
    label: int
    forward_return_pct: float


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _features(bars: list[Bar], index: int) -> tuple[float, ...]:
    closes = [bar.close for bar in bars]
    close = closes[index]
    returns = [math.log(closes[j] / closes[j - 1]) for j in range(index - 19, index + 1)]
    mean_return = _mean(returns)
    volatility = math.sqrt(sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1))
    changes = [closes[j] - closes[j - 1] for j in range(index - 13, index + 1)]
    gains = [max(value, 0.0) for value in changes]
    losses = [max(-value, 0.0) for value in changes]
    avg_gain, avg_loss = _mean(gains), _mean(losses)
    rsi = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    average_volume = _mean([bars[j].volume for j in range(index - 19, index + 1)])
    volume_ratio = math.log((bars[index].volume + 1.0) / (average_volume + 1.0))
    distances = [close / _mean(closes[index - period + 1:index + 1]) - 1 for period in (5, 20, 50)]
    return (
        close / closes[index - 1] - 1,
        close / closes[index - 5] - 1,
        close / closes[index - 10] - 1,
        close / closes[index - 20] - 1,
        *distances,
        volatility,
        (rsi - 50) / 50,
        volume_ratio,
    )


def _make_samples(bars: list[Bar], horizon: int, cost_bps_per_side: float) -> list[Sample]:
    hurdle = 2 * cost_bps_per_side / 10_000
    samples: list[Sample] = []
    for index in range(MIN_HISTORY, len(bars) - horizon):
        # Features are known at today's close. Entry is next session's open;
        # exit is the close horizon sessions after the signal.
        entry = bars[index + 1].open
        exit_price = bars[index + horizon].close
        forward_return = exit_price / entry - 1
        samples.append(Sample(index, index + horizon, _features(bars, index),
                              int(forward_return > hurdle), forward_return * 100))
    return samples


def _classification_metrics(actual: list[int], probabilities: list[float]) -> dict:
    if not actual:
        return {"samples": 0}
    predicted = [int(probability >= 0.5) for probability in probabilities]
    result = {
        "samples": len(actual),
        "positive_label_pct": round(sum(actual) / len(actual) * 100, 2),
        "accuracy_pct": round(accuracy_score(actual, predicted) * 100, 2),
        "balanced_accuracy_pct": round(balanced_accuracy_score(actual, predicted) * 100, 2),
        "precision_pct": round(precision_score(actual, predicted, zero_division=0) * 100, 2),
        "recall_pct": round(recall_score(actual, predicted, zero_division=0) * 100, 2),
        "brier_score": round(brier_score_loss(actual, probabilities), 5),
    }
    result["roc_auc"] = round(roc_auc_score(actual, probabilities), 4) if len(set(actual)) > 1 else None
    return result


def _fit_model(samples: list[Sample]):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs", random_state=7),
    )
    model.fit([sample.features for sample in samples], [sample.label for sample in samples])
    return model


def _simulate(
    bars: list[Bar], samples: list[Sample], probabilities: list[float], start_index: int,
    end_index: int, capital: float, cost_bps: float, allocation_pct: float,
    probability_threshold: float, horizon: int, stop_loss_pct: float | None = None,
) -> dict:
    if not 0 < allocation_pct <= 100:
        raise ValueError("Allocation must be greater than 0 and at most 100 percent")
    if not 0 < probability_threshold <= 1:
        raise ValueError("Probability threshold must be greater than 0 and at most 1")
    if stop_loss_pct is not None and not 0 < stop_loss_pct < 100:
        raise ValueError("Stop loss must be greater than 0 and less than 100 percent")
    probability_by_index = {sample.index: probability for sample, probability in zip(samples, probabilities)}
    fee_rate = cost_bps / 10_000
    sell_threshold = 1 - probability_threshold
    cash, shares = capital, 0
    entry_price = entry_fee = 0.0
    entry_index = None
    held_sessions = 0
    pending_action: tuple[str, float, str] | None = None
    orders: list[dict] = []
    trades: list[dict] = []
    curve: list[dict] = []

    def close_position(bar: Bar, price: float, probability: float | None, reason: str) -> None:
        nonlocal cash, shares, entry_price, entry_fee, entry_index, held_sessions
        gross = shares * price
        exit_fee = gross * fee_rate
        cash += gross - exit_fee
        pnl = gross - exit_fee - (shares * entry_price + entry_fee)
        trades.append({"entry_date": bars[entry_index].date.isoformat(), "exit_date": bar.date.isoformat(),
                       "shares": shares, "entry_price": round(entry_price, 4), "exit_price": round(price, 4),
                       "pnl": round(pnl, 2), "return_pct": round(pnl / (shares * entry_price + entry_fee) * 100, 4)})
        orders.append({"date": bar.date.isoformat(), "side": "SELL", "shares": shares,
                       "price": round(price, 4), "cost": round(exit_fee, 2),
                       "probability": round(probability, 4) if probability is not None else None,
                       "reason": reason})
        shares = 0
        entry_price = entry_fee = 0.0
        entry_index = None
        held_sessions = 0

    for index in range(start_index, end_index):
        bar = bars[index]
        stopped_today = False
        if pending_action:
            action, signal_probability, reason = pending_action
            if action == "BUY" and shares == 0:
                budget = min(cash, capital * allocation_pct / 100)
                quantity = int(budget / (bar.open * (1 + fee_rate)))
                if quantity:
                    gross = quantity * bar.open
                    entry_fee = gross * fee_rate
                    cash -= gross + entry_fee
                    shares = quantity
                    entry_price, entry_index = bar.open, index
                    held_sessions = 0
                    orders.append({"date": bar.date.isoformat(), "side": "BUY", "shares": shares,
                                   "price": round(bar.open, 4), "cost": round(entry_fee, 2),
                                   "probability": round(signal_probability, 4), "reason": reason})
            elif action == "SELL" and shares:
                close_position(bar, bar.open, signal_probability, reason)
            pending_action = None

        # Daily-bar stop simulation is conservative for overnight gaps: a gap
        # through the stop fills at the open; otherwise it fills at the stop.
        if shares and stop_loss_pct is not None:
            stop_price = entry_price * (1 - stop_loss_pct / 100)
            if bar.open <= stop_price or bar.low <= stop_price:
                fill_price = min(bar.open, stop_price)
                close_position(bar, fill_price, probability_by_index.get(index),
                               f"{stop_loss_pct:.2f}% protective stop; daily OHLC fill estimate")
                pending_action = None
                stopped_today = True

        if shares:
            held_sessions += 1
        equity = cash + shares * bar.close * (1 - fee_rate)
        curve.append({"date": bar.date.isoformat(), "equity": round(equity, 2), "cash": round(cash, 2),
                      "shares": shares, "drawdown_pct": 0.0})

        probability = probability_by_index.get(index)
        if shares and probability is not None and probability < sell_threshold:
            pending_action = ("SELL", probability, "model probability fell below sell threshold; exit next-session open")
        elif shares and held_sessions >= horizon:
            pending_action = ("SELL", probability if probability is not None else 0.0,
                              f"maximum {horizon}-session holding period; exit next-session open")
        elif (not shares and not stopped_today and probability is not None
              and probability >= probability_threshold and index + 1 < end_index):
            pending_action = ("BUY", probability, "model probability cleared buy threshold; enter next-session open")

    # Force liquidation at the final close for comparable segment-end equity.
    if shares:
        bar = bars[end_index - 1]
        close_position(bar, bar.close, None, "segment ended; liquidated at final close for measurement")
        curve[-1].update({"equity": round(cash, 2), "cash": round(cash, 2), "shares": 0})

    peak = capital
    max_drawdown = 0.0
    for point in curve:
        peak = max(peak, point["equity"])
        max_drawdown = max(max_drawdown, (peak - point["equity"]) / peak if peak else 0.0)
        point["drawdown_pct"] = round(max_drawdown * 100, 4)
    benchmark_shares = int(capital / (bars[start_index].close * (1 + fee_rate)))
    benchmark_cash = capital - benchmark_shares * bars[start_index].close * (1 + fee_rate)
    benchmark_end = benchmark_cash + benchmark_shares * bars[end_index - 1].close * (1 - fee_rate)
    ending_equity = curve[-1]["equity"] if curve else capital
    trade_pnls = [float(trade["pnl"]) for trade in trades]
    gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
    gross_loss = -sum(pnl for pnl in trade_pnls if pnl < 0)
    winning_trades = sum(pnl > 0 for pnl in trade_pnls)
    largest_loss = min((pnl for pnl in trade_pnls if pnl < 0), default=0.0)
    return {
        "starting_capital": round(capital, 2), "ending_equity": round(ending_equity, 2),
        "net_pnl": round(ending_equity - capital, 2),
        "net_return_pct": round((ending_equity / capital - 1) * 100, 4),
        "buy_and_hold_return_pct": round((benchmark_end / capital - 1) * 100, 4),
        "buy_and_hold_shares": benchmark_shares,
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "closed_trades": len(trades),
        "winning_trades": winning_trades,
        "win_rate_pct": round(winning_trades / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else None,
        "largest_trade_loss": round(largest_loss, 2),
        "orders": orders, "trades": trades,
        "equity_curve": curve,
    }


def _select_validation_policy(
    bars: list[Bar], validation: list[Sample], validation_prob: list[float],
    start_index: int, end_index: int, capital: float, cost_bps: float,
    allocation_pct: float, base_threshold: float, horizon: int,
    max_drawdown_pct: float,
    cv_folds: list[tuple[list[Sample], list[float], int, int]],
) -> tuple[dict, list[dict]]:
    """Select a small risk-policy grid using training folds and validation.

    A policy must have positive net return, at least three closed trades, and
    drawdown no greater than the configured research guardrail in every
    rolling-origin fold and the validation segment. If none qualifies, the
    model stands aside (cash) instead of forcing trades.
    """
    thresholds = sorted({round(max(0.50, base_threshold - 0.05), 4),
                         round(base_threshold, 4),
                         round(min(0.70, base_threshold + 0.05), 4),
                         round(min(0.70, base_threshold + 0.10), 4)})
    candidates = []
    for threshold in thresholds:
        for stop_loss in (None, 3.0, 5.0):
            validation_simulation = _simulate(
                bars, validation, validation_prob, start_index, end_index,
                capital, cost_bps, allocation_pct, threshold, horizon, stop_loss)
            fold_simulations = [
                _simulate(bars, fold_samples, fold_prob, fold_start, fold_end,
                          capital, cost_bps, allocation_pct, threshold, horizon, stop_loss)
                for fold_samples, fold_prob, fold_start, fold_end in cv_folds]
            candidates.append({
                "probability_threshold": threshold,
                "stop_loss_pct": stop_loss,
                "validation_net_pnl": validation_simulation["net_pnl"],
                "validation_return_pct": validation_simulation["net_return_pct"],
                "validation_max_drawdown_pct": validation_simulation["max_drawdown_pct"],
                "validation_closed_trades": validation_simulation["closed_trades"],
                "validation_win_rate_pct": validation_simulation["win_rate_pct"],
                "validation_profit_factor": validation_simulation["profit_factor"],
                "validation_largest_trade_loss": validation_simulation["largest_trade_loss"],
                "cv_fold_returns_pct": [item["net_return_pct"] for item in fold_simulations],
                "cv_fold_drawdowns_pct": [item["max_drawdown_pct"] for item in fold_simulations],
                "cv_fold_trade_counts": [item["closed_trades"] for item in fold_simulations],
            })

    eligible = [candidate for candidate in candidates
                if candidate["cv_fold_returns_pct"]
                and candidate["validation_closed_trades"] >= 3
                and candidate["validation_return_pct"] > 0
                and candidate["validation_max_drawdown_pct"] <= max_drawdown_pct
                and all(value > 0 for value in candidate["cv_fold_returns_pct"])
                and all(value <= max_drawdown_pct for value in candidate["cv_fold_drawdowns_pct"])]
    if eligible:
        selected = max(eligible, key=lambda item: (
            sum(item["cv_fold_returns_pct"]) / len(item["cv_fold_returns_pct"]),
            item["validation_return_pct"], -max(item["cv_fold_drawdowns_pct"]),
            -item["validation_max_drawdown_pct"]))
        policy = {"mode": "selected_rolling_validation_policy",
                  "probability_threshold": selected["probability_threshold"],
                  "stop_loss_pct": selected["stop_loss_pct"],
                  "selection_reason": "highest mean rolling-fold return among candidates positive in every training fold and validation, with at least 3 validation trades and drawdown within the configured limit in every segment"}
    else:
        policy = {"mode": "stand_aside",
                  "probability_threshold": 1.0,
                  "stop_loss_pct": None,
                  "selection_reason": "no candidate was positive in every rolling training fold and validation while meeting the drawdown and trade-count requirements"}
    return policy, candidates


def train_and_evaluate(
    bars: list[Bar], symbol: str, horizon: int, capital: float,
    cost_bps: float, allocation_pct: float, probability_threshold: float = 0.55,
    max_drawdown_pct: float = 5.0,
) -> tuple[dict, dict, list[dict]]:
    if horizon < 1:
        raise ValueError("Prediction horizon must be at least one session")
    if len(bars) < 300:
        raise ValueError("Model research needs at least 300 daily bars; use a longer adjusted history")
    if capital <= 0 or not math.isfinite(capital):
        raise ValueError("Capital must be a positive finite number")
    if not 0 < max_drawdown_pct < 100 or not math.isfinite(max_drawdown_pct):
        raise ValueError("Maximum validation drawdown must be greater than 0 and less than 100 percent")
    if not 0 < probability_threshold < 1 or not math.isfinite(probability_threshold):
        raise ValueError("Starting probability threshold must be greater than 0 and less than 1")
    samples = _make_samples(bars, horizon, cost_bps)
    train_cut, validation_cut = int(len(bars) * 0.6), int(len(bars) * 0.8)
    train = [sample for sample in samples if sample.exit_index < train_cut]
    validation = [sample for sample in samples if sample.index >= train_cut and sample.exit_index < validation_cut]
    test = [sample for sample in samples if sample.index >= validation_cut]
    if len(train) < 100 or len(validation) < 30 or len(test) < 30:
        raise ValueError("Not enough samples in chronological splits after purging overlapping outcome windows")
    if len({sample.label for sample in train}) < 2:
        raise ValueError("Training labels contain only one class; use more history or another target horizon")

    train_model = _fit_model(train)
    validation_prob = train_model.predict_proba([sample.features for sample in validation])[:, 1].tolist()
    test_prob = train_model.predict_proba([sample.features for sample in test])[:, 1].tolist()

    cv_reports = []
    policy_cv_folds: list[tuple[list[Sample], list[float], int, int]] = []
    splitter = TimeSeriesSplit(n_splits=3, gap=horizon)
    for fold, (fit_idx, check_idx) in enumerate(splitter.split(train), start=1):
        fit_samples = [train[int(i)] for i in fit_idx]
        check_samples = [train[int(i)] for i in check_idx]
        if len({sample.label for sample in fit_samples}) < 2:
            continue
        fold_model = _fit_model(fit_samples)
        fold_prob = fold_model.predict_proba([sample.features for sample in check_samples])[:, 1].tolist()
        policy_cv_folds.append((check_samples, fold_prob, check_samples[0].index,
                                check_samples[-1].exit_index + 1))
        fold_metrics = _classification_metrics([sample.label for sample in check_samples], fold_prob)
        cv_reports.append({"fold": fold, **fold_metrics,
                           "fit_end": bars[fit_samples[-1].index].date.isoformat(),
                           "check_start": bars[check_samples[0].index].date.isoformat()})

    validation_baseline = _simulate(
        bars, validation, validation_prob, train_cut, validation_cut,
        capital, cost_bps, allocation_pct, probability_threshold, horizon)
    policy, policy_candidates = _select_validation_policy(
        bars, validation, validation_prob, train_cut, validation_cut, capital,
        cost_bps, allocation_pct, probability_threshold, horizon, max_drawdown_pct,
        policy_cv_folds)
    selected_threshold = float(policy["probability_threshold"])
    selected_stop_loss = policy["stop_loss_pct"]
    validation_sim = _simulate(
        bars, validation, validation_prob, train_cut, validation_cut,
        capital, cost_bps, allocation_pct, selected_threshold, horizon, selected_stop_loss)
    test_sim = _simulate(
        bars, test, test_prob, validation_cut, len(bars),
        capital, cost_bps, allocation_pct, selected_threshold, horizon, selected_stop_loss)
    test_baseline = _simulate(
        bars, test, test_prob, validation_cut, len(bars),
        capital, cost_bps, allocation_pct, probability_threshold, horizon)
    fixed_sma = run_backtest(bars[validation_cut:], 5, 20, capital, cost_bps, bars[:validation_cut])
    report = {
        "model": "standardized_logistic_regression",
        "symbol": symbol,
        "data_start": bars[0].date.isoformat(), "data_end": bars[-1].date.isoformat(),
        "feature_names": FEATURE_NAMES,
        "target": f"next-open to close after {horizon} sessions return exceeds estimated round-trip costs",
        "horizon_sessions": horizon,
        "estimated_cost_bps_per_side": cost_bps,
        "allocation_pct": allocation_pct,
        "probability_threshold": selected_threshold,
        "sell_threshold": round(1 - selected_threshold, 4),
        "max_validation_drawdown_pct": max_drawdown_pct,
        "policy_selection": {"selected": policy, "candidates": policy_candidates,
                             "selection_period": "training rolling-origin folds and validation only",
                             "test_usage": "evaluation only; never used to select the policy"},
        "order_policy": ("Stand aside; no candidate passed rolling-fold and validation risk/return requirements."
                         if policy["mode"] == "stand_aside" else
                         f"BUY at next open when flat and probability >= {selected_threshold:.2f}; "
                         f"SELL at next open when holding and probability < {1-selected_threshold:.2f}; "
                         f"protective stop {selected_stop_loss}% when configured; maximum holding period {horizon} sessions."),
        "split_method": "chronological 60/20/20; outcomes crossing split boundaries are purged by label exit date",
        "training": {"start": bars[train[0].index].date.isoformat(), "end": bars[train[-1].index].date.isoformat(),
                     "samples": len(train), "positive_pct": round(sum(s.label for s in train) / len(train) * 100, 2)},
        "validation": {"start": bars[validation[0].index].date.isoformat(), "end": bars[validation[-1].index].date.isoformat(),
                       "classification": _classification_metrics([s.label for s in validation], validation_prob),
                       "baseline_portfolio": {key: value for key, value in validation_baseline.items()
                                              if key not in {"orders", "trades", "equity_curve"}},
                       "portfolio": {key: value for key, value in validation_sim.items()
                                     if key not in {"orders", "trades", "equity_curve"}}},
        "test": {"start": bars[test[0].index].date.isoformat(), "end": bars[test[-1].index].date.isoformat(),
                 "classification": _classification_metrics([s.label for s in test], test_prob),
                 "baseline_portfolio": {key: value for key, value in test_baseline.items()
                                        if key not in {"orders", "trades", "equity_curve"}},
                 "portfolio": {key: value for key, value in test_sim.items()
                               if key not in {"orders", "trades", "equity_curve"}},
                 "holdout_status": "previously inspected historical audit period; not a pristine final holdout",
                 "fixed_sma_5_20_return_pct": fixed_sma["net_return_pct"],
                 "buy_and_hold_return_pct": test_sim["buy_and_hold_return_pct"]},
        "train_time_series_cv": cv_reports,
        "limitations": [
            "One symbol and a few years of daily history are not enough to establish a durable trading edge.",
            "This is a small interpretable baseline, not a pretrained model and not a live trading signal.",
            "Costs are a rough estimate; taxes, exchange fees, spread, slippage, and liquidity impact are incomplete.",
            "The 2025 historical test period has been inspected and is not a pristine final holdout.",
            "The validation-selected policy can overfit the single validation period; any protective stop can gap through its threshold.",
        ],
    }

    def artifact_for(fitted_model, first_sample: Sample, last_sample: Sample,
                     sample_count: int, note: str) -> dict:
        scaler = fitted_model.named_steps["standardscaler"]
        classifier = fitted_model.named_steps["logisticregression"]
        return {
            "model": report["model"], "research_only": True, "symbol": symbol,
            "training_start": bars[first_sample.index].date.isoformat(),
            "training_end": bars[last_sample.exit_index].date.isoformat(),
            "training_samples": sample_count, "feature_names": FEATURE_NAMES,
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]), "positive_class": 1,
            "horizon_sessions": horizon, "probability_threshold": selected_threshold,
            "stop_loss_pct": selected_stop_loss, "policy_mode": policy["mode"],
            "estimated_cost_bps_per_side": cost_bps, "warning": note,
        }

    final_model = _fit_model(samples)
    artifact = artifact_for(
        final_model, samples[0], samples[-1], len(samples),
        "Research-only final refit over all labeled history after the holdout evaluation. Not approved for live orders.")
    evaluation_artifact = artifact_for(
        train_model, train[0], train[-1], len(train),
        "Training-split-only evaluation artifact. Validation and test results in model_report.json were produced before any full-history refit.")
    report["final_refit"] = {
        "training_start": artifact["training_start"],
        "training_end": artifact["training_end"],
        "samples": len(samples),
        "note": "The saved model.json is refit on all labeled rows after evaluation for exploratory scoring; 2025 is already inspected, and these weights/policies are not approved for live orders.",
    }
    prediction_rows = []
    for split_name, split_samples, split_prob in (("validation", validation, validation_prob), ("test", test, test_prob)):
        for sample, probability in zip(split_samples, split_prob):
            prediction_rows.append({
                "split": split_name, "signal_date": bars[sample.index].date.isoformat(),
                "target_exit_date": bars[sample.exit_index].date.isoformat(),
                "probability_positive_net_return": round(probability, 6),
                "predicted_action": ("BUY_IF_FLAT" if probability >= selected_threshold else
                                     "SELL_IF_HOLDING" if probability < 1 - selected_threshold else "HOLD"),
                "actual_positive_net_return": sample.label,
                "realized_forward_return_pct": round(sample.forward_return_pct, 4),
            })
    extras = {"model_artifact": artifact, "evaluation_model_artifact": evaluation_artifact,
              "predictions": prediction_rows,
              "validation_detail": validation_sim, "test_detail": test_sim}
    return report, extras, prediction_rows


def write_model_outputs(report: dict, extras: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "model.json").write_text(json.dumps(extras["model_artifact"], indent=2) + "\n", encoding="utf-8")
    (output_dir / "evaluation_model.json").write_text(
        json.dumps(extras["evaluation_model_artifact"], indent=2) + "\n", encoding="utf-8")
    with (output_dir / "model_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(extras["predictions"][0]) if extras["predictions"] else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(extras["predictions"])
    for name in ("validation", "test"):
        details = extras[f"{name}_detail"]
        (output_dir / f"model_{name}_orders.json").write_text(
            json.dumps({"orders": details["orders"], "trades": details["trades"],
                        "equity_curve": details["equity_curve"]}, indent=2) + "\n", encoding="utf-8")


def predict_latest(bars: list[Bar], symbol: str, model_path: Path) -> dict:
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    if artifact.get("symbol", "UNKNOWN").upper() not in {"UNKNOWN", symbol.upper()}:
        raise ValueError(f"Model is for {artifact['symbol']}, but the CSV symbol is {symbol}")
    if len(bars) <= MIN_HISTORY:
        raise ValueError(f"Need more than {MIN_HISTORY} daily bars to calculate model features")
    features = _features(bars, len(bars) - 1)
    scaled = [(value - mean) / scale for value, mean, scale in
              zip(features, artifact["scaler_mean"], artifact["scaler_scale"])]
    score = float(artifact["intercept"]) + sum(weight * value for weight, value in
                                               zip(artifact["coefficients"], scaled))
    probability = 1 / (1 + math.exp(-max(min(score, 700), -700)))
    threshold = float(artifact["probability_threshold"])
    action = "BUY_IF_FLAT" if probability >= threshold else "SELL_IF_HOLDING" if probability < 1 - threshold else "HOLD"
    return {
        "research_only": True, "symbol": symbol.upper(), "as_of_date": bars[-1].date.isoformat(),
        "last_close": bars[-1].close, "forecast_horizon_sessions": artifact["horizon_sessions"],
        "probability_return_exceeds_estimated_costs": round(probability, 6),
        "probability_threshold": threshold, "action": action,
        "policy_mode": artifact.get("policy_mode", "legacy_fixed_threshold"),
        "stop_loss_pct": artifact.get("stop_loss_pct"),
        "model_training_end": artifact["training_end"],
        "warning": "A historical model probability is not a guaranteed forecast or a live trading instruction. Refresh data and evaluate across multiple symbols before any real-money use.",
    }
