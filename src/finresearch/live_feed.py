"""Read-only Upstox market-data stream for a local live-price monitor."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import threading
import time
import csv
import json
import ssl
import uuid
from typing import Any

from .intraday import MinuteBar, features_at, score_intraday_features


def _find_ltp(value: Any) -> tuple[float | None, str | None, int | None]:
    """Extract last traded price/time/volume from the SDK's decoded feed dict."""
    if isinstance(value, dict):
        if "ltp" in value:
            try:
                return float(value["ltp"]), str(value.get("ltt") or "") or None, int(value.get("ltq", 0))
            except (TypeError, ValueError):
                return None, None, None
        for child in value.values():
            found = _find_ltp(child)
            if found[0] is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_ltp(child)
            if found[0] is not None:
                return found
    return None, None, None


class PaperSmaBaseline:
    """One-minute paper benchmark or ML strategy; never sends broker orders."""

    def __init__(self, capital: float = 10000.0, cost_bps_per_side: float = 30.0,
                 max_drawdown_pct: float = 5.0, model_artifact: dict | None = None):
        self.starting_capital = capital
        self.cash = capital
        self.cost_rate = cost_bps_per_side / 10000
        self.max_loss_pct = float((model_artifact or {}).get("max_loss_pct", max_drawdown_pct))
        self.max_loss = capital * self.max_loss_pct / 100
        self.shares = 0
        self.current_price: float | None = None
        self.current_minute: str | None = None
        self.current_bar: dict[str, float] | None = None
        self.closes: list[float] = []
        self.previous_fast: float | None = None
        self.previous_slow: float | None = None
        self.pending_action: str | None = None
        self.pending_reason = ""
        self.orders: list[dict[str, Any]] = []
        self.model_artifact = model_artifact
        self.bars: list[MinuteBar] = []
        self.bar_sequence = 0
        self.current_bar_time: datetime | None = None
        self.entry_bar_index: int | None = None
        self.model_score: float | None = None
        self.halted = False
        self.last_signal = "Collecting one-minute bars"
        self._timezone = ZoneInfo("Asia/Kolkata")

    def _execute(self, action: str, price: float, received_at: str, reason: str) -> None:
        if action == "BUY" and self.shares == 0 and not self.halted:
            allocation = min(self.cash, self.starting_capital * 0.5)
            quantity = int(allocation / (price * (1 + self.cost_rate)))
            if quantity < 1:
                return
            self.cash -= quantity * price * (1 + self.cost_rate)
            self.shares = quantity
            if self.model_artifact:
                self.entry_bar_index = self.bar_sequence
        elif action == "SELL" and self.shares:
            quantity = self.shares
            self.cash += quantity * price * (1 - self.cost_rate)
            self.shares = 0
            self.entry_bar_index = None
        else:
            return
        self.orders.append({"side": action, "price": price, "shares": quantity,
                            "received_at": received_at, "reason": reason})
        del self.orders[:-30]

    def on_tick(self, price: float, received_at: str, last_quantity: int = 0) -> None:
        self.current_price = price
        instant = datetime.fromisoformat(received_at).astimezone(self._timezone)
        minute = instant.strftime("%Y-%m-%d %H:%M")
        if minute != self.current_minute:
            if self.current_bar is not None:
                self._finish_bar()
            self.current_minute = minute
            self.current_bar_time = instant.replace(second=0, microsecond=0)
            self.current_bar = {"open": price, "high": price, "low": price,
                                "close": price, "volume": max(0, last_quantity)}
            if self.pending_action:
                action, self.pending_action = self.pending_action, None
                reason, self.pending_reason = self.pending_reason, ""
                self._execute(action, price, received_at,
                              reason or "Signal; next-minute-open paper fill")
        elif self.current_bar is not None:
            self.current_bar["high"] = max(self.current_bar["high"], price)
            self.current_bar["low"] = min(self.current_bar["low"], price)
            self.current_bar["close"] = price
            self.current_bar["volume"] += max(0, last_quantity)

        equity = self.cash + self.shares * price * (1 - self.cost_rate)
        if not self.halted and self.starting_capital - equity >= self.max_loss:
            if self.shares:
                self._execute("SELL", price, received_at, "5% paper-account loss circuit breaker")
            self.halted = True
            self.pending_action = None
            self.pending_reason = ""
            self.last_signal = "HALTED · paper loss guardrail reached"

    def _finish_bar(self) -> None:
        assert self.current_bar is not None
        assert self.current_bar_time is not None
        completed = MinuteBar(self.current_bar_time, self.current_bar["open"],
                              self.current_bar["high"], self.current_bar["low"],
                              self.current_bar["close"], self.current_bar["volume"])
        self.bars.append(completed)
        self.bar_sequence += 1
        self.bars = self.bars[-2500:]
        if self.model_artifact:
            if self.shares and self.entry_bar_index is not None:
                if self.bar_sequence - self.entry_bar_index >= int(self.model_artifact["horizon_minutes"]):
                    self.pending_action = "SELL"
                    self.pending_reason = "Intraday model fixed holding horizon; next-minute-open paper exit"
                    self.last_signal = "SELL queued · model holding horizon reached"
                return
            features = features_at(self.bars, len(self.bars) - 1)
            if features is None:
                self.last_signal = "Collecting 21 contiguous one-minute bars"
                return
            self.model_score = score_intraday_features(self.model_artifact, features)
            threshold = float(self.model_artifact["score_threshold"])
            if self.model_score >= threshold and not self.halted:
                self.pending_action = "BUY"
                self.pending_reason = (f"Intraday model score {self.model_score:.3f} met "
                                       f"threshold {threshold:.3f}; next-minute-open paper fill")
                self.last_signal = f"BUY queued · model score {self.model_score:.3f}"
            else:
                self.last_signal = f"HOLD · model score {self.model_score:.3f}"
            return
        self.closes.append(self.current_bar["close"])
        self.closes = self.closes[-20:]
        if self.halted or len(self.closes) < 20:
            return
        fast = sum(self.closes[-5:]) / 5
        slow = sum(self.closes) / 20
        if self.previous_fast is not None and self.previous_slow is not None:
            crossed_up = self.previous_fast <= self.previous_slow and fast > slow
            crossed_down = self.previous_fast >= self.previous_slow and fast < slow
            if crossed_up and not self.shares:
                self.pending_action = "BUY"
                self.pending_reason = "5/20 SMA signal; next-minute-open paper fill"
                self.last_signal = "BUY queued · next one-minute bar"
            elif crossed_down and self.shares:
                self.pending_action = "SELL"
                self.pending_reason = "5/20 SMA signal; next-minute-open paper fill"
                self.last_signal = "SELL queued · next one-minute bar"
            else:
                self.last_signal = "HOLD"
        self.previous_fast, self.previous_slow = fast, slow

    def snapshot(self) -> dict[str, Any]:
        mark = self.current_price or 0.0
        # Mark open positions at estimated net liquidation value, including exit-side costs.
        equity = self.cash + self.shares * mark * (1 - self.cost_rate)
        pnl = equity - self.starting_capital
        return {"starting_capital": round(self.starting_capital, 2), "cash": round(self.cash, 2),
                "shares": self.shares, "mark_price": mark, "equity": round(equity, 2),
                "net_pnl": round(pnl, 2), "net_return_pct": round(pnl / self.starting_capital * 100, 3),
                "orders": list(self.orders), "signal": self.last_signal, "halted": self.halted,
                "model_score": round(self.model_score, 5) if self.model_score is not None else None,
                "max_loss_pct": self.max_loss_pct,
                "strategy": ("minute logistic-regression model" if self.model_artifact else
                             "5/20 simple moving-average crossover on one-minute bars"),
                "estimated_cost_bps_per_side": round(self.cost_rate * 10000, 2)}


class UpstoxLiveFeed:
    """Keep recent live ticks in memory; never exposes or logs the API token."""

    def __init__(self, access_token: str, instrument_keys: list[str],
                 capture_dir: Path = Path("data/live-capture"),
                 model_artifact: dict | None = None):
        self._token = access_token
        self.instrument_keys = instrument_keys
        self._lock = threading.Lock()
        self._ticks: dict[str, list[dict[str, Any]]] = {key: [] for key in instrument_keys}
        self._paper = {key: PaperSmaBaseline(model_artifact=model_artifact)
                       for key in instrument_keys}
        self._connected = False
        self._last_message: str | None = None
        self._error: str | None = None
        self._capture_dir = capture_dir

    def start(self) -> None:
        thread = threading.Thread(target=self._run, name="upstox-live-feed", daemon=True)
        thread.start()

    def _run(self) -> None:
        try:
            import certifi
            import websocket
            from google.protobuf import json_format
            from upstox_client.feeder.proto import MarketDataFeedV3_pb2

            feed_response_type = MarketDataFeedV3_pb2.FeedResponse

            def on_open(ws: Any) -> None:
                with self._lock:
                    self._connected = True
                    self._error = None
                request = {"guid": str(uuid.uuid4()), "method": "sub",
                           "data": {"instrumentKeys": self.instrument_keys, "mode": "ltpc"}}
                ws.send(json.dumps(request).encode("utf-8"), opcode=websocket.ABNF.OPCODE_BINARY)

            def on_close(*_args: Any) -> None:
                with self._lock:
                    self._connected = False

            def on_error(*_args: Any) -> None:
                # Never expose transport exception strings or authorization headers.
                with self._lock:
                    self._connected = False
                    self._error = "Market-data connection error; check token, subscription, and network."

            def on_message(_ws: Any, message: Any) -> None:
                if not isinstance(message, (bytes, bytearray)):
                    return
                try:
                    decoded = feed_response_type.FromString(message)
                    message = json_format.MessageToDict(decoded)
                except Exception:
                    with self._lock:
                        self._error = "Received an unreadable market-data packet."
                    return
                now = datetime.now(timezone.utc).isoformat()
                feeds = message.get("feeds", {}) if isinstance(message, dict) else {}
                updates = []
                for key, feed in feeds.items():
                    if key not in self._ticks:
                        continue
                    price, exchange_time, quantity = _find_ltp(feed)
                    if price is None or price <= 0:
                        continue
                    updates.append((key, {"received_at": now, "exchange_time": exchange_time,
                                          "price": price, "last_quantity": quantity}))
                if updates:
                    with self._lock:
                        for key, tick in updates:
                            self._ticks[key].append(tick)
                            del self._ticks[key][:-1200]
                            self._paper[key].on_tick(tick["price"], tick["received_at"],
                                                     tick["last_quantity"] or 0)
                        self._last_message = now
                        self._connected = True
                    try:
                        self._capture_dir.mkdir(parents=True, exist_ok=True)
                        capture_path = self._capture_dir / f"upstox_ticks_{datetime.now().date().isoformat()}.csv"
                        with capture_path.open("a", newline="", encoding="utf-8") as handle:
                            writer = csv.DictWriter(handle, fieldnames=("received_at", "exchange_time", "instrument_key", "price", "last_quantity"))
                            if capture_path.stat().st_size == 0:
                                writer.writeheader()
                            for key, tick in updates:
                                writer.writerow({**tick, "instrument_key": key})
                    except OSError:
                        # A filesystem issue should not interrupt visualization.
                        pass

            attempts = 0
            while attempts < 20:
                app = websocket.WebSocketApp(
                    "wss://api.upstox.com/v3/feed/market-data-feed",
                    header={"Authorization": f"Bearer {self._token}"},
                    on_open=on_open, on_message=on_message,
                    on_error=on_error, on_close=on_close)
                # Validate the server certificate and hostname before sending the read-only token.
                app.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED,
                                        "ca_certs": certifi.where(),
                                        "check_hostname": True},
                                ping_interval=30, ping_timeout=10)
                attempts += 1
                if attempts < 20:
                    time.sleep(5)
            with self._lock:
                self._connected = False
                self._error = "Market-data reconnect limit reached; restart the dashboard to retry."
        except Exception:
            with self._lock:
                self._connected = False
                self._error = "Could not start Upstox feed; install dependencies and check the read-only token."

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"provider": "Upstox", "connected": self._connected,
                    "instrument_keys": self.instrument_keys, "last_message": self._last_message,
                    "error": self._error,
                    "ticks": {key: list(values) for key, values in self._ticks.items()},
                    "paper": {key: value.snapshot() for key, value in self._paper.items()}}
