"""Read-only Upstox market-data stream for a local live-price monitor."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
import time
import csv
import json
import ssl
import uuid
from typing import Any


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


class UpstoxLiveFeed:
    """Keep recent live ticks in memory; never exposes or logs the API token."""

    def __init__(self, access_token: str, instrument_keys: list[str], capture_dir: Path = Path("data/live-capture")):
        self._token = access_token
        self.instrument_keys = instrument_keys
        self._lock = threading.Lock()
        self._ticks: dict[str, list[dict[str, Any]]] = {key: [] for key in instrument_keys}
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
                    "ticks": {key: list(values) for key, values in self._ticks.items()}}
