"""Local-only web server that lets the dashboard run all research models."""

from __future__ import annotations

from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os

from .backtest import load_bars
from .dashboard import build_dashboard, _symbol_for_csv
from .model import predict_latest, train_and_evaluate, write_model_outputs


def run_model_cycle(
    data_dir: Path,
    results_dir: Path,
    dashboard_path: Path,
    horizon: int = 5,
    capital: float = 10000.0,
    cost_bps: float = 30.0,
    allocation_pct: float = 50.0,
    probability_threshold: float = 0.55,
    max_drawdown_pct: float = 5.0,
    include_live: bool = False,
) -> dict:
    """Train/evaluate each available symbol and score its latest local bar."""
    csvs = sorted(data_dir.glob("*_adjusted.csv"))
    if not csvs:
        raise ValueError(f"No *_adjusted.csv files found in {data_dir}")

    # Prefer the longest available history for each symbol.
    selected: dict[str, tuple[int, Path]] = {}
    for csv_path in csvs:
        symbol = _symbol_for_csv(csv_path)
        metadata_path = csv_path.with_suffix(".source.json")
        row_count = 0
        if metadata_path.exists():
            try:
                row_count = int(json.loads(metadata_path.read_text(encoding="utf-8")).get("row_count", 0))
            except (OSError, ValueError, json.JSONDecodeError):
                row_count = 0
        previous = selected.get(symbol)
        if previous is None or row_count > previous[0]:
            selected[symbol] = (row_count, csv_path)

    signals, failures = [], []
    for symbol, (_, csv_path) in sorted(selected.items()):
        try:
            bars = load_bars(csv_path)
            report, extras, _ = train_and_evaluate(
                bars, symbol, horizon, capital, cost_bps, allocation_pct,
                probability_threshold, max_drawdown_pct)
            symbol_dir = results_dir / symbol
            write_model_outputs(report, extras, symbol_dir)
            signal = predict_latest(bars, symbol, symbol_dir / "model.json")
            signals.append(signal)
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"symbol": symbol, "error": str(exc)})

    if not signals:
        reason = "; ".join(f"{item['symbol']}: {item['error']}" for item in failures)
        raise ValueError(f"No symbol could be scored. {reason}")

    results_dir.mkdir(parents=True, exist_ok=True)
    signal_path = results_dir / "latest_signals.json"
    signal_path.write_text(json.dumps(signals, indent=2) + "\n", encoding="utf-8")
    build_dashboard(data_dir, results_dir, dashboard_path, include_live=include_live)
    return {"signal_count": len(signals), "signals": signals, "failures": failures}


class _DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FinResearchLocal/0.1"

    def _is_local_request(self) -> bool:
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "").lower()
        origin = self.headers.get("Origin")
        return host in allowed and (origin is None or origin.lower() in {f"http://{item}" for item in allowed})

    def do_GET(self) -> None:
        if not self._is_local_request():
            self.send_error(403, "Dashboard is restricted to localhost")
            return
        if self.path == "/api/live":
            payload = json.dumps(self.server.live_callback()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        try:
            body = self.server.dashboard_path.read_bytes()
        except OSError:
            self.send_error(503, "Dashboard file is not available")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if not self._is_local_request():
            self.send_error(403, "Dashboard is restricted to localhost")
            return
        if self.path != "/api/run-model":
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return
        if content_length:
            self.send_error(400, "This endpoint does not accept a request body")
            return
        try:
            result = self.server.run_callback()
            payload = json.dumps({"signal_count": result["signal_count"],
                                  "symbols": [item["symbol"] for item in result["signals"]],
                                  "failures": result["failures"]}).encode("utf-8")
            self.send_response(200)
        except (OSError, ValueError) as exc:
            payload = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")


def serve_dashboard(data_dir: Path, results_dir: Path, dashboard_path: Path,
                    port: int = 8765, horizon: int = 5, capital: float = 10000.0,
                    cost_bps: float = 30.0, allocation_pct: float = 50.0,
                    probability_threshold: float = 0.55,
                    max_drawdown_pct: float = 5.0) -> None:
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    build_dashboard(data_dir, results_dir, dashboard_path, include_live=True)

    def callback() -> dict:
        return run_model_cycle(data_dir, results_dir, dashboard_path, horizon,
                               capital, cost_bps, allocation_pct, probability_threshold,
                               max_drawdown_pct, include_live=True)

    handler = partial(_DashboardHandler)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.dashboard_path = dashboard_path
    server.run_callback = callback
    token = os.environ.get("FINRESEARCH_UPSTOX_ANALYTICS_TOKEN", "").strip()
    instrument_keys = [item.strip() for item in os.environ.get(
        "FINRESEARCH_UPSTOX_INSTRUMENT_KEYS", "NSE_EQ|INE467B01029").split(",") if item.strip()]
    if token and instrument_keys:
        from .live_feed import UpstoxLiveFeed
        live_feed = UpstoxLiveFeed(token, instrument_keys)
        live_feed.start()
        server.live_callback = live_feed.snapshot
        live_message = f"Read-only Upstox feed starting for {len(instrument_keys)} instrument(s)."
    elif not token:
        from .live_feed import PaperSmaBaseline
        initial_paper = {key: PaperSmaBaseline().snapshot() for key in instrument_keys}
        for account in initial_paper.values():
            account["signal"] = "Waiting for live feed"
        server.live_callback = lambda: {"provider": "Upstox", "connected": False,
            "instrument_keys": instrument_keys, "ticks": {key: [] for key in instrument_keys},
            "paper": initial_paper,
            "error": "Not configured. Set FINRESEARCH_UPSTOX_ANALYTICS_TOKEN and restart."}
        live_message = "Live chart is waiting for a read-only Upstox Analytics Token (default instrument: TCS)."
    else:
        server.live_callback = lambda: {"provider": "Upstox", "connected": False,
            "instrument_keys": [], "ticks": {},
            "error": "No instrument keys configured. Set FINRESEARCH_UPSTOX_INSTRUMENT_KEYS and restart."}
        live_message = "Live chart is waiting for Upstox instrument key(s)."
    print(f"Research dashboard: http://127.0.0.1:{port}/")
    print("Click ‘Run model · all stocks’ to train, evaluate, and score all local adjusted histories.")
    print(live_message)
    print("Market data only; no orders are sent. Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()
