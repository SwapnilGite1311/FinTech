"""Small client for TejHQ's keyless end-of-day OHLCV endpoint."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import csv
import json


BASE_URL = "https://api.tejhq.dev/v1/ohlcv/nse"
ADJUSTED_URL = "https://api.tejhq.dev/v1/adjusted/nse"
UPSTOX_INTRADAY_URL = "https://api.upstox.com/v3/historical-candle"


def _fetch_json(url: str, api_key: str | None = None) -> dict:
    headers = {"User-Agent": "fintech-trading-research/0.1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ValueError(f"Data provider returned HTTP {exc.code}; check the API key and access tier") from exc
    except URLError as exc:
        raise ValueError(f"Could not reach TejHQ data API: {exc.reason}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Data provider returned an invalid JSON response") from exc
    if not isinstance(payload, dict):
        raise ValueError("Data provider returned an unexpected response")
    return payload


def _data_rows(payload: dict) -> list[dict]:
    rows = payload.get("data")
    if rows is None and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("Provider response did not contain a data row list")
    return rows


def fetch_eod_csv(symbol: str, start: date, end: date, output_dir: Path) -> Path:
    if start > end:
        raise ValueError("Start date must be on or before end date")
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol or not normalized_symbol.replace("&", "").replace("-", "").isalnum():
        raise ValueError("Symbol must contain letters or numbers, with optional hyphens or ampersands")

    query = urlencode({"from": start.isoformat(), "to": end.isoformat()})
    url = f"{BASE_URL}/{quote(normalized_symbol, safe='')}?{query}"
    rows = _data_rows(_fetch_json(url))
    if not rows:
        raise ValueError("Provider returned no rows for that symbol and date range")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"NSE_{normalized_symbol}_{start.isoformat()}_{end.isoformat()}.csv"
    fields = ["date", "open", "high", "low", "close", "volume"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            try:
                writer.writerow({field: row[field] for field in fields})
            except (KeyError, TypeError) as exc:
                raise ValueError("Provider response is missing required OHLCV fields") from exc

    metadata = {
        "provider": "TejHQ",
        "endpoint": url,
        "exchange": "NSE",
        "symbol": normalized_symbol,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "row_count": len(rows),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_type": "raw end-of-day OHLCV; corporate-action adjusted prices are not included",
        "license_note": "Provider describes data as exchange-published and redistributable; verify applicable exchange terms before commercial use.",
    }
    metadata_path = csv_path.with_suffix(".source.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path


def fetch_adjusted_eod_csv(symbol: str, start: date, end: date, output_dir: Path,
                           api_key: str) -> Path:
    """Join raw OHLCV with provider adjusted closes and scale OHLC by its factor."""
    if start > end:
        raise ValueError("Start date must be on or before end date")
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol or not normalized_symbol.replace("&", "").replace("-", "").isalnum():
        raise ValueError("Symbol must contain letters or numbers, with optional hyphens or ampersands")
    if not api_key.strip():
        raise ValueError("Set FINRESEARCH_TEJHQ_API_KEY to a free TejHQ API key before fetching adjusted prices")

    query = urlencode({"from": start.isoformat(), "to": end.isoformat()})
    raw_url = f"{BASE_URL}/{quote(normalized_symbol, safe='')}?{query}"
    adjusted_url = f"{ADJUSTED_URL}/{quote(normalized_symbol, safe='')}?{query}"
    raw_rows = _data_rows(_fetch_json(raw_url))
    adjusted_rows = _data_rows(_fetch_json(adjusted_url, api_key.strip()))
    if not raw_rows or not adjusted_rows:
        raise ValueError("Provider returned no rows for that symbol and date range")

    adjusted_by_date: dict[str, float] = {}
    try:
        for row in adjusted_rows:
            adjusted_by_date[str(row["date"])] = float(row["adj_close"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Adjusted-price response is missing date or adj_close") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"NSE_{normalized_symbol}_{start.isoformat()}_{end.isoformat()}_adjusted.csv"
    fields = ["date", "open", "high", "low", "close", "volume"]
    rows_written = 0
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in raw_rows:
                day = str(row["date"])
                adjusted_close = adjusted_by_date.get(day)
                raw_close = float(row["close"])
                if adjusted_close is None or raw_close <= 0:
                    continue
                factor = adjusted_close / raw_close
                writer.writerow({
                    "date": day,
                    "open": float(row["open"]) * factor,
                    "high": float(row["high"]) * factor,
                    "low": float(row["low"]) * factor,
                    "close": adjusted_close,
                    "volume": row["volume"],
                })
                rows_written += 1
    except (KeyError, TypeError, ValueError) as exc:
        csv_path.unlink(missing_ok=True)
        raise ValueError("Provider raw-price response is missing valid OHLCV fields") from exc
    if rows_written == 0:
        csv_path.unlink(missing_ok=True)
        raise ValueError("No matching adjusted and raw dates were returned")

    metadata = {
        "provider": "TejHQ",
        "exchange": "NSE",
        "symbol": normalized_symbol,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "row_count": rows_written,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_type": "OHLC adjusted by the provider adjusted-close/raw-close factor; volume remains raw",
        "adjustment_source": "TejHQ /v1/adjusted adj_close joined to /v1/ohlcv by date",
        "license_note": "Provider states its published dataset is MIT licensed; check exchange and provider terms for intended use.",
    }
    csv_path.with_suffix(".source.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path


def fetch_upstox_minute_csv(instrument_key: str, start: date, end: date,
                            output_dir: Path, access_token: str) -> Path:
    """Fetch one-minute NSE candles in month-sized V3 API requests.

    Upstox limits one-minute historical requests to a month per call. The
    function keeps credentials in the Authorization header and only writes
    candle data and non-secret provenance to disk.
    """
    if start > end:
        raise ValueError("Start date must be on or before end date")
    key = instrument_key.strip()
    if not key or "|" not in key or any(char.isspace() for char in key):
        raise ValueError("Provide a valid Upstox instrument key, for example NSE_EQ|INE467B01029")
    token = access_token.strip()
    if not token:
        raise ValueError("Set FINRESEARCH_UPSTOX_ANALYTICS_TOKEN before fetching minute candles")

    rows_by_timestamp: dict[str, list] = {}
    month_start = start.replace(day=1)
    while month_start <= end:
        next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        chunk_start = max(start, month_start)
        chunk_end = min(end, next_month - timedelta(days=1))
        encoded_key = quote(key, safe="")
        url = (f"{UPSTOX_INTRADAY_URL}/{encoded_key}/minutes/1/"
               f"{chunk_end.isoformat()}/{chunk_start.isoformat()}")
        request = Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "fintech-trading-research/0.1",
            "Authorization": f"Bearer {token}",
        })
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ValueError(
                f"Upstox returned HTTP {exc.code} for {chunk_start} to {chunk_end}; "
                "check the token, instrument key, and historical-data access"
            ) from exc
        except URLError as exc:
            raise ValueError(f"Could not reach Upstox historical-data API: {exc.reason}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Upstox returned an invalid JSON response") from exc
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise ValueError("Upstox did not return a successful minute-candle response")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("Upstox response is missing its candle data object")
        candles = data.get("candles", [])
        if not isinstance(candles, list):
            raise ValueError("Upstox response is missing its candle list")
        for candle in candles:
            if not isinstance(candle, list) or len(candle) < 6:
                raise ValueError("Upstox returned a malformed minute candle")
            rows_by_timestamp[str(candle[0])] = candle[:6]
        month_start = next_month

    if not rows_by_timestamp:
        raise ValueError("Upstox returned no minute candles for the requested dates")

    safe_key = key.replace("|", "_").replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"UPSTOX_{safe_key}_{start.isoformat()}_{end.isoformat()}_1m.csv"
    fields = ["timestamp", "open", "high", "low", "close", "volume"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for timestamp, candle in sorted(rows_by_timestamp.items()):
            writer.writerow(dict(zip(fields, candle)))

    metadata = {
        "provider": "Upstox V3 historical candle API",
        "exchange": key.split("|", 1)[0],
        "instrument_key": key,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "interval": "1 minute",
        "row_count": len(rows_by_timestamp),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_type": "intraday OHLCV; provider values and timestamps as returned",
        "note": "Check current Upstox data access and exchange terms before commercial use.",
    }
    csv_path.with_suffix(".source.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path
