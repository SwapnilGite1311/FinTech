"""Small client for TejHQ's keyless end-of-day OHLCV endpoint."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import csv
import json


BASE_URL = "https://api.tejhq.dev/v1/ohlcv/nse"


def fetch_eod_csv(symbol: str, start: date, end: date, output_dir: Path) -> Path:
    if start > end:
        raise ValueError("Start date must be on or before end date")
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol or not normalized_symbol.replace("&", "").replace("-", "").isalnum():
        raise ValueError("Symbol must contain letters or numbers, with optional hyphens or ampersands")

    query = urlencode({"from": start.isoformat(), "to": end.isoformat()})
    url = f"{BASE_URL}/{quote(normalized_symbol, safe='')}?{query}"
    request = Request(url, headers={"User-Agent": "fintech-trading-research/0.1"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ValueError(f"Data provider returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise ValueError(f"Could not reach TejHQ data API: {exc.reason}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Data provider returned an invalid JSON response") from exc

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
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
