"""Lightweight daily-close fetcher hitting Yahoo's v8 chart API directly.

Exists because yfinance 1.4.1 on this machine returns "possibly delisted"
for every ticker (a broken crumb/cookie path), while the plain chart endpoint
works fine once we send a browser User-Agent and route TLS through the system
trust store (Norton re-signs HTTPS here — see the repo's sitecustomize.py).

Pure stdlib + pandas, no Streamlit import, so both build_fcn_excel.py and
train_fcn.py can reuse it off the main app.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request

import pandas as pd

try:  # match the app's TLS handling for the Norton-re-signed cert chain
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?period1={p1}&period2={p2}&interval=1d"
)
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _to_epoch(d: dt.date) -> int:
    return int(time.mktime(d.timetuple()))


def fetch_closes(ticker: str, start: str = "2024-06-01", end: str | None = None,
                 tries: int = 4) -> pd.Series:
    """Adjusted daily closes for ``ticker`` between ``start`` and ``end``
    (ISO dates, inclusive), indexed by date. Empty Series on failure."""
    end_date = dt.date.fromisoformat(end) if end else dt.date.today()
    url = _CHART_URL.format(
        ticker=ticker,
        p1=_to_epoch(dt.date.fromisoformat(start)),
        p2=_to_epoch(end_date) + 86400,
    )
    req = urllib.request.Request(url, headers=_HEADERS)
    for attempt in range(tries):
        try:
            payload = json.loads(urllib.request.urlopen(req, timeout=20).read())
            res = payload["chart"]["result"][0]
            ts = res["timestamp"]
            quote = res["indicators"]["quote"][0]["close"]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", quote)
            idx = pd.to_datetime([dt.datetime.fromtimestamp(t, dt.UTC).date() for t in ts])
            return pd.Series(adj, index=idx, dtype="float64").dropna()
        except Exception:
            if attempt < tries - 1:
                time.sleep(1.5)
    return pd.Series(dtype="float64")


def close_on(series: pd.Series, date: str) -> float | None:
    """The last close on or before ``date`` (ISO), or None if none exists."""
    sub = series[series.index <= pd.Timestamp(date)]
    return round(float(sub.iloc[-1]), 2) if len(sub) else None
