"""Cached data access layer around yfinance."""
import datetime as dt
import json
import urllib.request

import pandas as pd
import streamlit as st
import yfinance as yf

# 三大法人買賣超日報 (T86) on TWSE's RWD endpoint — per-stock daily institutional
# net buy/sell for the whole market. The openapi.twse.com.tw feed only exposes
# aggregate/top-20 foreign holdings, not this per-stock table, so we use RWD.
_TWSE_T86_URL = (
    "https://www.twse.com.tw/rwd/zh/fund/T86?date={date}&selectType=ALL&response=json"
)
_TWSE_T86_HEADERS = {"User-Agent": "Mozilla/5.0 (stock-market-trends-app)"}


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_multi_close(tickers: list[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    closes = {}
    for t in tickers:
        df = get_price_history(t, period=period, interval=interval)
        if not df.empty:
            closes[t] = df["Close"]
    return pd.DataFrame(closes).dropna(how="all")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _get_company_info_cached(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).get_info() or {}
    except Exception:
        return {}


def get_company_info(ticker: str) -> dict:
    """Company info dict from yfinance, cached 6h. A transient network failure
    returns {}; we evict that empty result from the cache so the next render
    retries instead of showing e.g. a US ticker with no company name for 6h
    (TW names come from a local dict and don't have this problem)."""
    info = _get_company_info_cached(ticker)
    if not info:
        try:
            _get_company_info_cached.clear(ticker)
        except Exception:
            pass
    return info


@st.cache_data(ttl=3600, show_spinner=False)
def get_twse_institutional_net(window_days: int = 5) -> dict[str, int]:
    """Per-stock 三大法人買賣超股數 summed over the most recent up to
    `window_days` TWSE trading days, keyed by bare stock code (e.g. "2330").

    Positive = net institutional buying. Walks back day by day (skipping
    weekends and any non-trading day, where the feed returns stat != "OK"),
    accumulating until `window_days` trading days are collected or a ~3-week
    calendar cap is hit. Returns {} on total failure. The net figure is the
    last column of each T86 row (三大法人買賣超股數).
    """
    net: dict[str, int] = {}
    collected = 0
    today = dt.date.today()
    for back in range(0, 21):
        if collected >= window_days:
            break
        d = today - dt.timedelta(days=back)
        if d.weekday() >= 5:  # skip Sat/Sun before spending a request
            continue
        url = _TWSE_T86_URL.format(date=d.strftime("%Y%m%d"))
        try:
            req = urllib.request.Request(url, headers=_TWSE_T86_HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                payload = json.loads(resp.read())
        except Exception:
            continue
        if payload.get("stat") != "OK" or not payload.get("data"):
            continue
        for row in payload["data"]:
            if not row:
                continue
            code = (row[0] or "").strip()
            try:
                val = int(str(row[-1]).replace(",", "").strip())
            except (ValueError, TypeError, AttributeError):
                continue
            net[code] = net.get(code, 0) + val
        collected += 1
    return net


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_fundamentals_table(tickers: list[str]) -> pd.DataFrame:
    fields = {
        "shortName": "公司名稱",
        "sector": "產業",
        "marketCap": "市值",
        "trailingPE": "P/E (TTM)",
        "forwardPE": "預估 P/E",
        "priceToBook": "P/B",
        "trailingEps": "EPS (TTM)",
        "revenueGrowth": "營收成長率",
        "earningsGrowth": "盈餘成長率",
        "profitMargins": "淨利率",
        "returnOnEquity": "ROE",
        "dividendYield": "股息率",
        "beta": "Beta",
        "fiftyTwoWeekHigh": "52週高",
        "fiftyTwoWeekLow": "52週低",
    }
    rows = {}
    for t in tickers:
        info = get_company_info(t)
        rows[t] = {label: info.get(key) for key, label in fields.items()}
    return pd.DataFrame(rows).T
