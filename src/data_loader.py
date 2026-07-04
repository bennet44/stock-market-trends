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

# 上市公司基本資料 — Chinese short name (公司簡稱) for every TWSE-listed company,
# used to give any TWSE ticker a Chinese name (header + zh-TW news query) even
# when it isn't in universe._TW_NAMES's small curated list.
_TWSE_COMPANY_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"


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


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_etf_top_holdings(ticker: str) -> pd.DataFrame:
    """Top holdings (Symbol, Name, Holding Percent) for an ETF ticker, from
    yfinance's funds_data scraper. Returns an empty DataFrame for non-ETF
    tickers or on any failure (e.g. yfinance has no funds data for it)."""
    try:
        holdings = yf.Ticker(ticker).funds_data.top_holdings
    except Exception:
        return pd.DataFrame()
    return holdings if holdings is not None else pd.DataFrame()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_dividend_history(ticker: str) -> pd.Series:
    """Full historical per-share dividend payments (ex-div date -> amount)
    from yfinance. Returns an empty Series for non-payers or on any failure."""
    try:
        s = yf.Ticker(ticker).dividends
    except Exception:
        return pd.Series(dtype=float)
    return s if s is not None else pd.Series(dtype=float)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_twse_company_names() -> dict[str, str]:
    """{stock code: 公司簡稱 (Chinese short name)} for all TWSE-listed companies,
    from the 上市公司基本資料 open-data feed. Returns {} on any failure."""
    try:
        req = urllib.request.Request(_TWSE_COMPANY_URL, headers=_TWSE_T86_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            rows = json.loads(resp.read())
    except Exception:
        return {}
    names = {}
    for row in rows:
        code = (row.get("公司代號") or "").strip()
        name = (row.get("公司簡稱") or "").strip()
        if code and name:
            names[code] = name
    return names


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


# ── 市場焦點 Dashboard ───────────────────────────────────────────────────────
_US_SECTOR_ETFS: dict[str, str] = {
    "XLK": "科技", "XLC": "通訊", "XLY": "非必需消費", "XLF": "金融",
    "XLI": "工業", "XLV": "醫療保健", "XLE": "能源",
    "XLP": "必需消費", "XLB": "原物料", "XLRE": "房地產", "XLU": "公用事業",
}
_MACRO_FLOW_TICKERS: dict[str, str] = {
    "SPY": "美股大盤", "QQQ": "那斯達克",
    "TLT": "長期公債", "GLD": "黃金", "BTC-USD": "比特幣",
}
_TWSE_BFI82U_URL = (
    "https://www.twse.com.tw/rwd/zh/fund/BFI82U?type=day&dayDate={date}&response=json"
)
_TWSE_FMTQIK_URL = "https://openapi.twse.com.tw/v1/exchangeReport/FMTQIK"


def _pct_returns(tickers: list[str]) -> pd.DataFrame:
    """1d/5d % returns + last close for a list of tickers via get_multi_close."""
    mc = get_multi_close(tickers, period="10d")
    rows = []
    for t in tickers:
        c = mc[t].dropna() if (not mc.empty and t in mc.columns) else pd.Series(dtype=float)
        if len(c) >= 2:
            r1d = float((c.iloc[-1] / c.iloc[-2] - 1) * 100)
            r5d = float((c.iloc[-1] / c.iloc[max(0, len(c) - 6)] - 1) * 100) if len(c) >= 6 else float("nan")
            last = float(c.iloc[-1])
        else:
            r1d = r5d = last = float("nan")
        rows.append({"ticker": t, "return_1d": round(r1d, 2), "return_5d": round(r5d, 2), "last_close": last})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def get_us_sector_returns() -> pd.DataFrame:
    """1d/5d % returns for 11 S&P 500 sector ETFs. Columns: ticker, sector, return_1d, return_5d."""
    df = _pct_returns(list(_US_SECTOR_ETFS))
    df["sector"] = df["ticker"].map(_US_SECTOR_ETFS)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def get_macro_flow_returns() -> pd.DataFrame:
    """1d/5d % returns for SPY/QQQ/TLT/GLD/BTC-USD. Columns: ticker, label, return_1d, return_5d, last_close."""
    df = _pct_returns(list(_MACRO_FLOW_TICKERS))
    df["label"] = df["ticker"].map(_MACRO_FLOW_TICKERS)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def get_vix() -> float | None:
    """Current VIX level, or None on failure."""
    try:
        h = yf.Ticker("^VIX").history(period="2d", interval="1d", auto_adjust=True)
        return round(float(h["Close"].iloc[-1]), 2) if not h.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_twse_institutional_summary() -> list[dict]:
    """三大法人今日買賣超（億元）from TWSE BFI82U.
    Returns [{"name": "外資", "net_bn": 12.3}, ...], empty list on failure."""
    today = dt.date.today()
    for back in range(5):
        d = today - dt.timedelta(days=back)
        if d.weekday() >= 5:
            continue
        url = _TWSE_BFI82U_URL.format(date=d.strftime("%Y%m%d"))
        try:
            req = urllib.request.Request(url, headers=_TWSE_T86_HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                payload = json.loads(resp.read())
        except Exception:
            continue
        if payload.get("stat") != "OK" or not payload.get("data"):
            continue
        # rows: [機構名稱, 買進金額(千), 賣出金額(千), 買賣差額(千)]
        buckets: dict[str, int] = {"外資": 0, "投信": 0, "自營商": 0}
        for row in payload["data"]:
            if not row or len(row) < 4:
                continue
            raw_name = str(row[0]).strip()
            try:
                net_k = int(str(row[3]).replace(",", "").strip())
            except (ValueError, TypeError):
                continue
            if "外資" in raw_name:
                buckets["外資"] += net_k
            elif "投信" in raw_name:
                buckets["投信"] += net_k
            elif "自營商" in raw_name:
                buckets["自營商"] += net_k
        result = [{"name": k, "net_bn": round(v / 100_000, 1)} for k, v in buckets.items()]
        if any(r["net_bn"] != 0 for r in result):
            return result
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_tw_industry_flow() -> pd.DataFrame:
    """各類股今日成交資料 from TWSE FMTQIK.
    Returns DataFrame[industry, count_up, count_flat, count_down, net_score], or empty."""
    try:
        req = urllib.request.Request(_TWSE_FMTQIK_URL, headers=_TWSE_T86_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            rows = json.loads(resp.read())
    except Exception:
        return pd.DataFrame()
    records = []
    for row in rows:
        name = str(row.get("類股名稱") or "").strip()
        if not name or name == "合計":
            continue
        try:
            up = int(row.get("今日漲股數") or 0)
            flat = int(row.get("今日平盤股數") or 0)
            down = int(row.get("今日跌股數") or 0)
        except (ValueError, TypeError):
            up = flat = down = 0
        records.append({"industry": name, "count_up": up, "count_flat": flat,
                        "count_down": down, "net_score": up - down})
    return pd.DataFrame(records) if records else pd.DataFrame()


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
