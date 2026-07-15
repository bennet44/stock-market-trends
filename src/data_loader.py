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


# Fallback price source: Yahoo's v8 chart REST endpoint, hit directly. Some
# environments have a broken yfinance (e.g. yfinance 1.4.1 here returns
# "possibly delisted" for every ticker via its crumb/cookie path) while this
# plain endpoint works once given a browser User-Agent. Only used when yfinance
# yields nothing, so cloud behaviour is unchanged.
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?range={range}&interval={interval}"
_CHART_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _chart_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """OHLCV DataFrame from the chart endpoint, split/dividend-adjusted to match
    yfinance's auto_adjust=True. Empty DataFrame on any failure."""
    url = _CHART_URL.format(t=ticker, range=period, interval=interval)
    req = urllib.request.Request(url, headers=_CHART_HEADERS)
    try:
        payload = json.loads(urllib.request.urlopen(req, timeout=20).read())
        res = payload["chart"]["result"][0]
        idx = pd.to_datetime([dt.datetime.fromtimestamp(t, dt.UTC).date() for t in res["timestamp"]])
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame(
            {"Open": q["open"], "High": q["high"], "Low": q["low"],
             "Close": q["close"], "Volume": q["volume"]},
            index=idx,
        )
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
        if adj is not None:
            ratio = (pd.Series(adj, index=idx) / df["Close"]).where(df["Close"] > 0, 1.0)
            for c in ("Open", "High", "Low", "Close"):
                df[c] = df[c] * ratio
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception:
        df = pd.DataFrame()
    if df.empty:  # broken/rate-limited yfinance → try the chart endpoint directly
        df = _chart_ohlcv(ticker, period, interval)
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
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


# 台股 ETF 成份股：yfinance 的 funds_data 只涵蓋部分台股 ETF（主動式 ETF 如
# 00997A 完全沒有），改抓 MoneyDJ 的 ETF 持股明細頁作為 fallback 資料源。
_MONEYDJ_ETF_HOLDINGS_URL = (
    "https://www.moneydj.com/ETF/X/Basic/Basic0007a.xdjhtm?etfid={code}.TW"
)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_tw_etf_holdings(ticker: str) -> pd.DataFrame:
    """台股 ETF 持股明細 from MoneyDJ, for ETFs yfinance has no holdings for.
    Returns DataFrame[名稱, 權重] (權重 = float %, 依權重降冪), empty on failure."""
    import io

    code = ticker.split(".")[0].strip().upper()
    url = _MONEYDJ_ETF_HOLDINGS_URL.format(code=code)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return pd.DataFrame()
    for t in tables:
        cols = [str(c) for c in t.columns]
        if "股票名稱" in cols and "比例" in cols:
            df = t[["股票名稱", "比例"]].copy()
            df.columns = ["名稱", "權重"]
            df["權重"] = pd.to_numeric(df["權重"], errors="coerce")
            df = df.dropna(subset=["名稱", "權重"])
            if not df.empty:
                return df.sort_values("權重", ascending=False).reset_index(drop=True)
    return pd.DataFrame()


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
_TWSE_MI_INDEX_URL = (
    "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
    "?date={date}&type=ALLBUT0999&response=json"
)
# TWSE 上市產業別代碼 (t187ap03_L「產業別」欄位)
_TW_INDUSTRY_NAMES: dict[str, str] = {
    "01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織纖維",
    "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙",
    "10": "鋼鐵", "11": "橡膠", "12": "汽車", "14": "建材營造",
    "15": "航運", "16": "觀光餐旅", "17": "金融保險", "18": "貿易百貨",
    "20": "其他", "21": "化學", "22": "生技醫療", "23": "油電燃氣",
    "24": "半導體", "25": "電腦及週邊", "26": "光電", "27": "通信網路",
    "28": "電子零組件", "29": "電子通路", "30": "資訊服務", "31": "其他電子",
    "35": "綠能環保", "36": "數位雲端", "37": "運動休閒", "38": "居家生活",
    "91": "存託憑證",
}


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
        # rows: [單位名稱, 買進金額(元), 賣出金額(元), 買賣差額(元)]
        buckets: dict[str, int] = {"外資": 0, "投信": 0, "自營商": 0}
        for row in payload["data"]:
            if not row or len(row) < 4:
                continue
            raw_name = str(row[0]).strip()
            try:
                net = int(str(row[3]).replace(",", "").strip())
            except (ValueError, TypeError):
                continue
            if "外資" in raw_name:
                buckets["外資"] += net
            elif "投信" in raw_name:
                buckets["投信"] += net
            elif "自營商" in raw_name:
                buckets["自營商"] += net
        result = [{"name": k, "net_bn": round(v / 100_000_000, 1)} for k, v in buckets.items()]
        if any(r["net_bn"] != 0 for r in result):
            return result
    return []


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _get_tw_industry_codes() -> dict[str, str]:
    """{stock code: 產業別代碼} for all TWSE-listed companies, from the same
    上市公司基本資料 feed as get_twse_company_names. Returns {} on any failure."""
    try:
        req = urllib.request.Request(_TWSE_COMPANY_URL, headers=_TWSE_T86_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            rows = json.loads(resp.read())
    except Exception:
        return {}
    return {
        code: ind
        for row in rows
        if (code := (row.get("公司代號") or "").strip())
        and (ind := (row.get("產業別") or "").strip())
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_tw_industry_flow() -> pd.DataFrame:
    """各產業今日漲跌家數, from TWSE MI_INDEX 每日收盤行情 (per-stock 漲跌(+/-))
    grouped by 上市公司基本資料 的產業別.
    Returns DataFrame[industry, count_up, count_flat, count_down, net_score], or empty."""
    industry_of = _get_tw_industry_codes()
    if not industry_of:
        return pd.DataFrame()
    today = dt.date.today()
    for back in range(7):
        d = today - dt.timedelta(days=back)
        if d.weekday() >= 5:
            continue
        url = _TWSE_MI_INDEX_URL.format(date=d.strftime("%Y%m%d"))
        try:
            req = urllib.request.Request(url, headers=_TWSE_T86_HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                payload = json.loads(resp.read())
        except Exception:
            continue
        if payload.get("stat") != "OK":
            continue
        quote_table = next(
            (t for t in payload.get("tables", [])
             if "證券代號" in (t.get("fields") or []) and t.get("data")),
            None,
        )
        if quote_table is None:
            continue
        sign_idx = quote_table["fields"].index("漲跌(+/-)")
        buckets: dict[str, dict[str, int]] = {}
        for row in quote_table["data"]:
            if not row or len(row) <= sign_idx:
                continue
            ind_code = industry_of.get(str(row[0]).strip())
            name = _TW_INDUSTRY_NAMES.get(ind_code or "")
            if not name:
                continue  # ETF/權證等非上市公司，或未知產業
            sign = str(row[sign_idx])  # e.g. "<p style= color:red>+</p>"
            b = buckets.setdefault(name, {"up": 0, "flat": 0, "down": 0})
            if "+" in sign:
                b["up"] += 1
            elif "-" in sign:
                b["down"] += 1
            else:
                b["flat"] += 1
        if buckets:
            return pd.DataFrame(
                {"industry": name, "count_up": b["up"], "count_flat": b["flat"],
                 "count_down": b["down"], "net_score": b["up"] - b["down"]}
                for name, b in buckets.items()
            )
    return pd.DataFrame()


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
