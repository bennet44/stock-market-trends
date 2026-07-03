"""Stock universe helpers — S&P 500/US watchlist and Taiwan stock/ETF lists."""
import json
import re
import ssl
import urllib.request

import pandas as pd
import streamlit as st

from . import data_loader as dl

_TWSE_HEADERS = {"User-Agent": "Mozilla/5.0 (stock-market-trends-app)"}
_TWSE_DAY_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
_TPEX_QUOTES_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"

# Some Windows environments fail TWSE/TPEx SSL chain verification; bypass it for
# these read-only market-data endpoints (no sensitive data transmitted).
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _fetch_json(url: str, timeout: int = 10) -> list[dict]:
    req = urllib.request.Request(url, headers=_TWSE_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_common_stock(code: str) -> bool:
    return bool(code) and code.isdigit() and len(code) == 4


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_twse_top_market_cap(n: int = 200) -> list[str]:
    """Top n TWSE stocks by daily trading value (STOCK_DAY_ALL, cached 24 h).

    Uses TradeValue (成交金額) as a liquid-cap proxy — high-turnover stocks are
    almost always the largest-cap names on the TWSE. Falls back to the curated
    list if the API is unreachable.
    """
    try:
        rows = _fetch_json(_TWSE_DAY_ALL_URL)
        df = pd.DataFrame(rows)
        df = df[df["Code"].apply(_is_common_stock)].copy()
        df["_tv"] = pd.to_numeric(
            df["TradeValue"].astype(str).str.replace(",", ""), errors="coerce"
        )
        df = df.dropna(subset=["_tv"]).sort_values("_tv", ascending=False)
        return [f"{c}.TW" for c in df["Code"].head(n).tolist()]
    except Exception:
        return _TW_STOCK_TICKERS


@st.cache_data(ttl=3600, show_spinner=False)
def get_twse_top_volume(n: int = 20) -> list[str]:
    """Top n TWSE stocks by single-day trading volume (STOCK_DAY_ALL, cached 1 h)."""
    try:
        rows = _fetch_json(_TWSE_DAY_ALL_URL)
        df = pd.DataFrame(rows)
        df = df[df["Code"].apply(_is_common_stock)].copy()
        df["_vol"] = pd.to_numeric(
            df["TradeVolume"].astype(str).str.replace(",", ""), errors="coerce"
        )
        df = df.dropna(subset=["_vol"]).sort_values("_vol", ascending=False)
        return [f"{c}.TW" for c in df["Code"].head(n).tolist()]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_tpex_top_volume(n: int = 20) -> list[str]:
    """Top n TPEx (上櫃) stocks by trading volume (tpex_mainboard_quotes, cached 1 h)."""
    try:
        rows = _fetch_json(_TPEX_QUOTES_URL)
        df = pd.DataFrame(rows)
        # TPEx API uses 'SecuritiesCompanyCode' for the stock code
        code_col = next((c for c in df.columns if "code" in c.lower()), None)
        vol_col = next((c for c in df.columns if "share" in c.lower() or "volume" in c.lower()), None)
        if code_col is None or vol_col is None:
            return []
        df = df[df[code_col].apply(_is_common_stock)].copy()
        df["_vol"] = pd.to_numeric(
            df[vol_col].astype(str).str.replace(",", ""), errors="coerce"
        )
        df = df.dropna(subset=["_vol"]).sort_values("_vol", ascending=False)
        return [f"{c}.TWO" for c in df[code_col].head(n).tolist()]
    except Exception:
        return []


_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
_DOW_WIKI_URL = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"

# Small, definitely-correct fallback used only if the live Wikipedia fetch
# fails (e.g. no network access or the page structure changed), so "ALL"
# mode still returns something usable.
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "LLY", "AVGO",
    "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "MRK", "COST",
    "ABBV", "CVX", "CRM", "NFLX", "AMD", "PEP", "KO", "WMT", "BAC", "TMO",
    "ADBE", "MCD", "CSCO", "ABT", "ORCL", "ACN", "LIN", "DHR", "WFC", "DIS",
    "TXN", "PM", "INTU", "VZ", "CMCSA", "IBM", "NOW", "CAT", "GE", "UNP",
]

# Candidate pool of historically high-volume US tickers (large caps, popular
# retail/momentum names, leveraged ETFs) used to derive a "top N by recent
# volume" sub-universe. This is a heuristic watchlist, not a live market-wide
# volume screener.
_HIGH_VOLUME_CANDIDATES = [
    "AAPL", "TSLA", "NVDA", "AMD", "AMZN", "META", "MSFT", "GOOGL", "NFLX", "BAC",
    "F", "T", "INTC", "PFE", "NIO", "SOFI", "PLTR", "RIVN", "LCID", "AAL",
    "CCL", "PLUG", "SNAP", "UBER", "PYPL", "XOM", "WBD", "KVUE", "VALE", "ITUB",
    "SIRI", "GRAB", "MARA", "RIOT", "COIN", "SOXL", "TQQQ", "SQQQ", "SPY", "QQQ",
]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    """Live S&P 500 ticker list scraped from Wikipedia, cached for a day.

    Falls back to a short list of well-known constituents if the fetch
    fails, so "ALL" mode keeps working without network access to Wikipedia.
    """
    try:
        tables = pd.read_html(_WIKI_URL)
        symbols = (
            tables[0]["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
        )
        tickers = sorted(set(symbols.tolist()))
        if len(tickers) >= 400:
            return tickers
    except Exception:
        pass
    return _FALLBACK_TICKERS


# Small fallbacks for the Nasdaq-100/Dow fetchers below, same role as
# _FALLBACK_TICKERS — only used if the live Wikipedia fetch fails.
_NASDAQ100_FALLBACK = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST",
    "NFLX", "ADBE", "PEP", "CSCO", "AMD", "INTC", "QCOM", "TXN", "INTU", "AMGN",
]
_DOW_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "JPM", "JNJ", "V", "PG", "HD", "UNH", "MRK",
    "CVX", "KO", "MCD", "CAT", "DIS", "IBM", "GS", "CSCO", "NKE", "WMT",
]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_nasdaq100_tickers() -> list[str]:
    """Live Nasdaq-100 index constituent list scraped from Wikipedia, cached
    for a day. Falls back to a short well-known subset if the fetch fails."""
    try:
        tables = pd.read_html(_NASDAQ100_WIKI_URL)
        for table in tables:
            cols = [str(c) for c in table.columns]
            ticker_col = next((c for c in cols if c.lower() in ("ticker", "symbol")), None)
            if ticker_col and len(table) >= 90:
                symbols = table[ticker_col].astype(str).str.strip().str.replace(".", "-", regex=False)
                return sorted(set(symbols.tolist()))
    except Exception:
        pass
    return _NASDAQ100_FALLBACK


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_dow_tickers() -> list[str]:
    """Live Dow Jones Industrial Average constituent list scraped from
    Wikipedia, cached for a day. Falls back to a short well-known subset if
    the fetch fails."""
    try:
        tables = pd.read_html(_DOW_WIKI_URL)
        for table in tables:
            cols = [str(c) for c in table.columns]
            ticker_col = next((c for c in cols if c.lower() in ("symbol", "ticker")), None)
            if ticker_col and 25 <= len(table) <= 35:
                symbols = table[ticker_col].astype(str).str.strip().str.replace(".", "-", regex=False)
                return sorted(set(symbols.tolist()))
    except Exception:
        pass
    return _DOW_FALLBACK


@st.cache_data(ttl=3600, show_spinner=False)
def get_top_volume_tickers(n: int = 30) -> list[str]:
    """Rank a curated watchlist of typically-liquid tickers by recent average
    daily volume (last 10 trading days) and return the top n symbols.
    """
    volumes = {}
    for t in _HIGH_VOLUME_CANDIDATES:
        df = dl.get_price_history(t, period="1mo")
        if not df.empty:
            volumes[t] = df["Volume"].tail(10).mean()
    ranked = sorted(volumes, key=volumes.get, reverse=True)
    return ranked[:n]


# Curated list of large/liquid Taiwan individual stocks (TWSE-listed), given
# as bare 4-digit codes; Yahoo Finance needs the ".TW" suffix to resolve them.
_TW_STOCK_CODES = [
    "2330", "2317", "2454", "2412", "2882", "2881", "1301", "2308", "2303", "2002",
    "3008", "2891", "2884", "2885", "1216", "2207", "2603", "2609", "2615", "3034",
    "3037", "3711", "2379", "6505", "5871", "2890", "2880", "1303", "1101", "9910",
    "2912", "4904", "3045", "2357", "2356", "2382", "2395", "6669", "3661", "6446",
]

# Curated list of popular Taiwan-listed ETFs (bare codes, same ".TW" suffix rule).
_TW_ETF_CODES = [
    "0050", "0056", "006208", "00878", "00919", "00929", "00940", "00713",
    "00692", "00701", "00733", "00850", "00891", "00900", "00905", "00961",
]

_TW_STOCK_TICKERS = [f"{c}.TW" for c in _TW_STOCK_CODES]
_TW_ETF_TICKERS = [f"{c}.TW" for c in _TW_ETF_CODES]

# Full ISIN-by-security-type listing — strMode=2 is the only mode that
# includes newly issued actively-managed ETFs (e.g. 00997A); strMode=4
# (which looks like the dedicated "ETF" mode) turned out to be stale and
# missing them, so this scrapes the comprehensive listing and slices out
# the "ETF" section instead.
_TWSE_ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
_TWSE_ISIN_HEADERS = {"User-Agent": "Mozilla/5.0 (stock-market-trends-app)"}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _fetch_twse_etf_rows() -> list[tuple[str, str]] | None:
    """(code, Chinese name) for every TWSE-listed ETF, parsed from the same
    ISIN page get_twse_etf_tickers/get_twse_etf_names both build on. Returns
    None on any fetch/parse failure or a too-thin result (TWSE has had 150+
    listed ETFs for years, so a thin parse means the page layout changed,
    not that ETFs were delisted) — callers fall back to their own curated
    data in that case.
    """
    try:
        req = urllib.request.Request(_TWSE_ISIN_URL, headers=_TWSE_ISIN_HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("big5", errors="replace")
        section_starts = {m.group(1).strip(): m.start() for m in re.finditer(r"<B>\s*([^<]+?)\s*<B>", html)}
        etf_start, etn_start = section_starts.get("ETF"), section_starts.get("ETN")
        if etf_start is None or etn_start is None or etn_start <= etf_start:
            return None
        section = html[etf_start:etn_start]
        rows = re.findall(r"<td bgcolor=#FAFAD2>(\d{4,6}[A-Z]?)\s*([^<]*?)</td><td bgcolor=#FAFAD2>TW", section)
        return rows if len(rows) >= 100 else None
    except Exception:
        return None


def get_twse_etf_tickers() -> list[str]:
    """Live list of every TWSE-listed ETF, scraped from TWSE's full
    securities-by-ISIN page. The curated _TW_ETF_TICKERS fallback above
    predates most actively-managed ETFs (00xxxA-style codes) and isn't
    maintained by hand, so this is what get_twse_tickers() actually uses;
    the curated list only kicks in if this fetch/parse fails outright.
    """
    rows = _fetch_twse_etf_rows()
    if rows is None:
        return _TW_ETF_TICKERS
    return sorted({f"{code}.TW" for code, _ in rows})


def get_twse_etf_names() -> dict[str, str]:
    """{bare code: Chinese name} for every TWSE-listed ETF, from the same
    ISIN page scrape as get_twse_etf_tickers — fills the gap left by
    get_twse_company_names(), which only covers 公司 (companies), not ETFs
    (funds), so ETF codes outside the small curated _TW_NAMES list used to
    fall back to yfinance's English shortName. Returns {} on fetch failure.
    """
    rows = _fetch_twse_etf_rows()
    return {code: name for code, name in rows} if rows else {}

# Traditional-Chinese names for the curated codes above. Yahoo Finance's
# "shortName" for TWSE tickers comes back in English (e.g. "Taiwan
# Semiconductor Mfg"), so the curated lists carry their own Chinese names.
_TW_NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2412": "中華電",
    "2882": "國泰金", "2881": "富邦金", "1301": "台塑", "2308": "台達電",
    "2303": "聯電", "2002": "中鋼", "3008": "大立光", "2891": "中信金",
    "2884": "玉山金", "2885": "元大金", "1216": "統一", "2207": "和泰車",
    "2603": "長榮", "2609": "陽明", "2615": "萬海", "3034": "聯詠",
    "3037": "欣興", "3711": "日月光投控", "2379": "瑞昱", "6505": "台塑化",
    "5871": "中租-KY", "2890": "永豐金", "2880": "華南金", "1303": "南亞",
    "1101": "台泥", "9910": "豐泰", "2912": "統一超", "4904": "遠傳",
    "3045": "台灣大", "2357": "華碩", "2356": "英業達", "2382": "廣達",
    "2395": "研華", "6669": "緯穎", "3661": "世芯-KY", "6446": "藥華藥",
    "0050": "元大台灣50", "0056": "元大高股息", "006208": "富邦台50",
    "00878": "國泰永續高股息", "00919": "群益台灣精選高息",
    "00929": "復華台灣科技優息", "00940": "元大台灣價值高息",
    "00713": "元大台灣高息低波", "00692": "富邦公司治理",
    "00701": "國泰股息精選30", "00733": "富邦臺灣中小",
    "00850": "元大臺灣ESG永續", "00891": "中信關鍵半導體",
    "00900": "富邦特選高股息30", "00905": "FT臺灣Smart",
    "00961": "中信成長高股息",
}


def get_tw_company_name(ticker: str) -> str | None:
    """Chinese name for a TW code. Prefers the small curated list (stays
    stable offline), then the full TWSE 上市公司 name map (covers individual
    stocks generally, but not ETFs — funds aren't "公司"), then the ETF-name
    scrape (covers ETFs specifically, including 00xxxA-style actively-managed
    ones). None if unknown (e.g. an OTC/.TWO code, or offline with nothing
    curated)."""
    code = ticker.split(".")[0]
    return (
        _TW_NAMES.get(code)
        or dl.get_twse_company_names().get(code)
        or get_twse_etf_names().get(code)
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_twse_tickers() -> list[str]:
    """Taiwan universe: TWSE top-200 by trading value + TWSE top-20 volume +
    TPEx top-20 volume + curated ETFs. Falls back to the curated stock+ETF
    lists if all live fetches fail. Shared by 買賣建議 and 存股區 so both
    tabs scan the same universe.
    """
    mc200 = get_twse_top_market_cap(200)
    twvol20 = get_twse_top_volume(20)
    tpvol20 = get_tpex_top_volume(20)
    combined = sorted(set(mc200) | set(twvol20) | set(tpvol20) | set(_TW_ETF_TICKERS))
    return combined if combined else sorted(set(_TW_STOCK_TICKERS) | set(_TW_ETF_TICKERS))


def normalize_tw_ticker(raw: str) -> str:
    """Append the Yahoo Finance ".TW" suffix to a bare Taiwan stock/ETF code.

    Leaves tickers that already carry an exchange suffix (e.g. "2330.TW",
    "6188.TWO") untouched.
    """
    raw = raw.strip().upper()
    if not raw or "." in raw:
        return raw
    return f"{raw}.TW"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def resolve_tw_ticker(raw: str) -> str:
    """Resolve a bare Taiwan code to whichever Yahoo Finance suffix has data.

    Bare codes are ambiguous between TWSE-listed (".TW") and TPEx/OTC-listed
    (".TWO") stocks (e.g. 3685 is OTC, not TWSE), so a fixed ".TW" suffix
    silently fails for OTC codes. Tries ".TW" first (the common case), falls
    back to ".TWO" if that has no price history, and otherwise returns the
    ".TW" guess unchanged (e.g. when offline) so callers still get a usable
    ticker string.
    """
    candidate = normalize_tw_ticker(raw)
    if "." in raw.strip():
        return candidate
    if not dl.get_price_history(candidate, period="5d").empty:
        return candidate
    alt = f"{raw.strip().upper()}.TWO"
    if not dl.get_price_history(alt, period="5d").empty:
        return alt
    return candidate

