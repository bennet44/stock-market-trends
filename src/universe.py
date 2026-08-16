"""Stock universe helpers — S&P 500/US watchlist and Taiwan stock/ETF lists."""
import json
import re
import ssl
import urllib.parse
import urllib.request

import pandas as pd
import streamlit as st

from . import data_loader as dl

_TWSE_HEADERS = {"User-Agent": "Mozilla/5.0 (stock-market-trends-app)"}
_TWSE_DAY_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
_TPEX_QUOTES_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
_YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search?q={q}&quotesCount=8&newsCount=0"
_YAHOO_SEARCH_HEADERS = {"User-Agent": "Mozilla/5.0"}

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


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_tpex_company_names() -> dict[str, str]:
    """{bare TPEx code: 公司簡稱 (Chinese short name)} for every TPEx (上櫃)
    security, from the same tpex_mainboard_quotes feed as get_tpex_top_volume.
    Fills the 上櫃 gap left by get_twse_company_names (TWSE/上市-listed only),
    which otherwise leaves .TWO tickers with no Chinese name source and
    falling back to yfinance's English shortName. Returns {} on fetch failure.
    """
    try:
        rows = _fetch_json(_TPEX_QUOTES_URL)
    except Exception:
        return {}
    return {
        r["SecuritiesCompanyCode"]: r["CompanyName"]
        for r in rows
        if r.get("SecuritiesCompanyCode") and r.get("CompanyName")
    }


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
    ones), then the TPEx 上櫃 quotes feed (covers .TWO-listed stocks, which
    the TWSE-only sources above don't). None if unknown (e.g. offline with
    nothing curated)."""
    code = ticker.split(".")[0]
    return (
        _TW_NAMES.get(code)
        or dl.get_twse_company_names().get(code)
        or get_twse_etf_names().get(code)
        or get_tpex_company_names().get(code)
    )


def search_tw_local(query: str, limit: int = 8) -> list[tuple[str, str]]:
    """(ticker, Chinese name) candidates whose code or name contains `query`
    (case-insensitive), searched across the same sources get_tw_company_name
    reads from — curated/TWSE company/TWSE ETF get ".TW", TPEx gets ".TWO".
    Lets a 台股 user find a ticker by (partial) Chinese name, English/pinyin
    fragment, or bare code without needing Yahoo's search API, which rejects
    non-ASCII queries outright. A code already resolved by an earlier source
    keeps that source's name (same priority order as get_tw_company_name).
    """
    q = query.strip().upper()
    if not q:
        return []
    pool: dict[str, tuple[str, str]] = {}  # code -> (name, suffix)
    for code, name in _TW_NAMES.items():
        pool.setdefault(code, (name, ".TW"))
    for code, name in dl.get_twse_company_names().items():
        pool.setdefault(code, (name, ".TW"))
    for code, name in get_twse_etf_names().items():
        pool.setdefault(code, (name, ".TW"))
    for code, name in get_tpex_company_names().items():
        pool.setdefault(code, (name, ".TWO"))
    matches = [
        (f"{code}{suffix}", name) for code, (name, suffix) in pool.items()
        if q in name.upper() or q in code
    ]
    matches.sort(key=lambda x: x[0])
    return matches[:limit]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def yahoo_ticker_search(query: str) -> list[tuple[str, str, str]]:
    """[(symbol, name, exchange), ...] from Yahoo Finance's search
    autocomplete endpoint, filtered to actual tradeable securities
    (quoteType EQUITY/ETF — skips options/mutual funds/index/crypto noise).
    Returns [] on any failure, including the 400 "Invalid Search Query"
    Yahoo returns for non-ASCII input (e.g. Chinese) — callers combining
    this with search_tw_local() rely on that silent no-op rather than a
    raised exception for Chinese queries."""
    query = query.strip()
    if not query:
        return []
    url = _YAHOO_SEARCH_URL.format(q=urllib.parse.quote(query))
    try:
        req = urllib.request.Request(url, headers=_YAHOO_SEARCH_HEADERS)
        payload = json.loads(urllib.request.urlopen(req, timeout=10).read())
    except Exception:
        return []
    return [
        (r["symbol"], r.get("shortname") or r.get("longname") or r["symbol"], r.get("exchange", ""))
        for r in payload.get("quotes", [])
        if r.get("symbol") and r.get("quoteType") in ("EQUITY", "ETF")
    ]


def search_ticker(query: str, is_tw: bool, limit: int = 8) -> list[tuple[str, str]]:
    """(ticker, display name) candidates for the "忘記代號" search box.

    TW: local dict matches (search_tw_local — covers Chinese names, which
    Yahoo's search can't) come first, then Yahoo results filtered to
    .TW/.TWO symbols fill in anything the local lists miss. US: Yahoo
    results filtered to symbols without a "." (this app's existing
    convention for "not a TW/foreign-exchange ticker"). Deduped by ticker,
    capped to `limit`.
    """
    seen: dict[str, str] = {}
    if is_tw:
        for ticker, name in search_tw_local(query, limit):
            seen.setdefault(ticker, name)
        for symbol, name, _exch in yahoo_ticker_search(query):
            if symbol.endswith((".TW", ".TWO")):
                seen.setdefault(symbol, name)
    else:
        for symbol, name, _exch in yahoo_ticker_search(query):
            if "." not in symbol:
                seen.setdefault(symbol, name)
    return list(seen.items())[:limit]


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


# ---------- 債券 ETF ----------
# 台股債券 ETF 的代號慣例是末碼 B（00679B、00937B…）。這類代號被
# _is_common_stock 的「剛好 4 位數字」規則擋在成交量／市值榜之外，而
# _TW_ETF_CODES 白名單裡也只有股票型 ETF，所以債券完全進不了掃描池——
# 這是副作用而非刻意排除，需要一個獨立入口把它們找回來。
def _is_tw_bond_etf_code(code: str) -> bool:
    return bool(code) and code.upper().endswith("B") and code[:-1].isdigit()


# 美股債券 ETF 名單穩定（不像台股每年新發一堆），直接列出並附上
# (天期, 類別) — 這兩項是債券真正該看的維度，而 yfinance 的 info 拿不到。
_US_BOND_ETFS: dict[str, tuple[str, str]] = {
    "TLT": ("長天期", "美國公債"),
    "IEF": ("中天期", "美國公債"),
    "SHY": ("短天期", "美國公債"),
    "SHV": ("短天期", "美國公債"),
    "GOVT": ("綜合", "美國公債"),
    "AGG": ("綜合", "投資等級"),
    "BND": ("綜合", "投資等級"),
    "LQD": ("中天期", "投資級公司債"),
    "VCIT": ("中天期", "投資級公司債"),
    "VCSH": ("短天期", "投資級公司債"),
    "HYG": ("中天期", "非投資等級"),
    "JNK": ("中天期", "非投資等級"),
    "TIP": ("中天期", "抗通膨債"),
    "EMB": ("中天期", "新興市場債"),
}

# 台股債券 ETF 的天期／類別由中文名稱關鍵字推斷（名稱本身就寫著
# 「20年」「7-10」「1-3」）。推不出來就回「—」，不亂猜。
_TW_BOND_TENOR_KEYWORDS = [
    (("20年", "20+", "15+", "25年", "30年", "10Y+", "長天期"), "長天期"),
    (("7-10", "5-10", "10年", "7年", "中天期"), "中天期"),
    (("0-1", "1-3", "0-3", "0-5", "1-5", "3年", "短期", "短天期"), "短天期"),
]
_TW_BOND_CLASS_KEYWORDS = [
    # 非投等要排在投等前面：「非投等債」同時含「投等」，順序反了會歸錯類。
    # 「非投債」是常見簡寫（第一金優選非投債、玉山嚴選非投債）。
    # 「優選／嚴選」是行銷詞不是信用等級——第一金優選「非投債」正是非投等，
    # 曾因把「優選」當投資級關鍵字而誤判，故不列入。
    (("非投等", "非投資等級", "非投債", "高收益"), "非投資等級"),
    (("投等", "投資級", "投資等級", "A級"), "投資等級"),
    (("新興",), "新興市場債"),
    (("公債", "美債", "政府"), "公債"),
    (("金融債",), "金融債"),
    (("公司債",), "公司債"),
]


@st.cache_data(ttl=3600, show_spinner=False)
def get_tw_bond_etf_tickers(n: int = 20) -> list[str]:
    """流動性最好的 n 檔台股債券 ETF（代號末碼 B）。

    上櫃走 TPEx quotes feed（有 TradingShares 可排序流動性），上市補
    ISIN ETF 清單（該來源沒有成交量，排在上櫃之後）。回傳含 Yahoo 後綴。
    """
    picked: list[str] = []
    try:
        rows = _fetch_json(_TPEX_QUOTES_URL)
        bonds = [r for r in rows if _is_tw_bond_etf_code(r.get("SecuritiesCompanyCode", ""))]

        def _vol(r: dict) -> float:
            try:
                return float(str(r.get("TradingShares", "0")).replace(",", "") or 0)
            except ValueError:
                return 0.0

        bonds.sort(key=_vol, reverse=True)
        picked = [f"{r['SecuritiesCompanyCode']}.TWO" for r in bonds[:n]]
    except Exception:
        pass
    if len(picked) < n:
        listed = [f"{c}.TW" for c, _ in (_fetch_twse_etf_rows() or [])
                  if _is_tw_bond_etf_code(c)]
        picked += [t for t in listed if t not in picked][: n - len(picked)]
    return picked


def get_us_bond_etf_tickers() -> list[str]:
    """美股主要債券 ETF（公債／投等／非投等／抗通膨／新興市場，涵蓋短中長天期）。"""
    return sorted(_US_BOND_ETFS)


def bond_profile(ticker: str) -> tuple[str, str] | None:
    """(天期, 類別) for a bond ETF, or None when `ticker` isn't one.

    美股查固定名單；台股先確認代號末碼為 B，再從中文名稱關鍵字推斷——
    名稱本身就帶天期資訊（「元大美債20年」「富邦美債1-3」）。任一維度
    推不出來就回「—」，不用猜的填。
    """
    bare = ticker.split(".")[0].upper()
    if bare in _US_BOND_ETFS:
        return _US_BOND_ETFS[bare]
    if not _is_tw_bond_etf_code(bare):
        return None
    name = get_tw_company_name(ticker) or ""
    tenor = next((label for keys, label in _TW_BOND_TENOR_KEYWORDS
                  if any(k in name for k in keys)), "—")
    klass = next((label for keys, label in _TW_BOND_CLASS_KEYWORDS
                  if any(k in name for k in keys)), "—")
    return tenor, klass


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

