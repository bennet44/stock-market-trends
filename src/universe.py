"""Stock universe helpers — S&P 500/US watchlist and Taiwan stock/ETF lists."""
import pandas as pd
import streamlit as st

from . import data_loader as dl

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

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
    """Chinese name for a TW code. Prefers the small curated list (covers ETFs
    and stays stable offline), then falls back to the full TWSE 上市公司 name
    map so any TWSE-listed stock — not just the curated watchlist — gets a
    Chinese name for its header and zh-TW news query. None if unknown (e.g. an
    OTC/.TWO code, or offline with nothing curated)."""
    code = ticker.split(".")[0]
    return _TW_NAMES.get(code) or dl.get_twse_company_names().get(code)


def get_twse_tickers() -> list[str]:
    """Curated Taiwan universe: individual stocks + ETFs, both TWSE-listed."""
    return sorted(set(_TW_STOCK_TICKERS) | set(_TW_ETF_TICKERS))


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

