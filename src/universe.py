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

