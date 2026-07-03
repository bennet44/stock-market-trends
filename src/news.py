"""News/filing headlines for a ticker from several sources, plus a
lightweight keyword-based sentiment score used as a recommendation factor.

get_recent_news() (Google News RSS, zh-TW) is the only source wired into
src/recommend.py's bulk sentiment scan across hundreds of tickers. The
other fetchers here (Reuters, SEC EDGAR, TWSE, MOPS) are for the
single-ticker detail view only — they're slower/rate-limited sources that
wouldn't hold up well fetched for an entire stock universe.
"""
import datetime as dt
import json
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import streamlit as st

_RSS_URL = "https://news.google.com/rss/search?q={query}&hl={hl}&gl={gl}&ceid={ceid}"
_TIMEOUT = 6

# Some Windows environments (AV/proxy-injected cert chains) fail SSL chain
# verification for these read-only public endpoints; bypass it, same as
# universe.py's TWSE/TPEx fetches (no sensitive data transmitted).
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# SEC requires a descriptive User-Agent identifying the requester on every
# request, or it returns 403 — see https://www.sec.gov/os/webmaster-faq#developers.
_SEC_HEADERS = {"User-Agent": "stock-market-trends-app (contact: github.com/bennet44/stock-market-trends)"}
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

_TWSE_NEWS_LIST_URL = "https://openapi.twse.com.tw/v1/news/newsList"

_POSITIVE_WORDS = [
    "上漲", "大漲", "看好", "升評", "優於預期", "創新高", "買進", "增持",
    "獲利", "成長", "樂觀", "飆漲", "突破", "強勢", "上修",
]
_NEGATIVE_WORDS = [
    "下跌", "大跌", "看壞", "降評", "不如預期", "創新低", "賣出", "減持",
    "虧損", "衰退", "悲觀", "重挫", "跳水", "下修", "示警",
]
_POSITIVE_WORDS_EN = [
    "surge", "soar", "rally", "jump", "gain", "upgrade", "beat", "record high",
    "buy rating", "outperform", "bullish", "profit", "growth", "raises",
]
_NEGATIVE_WORDS_EN = [
    "plunge", "slump", "tumble", "drop", "fall", "downgrade", "miss", "record low",
    "sell rating", "underperform", "bearish", "loss", "decline", "cuts", "lawsuit",
]


def _fetch_google_news(query: str, hl: str, gl: str, ceid: str, days: int) -> list[dict]:
    """Shared Google News RSS fetch/parse/date-filter used by every
    Google-News-backed source below. Returns [] on any network or parse
    failure."""
    url = _RSS_URL.format(query=urllib.parse.quote(query), hl=hl, gl=gl, ceid=ceid)
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT, context=_SSL_CTX) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
    except Exception:
        return []

    today = dt.datetime.now(dt.timezone.utc).date()
    valid_dates = {today - dt.timedelta(days=d) for d in range(days)}

    items = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or ""
        source = item.findtext("source") or ""
        try:
            published = parsedate_to_datetime(pub_date)
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=dt.timezone.utc)
        if published.date() not in valid_dates:
            continue
        items.append({"title": title, "link": link, "source": source, "published": published})
    items.sort(key=lambda x: x["published"], reverse=True)
    return items


@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_news(ticker: str, company_name: str | None = None, days: int = 4) -> list[dict]:
    """Fetch recent Chinese-language headlines for a ticker from Google News RSS.

    Returns headlines whose publish date falls within the last `days` days
    (today and the `days - 1` days before it, UTC), newest first. Returns []
    on any network or parse failure.
    """
    search_ticker = ticker.split(".")[0]
    query = f"{company_name} {search_ticker} 股票" if company_name else f"{search_ticker} 股票"
    return _fetch_google_news(query, hl="zh-TW", gl="TW", ceid="TW:zh-Hant", days=days)


@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_news_en(ticker: str, company_name: str | None = None, days: int = 4) -> list[dict]:
    """Fetch recent English-language headlines for a (US) ticker from Google
    News RSS — same shape/cache contract as get_recent_news, just hl/gl=US
    and no site restriction (broader than get_reuters_news below).
    """
    search_ticker = ticker.split(".")[0]
    base = f"{company_name} {search_ticker}" if company_name else search_ticker
    query = f"{base} stock"
    return _fetch_google_news(query, hl="en-US", gl="US", ceid="US:en", days=days)


@st.cache_data(ttl=3600, show_spinner=False)
def get_reuters_news(ticker: str, company_name: str | None = None, days: int = 4) -> list[dict]:
    """Recent Reuters headlines for a (US) ticker, via a Google News RSS
    search restricted to site:reuters.com. Returns [] on any failure."""
    search_ticker = ticker.split(".")[0]
    base = f"{company_name} {search_ticker}" if company_name else search_ticker
    query = f"{base} stock site:reuters.com"
    return _fetch_google_news(query, hl="en-US", gl="US", ceid="US:en", days=days)


@st.cache_data(ttl=3600, show_spinner=False)
def get_mops_news(ticker: str, company_name: str | None = None, days: int = 4) -> list[dict]:
    """Recent 公開資訊觀測站 (MOPS) material-info disclosures for a TW ticker.

    MOPS's per-stock query page is a JS single-page app with no documented
    public API, so this goes through a Google News RSS search restricted to
    site:mops.twse.com.tw instead of a bespoke scraper — best-effort, since
    it depends on Google News having indexed the disclosure. Returns [] on
    any failure or if nothing matched.
    """
    search_ticker = ticker.split(".")[0]
    base = f"{company_name} {search_ticker}" if company_name else search_ticker
    query = f"{base} site:mops.twse.com.tw"
    return _fetch_google_news(query, hl="zh-TW", gl="TW", ceid="TW:zh-Hant", days=days)


@st.cache_data(ttl=3600, show_spinner=False)
def _twse_news_list() -> list[dict]:
    """All recent TWSE exchange news (not filtered by ticker), with each
    item's ROC-calendar "Date" (e.g. "1150618") converted to a UTC
    datetime. Returns [] on any failure."""
    try:
        with urllib.request.urlopen(_TWSE_NEWS_LIST_URL, timeout=_TIMEOUT, context=_SSL_CTX) as resp:
            raw = json.loads(resp.read())
    except Exception:
        return []

    items = []
    for row in raw:
        title, url, date_str = row.get("Title"), row.get("Url"), row.get("Date")
        if not title or not url or not date_str or len(date_str) != 7:
            continue
        try:
            year = int(date_str[:3]) + 1911
            month, day = int(date_str[3:5]), int(date_str[5:7])
            published = dt.datetime(year, month, day, tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        items.append({"title": title, "link": url, "source": "TWSE", "published": published})
    return items


def get_twse_news(ticker: str, company_name: str | None = None, days: int = 4) -> list[dict]:
    """Recent TWSE exchange news mentioning this ticker's code or company
    name, from the official /news/newsList open-data feed (exchange-wide,
    filtered here client-side since the feed isn't per-ticker)."""
    search_ticker = ticker.split(".")[0]
    today = dt.datetime.now(dt.timezone.utc).date()
    valid_dates = {today - dt.timedelta(days=d) for d in range(days)}
    return [
        n for n in _twse_news_list()
        if n["published"].date() in valid_dates
        and (search_ticker in n["title"] or (company_name and company_name in n["title"]))
    ]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _sec_ticker_to_cik() -> dict[str, str]:
    """Ticker -> zero-padded 10-digit CIK, from SEC's official ticker list.
    Returns {} on any failure."""
    try:
        req = urllib.request.Request(_SEC_TICKERS_URL, headers=_SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=_TIMEOUT, context=_SSL_CTX) as resp:
            raw = json.loads(resp.read())
        return {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in raw.values()}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_sec_filings(ticker: str, days: int = 4) -> list[dict]:
    """Recent SEC EDGAR filings for a US ticker. Returns [] if the ticker
    has no CIK on file, or on any network/parse failure."""
    cik = _sec_ticker_to_cik().get(ticker.upper())
    if not cik:
        return []
    try:
        req = urllib.request.Request(_SEC_SUBMISSIONS_URL.format(cik=cik), headers=_SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=_TIMEOUT, context=_SSL_CTX) as resp:
            data = json.loads(resp.read())
    except Exception:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    today = dt.datetime.now(dt.timezone.utc).date()
    valid_dates = {today - dt.timedelta(days=d) for d in range(days)}

    items = []
    for form, date_str, accession, doc in zip(forms, filing_dates, accessions, docs):
        try:
            filed = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if filed not in valid_dates:
            continue
        accession_nodash = accession.replace("-", "")
        link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{doc}"
        items.append({
            "title": f"{form} 申報文件",
            "link": link,
            "source": "SEC EDGAR",
            "published": dt.datetime.combine(filed, dt.time(), tzinfo=dt.timezone.utc),
        })
    items.sort(key=lambda x: x["published"], reverse=True)
    return items


_TRANSLATE_URL = (
    "https://translate.googleapis.com/translate_a/single"
    "?client=gtx&sl=auto&tl=zh-TW&dt=t&q={text}"
)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def translate_to_zh_tw(text: str) -> str:
    """Best-effort machine translation of an English headline to Traditional
    Chinese, via the unofficial (no-key) Google Translate endpoint. Returns
    the original text unchanged on any failure."""
    if not text:
        return text
    url = _TRANSLATE_URL.format(text=urllib.parse.quote(text))
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT, context=_SSL_CTX) as resp:
            data = json.loads(resp.read())
        return "".join(seg[0] for seg in data[0] if seg[0])
    except Exception:
        return text


def _strip_source_suffix(title: str) -> str:
    """Google News RSS titles come as 'headline - Publisher'; drop the
    publisher tail (the <source> element already carries it) so the headline
    reads clean and the translation doesn't waste tokens on the outlet name."""
    return title.rsplit(" - ", 1)[0] if " - " in title else title


def _dedupe_by_title(items: list[dict], max_items: int) -> list[dict]:
    """Drop near-duplicate headlines (same normalized title), keep newest first."""
    seen, out = set(), []
    for n in sorted(items, key=lambda x: x["published"], reverse=True):
        key = _strip_source_suffix(n["title"]).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
        if len(out) >= max_items:
            break
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_us_week_ahead(max_items: int = 10) -> list[dict]:
    """本週美國市場重要紀事：財經行事曆類頭條（Fed、CPI、就業報告、財報週
    等），中英文兩路 Google News RSS 查詢合併去重，最新在前。英文標題由呼叫
    端自行翻譯（title 欄位保留原文）。"""
    zh = _fetch_google_news("美股 本週 聯準會 OR CPI OR 財報 OR 就業 when:7d",
                            hl="zh-TW", gl="TW", ceid="TW:zh-Hant", days=7)
    en = _fetch_google_news("US stock market week ahead Fed OR CPI OR jobs OR earnings when:7d",
                            hl="en-US", gl="US", ceid="US:en", days=7)
    return _dedupe_by_title(zh + en, max_items)


@st.cache_data(ttl=1800, show_spinner=False)
def get_us_market_today(max_items: int = 10) -> list[dict]:
    """影響當天美國股市的頭條（英文來源，Google News RSS，近 2 天涵蓋美台
    時差），去重後最新在前。翻譯由呼叫端做（可逐則顯示進度）。"""
    en = _fetch_google_news("stock market today Dow OR Nasdaq OR 'S&P 500' when:2d",
                            hl="en-US", gl="US", ceid="US:en", days=2)
    return _dedupe_by_title(en, max_items)


# 今年重要國際事件的主題查詢：每個主題各抓一路 RSS，涵蓋範圍從今年 1/1 起。
_GLOBAL_EVENT_TOPICS = [
    ("地緣政治／戰爭", "美伊 OR 以色列 OR 烏克蘭 戰爭 OR 衝突"),
    ("貿易與關稅", "美國 關稅 OR 貿易戰"),
    ("聯準會與利率", "聯準會 升息 OR 降息 OR 利率決策"),
    ("能源與原油", "原油 OR OPEC 油價"),
]


def _one_per_date(items: list[dict], max_items: int) -> list[dict]:
    """重點記錄模式：每個日曆日只留一則（取該日最新的那則），日期由新到舊。
    讓清單讀起來像大事記，而不是同一事件多家媒體的重覆報導。"""
    seen_dates, out = set(), []
    for n in sorted(items, key=lambda x: x["published"], reverse=True):
        day = n["published"].date()
        if day in seen_dates:
            continue
        seen_dates.add(day)
        out.append(n)
        if len(out) >= max_items:
            break
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_global_events_this_year(max_per_topic: int = 6) -> list[tuple[str, list[dict]]]:
    """今年（1/1 起）重要國際事件，依主題分組的 zh-TW 頭條，每主題每個日期
    只留一則重點（先去除同標題重覆，再依日期去重）。Google News RSS 的搜尋
    結果偏重近期，較舊事件只能盡力涵蓋（best-effort）。"""
    today = dt.datetime.now(dt.timezone.utc).date()
    days_ytd = (today - dt.date(today.year, 1, 1)).days + 1
    grouped = []
    for topic, query in _GLOBAL_EVENT_TOPICS:
        items = _fetch_google_news(f"{query} when:{days_ytd}d",
                                   hl="zh-TW", gl="TW", ceid="TW:zh-Hant", days=days_ytd)
        if items:
            unique = _dedupe_by_title(items, max_items=len(items))
            grouped.append((topic, _one_per_date(unique, max_per_topic)))
    return grouped


def recent_news_date_label(days: int = 4) -> str:
    """Human-readable date range matching get_recent_news's window, e.g. '6/17-6/20'."""
    today = dt.datetime.now(dt.timezone.utc).date()
    start = today - dt.timedelta(days=days - 1)
    return f"{start.month}/{start.day}-{today.month}/{today.day}"


def news_sentiment_score(news_items: list[dict]) -> float:
    """Keyword-based sentiment score in [-1, 1] from headline text.

    Counts positive vs. negative keyword hits across all headline titles and
    normalizes by total hits; returns 0.0 when there are no headlines or no
    keyword matches (neutral / unknown).
    """
    if not news_items:
        return 0.0
    pos = neg = 0
    for n in news_items:
        title = n["title"]
        title_lower = title.lower()
        pos += sum(title.count(w) for w in _POSITIVE_WORDS)
        neg += sum(title.count(w) for w in _NEGATIVE_WORDS)
        pos += sum(title_lower.count(w) for w in _POSITIVE_WORDS_EN)
        neg += sum(title_lower.count(w) for w in _NEGATIVE_WORDS_EN)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total
