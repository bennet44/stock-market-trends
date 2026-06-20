"""Chinese-language news headlines via Google News RSS, plus a lightweight
keyword-based sentiment score used as a recommendation factor.
"""
import datetime as dt
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import streamlit as st

_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
_TIMEOUT = 6

_POSITIVE_WORDS = [
    "上漲", "大漲", "看好", "升評", "優於預期", "創新高", "買進", "增持",
    "獲利", "成長", "樂觀", "飆漲", "突破", "強勢", "上修",
]
_NEGATIVE_WORDS = [
    "下跌", "大跌", "看壞", "降評", "不如預期", "創新低", "賣出", "減持",
    "虧損", "衰退", "悲觀", "重挫", "跳水", "下修", "示警",
]


@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_news(ticker: str, company_name: str | None = None, days: int = 4) -> list[dict]:
    """Fetch recent Chinese-language headlines for a ticker from Google News RSS.

    Returns headlines whose publish date falls within the last `days` days
    (today and the `days - 1` days before it, UTC), newest first. Returns []
    on any network or parse failure.
    """
    search_ticker = ticker.split(".")[0]
    query = f"{company_name} {search_ticker} 股票" if company_name else f"{search_ticker} 股票"
    url = _RSS_URL.format(query=urllib.parse.quote(query))
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
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
        pos += sum(title.count(w) for w in _POSITIVE_WORDS)
        neg += sum(title.count(w) for w in _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total
