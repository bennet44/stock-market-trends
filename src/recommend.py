"""Buy/sell recommendation scoring — a fund-manager-style composite score.

Combines momentum, risk-adjusted return, trend, valuation, and news
sentiment into a single cross-sectional z-score so candidates can be ranked
against each other.
"""
import concurrent.futures

import numpy as np
import pandas as pd

from . import data_loader as dl
from . import news as news_mod
from . import risk as risk_mod
from . import technical as ta

FACTOR_WEIGHTS = {
    "期間報酬率": 0.25,
    "Sharpe Ratio": 0.25,
    "趨勢(價格/SMA50)": 0.15,
    "估值(1/預估PE)": 0.15,
    "新聞情緒": 0.2,
}

_NEWS_FETCH_WORKERS = 12

_FACTOR_LABELS = {
    "期間報酬率": "期間報酬率",
    "Sharpe Ratio": "風險調整後報酬(Sharpe)",
    "趨勢(價格/SMA50)": "價格趨勢(相對SMA50)",
    "估值(1/預估PE)": "估值水準",
    "新聞情緒": "新聞情緒",
}
_CONTRIB_PREFIX = "_contrib_"


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _news_sentiment(ticker: str, company_name: str | None) -> float:
    items = news_mod.get_recent_news(ticker, company_name)
    return news_mod.news_sentiment_score(items)


def build_recommendation_table(tickers: list[str], period: str, risk_free_rate: float = 0.0) -> pd.DataFrame:
    rows = {}
    company_names = {}
    for t in tickers:
        df = dl.get_price_history(t, period=period)
        if df.empty or len(df) < 2:
            continue
        close = df["Close"]
        rets = risk_mod.daily_returns(close)
        sma50 = ta.sma(close, 50).iloc[-1]
        info = dl.get_company_info(t)
        pe = info.get("forwardPE") or info.get("trailingPE")
        company_names[t] = info.get("shortName")
        rows[t] = {
            "最新收盤價": close.iloc[-1],
            "期間報酬率": close.iloc[-1] / close.iloc[0] - 1,
            "Sharpe Ratio": risk_mod.sharpe_ratio(rets, risk_free_rate),
            "趨勢(價格/SMA50)": (close.iloc[-1] / sma50 - 1) if pd.notna(sma50) and sma50 else np.nan,
            "估值(1/預估PE)": (1 / pe) if pe and pe > 0 else np.nan,
            "RSI (14)": ta.rsi(close).iloc[-1],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=_NEWS_FETCH_WORKERS) as pool:
        futures = {
            pool.submit(_news_sentiment, t, company_names.get(t)): t for t in rows
        }
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                rows[t]["新聞情緒"] = future.result()
            except Exception:
                rows[t]["新聞情緒"] = 0.0

    table = pd.DataFrame(rows).T
    if table.empty:
        return table

    score = pd.Series(0.0, index=table.index)
    for factor, weight in FACTOR_WEIGHTS.items():
        contribution = _zscore(table[factor].astype(float)) * weight
        table[_CONTRIB_PREFIX + factor] = contribution
        score = score + contribution
    table["綜合評分"] = score
    return table.sort_values("綜合評分", ascending=False)


def top_buy_sell(table: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the ranked table into disjoint buy/sell groups.

    Caps each side at len(table)//2 so a small candidate list never lets the
    same ticker appear in both the buy and sell lists.
    """
    if table.empty:
        return table, table
    sorted_desc = table.sort_values("綜合評分", ascending=False)
    total = len(sorted_desc)
    if total == 1:
        return sorted_desc, sorted_desc.iloc[0:0]
    n_eff = min(n, total // 2)
    buy = sorted_desc.head(n_eff)
    sell = sorted_desc.tail(n_eff).iloc[::-1]
    return buy, sell


PRICE_TARGET_HOLD_DAYS = 5


def add_price_targets(
    df: pd.DataFrame, side: str, currency: str = "$",
    win_rate_pct: float = 60, period: str = "1y",
) -> pd.DataFrame:
    """Attach an entry price plus a win-rate-driven target price.

    For each ticker, pulls its own price history and builds an empirical
    distribution of forward returns (PRICE_TARGET_HOLD_DAYS trading days
    ahead). side="buy": the target is the profit-taking move achieved with
    win_rate_pct%% probability among historically up periods. side="sell":
    the target is a pullback worth watching to buy back in, sized to the
    win_rate_pct%% percentile among historically down periods (since "sell"
    here means reduce/avoid, not short selling).
    """
    out = df.copy()
    price = out["最新收盤價"].astype(float)
    win_rates, profit_pcts, targets = [], [], []
    for t, p in zip(out.index, price):
        hist = dl.get_price_history(t, period=period)
        close = hist["Close"] if not hist.empty else pd.Series(dtype=float)
        fwd_returns = close.pct_change(periods=PRICE_TARGET_HOLD_DAYS).dropna()
        subset = fwd_returns[fwd_returns > 0] if side == "buy" else fwd_returns[fwd_returns < 0]
        move = np.percentile(subset, 100 - win_rate_pct) if not subset.empty else None
        win_rates.append(win_rate_pct if move is not None else None)
        profit_pcts.append(move * 100 if move is not None else None)
        targets.append(p * (1 + move) if move is not None and pd.notnull(p) else None)
    out["勝率(%)"] = win_rates
    out["獲利%"] = profit_pcts
    if side == "buy":
        out["建議買入價"] = price
        out["目標賣出價"] = targets
    else:
        out["建議賣出價"] = price
        out["逢低買回參考價"] = targets
    return out.drop(columns=["最新收盤價"])


def add_reason(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Append a "原因說明" column naming the factors driving each pick.

    side="buy": names the factors that scored best relative to the group.
    side="sell": names the factors that scored worst relative to the group.
    Reads the hidden per-factor score contributions stashed by
    build_recommendation_table and drops them once the explanation is built.
    """
    contrib_cols = [c for c in df.columns if c.startswith(_CONTRIB_PREFIX)]
    reasons = []
    for _, row in df.iterrows():
        contribs = {
            _FACTOR_LABELS[c[len(_CONTRIB_PREFIX):]]: row[c]
            for c in contrib_cols if pd.notnull(row[c])
        }
        if not contribs:
            reasons.append("資料不足")
            continue
        ranked = sorted(contribs.items(), key=lambda kv: kv[1], reverse=(side == "buy"))
        top_labels = [label for label, _ in ranked[:2]]
        verb = "領先同組" if side == "buy" else "落後同組"
        reasons.append(f"{'、'.join(top_labels)}{verb}")
    out = df.drop(columns=contrib_cols)
    out["原因說明"] = reasons
    return out
