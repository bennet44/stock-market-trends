"""Buy/sell recommendation scoring — a fund-manager-style composite score.

Combines momentum, risk-adjusted return, trend, valuation, and news
sentiment into a single cross-sectional z-score so candidates can be ranked
against each other.
"""
import concurrent.futures

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import data_loader as dl
from . import news as news_mod
from . import risk as risk_mod
from . import technical as ta

# Composite-score factor weights, switched by the chosen horizon: a 1-day scan
# and a 5-year scan shouldn't value the same things. Short windows lean on price
# action / news momentum and all but ignore valuation (fundamentals barely move
# a stock in days); long windows lean on valuation and risk-adjusted return and
# discount short-term momentum / headline sentiment. Each row sums to 1.0.
FACTOR_WEIGHTS_BY_HORIZON = {
    "short": {
        "期間報酬率": 0.18,
        "技術面": 0.20,
        "籌碼": 0.18,
        "新聞情緒": 0.15,
        "趨勢(價格/SMA50)": 0.12,
        "Sharpe Ratio": 0.07,
        "估值(1/預估PE)": 0.05,
        "基本面": 0.05,
    },
    "medium": {
        "期間報酬率": 0.15,
        "技術面": 0.13,
        "籌碼": 0.12,
        "新聞情緒": 0.12,
        "趨勢(價格/SMA50)": 0.10,
        "Sharpe Ratio": 0.13,
        "估值(1/預估PE)": 0.12,
        "基本面": 0.13,
    },
    "long": {
        "期間報酬率": 0.10,
        "技術面": 0.06,
        "籌碼": 0.06,
        "新聞情緒": 0.06,
        "趨勢(價格/SMA50)": 0.08,
        "Sharpe Ratio": 0.22,
        "估值(1/預估PE)": 0.20,
        "基本面": 0.22,
    },
}
# Default / backward-compatible weights when no horizon is specified.
FACTOR_WEIGHTS = FACTOR_WEIGHTS_BY_HORIZON["medium"]

_NEWS_FETCH_WORKERS = 12

_FACTOR_LABELS = {
    "期間報酬率": "期間報酬率",
    "Sharpe Ratio": "風險調整後報酬(Sharpe)",
    "趨勢(價格/SMA50)": "價格趨勢(相對SMA50)",
    "估值(1/預估PE)": "估值水準",
    "新聞情緒": "新聞情緒",
    "基本面": "基本面",
    "技術面": "技術面",
    "籌碼": "籌碼面",
}
_CONTRIB_PREFIX = "_contrib_"


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    # Tickers missing this metric (NaN) score neutral (0) rather than poisoning
    # the row's composite — mean/std already ignore NaN, so only the per-row
    # result needs neutralizing. Matters more now that several factors (基本面,
    # 技術面, 籌碼) are unavailable for some tickers / short windows.
    return ((series - series.mean()) / std).fillna(0.0)


def _news_sentiment(ticker: str, company_name: str | None) -> float:
    items = news_mod.get_recent_news(ticker, company_name)
    return news_mod.news_sentiment_score(items)


def build_recommendation_table(
    tickers: list[str], period: str, risk_free_rate: float = 0.0,
    lookback_days: int | None = None, weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Score each ticker into a cross-sectional composite rank.

    period is the yfinance window to fetch. lookback_days, when given, trims
    the fetched closes to that many recent trading days so the scoring window
    (期間報酬率, Sharpe, etc.) can be shorter than any yfinance period supports
    (e.g. 1天/1週); None uses the full fetched window. Indicators needing more
    bars than the window holds (e.g. SMA50, RSI) simply come back NaN and drop
    out of the scoring. weights selects the per-factor weighting (see
    FACTOR_WEIGHTS_BY_HORIZON); None falls back to the medium-horizon default.
    """
    weights = weights or FACTOR_WEIGHTS
    # TW tickers source the 籌碼 factor from real 三大法人 net buy/sell (fetched
    # once for the whole market); US tickers fall back to a Chaikin Money Flow
    # proxy computed from their own OHLCV. Only pay for the TW fetch if needed.
    inst_net = (
        dl.get_twse_institutional_net()
        if any(t.endswith((".TW", ".TWO")) for t in tickers) else {}
    )
    rows = {}
    company_names = {}
    for t in tickers:
        df = dl.get_price_history(t, period=period)
        if df.empty or len(df) < 2:
            continue
        win = df.iloc[-(lookback_days + 1):] if lookback_days is not None else df
        if len(win) < 2:
            continue
        close = win["Close"]
        high, low, volume = win["High"], win["Low"], win["Volume"]
        last_close = close.iloc[-1]
        rets = risk_mod.daily_returns(close)
        sma50 = ta.sma(close, 50).iloc[-1]
        info = dl.get_company_info(t)
        pe = info.get("forwardPE") or info.get("trailingPE")
        company_names[t] = info.get("shortName")

        # 技術面 sub-signals (each a "bullish momentum" continuous value, NaN when
        # the window is too short — neutralized by _zscore later).
        macd_hist = ta.macd(close)["hist"].iloc[-1]
        kd_df = ta.kd(high, low, close)
        rsi_last = ta.rsi(close).iloc[-1]

        # 籌碼 (capital flow): TW = recent 三大法人 net normalized by shares
        # outstanding; US = Chaikin Money Flow over the window.
        if t.endswith((".TW", ".TWO")):
            shares = info.get("sharesOutstanding") or info.get("floatShares")
            net_shares = inst_net.get(t.split(".")[0])
            chip = (net_shares / shares) if (net_shares is not None and shares) else np.nan
        else:
            cmf = ta.chaikin_money_flow(high, low, close, volume, window=min(20, max(2, len(win) - 1)))
            chip = cmf.iloc[-1] if len(cmf) else np.nan

        rows[t] = {
            "最新收盤價": last_close,
            "期間報酬率": last_close / close.iloc[0] - 1,
            "Sharpe Ratio": risk_mod.sharpe_ratio(rets, risk_free_rate),
            "趨勢(價格/SMA50)": (last_close / sma50 - 1) if pd.notna(sma50) and sma50 else np.nan,
            "估值(1/預估PE)": (1 / pe) if pe and pe > 0 else np.nan,
            "RSI (14)": rsi_last,
            # 基本面 sub-metrics (yfinance fundamentals)
            "_f_rev": info.get("revenueGrowth"),
            "_f_earn": info.get("earningsGrowth"),
            "_f_margin": info.get("profitMargins"),
            "_f_roe": info.get("returnOnEquity"),
            # 技術面 sub-metrics
            "_t_macd": macd_hist / last_close if last_close else np.nan,
            "_t_kd": kd_df["k"].iloc[-1] - kd_df["d"].iloc[-1],
            "_t_rsi": rsi_last - 50,
            # 籌碼 raw
            "_chip": chip,
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

    # Composite factors: 基本面 and 技術面 are the mean of their sub-metrics'
    # cross-sectional z-scores (so differently-scaled inputs combine fairly);
    # 籌碼 is the single chip signal standardized. Each is z-scored again in the
    # weighting loop, which is idempotent for already-standardized columns.
    table["基本面"] = pd.concat(
        [_zscore(table[c].astype(float)) for c in ["_f_rev", "_f_earn", "_f_margin", "_f_roe"]],
        axis=1,
    ).mean(axis=1)
    table["技術面"] = pd.concat(
        [_zscore(table[c].astype(float)) for c in ["_t_macd", "_t_kd", "_t_rsi"]],
        axis=1,
    ).mean(axis=1)
    table["籌碼"] = _zscore(table["_chip"].astype(float))
    table = table.drop(columns=[c for c in table.columns if c.startswith(("_f_", "_t_")) or c == "_chip"])

    score = pd.Series(0.0, index=table.index)
    for factor, weight in weights.items():
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


def forward_touch_rate(close, extreme, n_days: int, threshold: float, direction: str):
    """Path-based "touch" probability over a forward window.

    For every historical day, looks at the next `n_days` and computes the
    forward extreme return relative to that day's close — the max (direction
    "up") or min (direction "down"). Returns the % of days whose forward
    extreme reaches `threshold` (>= for up, <= for down): i.e. the historical
    chance the price *touches* a target move within the holding period.
    """
    c = np.asarray(close, dtype=float)
    e = np.asarray(extreme, dtype=float)
    n = len(c)
    if n <= n_days:
        return None
    # win[k] = e[k : k+n_days]; the window after day i is win[i+1].
    win = sliding_window_view(e, n_days)[1:n - n_days + 1]
    base = c[:n - n_days]
    if direction == "up":
        rets = win.max(axis=1) / base - 1
        return float((rets >= threshold).mean() * 100)
    rets = win.min(axis=1) / base - 1
    return float((rets <= threshold).mean() * 100)


def add_price_targets(
    df: pd.DataFrame, side: str, currency: str = "$",
    hold_days: int = PRICE_TARGET_HOLD_DAYS, hist_period: str = "10y",
) -> pd.DataFrame:
    """Attach a dynamic entry price, a target price, profit %, and 預測準確機率.

    Sized off a long history (hist_period) over `hold_days` trading days, with
    u = median of up moves and d = median of down moves (d < 0):
    - buy:  進場 = 現價×(1+d) (逢低承接), 賣出 = 現價×(1+u) (典型漲幅目標).
    - sell: 進場 = 現價×(1+u) (逢高減碼), 賣出 = 現價×(1+d) (逢低買回目標).
    - 獲利%: 賣出/進場 − 1 (buy 為正、sell 為負，獲利來自下跌).
    - 預測準確機率: path-based — the historical % of `hold_days` windows whose
      forward high (buy) reaches the up target, or forward low (sell) reaches
      the down target, relative to the day's close. Uses High/Low for the
      "touch", falling back to Close.
    """
    out = df.copy()
    price = out["最新收盤價"].astype(float)
    profit_pcts, entries, targets, future_wins = [], [], [], []
    for t, p in zip(out.index, price):
        hist = dl.get_price_history(t, period=hist_period)
        if hist.empty or not pd.notnull(p):
            entries.append(None); targets.append(None); profit_pcts.append(None); future_wins.append(None)
            continue
        close = hist["Close"]
        high = hist["High"] if "High" in hist else close
        low = hist["Low"] if "Low" in hist else close
        fwd = close.pct_change(periods=hold_days).dropna()
        ups, downs = fwd[fwd > 0], fwd[fwd < 0]
        u = float(np.median(ups)) if len(ups) else None
        d = float(np.median(downs)) if len(downs) else None
        if side == "buy":
            entry = p * (1 + d) if d is not None else None
            target = p * (1 + u) if u is not None else None
            fw = forward_touch_rate(close, high, hold_days, u, "up") if u is not None else None
        else:
            entry = p * (1 + u) if u is not None else None   # 逢高減碼
            target = p * (1 + d) if d is not None else None   # 逢低買回
            fw = forward_touch_rate(close, low, hold_days, d, "down") if d is not None else None
        entries.append(entry)
        targets.append(target)
        profit_pcts.append((target / entry - 1) * 100 if entry and target else None)
        future_wins.append(fw)
    out["獲利%"] = profit_pcts
    if side == "buy":
        out["建議買入價"] = entries
        out["目標賣出價"] = targets
    else:
        out["建議賣出價"] = entries
        out["逢低買回參考價"] = targets
    out["預測準確機率"] = future_wins  # placed last so it sits just before 原因說明
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
