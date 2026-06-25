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
        "趨勢(價格/均線)": 0.12,
        "Sharpe Ratio": 0.07,
        "估值(1/預估PE)": 0.05,
        "基本面": 0.05,
    },
    "medium": {
        "期間報酬率": 0.15,
        "技術面": 0.13,
        "籌碼": 0.12,
        "新聞情緒": 0.12,
        "趨勢(價格/均線)": 0.10,
        "Sharpe Ratio": 0.13,
        "估值(1/預估PE)": 0.12,
        "基本面": 0.13,
    },
    "long": {
        "期間報酬率": 0.10,
        "技術面": 0.06,
        "籌碼": 0.06,
        "新聞情緒": 0.06,
        "趨勢(價格/均線)": 0.08,
        "Sharpe Ratio": 0.22,
        "估值(1/預估PE)": 0.20,
        "基本面": 0.22,
    },
}
# Default / backward-compatible weights when no horizon is specified.
FACTOR_WEIGHTS = FACTOR_WEIGHTS_BY_HORIZON["medium"]

# Sub-weights *inside* the 技術面 factor, switched by horizon. Six sub-signals:
# momentum (MACD/KD/RSI), Bollinger %B (bb), SMA bullish-alignment (sma), and
# trend/pattern regression slope (pat). Short windows lean on momentum &
# Bollinger; long windows lean on SMA alignment & trend. Each row sums to 1.0.
TECH_SUBWEIGHTS_BY_HORIZON = {
    "short":  {"macd": 0.20, "kd": 0.15, "rsi": 0.15, "bb": 0.25, "sma": 0.10, "pat": 0.15},
    "medium": {"macd": 0.15, "kd": 0.10, "rsi": 0.10, "bb": 0.15, "sma": 0.30, "pat": 0.20},
    "long":   {"macd": 0.10, "kd": 0.05, "rsi": 0.05, "bb": 0.05, "sma": 0.50, "pat": 0.25},
}

# The 趨勢 factor's reference moving average, by horizon: short uses the 5-day
# line, medium the 20-day 月線 (the "強勢股需在月線上" benchmark), long the 60-day.
# (Being above SMA20 also feeds 技術面's SMA-alignment sub-signal.)
_TREND_SMA_BY_HORIZON = {"short": 5, "medium": 20, "long": 60}
# Flat penalty subtracted from 綜合評分 when 現價 is below the 20-day 月線
# (regardless of horizon) — a strength gate that demotes below-month-line stocks.
SMA20_PENALTY = 0.5

_NEWS_FETCH_WORKERS = 12

_FACTOR_LABELS = {
    "期間報酬率": "期間報酬率",
    "Sharpe Ratio": "風險調整後報酬(Sharpe)",
    "趨勢(價格/均線)": "價格趨勢(短SMA5/中SMA20月線/長SMA60)",
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


def _news_sentiment(ticker: str, company_name: str | None) -> tuple[float, str]:
    """Return (sentiment score, short headline summary) for the recent news.

    US tickers are covered far better by English-language financial media
    than by zh-TW Google News (which rarely indexes small/mid-cap US
    names), so US tickers search English first and only fall back to the
    zh-TW search if that comes back empty.
    """
    is_us = not ticker.endswith((".TW", ".TWO"))
    items = news_mod.get_recent_news_en(ticker, company_name) if is_us else []
    if not items:
        items = news_mod.get_recent_news(ticker, company_name)
    score = news_mod.news_sentiment_score(items)
    if not items:
        return score, "近4日無相關新聞"
    head = (items[0].get("title") or "").strip().replace("\n", " ")
    if len(head) > 22:
        head = head[:22] + "…"
    summary = head if len(items) == 1 else f"{head}（共{len(items)}則）"
    return score, summary


def _technical_summary(rsi, k, d, macd_hist, info: dict) -> str:
    """Compact quantified technical readout for the 原因說明 column:
    KD / MACD / RSI plus dividend per share & dividend yield."""
    parts = []
    if pd.notna(k) and pd.notna(d):
        parts.append("KD" + ("金叉" if k >= d else "死叉"))
    if pd.notna(macd_hist):
        parts.append("MACD" + ("翻紅" if macd_hist >= 0 else "翻黑"))
    if pd.notna(rsi):
        parts.append(f"RSI{rsi:.0f}")
    rate = info.get("dividendRate")
    if rate:
        parts.append(f"股息{rate:.2f}")
    dy = info.get("dividendYield") or 0.0
    if dy > 0.5:  # yfinance sometimes returns this as a percent, not a fraction
        dy /= 100.0
    if dy:
        parts.append(f"殖利率{dy * 100:.2f}%")
    return "、".join(parts)


def _fundamental_summary(info: dict) -> str:
    """Compact key-figure readout (營收成長/ROE/淨利率) for the 原因說明 column."""
    parts = []
    rev, roe, margin = info.get("revenueGrowth"), info.get("returnOnEquity"), info.get("profitMargins")
    if rev is not None:
        parts.append(f"營收成長{rev * 100:.0f}%")
    if roe is not None:
        parts.append(f"ROE{roe * 100:.0f}%")
    if margin is not None:
        parts.append(f"淨利率{margin * 100:.0f}%")
    return "、".join(parts)


def build_recommendation_table(
    tickers: list[str], period: str, risk_free_rate: float = 0.0,
    lookback_days: int | None = None, weights: dict[str, float] | None = None,
    horizon: str = "medium",
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
        # Computed on the full period-fetched df (before the lookback_days
        # trim below), since short-hold lookbacks (1/3/5 天) are too few bars
        # for a meaningful MA5/MA20 read — see zhu_breakout_signal.
        zhu_signal = zhu_breakout_signal(df["Close"], df["High"])
        zhu_vol_ok = zhu_volume_confirmed(df["Volume"])
        win = df.iloc[-(lookback_days + 1):] if lookback_days is not None else df
        if len(win) < 2:
            continue
        close = win["Close"]
        high, low, volume = win["High"], win["Low"], win["Volume"]
        last_close = close.iloc[-1]
        rets = risk_mod.daily_returns(close)
        sma_trend = ta.sma(close, _TREND_SMA_BY_HORIZON.get(horizon, 20)).iloc[-1]
        sma20_val = ta.sma(close, 20).iloc[-1]  # 月線, for the strength penalty
        info = dl.get_company_info(t)
        pe = info.get("forwardPE") or info.get("trailingPE")
        company_names[t] = info.get("shortName")

        # 技術面 sub-signals (each a "more bullish = higher" continuous value, NaN
        # when the window is too short — neutralized by _zscore later).
        macd_hist = ta.macd(close)["hist"].iloc[-1]
        kd_df = ta.kd(high, low, close)
        rsi_last = ta.rsi(close).iloc[-1]
        # 布林 %B: where the close sits in the band (centred at 0); >0 upper half.
        bb = ta.bollinger_bands(close)
        bb_up, bb_lo = bb["upper"].iloc[-1], bb["lower"].iloc[-1]
        pct_b = ((last_close - bb_lo) / (bb_up - bb_lo) - 0.5) if pd.notna(bb_up) and bb_up != bb_lo else np.nan
        # SMA 多頭排列: how many of SMA5>SMA10, SMA10>SMA20, 收盤>SMA20 hold (centred).
        sma5, sma10, sma20 = ta.sma(close, 5).iloc[-1], ta.sma(close, 10).iloc[-1], ta.sma(close, 20).iloc[-1]
        if pd.notna(sma5) and pd.notna(sma10) and pd.notna(sma20):
            sma_align = (sma5 > sma10) + (sma10 > sma20) + (last_close > sma20) - 1.5
        else:
            sma_align = np.nan
        # 型態趨勢: normalized slope of a linear fit over the window (trend strength).
        pattern = (np.polyfit(np.arange(len(close)), close.values, 1)[0] / last_close
                   if last_close and len(close) >= 2 else np.nan)

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
            "趨勢(價格/均線)": (last_close / sma_trend - 1) if pd.notna(sma_trend) and sma_trend else np.nan,
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
            "_t_bb": pct_b,
            "_t_sma": sma_align,
            "_t_pat": pattern,
            # 籌碼 raw
            "_chip": chip,
            # 1.0 when 現價 is below the 20-day 月線 → 綜合評分 penalty.
            "_below_sma20": 1.0 if (pd.notna(sma20_val) and last_close < sma20_val) else 0.0,
            # 朱家泓 short-term breakout trigger; only consulted for short-horizon
            # buy picks (see top_buy_sell's require_signal_col), not scored.
            "_zhu_signal": 1.0 if zhu_signal else 0.0,
            "_zhu_vol_ok": 1.0 if zhu_vol_ok else 0.0,
            # 原因說明 enrichments (display-only strings)
            "技術摘要": _technical_summary(rsi_last, kd_df["k"].iloc[-1], kd_df["d"].iloc[-1], macd_hist, info),
            "基本面摘要": _fundamental_summary(info),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=_NEWS_FETCH_WORKERS) as pool:
        futures = {
            pool.submit(_news_sentiment, t, company_names.get(t)): t for t in rows
        }
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                rows[t]["新聞情緒"], rows[t]["新聞摘要"] = future.result()
            except Exception:
                rows[t]["新聞情緒"], rows[t]["新聞摘要"] = 0.0, ""

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
    # 技術面 = horizon-weighted blend of six sub-signals' cross-sectional z-scores
    # (momentum MACD/KD/RSI + 布林 + SMA多頭排列 + 型態趨勢); see TECH_SUBWEIGHTS_BY_HORIZON.
    _subw = TECH_SUBWEIGHTS_BY_HORIZON.get(horizon, TECH_SUBWEIGHTS_BY_HORIZON["medium"])
    table["技術面"] = sum(
        _subw[k] * _zscore(table[f"_t_{k}"].astype(float)) for k in _subw
    )
    table["籌碼"] = _zscore(table["_chip"].astype(float))
    table = table.drop(columns=[c for c in table.columns if c.startswith(("_f_", "_t_")) or c == "_chip"])

    score = pd.Series(0.0, index=table.index)
    for factor, weight in weights.items():
        contribution = _zscore(table[factor].astype(float)) * weight
        table[_CONTRIB_PREFIX + factor] = contribution
        score = score + contribution
    # Strength gate: stocks trading below the 20-day 月線 are penalized.
    table["綜合評分"] = score - SMA20_PENALTY * table["_below_sma20"].astype(float)
    return table.sort_values("綜合評分", ascending=False)


def top_buy_sell(
    table: pd.DataFrame, n: int, require_signal_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the ranked table into disjoint buy/sell groups.

    Caps each side at len(table)//2 so a small candidate list never lets the
    same ticker appear in both the buy and sell lists.

    require_signal_col, when given (e.g. "_zhu_signal"), gates the *buy* side
    to only rows where that column is truthy — a hard entry-trigger filter on
    top of the composite ranking, rather than a score contributor. Used for
    short-horizon buy picks (see zhu_breakout_signal). If nothing qualifies,
    buy comes back empty rather than falling back to the unfiltered ranking.

    The signal is independent of 綜合評分 rank, so a low-scoring row can still
    qualify and get selected for buy — sell is then drawn from the remaining
    rows (excluding whatever got picked for buy) so the two never share a
    ticker; concat'ing them downstream into one display table needs a unique
    index, and a duplicate ticker there raises "Styler.apply ... not
    compatible with non-unique index" instead of a usable error message.
    """
    if table.empty:
        return table, table
    sorted_desc = table.sort_values("綜合評分", ascending=False)
    total = len(sorted_desc)
    if total == 1:
        return sorted_desc, sorted_desc.iloc[0:0]
    n_eff = min(n, total // 2)
    if require_signal_col and require_signal_col in sorted_desc.columns:
        qualifying = sorted_desc[sorted_desc[require_signal_col] >= 1]
        buy = qualifying.head(n_eff)
    else:
        buy = sorted_desc.head(n_eff)
    sell_pool = sorted_desc[~sorted_desc.index.isin(buy.index)]
    sell = sell_pool.tail(n_eff).iloc[::-1]
    return buy, sell


PRICE_TARGET_HOLD_DAYS = 5

# How much the per-stock technical bias may stretch/shrink the median move when
# pricing the buy/sell levels (bias∈[-1,1] → move scaled within ±this fraction).
TECH_BIAS_BETA = 0.4


def zhu_breakout_signal(close: pd.Series, high: pd.Series) -> bool:
    """朱家泓-style short-term entry trigger: uptrend filter (現價 above a
    rising MA20) plus a breakout trigger (收盤突破MA5 且 收盤突破前一日最高點).
    Backtested over 2026-02~06: lifts the 1-day-hold win rate among 美股
    Top-10 picks from ~54% to ~60%; used as a hard gate (not a score
    contributor) for short-horizon buy candidates — only tickers actually
    firing the breakout qualify, rather than just ranking highest.
    Returns False (no signal) on any insufficient-data case.
    """
    if len(close) < 21:
        return False
    sma5, sma20 = ta.sma(close, 5), ta.sma(close, 20)
    c = close.iloc[-1]
    prev_high = high.iloc[-2]
    ma5, ma20, ma20_prev = sma5.iloc[-1], sma20.iloc[-1], sma20.iloc[-2]
    if pd.isna(ma5) or pd.isna(ma20) or pd.isna(ma20_prev):
        return False
    uptrend = c > ma20 and ma20 > ma20_prev
    breakout = c > ma5 and c > prev_high
    return bool(uptrend and breakout)


def zhu_volume_confirmed(volume: pd.Series, ratio: float = 1.2) -> bool:
    """朱家泓 also requires 放量 (a volume pickup) to confirm a breakout is real
    buying, not a thin no-volume false move. This isn't wired in as a hard
    gate (untested whether it actually improves win rate) — it's used to
    flag, in 備註, breakout picks whose volume didn't actually confirm, so
    the trade can be entered with stop-loss already top of mind.
    Returns False (not confirmed) on any insufficient-data case.
    """
    if len(volume) < 6:
        return False
    last_vol = volume.iloc[-1]
    avg_vol = volume.iloc[-6:-1].mean()
    if pd.isna(last_vol) or pd.isna(avg_vol) or not avg_vol:
        return False
    return bool(last_vol >= ratio * avg_vol)


def horizon_for_hold_days(hold_days: int) -> str:
    """Map a holding period (trading days) to a short/medium/long horizon, so
    the price-target technical adjustment uses the matching sub-weights."""
    if hold_days <= 10:
        return "short"
    if hold_days <= 63:
        return "medium"
    return "long"


def technical_bias(close: pd.Series, high: pd.Series, low: pd.Series, horizon: str = "medium") -> float:
    """Per-stock "how bullish" score in roughly [-1, 1], a horizon-weighted blend
    of six self-contained (absolute, not cross-sectional) technical signals —
    the same six that make up the 技術面 factor. Positive = technically bullish.
    Returns 0.0 (neutral) when nothing can be computed (e.g. too few bars).
    """
    if len(close) < 2:
        return 0.0
    last = close.iloc[-1]
    sigs: dict[str, float] = {}
    hist = ta.macd(close)["hist"].iloc[-1]
    if pd.notna(hist) and last:
        sigs["macd"] = float(np.tanh(hist / last * 50))
    kd = ta.kd(high, low, close)
    if pd.notna(kd["k"].iloc[-1]) and pd.notna(kd["d"].iloc[-1]):
        sigs["kd"] = float(np.clip((kd["k"].iloc[-1] - kd["d"].iloc[-1]) / 100, -1, 1))
    rsi = ta.rsi(close).iloc[-1]
    if pd.notna(rsi):
        sigs["rsi"] = float(np.clip((rsi - 50) / 50, -1, 1))
    bb = ta.bollinger_bands(close)
    up, lo = bb["upper"].iloc[-1], bb["lower"].iloc[-1]
    if pd.notna(up) and up != lo:
        sigs["bb"] = float(np.clip(((last - lo) / (up - lo) - 0.5) * 2, -1, 1))
    s5, s10, s20 = ta.sma(close, 5).iloc[-1], ta.sma(close, 10).iloc[-1], ta.sma(close, 20).iloc[-1]
    if pd.notna(s5) and pd.notna(s10) and pd.notna(s20):
        sigs["sma"] = float(((s5 > s10) + (s10 > s20) + (last > s20) - 1.5) / 1.5)
    if last:
        slope = np.polyfit(np.arange(len(close)), close.values, 1)[0]
        sigs["pat"] = float(np.tanh(slope / last * len(close)))

    subw = TECH_SUBWEIGHTS_BY_HORIZON.get(horizon, TECH_SUBWEIGHTS_BY_HORIZON["medium"])
    wsum = sum(subw[k] for k in sigs)
    if not wsum:
        return 0.0
    return sum(subw[k] * v for k, v in sigs.items()) / wsum


def technical_analysis_brief(close: pd.Series, high: pd.Series, low: pd.Series,
                              horizon: str = "medium") -> tuple[pd.DataFrame, str]:
    """Per-indicator technical readout for Page 1's "技術分析" table: each row's
    現況/狀態/說明, plus a one-line 結論建議 derived from the same bias used to
    nudge price targets (technical_bias) — so the table's conclusion always
    agrees with the price-target adjustment shown above it.
    """
    rows = []
    last = close.iloc[-1] if len(close) else float("nan")

    rsi = ta.rsi(close).iloc[-1] if len(close) >= 2 else float("nan")
    if pd.notna(rsi):
        status = "超買" if rsi >= 70 else ("超賣" if rsi <= 30 else "中性")
        rows.append(("RSI (14)", f"{rsi:.1f}", status,
                     "≥70視為超買、≤30視為超賣，中間區間視為中性動能。"))

    macd_df = ta.macd(close) if len(close) >= 2 else None
    if macd_df is not None and pd.notna(macd_df["hist"].iloc[-1]):
        hist = macd_df["hist"].iloc[-1]
        status = "翻紅（偏多）" if hist >= 0 else "翻黑（偏空）"
        rows.append(("MACD", f"{hist:+.2f}", status, "柱狀體（DIF−訊號線）由負轉正視為偏多訊號，反之偏空。"))

    kd_df = ta.kd(high, low, close) if len(close) >= 2 else None
    if kd_df is not None and pd.notna(kd_df["k"].iloc[-1]) and pd.notna(kd_df["d"].iloc[-1]):
        k, d = kd_df["k"].iloc[-1], kd_df["d"].iloc[-1]
        cross = "金叉" if k >= d else "死叉"
        zone = "超買區" if k >= 80 else ("超賣區" if k <= 20 else "中性區")
        rows.append(("KD (9)", f"K{k:.0f} / D{d:.0f}", f"{cross}・{zone}",
                     "K≥D為金叉偏多，K≤D為死叉偏空；K≥80超買、K≤20超賣。"))

    bb = ta.bollinger_bands(close) if len(close) >= 2 else None
    if bb is not None and pd.notna(bb["upper"].iloc[-1]) and bb["upper"].iloc[-1] != bb["lower"].iloc[-1]:
        up, lo = bb["upper"].iloc[-1], bb["lower"].iloc[-1]
        pct_b = (last - lo) / (up - lo)
        status = "貼上軌（偏多）" if pct_b >= 0.8 else ("貼下軌（偏空）" if pct_b <= 0.2 else "區間中段")
        rows.append(("布林通道 %B", f"{pct_b * 100:.0f}%", status,
                     "現價在通道中的相對位置；越貼上軌動能越強，越貼下軌越弱。"))

    s5, s10, s20 = ta.sma(close, 5).iloc[-1], ta.sma(close, 10).iloc[-1], ta.sma(close, 20).iloc[-1]
    if pd.notna(s5) and pd.notna(s10) and pd.notna(s20):
        if s5 > s10 > s20:
            status = "多頭排列"
        elif s5 < s10 < s20:
            status = "空頭排列"
        else:
            status = "糾結整理"
        rows.append(("均線排列 (5/10/20)", f"{s5:.1f} / {s10:.1f} / {s20:.1f}", status,
                     "短中長均線依大小排序：多頭排列＝5>10>20，空頭排列＝5<10<20。"))

    df = pd.DataFrame(rows, columns=["指標", "現況數值", "狀態", "說明"])

    bias = technical_bias(close, high, low, horizon)
    if bias >= 0.2:
        verdict = "技術面偏多"
    elif bias <= -0.2:
        verdict = "技術面偏空"
    else:
        verdict = "技術面中性"
    conclusion = (
        f"{verdict}（綜合技術偏多偏空指數 {bias:+.2f}，範圍−1~+1）。"
        "本指數已用於上方建議買入/賣出價的微調，方向一致。"
    )
    return df, conclusion


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
    horizon: str = "medium", aggressiveness: int = 50,
) -> pd.DataFrame:
    """Attach a dynamic entry price, a target price, profit %, and 預測準確機率.

    Sized off a long history (hist_period) over `hold_days` trading days, with
    u = the `aggressiveness`-th percentile of up moves and d the matching
    percentile of down moves (aggressiveness=50 → median; lower = more
    conservative/closer targets, higher = more aggressive/farther), then nudged by
    the stock's technical_bias (∈[-1,1], horizon-weighted): bullish stretches the
    up target and shrinks the buy dip, bearish does the reverse — bounded by
    TECH_BIAS_BETA so 買<現價<賣 always holds.
    - buy:  進場 = 現價×(1+d_adj) (逢低承接), 賣出 = 現價×(1+u_adj) (漲幅目標).
    - sell: 進場 = 現價×(1+u_adj) (逢高減碼), 賣出 = 現價×(1+d_adj) (逢低買回).
    - 獲利%: 賣出/進場 − 1.
    - 預測準確機率: path-based — historical % of `hold_days` windows whose forward
      high (buy) / low (sell) touches the *adjusted* target relative to the day's
      close. Uses High/Low for the "touch", falling back to Close.
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
        u = float(np.percentile(ups, aggressiveness)) if len(ups) else None
        d = float(np.percentile(downs, 100 - aggressiveness)) if len(downs) else None
        # Technical nudge: bullish -> bigger up target & shallower buy dip.
        bias = technical_bias(close, high, low, horizon)
        if u is not None:
            u = u * (1 + TECH_BIAS_BETA * bias)
        if d is not None:
            d = d * (1 - TECH_BIAS_BETA * bias)
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
    # Factors that carry a detail string appended in parentheses in 原因說明.
    summary_col = {"技術面": "技術摘要", "新聞情緒": "新聞摘要", "基本面": "基本面摘要"}
    # Pure performance/return metrics are not "reasons" — the 原因說明 should read
    # as 基本面/技術面/籌碼面 analysis, so these are left out of it.
    reason_exclude = {"期間報酬率", "Sharpe Ratio"}
    contrib_cols = [c for c in df.columns if c.startswith(_CONTRIB_PREFIX)]
    reasons = []
    for _, row in df.iterrows():
        contribs = {
            c[len(_CONTRIB_PREFIX):]: row[c]  # keep the raw factor key
            for c in contrib_cols
            if pd.notnull(row[c]) and c[len(_CONTRIB_PREFIX):] not in reason_exclude
        }
        if not contribs:
            reasons.append("資料不足")
            continue
        # List every factor that contributes in the recommended direction
        # (positive for buy / negative for sell), ranked most→least important,
        # numbered so the priority order is explicit. Falls back to the two
        # strongest factors if none point the "right" way.
        favorable = [(k, v) for k, v in contribs.items() if (v > 0 if side == "buy" else v < 0)]
        ranked = sorted(favorable or contribs.items(), key=lambda kv: kv[1], reverse=(side == "buy"))
        if not favorable:
            ranked = ranked[:2]
        nums = "①②③④⑤⑥⑦⑧⑨⑩"
        parts = []
        for i, (key, _) in enumerate(ranked):
            label = _FACTOR_LABELS.get(key, key)
            col = summary_col.get(key)
            detail = row[col] if (col and col in df.columns and pd.notnull(row.get(col))) else ""
            detail = str(detail).strip()
            tag = nums[i] if i < len(nums) else f"{i + 1}."
            parts.append(f"{tag}{label}（{detail}）" if detail else f"{tag}{label}")
        text = " ".join(parts)
        if "_below_sma20" in df.columns and pd.notna(row.get("_below_sma20")) and row.get("_below_sma20") >= 1:
            text += "（未站上月線SMA20，已扣分）"
        reasons.append(text)

    # 備註: this pick is a 朱家泓 short-horizon breakout (passed the hard
    # entry gate in top_buy_sell) but lacked the 放量 confirmation he also
    # requires — flag it so the trade is entered with stop-loss already in
    # mind, since an unconfirmed breakout is more likely to be a fake-out.
    notes = []
    if side == "buy" and "_zhu_signal" in df.columns and "_zhu_vol_ok" in df.columns:
        for _, row in df.iterrows():
            if row.get("_zhu_signal", 0) >= 1 and row.get("_zhu_vol_ok", 0) < 1:
                notes.append("⚠️ 突破未放量確認，留意假突破，進場前先想好停損點")
            else:
                notes.append("")
    else:
        notes = [""] * len(df)

    drop_extra = [c for c in summary_col.values() if c in df.columns]
    for hidden_col in ("_below_sma20", "_zhu_signal", "_zhu_vol_ok"):
        if hidden_col in df.columns:
            drop_extra.append(hidden_col)
    out = df.drop(columns=contrib_cols + drop_extra)
    out["原因說明"] = reasons
    if any(notes):
        out["備註"] = notes
    return out
