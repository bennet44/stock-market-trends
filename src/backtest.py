"""Walk-forward training/backtest harness for the recommendation factor weights.

Goal: replace the hand-tuned recommend.FACTOR_WEIGHTS_BY_HORIZON with weights
learned from history, maximizing cross-sectional **Rank IC** — the Spearman
correlation, computed within each rebalance date, between a ticker's composite
score and its subsequent forward return. Higher mean Rank IC = the score is
better at ordering who outperforms.

Scope / honesty notes:
- Only the price/volume-derived factors are cleanly point-in-time and therefore
  trainable here: 期間報酬率, 趨勢(價格/SMA50), 技術面, Sharpe Ratio, 籌碼. For 籌碼
  the backtest uses the Chaikin Money Flow proxy for BOTH markets (clean from
  OHLCV); the live model still uses real 三大法人 for TW, which this factor's
  learned weight transfers to as the same "capital-flow" factor.
- 估值(1/預估PE), 基本面, 新聞情緒 come from yfinance snapshots / recent-only news,
  so backtesting them would leak future info. They are held at their current
  weights (TRAINABLE budget = 1 - their sum) and the five trainable factors are
  optimized to split the remainder.
- Always evaluate out-of-sample (walk_forward): train weights on a past window,
  measure Rank IC on a later window the optimizer never saw.

This module is offline research tooling — it is not imported by app.py. Run it
via train_weights.py (repo root) where network access works.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import recommend
from . import risk as risk_mod
from . import technical as ta

# Factors trained here (clean point-in-time); the rest keep their current weight.
TRAINABLE_FACTORS = ["期間報酬率", "趨勢(價格/SMA50)", "技術面", "Sharpe Ratio", "籌碼"]
FIXED_FACTORS = ["估值(1/預估PE)", "基本面", "新聞情緒"]

# Per horizon: how many recent trading days form the scoring window (mirrors the
# slicing recommend.build_recommendation_table does), and the forward holding
# period whose return defines "did this pick work".
HORIZON_SPEC = {
    "short": {"lookback": 10, "fwd_days": 5},
    "medium": {"lookback": 63, "fwd_days": 21},
    "long": {"lookback": 252, "fwd_days": 63},
}


def _trainable_factors_asof(win: pd.DataFrame) -> dict[str, float]:
    """The five trainable raw factor values from one ticker's OHLCV window.

    `win` is the price history sliced to the scoring lookback (inclusive of the
    as-of date as its last row). Mirrors the corresponding lines in
    recommend.build_recommendation_table so trained weights transfer directly.
    Missing/too-short values come back as NaN and are neutralized cross-
    sectionally later.
    """
    close, high, low, vol = win["Close"], win["High"], win["Low"], win["Volume"]
    last = close.iloc[-1]
    rets = risk_mod.daily_returns(close)
    sma50 = ta.sma(close, 50).iloc[-1]
    macd_hist = ta.macd(close)["hist"].iloc[-1]
    kd = ta.kd(high, low, close)
    rsi_last = ta.rsi(close).iloc[-1]
    cmf = ta.chaikin_money_flow(high, low, close, vol, window=min(20, max(2, len(win) - 1)))
    # 技術面 here is a single pre-combined momentum proxy; in the live model it's
    # the mean of three sub-signal z-scores. For ranking it's monotonic enough.
    tech = np.nanmean([
        macd_hist / last if last else np.nan,
        (kd["k"].iloc[-1] - kd["d"].iloc[-1]),
        rsi_last - 50,
    ])
    return {
        "期間報酬率": last / close.iloc[0] - 1,
        "趨勢(價格/SMA50)": (last / sma50 - 1) if pd.notna(sma50) and sma50 else np.nan,
        "技術面": tech,
        "Sharpe Ratio": risk_mod.sharpe_ratio(rets, 0.0),
        "籌碼": cmf.iloc[-1] if len(cmf) else np.nan,
    }


def _zscore_xs(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; NaN -> 0 (neutral), matching recommend._zscore."""
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / std).fillna(0.0)


def build_panel(
    price_data: dict[str, pd.DataFrame], horizon: str, rebalance_every: int = 5
) -> pd.DataFrame:
    """Point-in-time factor panel for one horizon across all tickers.

    For each rebalance date (every `rebalance_every` trading days, on the shared
    calendar), computes each ticker's five trainable factors from data up to and
    including that date, plus the realized forward return over the horizon's
    holding period. Factors are z-scored within each date; the forward return is
    kept raw (ranked later for IC). Returns a long DataFrame indexed by
    (date, ticker) with one column per factor + 'fwd_ret'.
    """
    spec = HORIZON_SPEC[horizon]
    lookback, fwd = spec["lookback"], spec["fwd_days"]
    # Common trading calendar = union of all tickers' dates, sorted.
    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    all_dates = pd.DatetimeIndex(all_dates)
    # Rebalance points that leave room for both the lookback and the forward window.
    idxs = range(lookback, len(all_dates) - fwd, rebalance_every)

    records = []
    for i in idxs:
        d = all_dates[i]
        d_fwd = all_dates[i + fwd]
        for t, df in price_data.items():
            win = df[df.index <= d]
            if len(win) < max(2, lookback // 4):
                continue
            win = win.iloc[-lookback:]
            if d_fwd not in df.index or d not in df.index:
                continue
            fwd_ret = df.loc[d_fwd, "Close"] / df.loc[d, "Close"] - 1
            rec = _trainable_factors_asof(win)
            rec.update({"date": d, "ticker": t, "fwd_ret": fwd_ret})
            records.append(rec)

    panel = pd.DataFrame.from_records(records)
    if panel.empty:
        return panel
    # Cross-sectional z-score of each factor within each date.
    for f in TRAINABLE_FACTORS:
        panel[f] = panel.groupby("date")[f].transform(lambda s: _zscore_xs(s.astype(float)))
    return panel.set_index(["date", "ticker"])


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return np.nan
    return a.rank().corr(b.rank())


def factor_ic(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-factor mean Rank IC (and IR = mean/std of the per-date IC series)."""
    rows = {}
    for f in TRAINABLE_FACTORS:
        per_date = panel.groupby(level="date").apply(lambda g: _spearman(g[f], g["fwd_ret"]))
        per_date = per_date.dropna()
        mean_ic = per_date.mean()
        ir = mean_ic / per_date.std() if per_date.std() else np.nan
        rows[f] = {"mean_IC": mean_ic, "IR": ir, "n_dates": len(per_date)}
    return pd.DataFrame(rows).T


def combined_ic(panel: pd.DataFrame, weights: dict[str, float]) -> float:
    """Mean Rank IC of the full weighted composite score over the panel."""
    score = sum(panel[f] * w for f, w in weights.items() if f in panel)
    tmp = pd.DataFrame({"score": score, "fwd_ret": panel["fwd_ret"]})
    per_date = tmp.groupby(level="date").apply(lambda g: _spearman(g["score"], g["fwd_ret"]))
    return per_date.dropna().mean()


def optimize_weights(panel: pd.DataFrame, horizon: str) -> dict[str, float]:
    """Learn trainable-factor weights that maximize Rank IC, numpy-only.

    Regresses the per-date rank-normalized forward return on the (already
    cross-sectionally z-scored) trainable factors — the least-squares solution
    is the IC-maximizing linear combination. Coefficients are clipped to be
    non-negative (a factor can't get a perverse negative weight) and scaled so
    the five trainable weights consume the budget left after the three fixed
    factors keep their current weights. Returns a full 8-factor weight dict.
    """
    fixed = recommend.FACTOR_WEIGHTS_BY_HORIZON[horizon]
    fixed_budget = sum(fixed[f] for f in FIXED_FACTORS)
    trainable_budget = 1.0 - fixed_budget

    # Rank-normalize the forward return within each date to [-0.5, 0.5] so the
    # regression target is the cross-sectional ordering (what Rank IC rewards).
    y = panel.groupby(level="date")["fwd_ret"].transform(
        lambda s: s.rank(pct=True) - 0.5
    ).values
    X = panel[TRAINABLE_FACTORS].astype(float).values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef = np.clip(coef, 0.0, None)
    if coef.sum() == 0:  # degenerate: fall back to equal split
        coef = np.ones(len(TRAINABLE_FACTORS))
    coef = coef / coef.sum() * trainable_budget

    weights = {f: float(w) for f, w in zip(TRAINABLE_FACTORS, coef)}
    for f in FIXED_FACTORS:
        weights[f] = fixed[f]
    return weights


def walk_forward(
    panel: pd.DataFrame, horizon: str, n_splits: int = 4
) -> dict:
    """Time-ordered out-of-sample evaluation.

    Splits the rebalance dates into `n_splits` contiguous blocks; for each block
    (after the first) trains weights on all earlier blocks and measures combined
    Rank IC on that held-out block, for both the trained weights and the current
    hand-tuned weights. Returns mean OOS IC for each, so you can see whether
    training actually helps before adopting the weights.
    """
    dates = panel.index.get_level_values("date").unique().sort_values()
    blocks = np.array_split(dates, n_splits)
    current = recommend.FACTOR_WEIGHTS_BY_HORIZON[horizon]

    trained_ics, current_ics, last_weights = [], [], None
    for k in range(1, n_splits):
        train_dates = dates[dates <= blocks[k - 1][-1]]
        test_dates = blocks[k]
        train_panel = panel[panel.index.get_level_values("date").isin(train_dates)]
        test_panel = panel[panel.index.get_level_values("date").isin(test_dates)]
        if train_panel.empty or test_panel.empty:
            continue
        w = optimize_weights(train_panel, horizon)
        last_weights = w
        trained_ics.append(combined_ic(test_panel, w))
        current_ics.append(combined_ic(test_panel, current))

    return {
        "oos_ic_trained": float(np.nanmean(trained_ics)) if trained_ics else np.nan,
        "oos_ic_current": float(np.nanmean(current_ics)) if current_ics else np.nan,
        "weights_last_fold": last_weights,
        "weights_full": optimize_weights(panel, horizon),
    }
