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
- 技術面 is a simplified proxy here; TECH_SUBWEIGHTS retunes are outside this
  harness's coverage (details at _trainable_factors_asof).

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
# The panel keys its trend factor by the fixed SMA it is computed from, while
# the live model (recommend.py) keys the same conceptual factor by its
# horizon-varying name. This pair is the single place the two names are
# bridged: walk_forward translates PANEL->LIVE on everything it returns, and
# LIVE->PANEL when evaluating the current hand-tuned weights against the panel.
TREND_KEY_PANEL = "趨勢(價格/SMA50)"
TREND_KEY_LIVE = "趨勢(價格/均線)"

# Only these horizons are trained. 'long' is excluded: with a few years of
# history there are too few non-overlapping 1-year forward windows to fit a
# long-hold model, so long weights stay hand-tuned (valuation/Sharpe-led).
TRAINABLE_HORIZONS = ["short", "medium"]
# Guardrails so a small-sample fit refines the hand-tuned (speculative-method)
# baseline rather than overwriting it:
#   TRAIN_BLEND — final = α·trained + (1-α)·baseline per factor. At 0.5 every
#     factor keeps ≥50% of its baseline (a zeroed core factor like 籌碼 can't
#     drop below half its hand-set weight) and any extreme is pulled halfway
#     back toward the baseline.
#   TRAIN_FACTOR_CAP — no single trainable factor may exceed this, so one
#     factor (e.g. Sharpe) can't dominate on an overfit sample.
TRAIN_BLEND = 0.5
TRAIN_FACTOR_CAP = 0.30

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
    # 技術面 here is a single pre-combined 3-signal momentum proxy. The live
    # model's 技術面 is an 8-sub-signal blend weighted by
    # recommend.TECH_SUBWEIGHTS_BY_HORIZON (macd/kd/rsi/bb/sma/pat/adx/shape);
    # this proxy is monotonic enough for ranking the factor's overall weight,
    # but it means TECH_SUBWEIGHTS retunes are not covered by this backtest.
    tech = np.nanmean([
        macd_hist / last if last else np.nan,
        float(np.clip((kd["j"].iloc[-1] - 50) / 50, -1, 1)),  # J normalised to [-1,1]
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


def _cap_renormalize(weights: dict[str, float], cap: float, budget: float) -> dict[str, float]:
    """Clamp each weight to <= cap, redistributing the excess proportionally to
    the factors still below cap, so the total stays == budget. Water-filling
    over a handful of factors: a few passes converge (a factor pushed to cap
    can't take more excess, so the excess flows to the rest)."""
    w = dict(weights)
    for _ in range(10):
        excess = sum(v - cap for v in w.values() if v > cap)
        if excess <= 1e-12:
            break
        room = {f: v for f, v in w.items() if v < cap}
        pool = sum(room.values())
        if pool <= 0:
            break
        for f in w:
            if w[f] > cap:
                w[f] = cap
            elif f in room:
                w[f] += excess * room[f] / pool
    return w


def optimize_weights(panel: pd.DataFrame, horizon: str) -> dict[str, float]:
    """Learn trainable-factor weights that maximize Rank IC, numpy-only, then
    apply the TRAIN_BLEND / TRAIN_FACTOR_CAP guardrails.

    Regresses the per-date rank-normalized forward return on the (already
    cross-sectionally z-scored) trainable factors — the least-squares solution
    is the IC-maximizing linear combination. Coefficients are clipped to be
    non-negative and scaled to the budget left after every non-trainable factor
    keeps its current weight. The raw fit is then blended halfway back toward
    the hand-tuned baseline (so a small-sample result refines rather than
    overwrites it — see TRAIN_BLEND) and capped per factor (see
    TRAIN_FACTOR_CAP), keeping the trainable total unchanged. The returned dict
    covers every factor of the live row (配息穩定性 included) and sums to 1.0.
    Guardrails live here (not in the caller) so walk_forward's trained IC is
    measured on the weights that would actually be applied. Keys use the panel's
    trend name (TREND_KEY_PANEL); walk_forward translates to live names.
    """
    current = recommend.FACTOR_WEIGHTS_BY_HORIZON[horizon]
    # Everything in the live row that isn't trained here keeps its weight.
    # The live trend key maps to the panel's TREND_KEY_PANEL (trainable), so
    # it is excluded from the fixed set.
    fixed = {f: w for f, w in current.items()
             if f not in TRAINABLE_FACTORS and f != TREND_KEY_LIVE}
    trainable_budget = 1.0 - sum(fixed.values())

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
    trained = dict(zip(TRAINABLE_FACTORS, (float(c) for c in coef)))

    # Guardrail 1 — blend halfway back to the baseline's trainable portion. The
    # baseline trend weight lives under the live key, everything else matches.
    # Both trained and baseline trainable parts sum to trainable_budget, so the
    # blend does too (no renormalization needed).
    baseline = {f: current[TREND_KEY_LIVE] if f == TREND_KEY_PANEL else current[f]
                for f in TRAINABLE_FACTORS}
    blended = {f: TRAIN_BLEND * trained[f] + (1 - TRAIN_BLEND) * baseline[f]
               for f in TRAINABLE_FACTORS}
    # Guardrail 2 — cap any single factor, redistributing to keep the budget.
    blended = _cap_renormalize(blended, TRAIN_FACTOR_CAP, trainable_budget)

    return {**blended, **fixed}


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
    # Key the current row against the panel's trend column so its trend weight
    # actually participates in the comparison (combined_ic skips keys the
    # panel doesn't have).
    current_panel_keyed = {
        (TREND_KEY_PANEL if f == TREND_KEY_LIVE else f): w for f, w in current.items()
    }

    def _live_keys(w: dict[str, float] | None) -> dict[str, float] | None:
        """Translate a panel-keyed weight dict to the live model's keys."""
        if w is None:
            return None
        return {(TREND_KEY_LIVE if f == TREND_KEY_PANEL else f): v for f, v in w.items()}

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
        current_ics.append(combined_ic(test_panel, current_panel_keyed))

    return {
        "oos_ic_trained": float(np.nanmean(trained_ics)) if trained_ics else np.nan,
        "oos_ic_current": float(np.nanmean(current_ics)) if current_ics else np.nan,
        "weights_last_fold": _live_keys(last_weights),
        "weights_full": _live_keys(optimize_weights(panel, horizon)),
    }
