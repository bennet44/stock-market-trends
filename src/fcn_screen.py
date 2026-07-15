"""Forward-looking FCN planning tools built on the same Monte Carlo engine
(src/fcn.py) the app uses, aimed at *raising the capital-safety win rate* and
judging coupon fairness for notes you might buy next:

  • evaluate_structure — win rate / early-exit prob / fair coupon for one basket
    and one set of terms.
  • sweep — hold the basket fixed, vary one term (KI / tenor / KO / KI-style) and
    return how win rate and fair coupon move, so you can see the trade-off
    (e.g. a lower KI barrier or European vs American KI raises the win rate).
  • screen_baskets — over a candidate pool, rank every size-k basket under one
    fixed structure by win rate, with the fair coupon alongside.

All use a risk-neutral drift (the app's conservative default) and apply the
committed volatility calibration fcn.vol_scale(). Prices come through
data_loader.get_price_history (yfinance + chart-API fallback).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from src import data_loader as dl
from src import fcn as fcn_engine

RF = 0.04                    # matches app.DEFAULT_RISK_FREE_RATE
DEFAULT_N_SIMS = 8000
SCREEN_N_SIMS = 1500         # lighter for the many-combo screener
MAX_COMBOS = 150


def _pd_corr(corr: np.ndarray) -> np.ndarray:
    return 0.999 * corr + 0.001 * np.eye(len(corr))


def _recent_window(tickers: list[str], tenor_months: int) -> tuple[pd.DataFrame | None, list[str]]:
    """Aligned daily closes over the most-recent `tenor_months` window (the same
    window the app's FCN tab uses). Returns (df, missing_tickers)."""
    closes = {}
    for t in tickers:
        df = dl.get_price_history(t, period="5y")
        if df.empty:
            continue
        cutoff = df.index.max() - pd.DateOffset(months=tenor_months)
        closes[t] = df["Close"][df.index >= cutoff]
    missing = [t for t in tickers if t not in closes]
    if missing:
        return None, missing
    cd = pd.concat(closes, axis=1, join="inner")
    cd.columns = tickers
    return cd.dropna(), []


def _stats_from_window(win: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(annualized vols, correlation) from aligned closes."""
    log_ret = np.log(win / win.shift(1)).dropna()
    vols = (log_ret.std() * np.sqrt(fcn_engine.TRADING_DAYS_PER_YEAR)).to_numpy()
    corr = log_ret.corr().to_numpy()
    return vols, corr


def _simulate(vols, corr, *, tenor, strike, ki, ko, ki_style, n_sims) -> fcn_engine.PathStats:
    n = len(vols)
    return fcn_engine.simulate_basket(
        strike_pct=strike, ki_pct=ki, ko_pct=ko, tenor_months=tenor,
        vols=np.asarray(vols), drifts=np.full(n, RF), corr=_pd_corr(np.asarray(corr)),
        risk_free_rate=RF, ki_style=ki_style, n_sims=n_sims,
    )


def _metrics(stats: fcn_engine.PathStats) -> dict:
    return {
        "win_rate": fcn_engine.win_rate(stats),
        "prob_autocall": stats.prob_autocall,
        "prob_breach": stats.prob_breach,
        "avg_exit_month": stats.avg_exit_month,
        "fair_coupon": fcn_engine.fair_coupon_rate(stats),
    }


def evaluate_structure(tickers: list[str], *, tenor: int, strike: float, ki: float,
                       ko: float, ki_style: str, n_sims: int = DEFAULT_N_SIMS,
                       vol_scale: float | None = None) -> dict:
    """Win rate / early-exit prob / fair coupon for one basket + terms. Raises
    ValueError if any ticker has no price data."""
    vs = fcn_engine.vol_scale() if vol_scale is None else vol_scale
    win, missing = _recent_window(tickers, tenor)
    if missing:
        raise ValueError(f"找不到價格資料：{', '.join(missing)}")
    vols, corr = _stats_from_window(win)
    stats = _simulate(vols * vs, corr, tenor=tenor, strike=strike, ki=ki, ko=ko,
                      ki_style=ki_style, n_sims=n_sims)
    out = _metrics(stats)
    out["vols"] = vols
    out["mean_vol"] = float(np.mean(vols))
    out["mean_corr"] = float(corr[np.triu_indices(len(corr), 1)].mean()) if len(corr) > 1 else 1.0
    return out


def sweep(tickers: list[str], base: dict, param: str, values: list, *,
          n_sims: int = DEFAULT_N_SIMS, vol_scale: float | None = None) -> pd.DataFrame:
    """Vary one term over `values`, holding the basket and other terms fixed.
    `base` has keys tenor/strike/ki/ko/ki_style. `param` is one of those. Returns
    a DataFrame with the swept value plus win_rate/prob_autocall/fair_coupon."""
    vs = fcn_engine.vol_scale() if vol_scale is None else vol_scale
    stats_cache: dict[int, tuple] = {}

    def _stats_for(tenor: int):
        if tenor not in stats_cache:
            win, missing = _recent_window(tickers, tenor)
            if missing:
                raise ValueError(f"找不到價格資料：{', '.join(missing)}")
            stats_cache[tenor] = _stats_from_window(win)
        return stats_cache[tenor]

    rows = []
    for v in values:
        p = dict(base)
        p[param] = v
        vols, corr = _stats_for(int(p["tenor"]))
        st = _simulate(vols * vs, corr, tenor=int(p["tenor"]), strike=p["strike"],
                       ki=p["ki"], ko=p["ko"], ki_style=p["ki_style"], n_sims=n_sims)
        m = _metrics(st)
        rows.append({param: v, "win_rate": m["win_rate"],
                     "prob_autocall": m["prob_autocall"], "fair_coupon": m["fair_coupon"]})
    return pd.DataFrame(rows)


def count_combos(pool_size: int, combo_size: int) -> int:
    return math.comb(pool_size, combo_size) if 0 < combo_size <= pool_size else 0


def screen_baskets(pool: list[str], combo_size: int, structure: dict, *,
                   target_coupon: float | None = None, n_sims: int = SCREEN_N_SIMS,
                   vol_scale: float | None = None, max_combos: int = MAX_COMBOS) -> pd.DataFrame:
    """Rank every size-`combo_size` basket drawn from `pool` under one fixed
    `structure` (tenor/strike/ki/ko/ki_style) by capital-safety win rate.

    Vols and the full correlation matrix are computed once over the pool's shared
    window, then each combo just slices sub-arrays — so only the simulation runs
    per combo. Raises ValueError if the combo count exceeds `max_combos` or too
    few pool tickers have data."""
    n_total = count_combos(len(pool), combo_size)
    if n_total == 0:
        raise ValueError("basket 大小需介於 1 與候選池數量之間。")
    if n_total > max_combos:
        raise ValueError(f"組合數 {n_total} 超過上限 {max_combos}，請縮小候選池或 basket 大小。")

    win, missing = _recent_window(pool, int(structure["tenor"]))
    if missing:
        raise ValueError(f"以下候選標的無價格資料，請移除：{', '.join(missing)}")
    vs = fcn_engine.vol_scale() if vol_scale is None else vol_scale
    vols, corr = _stats_from_window(win)

    rows = []
    for combo in itertools.combinations(range(len(pool)), combo_size):
        idx = list(combo)
        sub_vols = vols[idx] * vs
        sub_corr = corr[np.ix_(idx, idx)]
        st = _simulate(sub_vols, sub_corr, tenor=int(structure["tenor"]),
                       strike=structure["strike"], ki=structure["ki"], ko=structure["ko"],
                       ki_style=structure["ki_style"], n_sims=n_sims)
        m = _metrics(st)
        mean_corr = (sub_corr[np.triu_indices(len(idx), 1)].mean() if len(idx) > 1 else 1.0)
        row = {
            "標的": " / ".join(pool[i] for i in idx),
            "本金安全勝率": m["win_rate"],
            "提前贖回機率": m["prob_autocall"],
            "合理年化票息": m["fair_coupon"],
            "平均年化波動": float(np.mean(vols[idx])),
            "平均相關": float(mean_corr),
        }
        if target_coupon is not None:
            row["利差(目標-合理)"] = target_coupon - m["fair_coupon"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["本金安全勝率", "合理年化票息"], ascending=[False, False]).reset_index(drop=True)
    return df
