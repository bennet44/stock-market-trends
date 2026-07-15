"""Backtest the FCN Monte Carlo engine (src/fcn.py) against the real FCN
purchases in src/fcn_records.py, and calibrate a volatility scale from them.

For each note, two things are computed as of its 2026 pricing date:

  • MODEL  — estimate each underlying's vol/correlation from the tenor-length
    daily window *ending on the pricing date* (point-in-time, no look-ahead),
    then run the same simulation the app uses under a risk-neutral drift to get
    the capital-safety win rate (1 − P(assigned shares)), P(early-exit) and the
    fair coupon.

  • ACTUAL — replay the real worst-of price path from the pricing date to
    `as_of` to see whether/when the note actually autocalled and whether the KI
    barrier was actually breached (→ assigned the worst stock, a loss). Notes
    that have neither autocalled nor matured yet are PENDING (censored).

Shared by train_fcn.py (CLI + `--apply` calibration) and the app's 實際戰績
panel, so both read one implementation. Prices come through
data_loader.get_price_history (yfinance with a chart-API fallback), matching the
rest of the app.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import data_loader as dl
from src import fcn as fcn_engine

RF = 0.04                                  # matches app.DEFAULT_RISK_FREE_RATE
VOL_SCALE_BOUNDS = (0.7, 1.5)
GRID = [round(x, 2) for x in np.arange(0.70, 1.51, 0.05)]
DEFAULT_N_SIMS = 6000
TD_MONTH = fcn_engine.TRADING_DAYS_PER_MONTH

_PRICE_CACHE: dict[str, pd.Series] = {}


def _closes(ticker: str) -> pd.Series:
    if ticker not in _PRICE_CACHE:
        df = dl.get_price_history(ticker, period="5y")
        _PRICE_CACHE[ticker] = df["Close"] if not df.empty else pd.Series(dtype="float64")
    return _PRICE_CACHE[ticker]


def _aligned(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cols = {}
    for t in tickers:
        s = _closes(t)
        cols[t] = s[(s.index >= start) & (s.index <= end)]
    df = pd.concat(cols, axis=1, join="inner")
    df.columns = tickers
    return df.dropna()


def _pd_corr(corr: np.ndarray) -> np.ndarray:
    """Nudge toward identity so a thin-window correlation stays Cholesky-able."""
    return 0.999 * corr + 0.001 * np.eye(len(corr))


def point_in_time_stats(r) -> tuple | None:
    """(vols, risk-neutral drifts, corr) from the tenor-length daily window
    ending on the pricing date, or None if the window is too short."""
    price_date = pd.Timestamp(r.pricing_date)
    start = price_date - pd.DateOffset(months=r.tenor_months)
    win = _aligned(r.tickers, start, price_date)
    if len(win) < 20:
        return None
    vols, _drift_hist, corr = fcn_engine.historical_stats(win)
    drifts = np.full(len(r.tickers), RF)   # risk-neutral; dividend yield ~0 for these names
    return vols, drifts, _pd_corr(corr)


def model_predict(r, stats, vol_scale: float = 1.0, n_sims: int = DEFAULT_N_SIMS) -> fcn_engine.PathStats:
    vols, drifts, corr = stats
    return fcn_engine.simulate_basket(
        strike_pct=r.strike, ki_pct=r.ki, ko_pct=r.ko, tenor_months=r.tenor_months,
        vols=vols * vol_scale, drifts=drifts, corr=corr, risk_free_rate=RF,
        ki_style=r.ki_style, n_sims=n_sims,
    )


def realized_outcome(r, as_of: dt.date) -> dict:
    """Replay the real worst-of path from pricing date to min(maturity, as_of)."""
    price_date = pd.Timestamp(r.pricing_date)
    maturity = price_date + pd.DateOffset(months=r.tenor_months)
    end = min(maturity, pd.Timestamp(as_of))
    path = _aligned(r.tickers, price_date, end)
    if len(path) < 2:
        return {"status": "no_data", "exit_month": None, "breached": None,
                "realized_return": None, "worst_final": None}

    ratio = path / path.iloc[0]
    worst = ratio.min(axis=1).to_numpy()
    n_days = len(worst)
    matured = pd.Timestamp(as_of) >= maturity

    exit_month = None
    for m in range(1, r.tenor_months + 1):
        idx = m * TD_MONTH - 1
        if idx >= n_days:
            break
        if worst[idx] >= r.ko:
            exit_month = m
            break

    breached_cont = bool(worst.min() < r.ki)
    worst_final = float(worst[-1])

    if exit_month is not None:
        return {"status": "autocalled", "exit_month": exit_month, "breached": False,
                "realized_return": r.coupon * exit_month / 12, "worst_final": worst_final}
    if matured:
        breached = breached_cont if r.ki_style == "continuous" else (worst_final < r.ki)
        principal = (worst_final / r.strike) if breached else 1.0
        ret = (principal - 1.0) + r.coupon * r.tenor_months / 12
        return {"status": "matured", "exit_month": r.tenor_months, "breached": breached,
                "realized_return": ret, "worst_final": worst_final}
    return {"status": "pending", "exit_month": None,
            "breached": breached_cont if r.ki_style == "continuous" else None,
            "realized_return": None, "worst_final": worst_final}


@dataclass
class NoteResult:
    record: object
    stats: tuple | None            # point-in-time (vols, drifts, corr)
    pred: fcn_engine.PathStats | None
    outcome: dict

    @property
    def model_win_rate(self) -> float | None:
        return None if self.pred is None else 1.0 - self.pred.prob_breach

    @property
    def fair_coupon(self) -> float | None:
        return None if self.pred is None else fcn_engine.fair_coupon_rate(self.pred)

    @property
    def resolved(self) -> bool:
        return self.outcome["status"] in ("autocalled", "matured")


def prepare(records, *, vol_scale: float = 1.0, n_sims: int = DEFAULT_N_SIMS,
            as_of: dt.date | None = None) -> list[NoteResult]:
    """One NoteResult per record: point-in-time model prediction + realized
    outcome. `as_of` defaults to today (the notes are 2026-dated)."""
    as_of = as_of or dt.date.today()
    results = []
    for r in records:
        stats = point_in_time_stats(r)
        pred = None if stats is None else model_predict(r, stats, vol_scale, n_sims)
        outcome = realized_outcome(r, as_of) if stats is not None else {
            "status": "no_window", "exit_month": None, "breached": None,
            "realized_return": None, "worst_final": None}
        results.append(NoteResult(r, stats, pred, outcome))
    return results


def summarize(results: list[NoteResult]) -> dict:
    resolved = [x for x in results if x.resolved]
    n_auto = sum(x.outcome["status"] == "autocalled" for x in results)
    n_mat = sum(x.outcome["status"] == "matured" for x in results)
    n_pend = sum(x.outcome["status"] == "pending" for x in results)
    realized_win = (np.mean([not bool(x.outcome["breached"]) for x in resolved])
                    if resolved else None)
    model_win = (np.mean([x.model_win_rate for x in resolved])
                 if resolved else None)
    return {
        "n_total": len(results), "n_autocalled": n_auto, "n_matured": n_mat,
        "n_pending": n_pend, "n_resolved": len(resolved),
        "realized_win_rate": None if realized_win is None else float(realized_win),
        "realized_exit_rate": (float(np.mean([x.outcome["status"] == "autocalled" for x in resolved]))
                               if resolved else None),
        "realized_loss_rate": (float(np.mean([bool(x.outcome["breached"]) for x in resolved]))
                               if resolved else None),
        "mean_model_win_rate": None if model_win is None else float(model_win),
    }


def calibrate(resolved: list[NoteResult], *, grid=GRID, n_sims: int = DEFAULT_N_SIMS) -> dict:
    """Grid-search a global vol_scale that best aligns average model
    P(early-exit)/P(loss) with the realized frequencies over resolved notes."""
    real_exit = np.mean([x.outcome["status"] == "autocalled" for x in resolved])
    real_loss = np.mean([bool(x.outcome["breached"]) for x in resolved])

    def err(vs: float) -> float:
        preds = [model_predict(x.record, x.stats, vs, n_sims) for x in resolved]
        m_exit = np.mean([p.prob_autocall for p in preds])
        m_loss = np.mean([p.prob_breach for p in preds])
        return abs(m_exit - real_exit) + abs(m_loss - real_loss)

    scores = {vs: err(vs) for vs in grid}
    best_vs = min(scores, key=scores.get)
    return {
        "real_exit_rate": float(real_exit), "real_loss_rate": float(real_loss),
        "scores": scores, "best_vol_scale": float(best_vs),
        "best_err": float(scores[best_vs]),
        "baseline_err": float(scores.get(1.0, err(1.0))),
    }


def run_backtest(records, *, vol_scale: float = 1.0, n_sims: int = DEFAULT_N_SIMS,
                 as_of: dt.date | None = None) -> tuple[list[NoteResult], dict]:
    """Convenience for the app panel: per-note results + aggregate summary."""
    results = prepare(records, vol_scale=vol_scale, n_sims=n_sims, as_of=as_of)
    return results, summarize(results)
