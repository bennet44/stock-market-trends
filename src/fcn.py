"""Monte Carlo pricing / risk engine for Fixed Coupon Notes (FCN), supporting
either a single underlying or a worst-of basket of underlyings.

An FCN is economically a zero-coupon-ish note that sells a downside put
(the knock-in, "KI") on its underlying(s) and is capped by an autocall/KO
feature: at each monthly observation the note redeems early at par once
the underlying(s) close at/above the KO barrier; otherwise it survives to
the next observation. If it is never called, principal is repaid at par
unless the KI barrier is breached, in which case the investor is repaid in
shares of the worst-performing underlying at the strike price. A coupon
accrues every month the note is held, regardless of the KI outcome.

For a basket of more than one underlying ("worst-of"), every barrier check
is driven by the worst-performing underlying at each point in time — the
note only calls once *every* underlying clears the KO level, and knocks in
if *any* underlying breaches the KI level.

All percentages here (strike_pct, ki_pct, ko_pct) are fractions of each
underlying's own initial spot, e.g. 0.85 = 85% of spot.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = TRADING_DAYS_PER_YEAR // 12


def historical_stats(close_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-asset annualized volatility/drift and the pairwise return
    correlation matrix, from a DataFrame of aligned daily close prices
    (one column per underlying)."""
    log_ret = np.log(close_df / close_df.shift(1)).dropna()
    vols = (log_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)).to_numpy()
    drifts = (log_ret.mean() * TRADING_DAYS_PER_YEAR).to_numpy()
    corr = log_ret.corr().to_numpy()
    return vols, drifts, corr


@dataclass
class PathStats:
    """Simulation results for one (tenor, strike, KI, KO) combo. ``principal_payoff``
    and ``exit_month`` are per-path arrays so callers can layer different
    coupon assumptions on top without re-simulating."""
    prob_autocall: float
    prob_breach: float
    avg_exit_month: float
    mean_pv_principal: float
    mean_pv_coupon_factor: float
    principal_payoff: np.ndarray
    exit_month: np.ndarray


def simulate_basket(
    *,
    strike_pct: float,
    ki_pct: float,
    ko_pct: float,
    tenor_months: int,
    vols: np.ndarray,
    drifts: np.ndarray,
    corr: np.ndarray,
    risk_free_rate: float,
    ki_style: str = "maturity",
    n_sims: int = 8000,
    seed: int = 42,
) -> PathStats:
    """Simulate correlated GBM paths for each underlying (each normalized to
    spot=1.0) and derive the worst-of basket's KO/KI outcome and payoff.

    ki_style="maturity" only checks the KI barrier at the final observation
    (the common retail-FCN convention); "continuous" checks it on every
    simulated trading day (a stricter, American-style barrier). A single
    underlying is just the n_assets=1 case (corr=[[1.0]]).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(vols)
    n_steps = tenor_months * TRADING_DAYS_PER_MONTH
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_sims, n_steps, n_assets))
    correlated_z = z @ chol.T

    mu = (drifts - 0.5 * vols ** 2) * dt
    sigma = vols * np.sqrt(dt)
    asset_paths = np.exp(np.cumsum(mu + sigma * correlated_z, axis=1))  # (n_sims, n_steps, n_assets)
    worst = asset_paths.min(axis=2)  # worst-of ratio across underlyings, per day

    obs_idx = np.arange(1, tenor_months + 1) * TRADING_DAYS_PER_MONTH - 1
    obs_prices = worst[:, obs_idx]

    called = obs_prices >= ko_pct
    any_called = called.any(axis=1)
    first_call_month = called.argmax(axis=1) + 1
    exit_month = np.where(any_called, first_call_month, tenor_months)

    if ki_style == "continuous":
        breached = worst.min(axis=1) < ki_pct
    else:
        breached = obs_prices[:, -1] < ki_pct
    loss_scenario = (~any_called) & breached

    worst_t = worst[:, -1]
    principal_payoff = np.ones(n_sims)
    principal_payoff[loss_scenario] = worst_t[loss_scenario] / strike_pct

    df_month = np.exp(-risk_free_rate * np.arange(1, tenor_months + 1) / 12)
    cum_pv_factor = np.cumsum(df_month) / 12
    pv_coupon_factor = cum_pv_factor[exit_month - 1]
    df_exit = np.exp(-risk_free_rate * exit_month / 12)
    pv_principal = principal_payoff * df_exit

    return PathStats(
        prob_autocall=float(any_called.mean()),
        prob_breach=float(loss_scenario.mean()),
        avg_exit_month=float(exit_month.mean()),
        mean_pv_principal=float(pv_principal.mean()),
        mean_pv_coupon_factor=float(pv_coupon_factor.mean()),
        principal_payoff=principal_payoff,
        exit_month=exit_month,
    )


def fair_coupon_rate(stats: PathStats) -> float:
    """Annualized coupon that makes the risk-neutral PV of the note equal
    par. Only meaningful when ``stats`` was simulated with a risk-neutral
    drift (risk-free rate minus dividend yield) for every underlying."""
    return (1.0 - stats.mean_pv_principal) / stats.mean_pv_coupon_factor


def realized_returns(stats: PathStats, coupon_rate: float) -> np.ndarray:
    """Per-path nominal (undiscounted) return on capital if the note pays
    ``coupon_rate`` annualized, given the already-simulated exit timing and
    principal payoff in ``stats``."""
    return (stats.principal_payoff - 1.0) + coupon_rate * stats.exit_month / 12
