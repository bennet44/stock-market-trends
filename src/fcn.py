"""Monte Carlo pricing / risk engine for Fixed Coupon Notes (FCN).

An FCN is economically a zero-coupon-ish note that sells a downside put
(the knock-in, "KI") on the underlying and is capped by an autocall
feature: at each monthly observation the note redeems early at par once
the underlying closes at/above the autocall barrier; otherwise it survives
to the next observation. If it is never autocalled, principal is repaid at
par unless the underlying breaches the KI barrier, in which case the
investor is repaid in shares at the strike price (i.e. takes the
underlying's downside below strike). A fixed coupon accrues every month
the note is held, regardless of the KI outcome.

All percentages here (strike_pct, ki_pct, autocall_pct) are fractions of
the initial spot, e.g. 0.85 = 85% of spot.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = TRADING_DAYS_PER_YEAR // 12


def historical_vol_and_drift(close: pd.Series) -> tuple[float, float]:
    """Annualized volatility and annualized real-world drift from daily
    log returns of a close-price series."""
    log_ret = np.log(close / close.shift(1)).dropna()
    if log_ret.empty:
        return float("nan"), float("nan")
    return float(log_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)), float(log_ret.mean() * TRADING_DAYS_PER_YEAR)


@dataclass
class PathStats:
    """Simulation results for one (tenor, strike, KI, autocall) combo under
    one choice of drift. ``principal_payoff`` and ``exit_month`` are
    per-path arrays so callers can layer different coupon assumptions on
    top without re-simulating."""
    prob_autocall: float
    prob_breach: float
    avg_exit_month: float
    mean_pv_principal: float
    mean_pv_coupon_factor: float
    principal_payoff: np.ndarray
    exit_month: np.ndarray


def simulate_paths(
    *,
    strike_pct: float,
    ki_pct: float,
    autocall_pct: float,
    tenor_months: int,
    vol_annual: float,
    drift_annual: float,
    risk_free_rate: float,
    ki_style: str = "maturity",
    n_sims: int = 8000,
    seed: int = 42,
) -> PathStats:
    """Simulate GBM price paths (spot normalized to 1.0) and derive the
    autocall / knock-in outcome and payoff for each path.

    ki_style="maturity" only checks the KI barrier at the final
    observation (the common retail-FCN convention); "continuous" checks it
    on every simulated trading day (a stricter, American-style barrier).
    """
    rng = np.random.default_rng(seed)
    n_steps = tenor_months * TRADING_DAYS_PER_MONTH
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    mu = (drift_annual - 0.5 * vol_annual ** 2) * dt
    sigma = vol_annual * np.sqrt(dt)
    z = rng.standard_normal((n_sims, n_steps))
    paths = np.exp(np.cumsum(mu + sigma * z, axis=1))

    obs_idx = np.arange(1, tenor_months + 1) * TRADING_DAYS_PER_MONTH - 1
    obs_prices = paths[:, obs_idx]

    called = obs_prices >= autocall_pct
    any_called = called.any(axis=1)
    first_call_month = called.argmax(axis=1) + 1
    exit_month = np.where(any_called, first_call_month, tenor_months)

    if ki_style == "continuous":
        breached = paths.min(axis=1) < ki_pct
    else:
        breached = obs_prices[:, -1] < ki_pct
    loss_scenario = (~any_called) & breached

    s_t = paths[:, -1]
    principal_payoff = np.ones(n_sims)
    principal_payoff[loss_scenario] = s_t[loss_scenario] / strike_pct

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
    drift (risk-free rate minus dividend yield)."""
    return (1.0 - stats.mean_pv_principal) / stats.mean_pv_coupon_factor


def realized_returns(stats: PathStats, coupon_rate: float) -> np.ndarray:
    """Per-path nominal (undiscounted) return on capital if the note pays
    ``coupon_rate`` annualized, given the already-simulated exit timing and
    principal payoff in ``stats``."""
    return (stats.principal_payoff - 1.0) + coupon_rate * stats.exit_month / 12
