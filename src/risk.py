"""Risk and statistical analysis helpers."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def annualized_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    growth = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return growth ** (1 / years) - 1 if years > 0 else np.nan


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS
    vol = returns.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Sharpe 的下跌風險版本：分母只取下跌日報酬的標準差（downside deviation），
    上漲波動不被懲罰 — 存股取向在意的是「跌的時候穩不穩」。無下跌日或資料不足
    時回傳 NaN（交叉 z 分數會視為中性）。"""
    excess = returns - risk_free_rate / TRADING_DAYS
    downside = returns[returns < 0]
    dvol = downside.std()
    if len(downside) < 2 or dvol == 0 or np.isnan(dvol):
        return np.nan
    return (excess.mean() / dvol) * np.sqrt(TRADING_DAYS)


def max_drawdown(close: pd.Series) -> float:
    cumulative = close / close.iloc[0]
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown.min()


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    if returns.empty:
        return np.nan
    return np.percentile(returns, (1 - confidence) * 100)


def risk_summary(close: pd.Series, risk_free_rate: float = 0.0) -> dict:
    rets = daily_returns(close)
    return {
        "年化報酬率": annualized_return(rets),
        "年化波動率": annualized_volatility(rets),
        "Sharpe Ratio": sharpe_ratio(rets, risk_free_rate),
        "最大回撤": max_drawdown(close),
        "VaR (95%, 日)": value_at_risk(rets, 0.95),
    }


def correlation_matrix(close_df: pd.DataFrame) -> pd.DataFrame:
    return close_df.pct_change().dropna().corr()
