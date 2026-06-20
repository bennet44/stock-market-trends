"""Technical indicator calculations."""
import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def kd(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 9) -> pd.DataFrame:
    """Stochastic KD oscillator using the conventional Taiwan-market formula.

    RSV is the close's position within the rolling high/low range; K and D
    are then smoothed with a 1/3 weight on the latest value (2/3 on the
    prior K/D), seeded at 50 for the first observation.
    """
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100

    k_values, d_values = [], []
    prev_k, prev_d = 50.0, 50.0
    for r in rsv:
        if pd.isna(r):
            k_values.append(np.nan)
            d_values.append(np.nan)
            continue
        prev_k = prev_k * 2 / 3 + r / 3
        prev_d = prev_d * 2 / 3 + prev_k / 3
        k_values.append(prev_k)
        d_values.append(prev_d)
    return pd.DataFrame({"k": k_values, "d": d_values}, index=close.index)
