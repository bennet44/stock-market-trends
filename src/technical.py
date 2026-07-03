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


def chaikin_money_flow(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Chaikin Money Flow — a volume-weighted accumulation/distribution gauge.

    For each bar the Money Flow Multiplier ((close-low)-(high-close))/(high-low)
    weights that bar's volume by where the close sat in the range; CMF is the
    rolling sum of that money-flow volume over `window` bars divided by total
    volume, giving a value in roughly [-1, 1]. Positive = net buying pressure
    (accumulation). Used as the US-market 籌碼 (capital-flow) proxy where true
    institutional flow data isn't available. A zero-height bar (high == low)
    contributes no multiplier (NaN), and is skipped by the rolling sums.
    """
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = mfm * volume
    return mfv.rolling(window).sum() / volume.rolling(window).sum()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX) with +DI and -DI.

    ADX > 25 = strong trend, < 20 = ranging. +DI > -DI = bullish direction.
    Uses Wilder's smoothing (equivalent to EMA with alpha=1/window).
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    both_pos = (plus_dm > 0) & (minus_dm > 0)
    plus_dm = plus_dm.where(~both_pos | (plus_dm > minus_dm), 0.0)
    minus_dm = minus_dm.where(~both_pos | (minus_dm > plus_dm), 0.0)

    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_line = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})


def kd(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 9) -> pd.DataFrame:
    """Stochastic KDJ oscillator using the conventional Taiwan-market formula.

    RSV is the close's position within the rolling high/low range; K and D
    are then smoothed with a 1/3 weight on the latest value (2/3 on the
    prior K/D), seeded at 50 for the first observation.
    J = 3K − 2D extends beyond [0, 100] and is more sensitive to market-
    sentiment turning points: J < 0 = deeply oversold, J > 100 = deeply overbought.
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
    k_ser = pd.Series(k_values, index=close.index)
    d_ser = pd.Series(d_values, index=close.index)
    j_ser = 3 * k_ser - 2 * d_ser
    return pd.DataFrame({"k": k_ser, "d": d_ser, "j": j_ser})


def support_resistance_levels(
    high: pd.Series, low: pd.Series, close: pd.Series,
    window: int = 10, tolerance: float = 0.015, max_levels: int = 4,
) -> tuple[list[float], list[float]]:
    """Local pivot-based support and resistance levels.

    Scans for bars where high/low is the extremum within ±window bars.
    Merges pivots closer than tolerance (fraction of current price).
    Returns (support_levels, resistance_levels): prices below/above the
    current close, ordered nearest-to-price first, capped at max_levels each.
    """
    n = len(high)
    if n < 2 * window + 1:
        return [], []

    last_price = float(close.iloc[-1])
    high_arr, low_arr = high.values, low.values

    raw_res, raw_sup = [], []
    for i in range(window, n - window):
        if high_arr[i] == max(high_arr[i - window: i + window + 1]):
            raw_res.append(float(high_arr[i]))
        if low_arr[i] == min(low_arr[i - window: i + window + 1]):
            raw_sup.append(float(low_arr[i]))

    def _dedupe_filter(levels: list[float], below: bool) -> list[float]:
        if not levels:
            return []
        merged: list[float] = [levels[0]]
        for lv in sorted(set(levels))[1:]:
            if abs(lv - merged[-1]) / max(last_price, 1e-9) > tolerance:
                merged.append(lv)
        if below:
            relevant = [lv for lv in merged if lv < last_price]
            return relevant[-max_levels:]   # closest below = highest
        relevant = [lv for lv in merged if lv > last_price]
        return relevant[:max_levels]        # closest above = lowest

    return _dedupe_filter(raw_sup, below=True), _dedupe_filter(raw_res, below=False)


def linear_regression_channel(close: pd.Series, std_mult: float = 2.0) -> pd.DataFrame:
    """Linear regression trend channel over the full series.

    Returns DataFrame with 'mid' (regression line), 'upper', 'lower' columns.
    std_mult: channel half-width in residual standard deviations.
    """
    x = np.arange(len(close))
    slope, intercept = np.polyfit(x, close.values, 1)
    mid = pd.Series(slope * x + intercept, index=close.index)
    std = float((close - mid).std())
    return pd.DataFrame({"mid": mid, "upper": mid + std_mult * std, "lower": mid - std_mult * std})
