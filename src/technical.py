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


def _pivot_points(high: pd.Series, low: pd.Series, window: int) -> list[tuple[int, float, str]]:
    """Local swing points: bars whose high/low is the extremum within ±window
    bars. Returns [(bar_index, price, "high"|"low"), ...] in time order."""
    n = len(high)
    if n < 2 * window + 1:
        return []
    high_arr, low_arr = high.values, low.values
    pivots: list[tuple[int, float, str]] = []
    for i in range(window, n - window):
        if high_arr[i] == max(high_arr[i - window: i + window + 1]):
            pivots.append((i, float(high_arr[i]), "high"))
        if low_arr[i] == min(low_arr[i - window: i + window + 1]):
            pivots.append((i, float(low_arr[i]), "low"))
    return pivots


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
    pivots = _pivot_points(high, low, window)
    if not pivots:
        return [], []

    last_price = float(close.iloc[-1])
    raw_res = [p for _, p, kind in pivots if kind == "high"]
    raw_sup = [p for _, p, kind in pivots if kind == "low"]

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


def candlestick_patterns(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    lookback: int = 10,
) -> list[dict]:
    """Rule-based classic candlestick patterns over the last `lookback` bars.

    Returns [{"bar": int (positional index), "name", "side" ("bull"/"bear"/
    "neutral"), "desc"}, ...] in time order. Single-bar reversal shapes
    (hammer/hanging man) take the prior 5-bar drift as trend context; at most
    one pattern is reported per bar (the first/strongest match wins).
    """
    n = len(close)
    if n < 4:
        return []
    o, h, l, c = (s.values.astype(float) for s in (open_, high, low, close))
    out: list[dict] = []
    for i in range(max(3, n - lookback), n):
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        upper = h[i] - max(o[i], c[i])
        lower = min(o[i], c[i]) - l[i]
        up_bar, dn_bar = c[i] > o[i], c[i] < o[i]
        pbody = abs(c[i - 1] - o[i - 1])
        # 前 5 根收盤漂移當背景趨勢（單根反轉形態需要上下文）
        ctx_start = max(0, i - 5)
        downtrend = c[i - 1] < c[ctx_start]
        uptrend = c[i - 1] > c[ctx_start]

        found = None
        # 三根形態優先（訊號較強）
        if i >= 2:
            b0, b1, b2 = abs(c[i - 2] - o[i - 2]), pbody, body
            r0 = h[i - 2] - l[i - 2]
            if (c[i - 2] < o[i - 2] and r0 > 0 and b0 >= 0.5 * r0
                    and b1 <= 0.5 * b0
                    and up_bar and c[i] >= (o[i - 2] + c[i - 2]) / 2):
                found = ("晨星", "bull", "長黑後出現小實體再長紅收復過半，底部反轉訊號。")
            elif (c[i - 2] > o[i - 2] and r0 > 0 and b0 >= 0.5 * r0
                    and b1 <= 0.5 * b0
                    and dn_bar and c[i] <= (o[i - 2] + c[i - 2]) / 2):
                found = ("暮星", "bear", "長紅後出現小實體再長黑跌破過半，頭部反轉訊號。")
            elif all(c[j] > o[j] and abs(c[j] - o[j]) >= 0.4 * (h[j] - l[j])
                     for j in (i - 2, i - 1, i) if h[j] > l[j]) \
                    and c[i - 2] < c[i - 1] < c[i] and h[i] > l[i]:
                found = ("紅三兵", "bull", "連三根中大實體收紅且收盤逐日墊高，多方動能延續。")
            elif all(c[j] < o[j] and abs(c[j] - o[j]) >= 0.4 * (h[j] - l[j])
                     for j in (i - 2, i - 1, i) if h[j] > l[j]) \
                    and c[i - 2] > c[i - 1] > c[i] and h[i] > l[i]:
                found = ("黑三鴉", "bear", "連三根中大實體收黑且收盤逐日下移，空方動能延續。")
        # 雙根形態
        if found is None and pbody > 0:
            if (up_bar and c[i - 1] < o[i - 1]
                    and o[i] <= c[i - 1] and c[i] >= o[i - 1]):
                found = ("多頭吞噬", "bull", "紅K實體完全包住前一根黑K，底部轉強訊號。")
            elif (dn_bar and c[i - 1] > o[i - 1]
                    and o[i] >= c[i - 1] and c[i] <= o[i - 1]):
                found = ("空頭吞噬", "bear", "黑K實體完全包住前一根紅K，頭部轉弱訊號。")
            elif (max(o[i], c[i]) <= max(o[i - 1], c[i - 1])
                    and min(o[i], c[i]) >= min(o[i - 1], c[i - 1])
                    and body <= 0.5 * pbody and body > 0):
                side = "bull" if c[i - 1] < o[i - 1] else "bear"
                found = ("母子線", side, "小實體縮在前一根大實體內，原趨勢動能收斂、留意反轉。")
        # 單根形態
        if found is None:
            if body <= 0.1 * rng:
                found = ("十字星", "neutral", "開收盤幾乎相同，多空拉鋸；趨勢末端出現時留意變盤。")
            elif lower >= 2 * body and upper <= max(body, 0.1 * rng) and body > 0:
                if downtrend:
                    found = ("錘子線", "bull", "下跌後留長下影線，低檔承接力道浮現的反轉訊號。")
                elif uptrend:
                    found = ("吊人線", "bear", "上漲後留長下影線，高檔獲利了結壓力浮現，留意轉弱。")
        if found:
            out.append({"bar": i, "name": found[0], "side": found[1], "desc": found[2]})
    return out


def chart_patterns(
    high: pd.Series, low: pd.Series, close: pd.Series,
    window: int = 5, tolerance: float = 0.02,
) -> list[dict]:
    """Pivot-sequence price-structure patterns: 雙重底(W底)/雙重頂(M頭)、
    頭肩頂/頭肩底、三角收斂. At most one (the most recent) match per type.

    Returns [{"name", "side", "desc", "points": [(bar_idx, price), ...],
    "neckline": float | None}, ...]. Only patterns whose last pivot falls in
    the most recent ~40% of the series are reported, so stale formations from
    months ago don't clutter the chart.
    """
    n = len(close)
    pivots = _pivot_points(high, low, window)
    # Flat stretches produce duplicate same-kind pivots a bar apart (the
    # extremum ties across the window); merge same-kind pivots closer than
    # `window` bars, keeping the more extreme one, so the alternating
    # high/low sequence the detectors expect isn't broken.
    merged: list[tuple[int, float, str]] = []
    for piv in pivots:
        if merged and merged[-1][2] == piv[2] and piv[0] - merged[-1][0] <= window:
            if (piv[2] == "high" and piv[1] >= merged[-1][1]) or \
                    (piv[2] == "low" and piv[1] <= merged[-1][1]):
                merged[-1] = piv
        else:
            merged.append(piv)
    pivots = merged
    if len(pivots) < 3:
        return []
    last_price = float(close.iloc[-1])
    recent_cut = n - max(int(n * 0.4), 3 * window)
    out: list[dict] = []

    def _near(a: float, b: float, tol: float) -> bool:
        return abs(a - b) / max(last_price, 1e-9) <= tol

    # 雙重底/頂：兩個相近、間隔夠遠的同向 pivot，中間夾反向 pivot（頸線）
    for kind, name, side, cmp_break in (
        ("low", "W底（雙重底）", "bull", lambda px, neck: px > neck),
        ("high", "M頭（雙重頂）", "bear", lambda px, neck: px < neck),
    ):
        hit = None
        for j in range(len(pivots) - 1, 0, -1):
            i2, p2, k2 = pivots[j]
            if k2 != kind or i2 < recent_cut:
                continue
            for q in range(j - 1, -1, -1):
                i1, p1, k1 = pivots[q]
                if k1 != kind:
                    continue
                between = [(i, p) for i, p, k in pivots if k != kind and i1 < i < i2]
                if not between or i2 - i1 <= window:
                    continue
                neck_i, neck = max(between, key=lambda t: t[1]) if kind == "low" \
                    else min(between, key=lambda t: t[1])
                # 形態高度門檻：頸線距雙底/雙頂需 ≥3%，濾掉盤整雜訊的偽形態
                deep_enough = abs(neck - (p1 + p2) / 2) / max(last_price, 1e-9) >= 0.03
                if _near(p1, p2, tolerance) and deep_enough:
                    status = "已突破頸線" if cmp_break(last_price, neck) else "形成中（未過頸線）"
                    hit = {
                        "name": name, "side": side, "neckline": neck,
                        "points": [(i1, p1), (neck_i, neck), (i2, p2)],
                        "desc": f"兩{'低' if kind == 'low' else '高'}點相近（頸線 {neck:.1f}），{status}。",
                    }
                break
            if hit:
                out.append(hit)
                break

    # 頭肩頂/底：三個同向 pivot，中間顯著高（低）於兩肩、兩肩相近
    for kind, name, side, is_head in (
        ("high", "頭肩頂", "bear", lambda hd, sh: hd > sh * 1.02),
        ("low", "頭肩底", "bull", lambda hd, sh: hd < sh * 0.98),
    ):
        same = [(i, p) for i, p, k in pivots if k == kind]
        if len(same) >= 3:
            for j in range(len(same) - 1, 1, -1):
                (iL, pL), (iH, pH), (iR, pR) = same[j - 2], same[j - 1], same[j]
                if iR < recent_cut:
                    break
                if is_head(pH, pL) and is_head(pH, pR) and _near(pL, pR, 0.03):
                    opp = [p for i, p, k in pivots if k != kind and iL < i < iR]
                    neck = (min(opp) if kind == "low" else max(opp)) if opp else None
                    out.append({
                        "name": name, "side": side, "neckline": neck,
                        "points": [(iL, pL), (iH, pH), (iR, pR)],
                        "desc": f"{'頭部' if kind == 'high' else '底部'}顯著{'高' if kind == 'high' else '低'}於兩肩，"
                                + (f"頸線約 {neck:.1f}，" if neck else "")
                                + ("跌破頸線確認反轉。" if kind == "high" else "突破頸線確認反轉。"),
                    })
                    break

    # 三角收斂：近段 pivot 高點遞降且低點遞升
    recent_highs = [(i, p) for i, p, k in pivots if k == "high" and i >= recent_cut]
    recent_lows = [(i, p) for i, p, k in pivots if k == "low" and i >= recent_cut]
    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        hs, ls = recent_highs[-2:], recent_lows[-2:]
        if hs[1][1] < hs[0][1] * (1 - tolerance / 2) and ls[1][1] > ls[0][1] * (1 + tolerance / 2):
            out.append({
                "name": "三角收斂", "side": "neutral", "neckline": None,
                "points": hs + ls,
                "desc": "高點遞降、低點遞升，波動收斂待變盤；突破方向通常決定後市。",
            })
    return out
