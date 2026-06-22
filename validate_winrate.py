"""Calibration check for the 買賣建議 tab's 預測準確機率 (run where network works):

    python validate_winrate.py            # 大立光 3008.TW
    python validate_winrate.py 2330.TW    # any ticker

For short/medium/long holding periods it prints the suggested 建議進場價/賣出價
and compares the *displayed* 預測準確機率 (from recommend.add_price_targets) against an
*independent* plain-loop recomputation of the same path-based "touch" rate,
asserting the two agree within 1%. This is offline research tooling — it does
not affect the app.
"""
import sys

import numpy as np
import pandas as pd

from src import data_loader as dl
from src import recommend


def _independent_touch(close, extreme, n_days, threshold, up):
    """Plain-loop reimplementation of the forward-touch rate (cross-checks the
    vectorized recommend._forward_touch_rate)."""
    c, e = close.values, extreme.values
    n = len(c)
    wins = tot = 0
    for i in range(n - n_days):
        w = e[i + 1:i + 1 + n_days]
        r = (w.max() if up else w.min()) / c[i] - 1
        tot += 1
        wins += (r >= threshold) if up else (r <= threshold)
    return wins / tot * 100 if tot else float("nan")


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "3008.TW"
    hist = dl.get_price_history(ticker, period="10y")
    if hist.empty:
        print(f"no data for {ticker}")
        return
    close, high, low = hist["Close"], hist["High"], hist["Low"]
    p = float(close.iloc[-1])
    print(f"{ticker}  現價={p:.1f}  樣本={len(close)}\n")
    df = pd.DataFrame({"最新收盤價": [p]}, index=[ticker])
    ok = True
    for name, n_days in [("短", 5), ("中", 63), ("長", 252)]:
        horizon = recommend.horizon_for_hold_days(n_days)
        fwd = close.pct_change(n_days).dropna()
        u = float(np.median(fwd[fwd > 0]))
        d = float(np.median(fwd[fwd < 0]))
        # Apply the same technical nudge add_price_targets uses, so the
        # independent recomputation thresholds match the displayed ones.
        bias = recommend.technical_bias(close, high, low, horizon)
        u_adj = u * (1 + recommend.TECH_BIAS_BETA * bias)
        d_adj = d * (1 - recommend.TECH_BIAS_BETA * bias)

        b = recommend.add_price_targets(df, "buy", "NT$", n_days, horizon=horizon)
        disp_b = b["預測準確機率"].iloc[0]
        ind_b = _independent_touch(close, high, n_days, u_adj, up=True)
        s = recommend.add_price_targets(df, "sell", "NT$", n_days, horizon=horizon)
        disp_s = s["預測準確機率"].iloc[0]
        ind_s = _independent_touch(close, low, n_days, d_adj, up=False)

        print(f"{name}({n_days}日) 買: 進場={b['建議買入價'].iloc[0]:.0f} 賣出={b['目標賣出價'].iloc[0]:.0f} "
              f"獲利={b['獲利%'].iloc[0]:.1f}% | 預測準確機率 顯示={disp_b:.2f}% 獨立={ind_b:.2f}% 誤差={abs(disp_b-ind_b):.3f}%")
        print(f"{name}({n_days}日) 賣: 進場={s['建議賣出價'].iloc[0]:.0f} 買回={s['逢低買回參考價'].iloc[0]:.0f} "
              f"| 預測準確機率 顯示={disp_s:.2f}% 獨立={ind_s:.2f}% 誤差={abs(disp_s-ind_s):.3f}%")
        ok = ok and abs(disp_b - ind_b) < 1.0 and abs(disp_s - ind_s) < 1.0
    print("\n" + ("ALL within 1% OK" if ok else "MISMATCH > 1%"))


if __name__ == "__main__":
    main()
