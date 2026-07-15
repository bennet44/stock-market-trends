"""The 18 real FCN (Fixed Coupon Note) purchases transcribed from the handwritten
notes in FCN/28236.jpg & 28237.jpg. Single source of truth shared by
build_fcn_excel.py (the spreadsheet) and train_fcn.py (the calibration backtest).

Percentages (ko/strike/ki) are fractions of each underlying's initial spot.
`coupon` is the annualized coupon rate as a fraction. All dates are 2026
(confirmed by the user; the notes only wrote month/day).

`handwritten` keeps the notes' own price digits for audit; the *actual* initial
price used for analysis is the real close on `pricing_date`, fetched live via
attach_actual_prices() — the two differ mostly by a few days of drift between
trade and pricing dates (the notes' 期初價 were eyeballed).

KI-observation style: the notes mark "AKI" (American / daily-observed) vs
"飛KI" or blank. We map AKI -> "continuous" and everything else -> "maturity"
(European, observed only at expiry — the common retail convention). Anything
marked uncertain in the notes is flagged in `notes`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FCNRecord:
    id: int
    trade_date: str          # ISO 2026 date the note was booked
    pricing_date: str        # ISO 2026 date the strike/期初價 was fixed
    tickers: list[str]
    handwritten: list[float | None]  # notes' own 期初價 digits (None = 未定價)
    tenor_months: int
    coupon: float            # annualized, fraction
    ko: float                # autocall / 提前出場 barrier, fraction of spot
    strike: float            # 執行價 (conversion), fraction of spot
    ki: float                # 下限 / KI barrier, fraction of spot
    ki_style: str            # "continuous" (AKI) | "maturity"
    ki_label: str            # raw note label: "AKI" / "飛KI" / "—"
    zhang: int               # 張數 (contracts)
    notes: str = ""
    date_uncertain: bool = False
    actual: list[float | None] = field(default_factory=list)  # filled at runtime


def _rec(*args, **kw) -> FCNRecord:
    return FCNRecord(*args, **kw)


# id, trade, pricing, tickers, handwritten, tenor, coupon, ko, strike, ki, style, label, zhang, notes[, date_uncertain]
RECORDS: list[FCNRecord] = [
    _rec(1, "2026-04-29", "2026-05-06", ["TSM", "TSLA", "NVDA"], [393.83, 372.8, 209.25],
         12, 0.1409, 1.00, 0.60, 0.50, "continuous", "AKI", 6,
         "綁13/雄安13；6/6提前出場（手寫KO疑為200%，實務視為100%）"),
    _rec(2, "2026-05-05", "2026-05-05", ["TSM", "TSLA", "NVDA"], [404.35, 422.24, 225.32],
         9, 0.1581, 1.00, 0.65, 0.55, "maturity", "飛KI", 20, "布建15/綁13/雄安2"),
    _rec(3, "2026-05-07", "2026-05-07", ["TSLA", "NVDA", "TSM"], [440.36, 212.6, 422.93],
         9, 0.1477, 1.00, 0.65, 0.55, "maturity", "—", 16, "綁12/綁8/雄安13"),
    _rec(4, "2026-05-08", "2026-05-08", ["NVDA", "TSM", "AVGO"], [214.25, 424.86, 426.58],
         5, 0.1421, 1.00, 0.70, 0.60, "maturity", "飛KI", 10, "綁8/廣2"),
    _rec(5, "2026-05-28", "2026-05-28", ["MU", "NVDA", "TSM"], [923.52, 214.25, 424.86],
         4, 0.238, 0.90, 0.60, 0.50, "maturity", "飛KI", 5, ""),
    _rec(6, "2026-06-01", "2026-06-01", ["NVDA", "AVGO", "ORCL"], [224.36, 459.97, 248.15],
         4, 0.1928, 1.00, 0.70, 0.60, "maturity", "—", 8, "綁6.8/雄安12"),
    _rec(7, "2026-06-15", "2026-06-15", ["TSLA", "NVDA", "TSM"], [411.15, 212.45, 441.4],
         9, 0.1572, 1.00, 0.65, 0.55, "maturity", "—", 10, ""),
    _rec(8, "2026-06-05", "2026-06-05", ["TSLA", "AMZN", "GOOG", "ORCL"], [412.01, 245, 368.09, 186.66],
         6, 0.1619, 1.00, 0.65, 0.55, "maturity", "—", 5, "雄安（KO手寫空白，視為100%）"),
    _rec(9, "2026-06-17", "2026-06-17", ["TSLA", "AMZN", "GOOGL", "ORCL"], [396.38, 227.5, 262.1, 183.53],
         9, 0.1568, 1.00, 0.65, 0.55, "maturity", "飛KI", 18,
         "綁13/雄安3；手寫日期不清，依期初價回推最接近 2026-06-17", True),
    _rec(10, "2026-06-07", "2026-06-07", ["QQQ", "SMH", "SOXX", "IWM"], [722.51, 623.97, 599.93, 289.88],
         5, 0.12, 1.00, 0.8036, 0.70, "maturity", "飛KI", 5, "ETF 標的"),
    _rec(11, "2026-06-25", "2026-06-25", ["MRVL", "INTC", "AMD"], [279.04, 132.28, 159.85],
         3, 0.2268, 0.90, 0.65, 0.50, "maturity", "飛KI", 6, "綁3/雄安3（AMD 期初價字跡不清）"),
    _rec(12, "2026-06-09", "2026-06-09", ["MU", "INTC"], [1145.28, 131.92],
         2, 0.2154, 0.80, 0.65, 0.50, "maturity", "飛KI", 3, "蓮"),
    _rec(13, "2026-06-09", "2026-06-09", ["MU", "SNDK"], [1145.28, 2050.39],
         2, 0.2599, 0.85, 0.65, 0.55, "maturity", "飛KI", 3, "蓮"),
    _rec(14, "2026-06-10", "2026-06-10", ["MU", "TSM", "NVDA"], [1144.69, 455.93, 199.7],
         6, 0.1807, 1.00, 0.60, 0.50, "maturity", "飛KI", 3, "蓮"),
    _rec(15, "2026-06-10", "2026-06-10", ["INTC", "ARM", "MU"], [132, 349.31, 1144.69],
         2, 0.25, 0.90, 0.6505, 0.50, "continuous", "AKI", 3, "蓮"),
    _rec(16, "2026-07-02", "2026-07-02", ["SNDK", "INTC", "TSM", "NVDA"], [1745, 120.35, 434.16, 194.83],
         4, 0.272, 0.90, 0.65, 0.40, "maturity", "飛KI", 9, "綁4/雄安5"),
    _rec(17, "2026-07-03", "2026-07-03", ["ARM", "MU", "AMD", "INTC"], [None, None, None, None],
         4, 0.3793, 0.85, 0.60, 0.55, "maturity", "飛KI", 8, "綁2/雄安6；手寫未定價，期初價取當日實際收盤"),
    _rec(18, "2026-07-09", "2026-07-09", ["AMAT", "AMZN", "META"], [None, None, None],
         6, 0.2625, 0.90, 0.65, 0.50, "maturity", "—", 5, "雄安；手寫未定價，期初價取當日實際收盤"),
]


def attach_actual_prices(records: list[FCNRecord] | None = None,
                         verbose: bool = False) -> list[FCNRecord]:
    """Fill each record's `actual` list with the real close on its pricing_date
    (fetched via src.price_fetch). Mutates and returns the records. Missing data
    leaves None in that slot."""
    from src import price_fetch

    records = records if records is not None else RECORDS
    cache: dict[str, "object"] = {}
    for r in records:
        prices: list[float | None] = []
        for t in r.tickers:
            if t not in cache:
                cache[t] = price_fetch.fetch_closes(t, start="2024-06-01")
                if verbose:
                    s = cache[t]
                    print(f"  fetched {t}: {len(s)} rows" if len(s) else f"  fetched {t}: EMPTY")
            prices.append(price_fetch.close_on(cache[t], r.pricing_date))
        r.actual = prices
    return records
