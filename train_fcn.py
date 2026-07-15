"""Calibrate the FCN risk engine (src/fcn.py) against the 18 real FCN purchases
in src/fcn_records.py, and (with --apply) write a volatility-scale override that
the app's 📐 FCN風險評估 tab picks up.

    python train_fcn.py            # dry-run: print the backtest report only
    python train_fcn.py --apply    # also write fcn_calibration.json if it wins

The backtest itself (point-in-time model prediction vs realized outcome from the
real forward price path) lives in src/fcn_backtest.py, shared with the app's
實際戰績 panel. Here we just print the per-note report and grid-search a single
global `vol_scale` in [0.7, 1.5], writing it only if it beats vol_scale=1.0
(mirrors train_weights.py's "TRAINED WINS" gate). Caveat, stated loudly in the
report and the file's _meta: 18 notes over a single 2026 bull window is a tiny,
regime-biased sample — treat the number as a sanity nudge, not a fitted parameter.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from src import fcn as fcn_engine
from src import fcn_backtest as bt
from src import fcn_records

try:  # Windows console defaults to cp950, which can't encode ⚠/中文 — force UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CALIB_PATH = Path(__file__).resolve().parent / "fcn_calibration.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write fcn_calibration.json if the calibrated vol_scale beats 1.0")
    ap.add_argument("--n-sims", type=int, default=bt.DEFAULT_N_SIMS)
    args = ap.parse_args()

    records = fcn_records.RECORDS
    as_of = dt.date.today()
    print(f"Backtesting {len(records)} FCN notes as of {as_of} "
          f"(RF={bt.RF:.0%}, {args.n_sims} sims)\n")

    results = bt.prepare(records, vol_scale=1.0, n_sims=args.n_sims, as_of=as_of)

    hdr = (f"{'id':>2} {'tenor':>5} {'KO/KI':>9} {'標的':22} "
           f"{'勝率':>6} {'P(exit)':>8} {'fair':>7} {'offered':>8} {'margin':>7}  "
           f"{'實際':10} {'exitM':>5} {'實報酬':>8}")
    print(hdr)
    print("-" * len(hdr))
    for x in results:
        r, o = x.record, x.outcome
        if x.pred is None:
            print(f"{r.id:>2} {r.tenor_months:>4}m  (資料不足，略過)")
            continue
        margin = r.coupon - x.fair_coupon
        rr = (f"{o['realized_return']*100:+.1f}%" if o.get("realized_return") is not None else "—")
        exitm = str(o.get("exit_month") or "—")
        basket = "/".join(r.tickers)[:22]
        print(f"{r.id:>2} {r.tenor_months:>4}m {int(r.ko*100):>3}/{int(r.ki*100):<2}% {basket:22} "
              f"{x.model_win_rate*100:>5.1f}% {x.pred.prob_autocall*100:>7.1f}% "
              f"{x.fair_coupon*100:>6.1f}% {r.coupon*100:>7.1f}% {margin*100:>+6.1f}%  "
              f"{o['status']:10} {exitm:>5} {rr:>8}")

    s = bt.summarize(results)
    print(f"\n實際結果彙總：提前出場 {s['n_autocalled']}、到期 {s['n_matured']}、"
          f"尚未結束(censored) {s['n_pending']}")

    resolved = [x for x in results if x.resolved]
    if len(resolved) < 3:
        print("\n可校準(已結束)樣本不足 3 筆，僅輸出報告、不校準 vol_scale。")
        return

    cal = bt.calibrate(resolved, n_sims=args.n_sims)
    print(f"\n可校準樣本 {len(resolved)} 筆：實際本金安全勝率 {s['realized_win_rate']*100:.0f}%"
          f"（實際提前出場率 {cal['real_exit_rate']*100:.0f}%、實際虧損率 {cal['real_loss_rate']*100:.0f}%）")
    print("vol_scale 網格誤差（越低越貼近實際）：")
    for vs in bt.GRID:
        mark = "  <= best" if vs == cal["best_vol_scale"] else ("  (baseline)" if vs == 1.0 else "")
        print(f"    {vs:.2f}: {cal['scores'][vs]:.4f}{mark}")

    best_vs = cal["best_vol_scale"]
    lo, hi = bt.VOL_SCALE_BOUNDS
    wins = cal["best_err"] < cal["baseline_err"] - 1e-4 and lo <= best_vs <= hi
    print(f"\n建議 vol_scale = {best_vs:.2f}（誤差 {cal['best_err']:.4f} vs baseline1.0 "
          f"{cal['baseline_err']:.4f}）→ {'採用' if wins else '不優於基準，維持 1.0'}")
    print("⚠ 樣本小且全為 2026 多頭區間，僅作為研究參考的 sanity 微調，非嚴謹擬合參數。")

    if args.apply:
        if wins:
            payload = {
                "vol_scale": best_vs,
                "_meta": {
                    "calibrated_at": dt.date.today().isoformat(),
                    "as_of": as_of.isoformat(),
                    "n_records": len(records),
                    "n_resolved": len(resolved),
                    "real_exit_rate": round(cal["real_exit_rate"], 3),
                    "real_loss_rate": round(cal["real_loss_rate"], 3),
                    "bounds": list(bt.VOL_SCALE_BOUNDS),
                    "caveat": "18 notes over a single 2026 bull window; regime-biased sanity nudge, not a fitted parameter.",
                },
            }
            CALIB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
            print(f"\n[--apply] 已寫入 {CALIB_PATH.name}（app FCN 分頁下次載入即套用）。"
                  f"以 `git diff {CALIB_PATH.name}` 檢視、`git checkout {CALIB_PATH.name}` 還原。")
        else:
            print("\n[--apply] 校準未勝過基準，未寫入任何檔案。")


if __name__ == "__main__":
    main()
