"""Train recommendation factor weights from history (Rank IC objective).

Run this where network access works (your machine or a Streamlit Cloud shell):

    python train_weights.py --market tw --years 5            # run, review, confirm
    python train_weights.py --market us --years 5 --max-tickers 120

For each horizon (short/medium/long) it downloads price history for the market's
universe, builds a point-in-time factor panel, reports each factor's mean Rank
IC, and runs a walk-forward comparison of the *trained* weights vs the current
hand-tuned weights (out-of-sample).

At the end it prints a 訓練前/後 factor-weight diff for every horizon whose
trained weights beat the current ones ("TRAINED WINS") and then, in an
interactive terminal, asks y/N whether to apply them. Answering y (or passing
--apply to skip the prompt, e.g. in a script) writes those horizons to
trained_weights.json at the repo root, under this market's own section
({"tw": {...}, "us": {...}}) — 台股/美股 走勢不同，各自訓練互不覆蓋.
src/recommend.py overlays each market's section onto its shared hand-tuned
baseline (see weights_for()), so the running app picks the new weights up on
next import with no manual copy-paste. The JSON is committed (git-tracked), so
`git diff`/`git checkout trained_weights.json` reviews or reverts a run.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from src import backtest, recommend, universe
from src import data_loader as dl

_TRAINED_WEIGHTS_PATH = Path(__file__).resolve().parent / "trained_weights.json"


def _normalized_row(weights: dict[str, float], horizon: str) -> dict[str, float]:
    """A full FACTOR_WEIGHTS_BY_HORIZON row from trained weights, each value
    rounded to 2dp with the residual absorbed into the largest weight so the
    row sums to exactly 1.00 (recommend.py's import-time sum check rejects
    anything off 1.0). Indexes strictly — a missing key means trainer/recommend
    factor drift and should fail loudly."""
    rounded = {f: round(weights[f], 2)
               for f in recommend.FACTOR_WEIGHTS_BY_HORIZON[horizon]}
    residual = round(1.0 - sum(rounded.values()), 2)
    if residual:
        biggest = max(rounded, key=rounded.get)
        rounded[biggest] = round(rounded[biggest] + residual, 2)
    return rounded


def _write_overrides(market: str, wins: dict[str, dict], meta: dict) -> None:
    """Merge this market's TRAINED-WINS horizon rows into trained_weights.json
    under its own section, preserving the *other* market's section and any
    horizons from earlier runs of this market that weren't retrained now.
    A legacy flat file (pre per-market schema) is migrated in place first."""
    existing = {}
    if _TRAINED_WEIGHTS_PATH.exists():
        try:
            existing = json.loads(_TRAINED_WEIGHTS_PATH.read_text(encoding="utf-8-sig"))
        except ValueError:
            existing = {}
    # Migrate a legacy flat file ({"FACTOR_WEIGHTS_BY_HORIZON": ...}) into the
    # per-market schema, attributing it to the market its _meta names.
    if "FACTOR_WEIGHTS_BY_HORIZON" in existing:
        legacy_market = (existing.get("_meta") or {}).get("market")
        legacy = {"FACTOR_WEIGHTS_BY_HORIZON": existing.pop("FACTOR_WEIGHTS_BY_HORIZON"),
                  "_meta": existing.pop("_meta", {})}
        existing = {legacy_market: legacy} if legacy_market in ("tw", "us") else {}
    section = existing.setdefault(market, {})
    section.setdefault("FACTOR_WEIGHTS_BY_HORIZON", {}).update(wins)
    section["_meta"] = meta
    _TRAINED_WEIGHTS_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_diff(before: dict[str, float], after: dict[str, float]) -> None:
    """Print a 訓練前/後 per-factor weight table, ordered by trained weight."""
    print(f"    {'因子':<14}{'訓練前':>8}{'訓練後':>8}{'Δ':>9}")
    for f in sorted(before, key=lambda k: -after.get(k, 0.0)):
        b, a = before.get(f, 0.0), after.get(f, 0.0)
        d = a - b
        mark = "" if abs(d) < 0.005 else ("  ▲" if d > 0 else "  ▼")
        print(f"    {f:<14}{b:>8.2f}{a:>8.2f}{d:>+9.2f}{mark}")


def _load_prices(tickers: list[str], years: int) -> dict[str, pd.DataFrame]:
    period = f"{years}y"
    out = {}
    for i, t in enumerate(tickers, 1):
        df = dl.get_price_history(t, period=period)
        if not df.empty and len(df) > 260:
            out[t] = df
        if i % 25 == 0:
            print(f"  ...loaded {i}/{len(tickers)} ({len(out)} usable)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["tw", "us"], required=True)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--max-tickers", type=int, default=150,
                    help="cap universe size to keep the download tractable")
    ap.add_argument("--apply", action="store_true",
                    help="skip the y/N confirmation and write TRAINED-WINS "
                         "horizons to trained_weights.json (for scripts)")
    args = ap.parse_args()

    if args.market == "tw":
        tickers = universe.get_twse_tickers()
    else:
        tickers = sorted(set(universe.get_top_volume_tickers(30)) | set(universe.get_sp500_tickers()))
    tickers = tickers[: args.max_tickers]
    print(f"Loading {len(tickers)} {args.market.upper()} tickers, {args.years}y of history...")
    price_data = _load_prices(tickers, args.years)
    print(f"Usable tickers: {len(price_data)}\n")

    trained_all, wins = {}, {}
    for horizon in ["short", "medium", "long"]:
        print(f"==================== horizon: {horizon} ====================")
        panel = backtest.build_panel(price_data, horizon)
        if panel.empty or panel.index.get_level_values("date").nunique() < 8:
            print("  insufficient panel; skipping\n")
            continue
        print("Per-factor Rank IC:")
        print(backtest.factor_ic(panel).round(4).to_string())
        wf = backtest.walk_forward(panel, horizon)
        better = (wf["oos_ic_trained"] or 0) > (wf["oos_ic_current"] or 0)
        print(f"\nOut-of-sample mean Rank IC:  trained={wf['oos_ic_trained']:.4f}  "
              f"current={wf['oos_ic_current']:.4f}  -> {'TRAINED WINS' if better else 'keep current'}")
        trained_all[horizon] = _normalized_row(wf["weights_full"], horizon)
        if horizon not in backtest.TRAINABLE_HORIZONS:
            print("  (long 不採用：資料不足以穩健擬合長線持有，權重維持手調)")
        elif better:
            wins[horizon] = trained_all[horizon]
        print()  # 訓練前/後 並排表格集中在最後一次列出（見下方）

    # ---- 訓練前/後 因子權重並排差異表（所有可訓練 horizon，含未勝出者）----
    # "訓練前" is this market's currently-in-effect weights (baseline overlaid
    # with any override already committed for this market); "訓練後" is the row
    # just trained. Every trainable horizon (short/medium) is shown with its
    # verdict so you can always compare — only TRAINED WINS rows are offered for
    # apply (a losing row would degrade out-of-sample, so it's shown but not
    # adopted). long isn't trained, so it never appears here.
    current_market = recommend.weights_for(args.market == "tw")
    shown = [h for h in ["short", "medium", "long"]
             if h in trained_all and h in backtest.TRAINABLE_HORIZONS]
    if shown:
        print(f"\n# ===== 訓練前/後 因子權重差異（{args.market.upper()}）=====")
        for horizon in shown:
            # 純文字 verdict：Windows 主控台常是 cp950，emoji（如 ✅）會觸發
            # UnicodeEncodeError 讓整支中斷，因此不用任何 emoji。
            verdict = "TRAINED WINS（可套用）" if horizon in wins else "keep current（未勝出，不套用）"
            print(f"\n---- {horizon}：{verdict} ----")
            _print_diff(current_market[horizon], trained_all[horizon])

    if not wins:
        print(f"\n以上皆未在樣本外贏過 {args.market.upper()} 現行權重 → 無可套用的變更。")
        return

    if args.apply:
        apply = True
    elif not sys.stdin.isatty():
        print("\n（非互動環境且未加 --apply → 僅顯示、不套用。加 --apply 可自動套用。）")
        apply = False
    else:
        apply = input(
            f"\n是否將以上 {sorted(wins)} 套用到 {args.market.upper()}？(y/N) "
        ).strip().lower() in ("y", "yes")

    if apply:
        _write_overrides(args.market, wins, {
            "trained_at": dt.date.today().isoformat(),
            "market": args.market,
            "years": args.years,
            "max_tickers": args.max_tickers,
            "horizons": sorted(wins),
        })
        print(f"\n已寫入 {sorted(wins)} 到 {_TRAINED_WEIGHTS_PATH.name}（{args.market.upper()} 區段）。"
              f"審閱：git diff {_TRAINED_WEIGHTS_PATH.name}")
    else:
        print(f"\n未套用，{_TRAINED_WEIGHTS_PATH.name} 未變更。")


if __name__ == "__main__":
    main()
