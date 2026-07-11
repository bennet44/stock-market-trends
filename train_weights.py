"""Train recommendation factor weights from history (Rank IC objective).

Run this where network access works (your machine or a Streamlit Cloud shell):

    python train_weights.py --market tw --years 5            # dry-run: print only
    python train_weights.py --market tw --years 5 --apply    # write overrides
    python train_weights.py --market us --years 5 --max-tickers 120 --apply

For each horizon (short/medium/long) it downloads price history for the market's
universe, builds a point-in-time factor panel, reports each factor's mean Rank
IC, and runs a walk-forward comparison of the *trained* weights vs the current
hand-tuned weights (out-of-sample).

Without --apply (default) it only prints a ready-to-paste FACTOR_WEIGHTS_BY_HORIZON
block for review. With --apply it writes the horizons that beat the current
weights out-of-sample ("TRAINED WINS") to trained_weights.json at the repo root,
which src/recommend.py loads as an override layer over its hand-tuned baseline —
so the running app picks them up on next import, with no manual copy-paste.
The JSON is committed (git-tracked), so `git diff`/`git checkout
trained_weights.json` reviews or reverts a training run. Note the live weight
table is shared across markets, so re-running with a different --market
overwrites the same horizons.
"""
import argparse
import datetime as dt
import json
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


def _write_overrides(wins: dict[str, dict], meta: dict) -> None:
    """Merge the TRAINED-WINS horizon rows into trained_weights.json, preserving
    any horizons from earlier runs that weren't retrained this time."""
    existing = {}
    if _TRAINED_WEIGHTS_PATH.exists():
        try:
            existing = json.loads(_TRAINED_WEIGHTS_PATH.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    existing.setdefault("FACTOR_WEIGHTS_BY_HORIZON", {}).update(wins)
    existing["_meta"] = meta
    _TRAINED_WEIGHTS_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
                    help="write TRAINED-WINS horizons to trained_weights.json "
                         "(the app's override layer) instead of only printing")
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
        print("Trained weights (full panel):")
        for f, w in sorted(wf["weights_full"].items(), key=lambda kv: -kv[1]):
            print(f"    {f}: {w:.3f}")
        print()

    print("\n# ---- paste-ready (trainable horizons; only adopt where TRAINED WINS) ----")
    print("FACTOR_WEIGHTS_BY_HORIZON = {")
    for horizon, rounded in trained_all.items():
        if horizon not in backtest.TRAINABLE_HORIZONS:
            continue  # long is not trained — keep its hand-tuned row
        print(f'    "{horizon}": {{')
        for f, val in rounded.items():
            print(f'        "{f}": {val:.2f},')
        print("    },")
    print("}")

    if args.apply:
        if wins:
            _write_overrides(wins, {
                "trained_at": dt.date.today().isoformat(),
                "market": args.market,
                "years": args.years,
                "max_tickers": args.max_tickers,
                "horizons": sorted(wins),
            })
            print(f"\n[--apply] wrote {sorted(wins)} to {_TRAINED_WEIGHTS_PATH.name} "
                  f"(the app's override layer). Review with `git diff {_TRAINED_WEIGHTS_PATH.name}`.")
        else:
            print("\n[--apply] no horizon beat the current weights out-of-sample; "
                  "nothing written.")


if __name__ == "__main__":
    main()
