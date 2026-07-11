"""Train recommendation factor weights from history (Rank IC objective).

Run this where network access works (your machine or a Streamlit Cloud shell):

    python train_weights.py --market tw --years 5
    python train_weights.py --market us --years 5 --max-tickers 120

For each horizon (short/medium/long) it downloads price history for the market's
universe, builds a point-in-time factor panel, reports each factor's mean Rank
IC, and runs a walk-forward comparison of the *trained* weights vs the current
hand-tuned weights (out-of-sample). It then prints a ready-to-paste
FACTOR_WEIGHTS_BY_HORIZON block — but only adopt it if the trained out-of-sample
IC actually beats the current one for that horizon (the script flags this).

This is offline research tooling and does not affect the running app until you
copy the weights into src/recommend.py yourself.
"""
import argparse

import pandas as pd

from src import backtest, recommend, universe
from src import data_loader as dl


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
    args = ap.parse_args()

    if args.market == "tw":
        tickers = universe.get_twse_tickers()
    else:
        tickers = sorted(set(universe.get_top_volume_tickers(30)) | set(universe.get_sp500_tickers()))
    tickers = tickers[: args.max_tickers]
    print(f"Loading {len(tickers)} {args.market.upper()} tickers, {args.years}y of history...")
    price_data = _load_prices(tickers, args.years)
    print(f"Usable tickers: {len(price_data)}\n")

    trained_all = {}
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
        trained_all[horizon] = wf["weights_full"]
        print("Trained weights (full panel):")
        for f, w in sorted(wf["weights_full"].items(), key=lambda kv: -kv[1]):
            print(f"    {f}: {w:.3f}")
        print()

    print("\n# ---- paste-ready (only adopt horizons where TRAINED WINS) ----")
    print("FACTOR_WEIGHTS_BY_HORIZON = {")
    for horizon, w in trained_all.items():
        # walk_forward returns live-model keys covering the full current row,
        # so index strictly — a missing key means trainer/recommend factor
        # drift and should fail loudly, not silently print a stale weight.
        rounded = {f: round(w[f], 2)
                   for f in recommend.FACTOR_WEIGHTS_BY_HORIZON[horizon]}
        # Per-value rounding can drift the row sum off 1.00, which would trip
        # recommend.py's import-time sum check when pasted — absorb the
        # residual into the largest weight so every printed row sums to 1.00.
        residual = round(1.0 - sum(rounded.values()), 2)
        if residual:
            biggest = max(rounded, key=rounded.get)
            rounded[biggest] = round(rounded[biggest] + residual, 2)
        print(f'    "{horizon}": {{')
        for f, val in rounded.items():
            print(f'        "{f}": {val:.2f},')
        print("    },")
    print("}")


if __name__ == "__main__":
    main()
