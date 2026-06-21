"""US Stock Analyst Dashboard — Streamlit app."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_local_storage import LocalStorage

from src import data_loader as dl
from src import fcn
from src import news
from src import recommend
from src import risk
from src import technical as ta
from src import universe

st.set_page_config(page_title="美股分析師看板", layout="wide")

# Persist every tab's widget settings/inputs in the visitor's browser
# (localStorage) so they're restored next time the same browser opens the
# app — this round-trips through a hidden component on every rerun, so it
# survives Streamlit Cloud container restarts (unlike writing to a server
# file) but is local to each visitor's browser, not shared across users.
_SETTINGS_STORAGE_KEY = "dashboard_settings"
# "market" is deliberately excluded: remembering it silently re-selects
# 台股/美股 from a past visit, and a user who doesn't notice the toggle
# already flipped can end up typing a US ticker into the TW-labeled field
# (or vice versa) and seeing the wrong stock. Every other tab's
# settings/inputs are still remembered.
_PERSIST_EXCLUDE_KEYS = {"market"}
_local_storage = LocalStorage()
_saved_settings_raw = _local_storage.getItem(_SETTINGS_STORAGE_KEY)
try:
    _saved_settings = json.loads(_saved_settings_raw) if _saved_settings_raw else {}
except (TypeError, ValueError):
    _saved_settings = {}
for _k, _v in _saved_settings.items():
    if _k not in st.session_state and _k not in _PERSIST_EXCLUDE_KEYS:
        st.session_state[_k] = _v

PERIOD_OPTIONS = {
    "1個月": "1mo", "3個月": "3mo", "6個月": "6mo",
    "1年": "1y", "2年": "2y", "5年": "5y",
}
# Charts that render one trace/row per ticker (comparison overlay, correlation
# heatmap, distribution histogram) become unreadable and slow past this many
# tickers, so those views are capped — tables and the recommendation scan
# still use the full list.
MAX_CHART_TICKERS = 30
# Risk/statistics tab no longer exposes its own risk-free-rate control (that
# input now lives on the recommendation tab), so its Sharpe Ratio uses this
# fixed default instead.
DEFAULT_RISK_FREE_RATE = 0.04
# Holding-period choices for Tab 1's win-rate-based buy/sell price reference:
# trading days drive the historical return distribution, calendar days drive
# the displayed "query date ~ target date" label.
PRICE_TARGET_HORIZONS = {
    "預測期間(1天)": {"trading_days": 1, "calendar_days": 1},
    "預測期間(3天)": {"trading_days": 3, "calendar_days": 3},
    "短期（1週）": {"trading_days": 5, "calendar_days": 7},
    "中期（6個月）": {"trading_days": 126, "calendar_days": 182},
    "長期（1年）": {"trading_days": 252, "calendar_days": 365},
}
# FCN tab: Monte Carlo path count (vectorized, so this stays fast even for a
# multi-year tenor or a several-asset worst-of basket).
FCN_N_SIMS = 8000
FCN_MAX_ASSETS = 5

def _display_name(ticker: str) -> str:
    """Return "TICKER(公司名稱)", falling back to the bare ticker if the
    name is unavailable (e.g. offline or an unrecognized symbol).

    TW tickers use the curated Chinese name (Yahoo's "shortName" for TWSE
    tickers comes back in English); US tickers keep the English shortName.
    """
    name = universe.get_tw_company_name(ticker) if ticker.endswith((".TW", ".TWO")) else None
    if not name:
        info = dl.get_company_info(ticker)
        name = info.get("shortName")
    return f"{ticker}({name})" if name else ticker


def _render_chart(fig: go.Figure, analysis_mode: bool = False) -> None:
    """Render a Plotly chart.

    By default, drag/pan is disabled so mobile scroll isn't trapped by the
    chart; hover (desktop) / tap (mobile) still works for reading exact
    OHLC values regardless of this mode. When analysis_mode is on, drag/pan
    and scroll-to-zoom are re-enabled for users who want to zoom into a
    specific range — trading off easy page scrolling for that.
    """
    if analysis_mode:
        fig.update_layout(dragmode="zoom")
        config = {"displayModeBar": True, "scrollZoom": True}
    else:
        fig.update_layout(dragmode=False)
        config = {"displayModeBar": False, "scrollZoom": False}
    st.plotly_chart(fig, use_container_width=True, config=config)


@st.cache_data(show_spinner=False)
def _fcn_run(strike_pct, ki_pct, ko_pct, tenor_months, vols, drifts, corr, risk_free_rate, ki_style, n_sims):
    return fcn.simulate_basket(
        strike_pct=strike_pct, ki_pct=ki_pct, ko_pct=ko_pct, tenor_months=tenor_months,
        vols=vols, drifts=drifts, corr=corr, risk_free_rate=risk_free_rate,
        ki_style=ki_style, n_sims=n_sims,
    )


market = st.radio("市場", ["美股", "台股"], horizontal=True, key="market")
is_tw = market == "台股"
currency = "NT$" if is_tw else "$"

tab_overview, tab_reco, tab_compare_risk, tab_fcn = st.tabs(
    ["📈 價格、技術指標與基本面", "💡 買賣建議", "🔗 多股比較與風險統計", "📐 FCN風險評估"]
)

# ---------- Tab 1: Price, technical indicators & fundamentals (one ticker) ----------
with tab_overview:
    col_ticker, col_period = st.columns([2, 1])
    with col_ticker:
        if is_tw:
            default_ticker = "2330"
            ticker_label = "股票代號（台股代碼，例如 2330、0050）"
        else:
            default_ticker = "AAPL"
            ticker_label = "股票代號"
        raw_primary = st.text_input(
            ticker_label, value=default_ticker, key=f"price_ticker_{'tw' if is_tw else 'us'}"
        ).strip().upper() or default_ticker
        primary = universe.resolve_tw_ticker(raw_primary) if is_tw else raw_primary
    with col_period:
        period_label = st.selectbox(
            "時間範圍", list(PERIOD_OPTIONS.keys()), index=3, key=f"period_tab1_{'tw' if is_tw else 'us'}"
        )
    period = PERIOD_OPTIONS[period_label]
    primary_label = _display_name(primary)
    st.subheader(f"{primary_label} 價格與技術指標")
    df = dl.get_price_history(primary, period=period)
    if df.empty:
        st.error(f"找不到 {primary} 的資料，請確認代號是否正確。")
    else:
        close = df["Close"]
        sma5, sma10, sma20 = ta.sma(close, 5), ta.sma(close, 10), ta.sma(close, 20)
        bb = ta.bollinger_bands(close)

        st.caption("提示：將滑鼠移到圖上（手機點一下 K 棒）即可看到當天開盤／最高／最低／收盤價，不需開啟下方分析模式。")
        analysis_mode = st.checkbox(
            "📊 啟用圖表分析模式（可縮放、拖曳查看細節；行動裝置上頁面滑動會變得較不順手）",
            key=f"chart_analysis_mode_{'tw' if is_tw else 'us'}",
        )

        # "三竹股市" look for TW stocks: 漲=red／跌=green candles (the
        # reverse of the US green-up/red-down convention); background stays
        # white to match the US chart.
        up_color, down_color = ("#ff3333", "#00b300") if is_tw else ("#2ca02c", "#d62728")
        legend_top = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=primary_label,
            increasing_line_color=up_color, increasing_fillcolor=up_color,
            decreasing_line_color=down_color, decreasing_fillcolor=down_color,
        ))
        fig.add_trace(go.Scatter(x=df.index, y=sma5, name="SMA5", line=dict(width=1, color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df.index, y=sma10, name="SMA10", line=dict(width=1, color="#ff7f0e")))
        fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20", line=dict(width=1, color="#9467bd")))
        fig.add_trace(go.Scatter(x=df.index, y=bb["upper"], name="Bollinger Upper",
                                  line=dict(width=1, dash="dot"), opacity=0.5))
        fig.add_trace(go.Scatter(x=df.index, y=bb["lower"], name="Bollinger Lower",
                                  line=dict(width=1, dash="dot"), opacity=0.5))
        fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                           margin=dict(t=80, b=20), legend=legend_top)
        _render_chart(fig, analysis_mode)

        vol_colors = [
            up_color if c >= o else down_color
            for o, c in zip(df["Open"], df["Close"])
        ]
        vol_fig = go.Figure(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors))
        vol_fig.update_layout(height=180, margin=dict(t=10, b=10), title="成交量")
        _render_chart(vol_fig, analysis_mode)

        col1, col2, col3 = st.columns(3)
        with col1:
            rsi_series = ta.rsi(close)
            rsi_fig = go.Figure(go.Scatter(x=df.index, y=rsi_series, name="RSI"))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.update_layout(height=250, title="RSI (14)", margin=dict(t=30, b=10))
            _render_chart(rsi_fig, analysis_mode)
        with col2:
            macd_df = ta.macd(close)
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df.index, y=macd_df["macd"], name="MACD"))
            macd_fig.add_trace(go.Scatter(x=df.index, y=macd_df["signal"], name="Signal"))
            macd_fig.add_trace(go.Bar(x=df.index, y=macd_df["hist"], name="Histogram"))
            macd_fig.update_layout(height=250, title="MACD", margin=dict(t=30, b=10))
            _render_chart(macd_fig, analysis_mode)
        with col3:
            kd_df = ta.kd(df["High"], df["Low"], close)
            kd_fig = go.Figure()
            kd_fig.add_trace(go.Scatter(x=df.index, y=kd_df["k"], name="K"))
            kd_fig.add_trace(go.Scatter(x=df.index, y=kd_df["d"], name="D"))
            kd_fig.add_hline(y=80, line_dash="dash", line_color="red")
            kd_fig.add_hline(y=20, line_dash="dash", line_color="green")
            kd_fig.update_layout(height=250, title="KD (9)", margin=dict(t=30, b=10))
            _render_chart(kd_fig, analysis_mode)

        latest = close.iloc[-1]
        prev = close.iloc[-2] if len(close) > 1 else latest
        st.metric(f"{primary_label} 最新收盤價", f"{currency}{latest:,.2f}",
                   f"{(latest / prev - 1) * 100:.2f}%")

        st.markdown("##### 建議買入／賣出價格參考（依勝率設定）")
        col_horizon, col_winrate = st.columns(2)
        with col_horizon:
            horizon_label = st.selectbox(
                "預測期間", list(PRICE_TARGET_HORIZONS.keys()), index=2,
                key=f"price_target_horizon_{'tw' if is_tw else 'us'}",
            )
        with col_winrate:
            win_rate_pct = st.number_input(
                "設定勝率 (%)", min_value=50, max_value=95, value=60, step=5,
                key=f"win_rate_{'tw' if is_tw else 'us'}",
                help="以歷史上漲／下跌期間的報酬率分布，反推在此勝率下對應的漲跌幅。",
            )
        horizon = PRICE_TARGET_HORIZONS[horizon_label]
        hold_days, calendar_days = horizon["trading_days"], horizon["calendar_days"]
        query_date = dt.date.today()
        target_date = query_date + dt.timedelta(days=calendar_days)
        st.caption(
            f"依過去 {hold_days} 個交易日（{horizon_label}）的歷史報酬率分布估算，"
            f"對應查詢日 {query_date.year}/{query_date.month}/{query_date.day} ~ "
            f"{horizon_label}預測日 {target_date.year}/{target_date.month}/{target_date.day}，"
            "未考慮基本面或市場狀況，僅供參考，非投資建議。"
        )
        fwd_returns = close.pct_change(periods=hold_days).dropna()
        ups, downs = fwd_returns[fwd_returns > 0], fwd_returns[fwd_returns < 0]
        up_move = np.percentile(ups, 100 - win_rate_pct) if not ups.empty else None
        down_move = np.percentile(downs, 100 - win_rate_pct) if not downs.empty else None

        col_buy, col_sell = st.columns(2)
        with col_buy:
            if down_move is not None:
                st.metric("建議買入價（逢低承接）", f"{currency}{latest * (1 + down_move):,.2f}",
                           f"{down_move * 100:.2f}%")
                st.caption(f"歷史下跌期間中，有 {win_rate_pct}% 的機率跌幅不超過此價位。")
            else:
                st.metric("建議買入價（逢低承接）", "資料不足")
        with col_sell:
            if up_move is not None:
                st.metric("建議賣出價（目標停利）", f"{currency}{latest * (1 + up_move):,.2f}",
                           f"{up_move * 100:.2f}%")
                st.caption(f"歷史上漲期間中，有 {win_rate_pct}% 的機率可達此漲幅。")
            else:
                st.metric("建議賣出價（目標停利）", "資料不足")

    st.divider()
    st.subheader(f"{primary_label} 基本面財務")
    fdf = dl.get_fundamentals_table([primary])
    if fdf.empty:
        st.warning("無法取得基本面資料。")
    else:
        display = fdf.copy()
        if "市值" in display:
            display["市值"] = display["市值"].apply(
                lambda v: f"{currency}{v / 1e9:,.1f}B" if pd.notnull(v) else None)
        for pct_col in ["營收成長率", "盈餘成長率", "淨利率", "ROE", "股息率"]:
            if pct_col in display:
                display[pct_col] = display[pct_col].apply(
                    lambda v: f"{v * 100:.2f}%" if pd.notnull(v) else None)
        st.dataframe(display, use_container_width=True)

    st.divider()
    news_date_label = news.recent_news_date_label()
    st.subheader(f"{primary_label} 相關新聞（{news_date_label}）")
    # TW tickers: prefer the curated Chinese name for the news query — yfinance's
    # "shortName" comes back in English for TWSE tickers (see _display_name),
    # and an English company name paired with a zh-TW Google News search
    # routinely returns zero matches.
    company_name = universe.get_tw_company_name(primary) if is_tw else None
    if not company_name:
        company_name = (
            fdf["公司名稱"].iloc[0] if not fdf.empty and "公司名稱" in fdf and pd.notnull(fdf["公司名稱"].iloc[0]) else None
        )
    news_items = news.get_recent_news(primary, company_name)
    # Extra, market-appropriate sources layered on top of Google News: SEC
    # EDGAR filings + Reuters for US tickers, TWSE exchange news + MOPS
    # material-info disclosures (best-effort, see news.get_mops_news) for
    # TW tickers. Display-only — not fed into the recommendation scoring.
    if is_tw:
        extra_news = news.get_twse_news(primary, company_name) + news.get_mops_news(primary, company_name)
    else:
        extra_news = news.get_reuters_news(primary, company_name) + news.get_sec_filings(primary)
    news_items = sorted(news_items + extra_news, key=lambda n: n["published"], reverse=True)
    if not news_items:
        st.info(f"暫無 {news_date_label} 的相關新聞。")
    else:
        for n in news_items:
            published_str = n["published"].strftime("%Y-%m-%d %H:%M UTC")
            st.markdown(f"- [{n['title']}]({n['link']})　_{n['source']}｜{published_str}_")

# ---------- Tab 2: Multi-stock comparison, correlation & risk stats ----------
with tab_compare_risk:
    col_compare, col_period2 = st.columns([2, 1])
    with col_compare:
        if is_tw:
            compare_label = "比較用股票代號（逗號分隔，台股代碼；留空代表全部台股觀察清單，含ETF及個股）"
            compare_default = "2330, 0050"
        else:
            compare_label = "比較用股票代號（逗號分隔；留空代表全部 S&P 500 成分股）"
            compare_default = "AAPL, OKLO"
        compare_input = st.text_input(
            compare_label, value=compare_default, key=f"compare_input_{'tw' if is_tw else 'us'}")
    with col_period2:
        period_label = st.selectbox(
            "時間範圍", list(PERIOD_OPTIONS.keys()), index=3, key=f"period_tab2_{'tw' if is_tw else 'us'}"
        )
    period = PERIOD_OPTIONS[period_label]
    raw_compare = compare_input.strip()
    if raw_compare:
        if is_tw:
            compare_tickers = [universe.resolve_tw_ticker(t) for t in raw_compare.split(",") if t.strip()]
        else:
            compare_tickers = [t.strip().upper() for t in raw_compare.split(",") if t.strip()]
    elif is_tw:
        compare_tickers = universe.get_twse_tickers()
        st.caption(f"已自動帶入全部台股觀察清單，含ETF及個股（{len(compare_tickers)} 檔）。")
    else:
        compare_tickers = universe.get_sp500_tickers()
        st.caption(f"已自動帶入全部 S&P 500 成分股（{len(compare_tickers)} 檔）。首次掃描資料量大，"
                   "請耐心等候，結果會快取加速下次載入。")

    chart_tickers = compare_tickers
    if len(compare_tickers) > MAX_CHART_TICKERS:
        st.info(f"標的數量較多（{len(compare_tickers)} 檔），圖表僅顯示前 {MAX_CHART_TICKERS} 檔以維持可讀性與效能。")
        chart_tickers = compare_tickers[:MAX_CHART_TICKERS]

    st.subheader("多股票報酬比較與相關性")
    close_df = dl.get_multi_close(chart_tickers, period=period)
    if close_df.empty or len(chart_tickers) < 2:
        st.info("請輸入至少兩個股票代號以進行比較。")
    else:
        chart_labels = {t: _display_name(t) for t in chart_tickers}
        normalized = close_df / close_df.iloc[0] * 100
        norm_fig = go.Figure()
        for t in normalized.columns:
            norm_fig.add_trace(go.Scatter(x=normalized.index, y=normalized[t], name=chart_labels.get(t, t)))
        norm_fig.update_layout(height=400, title="累積報酬比較（基準=100）",
                                margin=dict(t=40, b=10))
        st.plotly_chart(norm_fig, use_container_width=True)

        corr = risk.correlation_matrix(close_df)
        corr_labels = [chart_labels.get(t, t) for t in corr.columns]
        heat_fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr_labels, y=corr_labels,
            colorscale="RdBu", zmid=0, text=corr.round(2).values,
            texttemplate="%{text}",
        ))
        heat_fig.update_layout(height=400, title="日報酬相關係數矩陣", margin=dict(t=40, b=10))
        st.plotly_chart(heat_fig, use_container_width=True)

    st.divider()
    st.subheader("風險與統計分析")
    rows = {}
    price_by_ticker = {}
    for t in compare_tickers:
        df_t = dl.get_price_history(t, period=period)
        if not df_t.empty:
            price_by_ticker[t] = df_t
            rows[t] = risk.risk_summary(df_t["Close"], DEFAULT_RISK_FREE_RATE)
    if rows:
        summary_df = pd.DataFrame(rows).T
        summary_df.index = [_display_name(t) for t in summary_df.index]
        fmt = summary_df.copy()
        for col in ["年化報酬率", "年化波動率", "最大回撤", "VaR (95%, 日)"]:
            fmt[col] = fmt[col].apply(lambda v: f"{v * 100:.2f}%" if pd.notnull(v) else None)
        fmt["Sharpe Ratio"] = fmt["Sharpe Ratio"].apply(
            lambda v: f"{v:.2f}" if pd.notnull(v) else None)
        st.dataframe(fmt, use_container_width=True)

        st.markdown("#### 日報酬率分布")
        hist_tickers = list(price_by_ticker.keys())
        if len(hist_tickers) > MAX_CHART_TICKERS:
            st.caption(f"標的數量較多，分布圖僅顯示前 {MAX_CHART_TICKERS} 檔以維持可讀性。")
            hist_tickers = hist_tickers[:MAX_CHART_TICKERS]
        hist_fig = go.Figure()
        for t in hist_tickers:
            rets = risk.daily_returns(price_by_ticker[t]["Close"]) * 100
            hist_fig.add_trace(go.Histogram(x=rets, name=_display_name(t), opacity=0.6, nbinsx=60))
        hist_fig.update_layout(barmode="overlay", height=350,
                                xaxis_title="日報酬率 (%)", margin=dict(t=20, b=10))
        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.warning("無可用資料以計算風險指標。")

# ---------- Tab 3: Buy/sell recommendations ----------
with tab_reco:
    col_period3, col_winrate3, col_topn = st.columns(3)
    with col_period3:
        period_label = st.selectbox(
            "時間範圍", list(PERIOD_OPTIONS.keys()), index=3, key=f"period_tab3_{'tw' if is_tw else 'us'}"
        )
        period = PERIOD_OPTIONS[period_label]
    with col_winrate3:
        win_rate_pct3 = st.number_input(
            "設定勝率 (%)", min_value=50, max_value=95, value=60, step=5,
            key=f"win_rate_tab3_{'tw' if is_tw else 'us'}",
            help="以歷史上漲／下跌期間的報酬率分布，反推在此勝率下對應的漲跌幅。",
        )
    with col_topn:
        top_n = st.selectbox(
            "建議買賣標的數量 (Top N)", [1, 5, 10, 15], index=1, key=f"topn_tab3_{'tw' if is_tw else 'us'}"
        )

    st.subheader("基金經理人觀點：建議買入 / 賣出")
    if is_tw:
        scope_desc = "篩選範圍為「台股觀察清單（含ETF及個股）」。"
    else:
        scope_desc = "篩選範圍為「美股交易量前 30 大（依近期平均成交量排序的觀察名單）」與「S&P 500 成分股」的聯集。"
    st.caption(
        scope_desc +
        "綜合「期間報酬率」「Sharpe Ratio」"
        "「價格趨勢（價格 / SMA50）」「估值（1/預估PE）」"
        "「新聞情緒（近 4 日中文新聞標題關鍵字判斷）」五項因子計算組內相對評分，"
        "僅反映目前範圍內標的之相對排序，非投資建議。"
        "買入價／賣出價以最新收盤價估算，目標漲跌幅依設定勝率反推歷史報酬率分布，"
        "未考慮基本面或市場狀況，僅供參考。各欄位可點選表頭由大至小／小至大排序。"
    )
    if is_tw:
        reco_universe = universe.get_twse_tickers()
    else:
        reco_universe = sorted(set(universe.get_top_volume_tickers(30)) | set(universe.get_sp500_tickers()))
    with st.spinner(f"正在掃描 {len(reco_universe)} 檔標的計算評分，資料量較大可能需要數分鐘…"):
        reco_table = recommend.build_recommendation_table(reco_universe, period, DEFAULT_RISK_FREE_RATE)
    if reco_table.empty:
        st.warning("無足夠資料產生建議，請確認時間範圍。")
    else:
        buy_df, sell_df = recommend.top_buy_sell(reco_table, top_n)
        buy_df = recommend.add_reason(
            recommend.add_price_targets(buy_df, "buy", currency, win_rate_pct3, period), "buy")
        sell_df = recommend.add_reason(
            recommend.add_price_targets(sell_df, "sell", currency, win_rate_pct3, period), "sell")

        _PCT_COLS = ["期間報酬率", "趨勢(價格/SMA50)"]
        _PLAIN_COLS = ["Sharpe Ratio", "估值(1/預估PE)", "新聞情緒", "RSI (14)", "綜合評分"]
        _PRICE_COLS = ["建議買入價", "建議賣出價", "目標賣出價", "逢低買回參考價"]

        def _format_reco(df: pd.DataFrame) -> pd.DataFrame:
            fmt = df.copy()
            for col in _PCT_COLS:
                fmt[col] = fmt[col] * 100
            fmt.index = [_display_name(t) for t in fmt.index]
            return fmt

        def _column_config(df: pd.DataFrame) -> dict:
            config = {}
            for col in _PCT_COLS + ["獲利%"]:
                if col in df:
                    config[col] = st.column_config.NumberColumn(col, format="%.2f%%")
            for col in _PLAIN_COLS:
                if col in df:
                    config[col] = st.column_config.NumberColumn(col, format="%.2f")
            for col in _PRICE_COLS:
                if col in df:
                    config[col] = st.column_config.NumberColumn(col, format=f"{currency}%.2f")
            return config

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🟢 建議買入 Top {len(buy_df)}")
            fmt_buy = _format_reco(buy_df)
            st.dataframe(fmt_buy, use_container_width=True, column_config=_column_config(fmt_buy))
        with col2:
            st.markdown(f"#### 🔴 建議賣出 Top {len(sell_df)}")
            fmt_sell = _format_reco(sell_df)
            st.dataframe(fmt_sell, use_container_width=True, column_config=_column_config(fmt_sell))

        if len(buy_df) < top_n:
            st.info(
                f"目前範圍共 {len(reco_table)} 檔標的，為避免買入／賣出名單重複，"
                f"已各自裁切為 {len(buy_df)} 檔（最多取清單一半），而非選擇的 Top {top_n}。"
            )

# ---------- Tab 4: FCN risk assessment ----------
with tab_fcn:
    st.subheader("FCN 風險評估與條款試算")
    st.caption(
        "- **主要輸入**：標的、敲出價、敲入價、到期日、配息率\n"
        "- **多標的時**：若任一標的觸及下限則視為「觸及」，但只有全部標的同時達到提前出場價才提前出場\n"
        "- **模型假設**：月配息、月觀察、固定波動率、幾何布朗運動\n"
        "- **參數來源**：標的歷史日報酬的年化波動率與相關性\n"
        "- **模擬結果**：提前出場機率、本金虧損機率\n"
        "- **限制**：不含發行商報價、信用風險、手續費，為簡化估算，**僅供研究參考，非投資建議**"
    )

    col_n_assets, col_tickers = st.columns([1, 3])
    with col_n_assets:
        n_assets = st.number_input(
            "標的數量", min_value=1, max_value=FCN_MAX_ASSETS, value=1, step=1,
            key=f"fcn_n_assets_{'tw' if is_tw else 'us'}",
        )
    with col_tickers:
        # Switching 市場 only builds the *active* market's FCN widgets, so the
        # other market's text_input (e.g. fcn_tickers_us_3) isn't rendered that
        # run and Streamlit garbage-collects its widget state — coming back, the
        # field would reset to the hardcoded default. Keep the last-entered value
        # in a plain (non-widget) session_state key, which is never GC'd, and use
        # it as the field's default so a 台股⇄美股 round-trip preserves the input.
        # The key includes n_assets so changing 標的數量 still repopulates with
        # that many default tickers.
        _fcn_tickers_shadow_key = f"fcn_tickers_value_{'tw' if is_tw else 'us'}_{n_assets}"
        fcn_default_tickers = st.session_state.get(_fcn_tickers_shadow_key) or ", ".join(
            (["2330", "2317", "2454", "2412", "2882"] if is_tw else ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"])[:n_assets]
        )
        fcn_ticker_label = f"標的股票代號（共 {n_assets} 個，以「,」區隔）" + ("（台股代碼，例如 2330）" if is_tw else "")
        fcn_raw_tickers = st.text_input(
            fcn_ticker_label, value=fcn_default_tickers, key=f"fcn_tickers_{'tw' if is_tw else 'us'}_{n_assets}"
        )
        st.session_state[_fcn_tickers_shadow_key] = fcn_raw_tickers
    fcn_tickers_input = [t.strip().upper() for t in fcn_raw_tickers.split(",") if t.strip()]
    fcn_tickers = [universe.resolve_tw_ticker(t) for t in fcn_tickers_input] if is_tw else fcn_tickers_input

    if len(fcn_tickers) != n_assets:
        st.warning(f"已設定標的數量為 {n_assets}，但目前以「,」區隔輸入了 {len(fcn_tickers)} 個代號，請調整一致。")
    else:
        _mkt_suffix = "tw" if is_tw else "us"
        col_tenor, col_strike, col_ko = st.columns(3)
        with col_tenor:
            tenor_months = st.number_input(
                "投資時間（個X月）", min_value=1, max_value=60, value=6, step=1, key=f"fcn_tenor_{_mkt_suffix}"
            )
        with col_strike:
            strike_pct = st.number_input(
                "執行價 STRIKE(%)", min_value=50.0, max_value=120.0, value=100.0, step=1.0,
                key=f"fcn_strike_{_mkt_suffix}",
            ) / 100
        with col_ko:
            ko_pct = st.number_input(
                "提前出場 KO(%)", min_value=50.0, max_value=130.0, value=100.0, step=1.0,
                key=f"fcn_ko_{_mkt_suffix}",
            ) / 100

        col_ki, col_coupon, col_ki_style = st.columns(3)
        with col_ki:
            ki_pct = st.number_input(
                "下限價 KI(%)", min_value=10.0, max_value=120.0, value=75.0, step=1.0, key=f"fcn_ki_{_mkt_suffix}"
            ) / 100
        with col_coupon:
            coupon_rate = st.number_input(
                "年化收益率(%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5,
                key=f"fcn_coupon_{_mkt_suffix}",
            ) / 100
        with col_ki_style:
            ki_style_label = st.radio(
                "KI 觀察方式", ["到期日觀察（歐式，較常見）", "每日觀察（美式，較嚴格）"],
                key=f"fcn_ki_style_{_mkt_suffix}",
            )
            ki_style = "maturity" if ki_style_label.startswith("到期日觀察") else "continuous"

        if ki_pct > strike_pct:
            st.warning("下限價 KI 通常不應高於執行價 STRIKE，請確認條款設定是否正確。")

        fcn_closes = {}
        for t in fcn_tickers:
            t_df = dl.get_price_history(t, period="5y")
            if not t_df.empty:
                cutoff = t_df.index.max() - pd.DateOffset(months=tenor_months)
                fcn_closes[t] = t_df["Close"][t_df.index >= cutoff]
        missing = [t for t in fcn_tickers if t not in fcn_closes]
        if missing:
            st.error(f"找不到以下代號的價格資料，請確認是否正確：{', '.join(missing)}")
        else:
            close_df = pd.concat(fcn_closes, axis=1, join="inner")
            close_df.columns = fcn_tickers
            vols, drift_hist, corr = fcn.historical_stats(close_df)

            dividend_yields = []
            for t in fcn_tickers:
                dy = dl.get_company_info(t).get("dividendYield") or 0.0
                if dy > 0.5:  # yfinance has, at times, returned this as a percent rather than a fraction
                    dy /= 100.0
                dividend_yields.append(dy)
            dividend_yields = np.array(dividend_yields)

            asset_summary = pd.DataFrame({
                "最新收盤價": close_df.iloc[-1].apply(lambda v: f"{currency}{v:,.2f}"),
                "年化歷史波動率": [f"{v * 100:.1f}%" for v in vols],
                "年化歷史漲跌幅": [f"{v * 100:.1f}%" for v in drift_hist],
                "股息率（估）": [f"{v * 100:.2f}%" for v in dividend_yields],
            }, index=[_display_name(t) for t in fcn_tickers])
            st.markdown("##### 標的概況")
            st.caption(f"歷史統計窗口：投資時間相同的最近 {tenor_months} 個月（若窗口較短，波動率／相關性估計可能不穩定）。")
            st.dataframe(asset_summary, use_container_width=True)
            if n_assets > 1:
                corr_df = pd.DataFrame(corr, index=[_display_name(t) for t in fcn_tickers],
                                        columns=[_display_name(t) for t in fcn_tickers])
                st.markdown("###### 標的間相關係數")
                st.dataframe(corr_df.style.format("{:.2f}"), use_container_width=True)

            drift_choice = st.radio(
                "風險評估的股價成長率假設",
                ["中性假設：預期報酬＝無風險利率，僅反映波動風險（較保守，預設）",
                 "延伸近期歷史走勢（依上表「年化歷史漲跌幅」，可能過度樂觀或悲觀，僅供對照）"],
                key=f"fcn_drift_choice_{_mkt_suffix}",
            )
            use_historical_drift = drift_choice.startswith("延伸近期歷史走勢")

            drift_risk_neutral = DEFAULT_RISK_FREE_RATE - dividend_yields
            drift_for_risk = drift_hist if use_historical_drift else drift_risk_neutral

            with st.spinner("模擬中…"):
                stats = _fcn_run(strike_pct, ki_pct, ko_pct, tenor_months, vols, drift_for_risk, corr,
                                  DEFAULT_RISK_FREE_RATE, ki_style, FCN_N_SIMS)
                rn_stats = (
                    stats if not use_historical_drift else
                    _fcn_run(strike_pct, ki_pct, ko_pct, tenor_months, vols, drift_risk_neutral, corr,
                             DEFAULT_RISK_FREE_RATE, ki_style, FCN_N_SIMS)
                )
            realized = fcn.realized_returns(stats, coupon_rate)
            fair_coupon = fcn.fair_coupon_rate(rn_stats)

            st.divider()
            st.markdown("##### 風險評估結果")
            r1, r2, r3 = st.columns(3)
            r1.metric("提前出場機率", f"{stats.prob_autocall * 100:.1f}%")
            r2.metric("平均出場月數", f"{stats.avg_exit_month:.1f} 個月")
            r3.metric("本金虧損機率", f"{stats.prob_breach * 100:.1f}%")

            r4, r5, r6 = st.columns(3)
            r4.metric("期望報酬（總報酬，非年化）", f"{realized.mean() * 100:.2f}%")
            r5.metric("5%最差情境報酬", f"{np.percentile(realized, 5) * 100:.2f}%")
            r6.metric("風險中性參考年化收益率", f"{fair_coupon * 100:.2f}%",
                      f"您輸入 {coupon_rate * 100:.2f}%")
            st.caption(
                "「風險中性參考年化收益率」為假設無風險利率"
                f"（年化 {DEFAULT_RISK_FREE_RATE * 100:.0f}%）下，使票券折現價值等於票面（100%）的"
                "理論票息，可與您輸入的「年化收益率」比較：您輸入值若明顯偏低，代表此票券條款對發行商"
                "較有利；若明顯偏高，請留意是否低估了風險（例如波動率估計過低）。"
                "「本金虧損機率」「期望報酬」則依您所選的風險評估假設計算，為您實際可能面對的風險參考。"
            )

    st.divider()
    st.caption(
        "模型限制：固定波動率與相關性（不含波動率微笑/期限結構、不含相關性隨市況變化）、"
        "不含股息再投資與交易成本、不含發行商信用風險與流動性折價，且歷史波動率／報酬率不保證代表"
        "未來。實際 FCN 報價請以發行商（券商/銀行）條款書為準。"
    )

# Snapshot every widget's current value back into the browser's localStorage
# (overwriting the dict loaded at startup) so it's there on the next visit.
# Excludes the local-storage component's own bookkeeping keys and "market"
# (see _PERSIST_EXCLUDE_KEYS above).
_settings_to_save = {
    k: v for k, v in st.session_state.items()
    if k != "storage_init" and not k.startswith("save_") and k not in _PERSIST_EXCLUDE_KEYS
}
_local_storage.setItem(_SETTINGS_STORAGE_KEY, json.dumps(_settings_to_save), key="save_dashboard_settings")
