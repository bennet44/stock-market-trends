"""US Stock Analyst Dashboard — Streamlit app."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data_loader as dl
from src import news
from src import recommend
from src import risk
from src import technical as ta
from src import universe

st.set_page_config(page_title="美股分析師看板", layout="wide")

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
# Holding-period window (trading days) used to build the historical return
# distribution behind Tab 1's win-rate-based buy/sell price reference.
PRICE_TARGET_HOLD_DAYS = 5

market = st.radio("市場", ["美股", "台股"], horizontal=True, key="market")
is_tw = market == "台股"

tab_overview, tab_compare_risk, tab_reco = st.tabs(
    ["📈 價格、技術指標與基本面", "🔗 多股比較、相關性與風險統計", "💡 買賣建議"]
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
        primary = universe.normalize_tw_ticker(raw_primary) if is_tw else raw_primary
    with col_period:
        period_label = st.selectbox("時間範圍", list(PERIOD_OPTIONS.keys()), index=3, key="period_tab1")
    period = PERIOD_OPTIONS[period_label]
    st.subheader(f"{primary} 價格與技術指標")
    df = dl.get_price_history(primary, period=period)
    if df.empty:
        st.error(f"找不到 {primary} 的資料，請確認代號是否正確。")
    else:
        close = df["Close"]
        sma20, sma50 = ta.sma(close, 20), ta.sma(close, 50)
        bb = ta.bollinger_bands(close)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=primary,
        ))
        fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA50", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=bb["upper"], name="Bollinger Upper",
                                  line=dict(width=1, dash="dot"), opacity=0.5))
        fig.add_trace(go.Scatter(x=df.index, y=bb["lower"], name="Bollinger Lower",
                                  line=dict(width=1, dash="dot"), opacity=0.5,
                                  fill="tonexty"))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False,
                           margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        vol_fig = go.Figure(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
        vol_fig.update_layout(height=180, margin=dict(t=10, b=10), title="成交量")
        st.plotly_chart(vol_fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            rsi_series = ta.rsi(close)
            rsi_fig = go.Figure(go.Scatter(x=df.index, y=rsi_series, name="RSI"))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.update_layout(height=250, title="RSI (14)", margin=dict(t=30, b=10))
            st.plotly_chart(rsi_fig, use_container_width=True)
        with col2:
            macd_df = ta.macd(close)
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df.index, y=macd_df["macd"], name="MACD"))
            macd_fig.add_trace(go.Scatter(x=df.index, y=macd_df["signal"], name="Signal"))
            macd_fig.add_trace(go.Bar(x=df.index, y=macd_df["hist"], name="Histogram"))
            macd_fig.update_layout(height=250, title="MACD", margin=dict(t=30, b=10))
            st.plotly_chart(macd_fig, use_container_width=True)

        latest = close.iloc[-1]
        prev = close.iloc[-2] if len(close) > 1 else latest
        st.metric(f"{primary} 最新收盤價", f"${latest:,.2f}",
                   f"{(latest / prev - 1) * 100:.2f}%")

        st.markdown("##### 建議買入／賣出價格參考（依勝率設定）")
        win_rate_pct = st.number_input(
            "設定勝率 (%)", min_value=50, max_value=95, value=60, step=5,
            key=f"win_rate_{'tw' if is_tw else 'us'}",
            help="以歷史上漲／下跌期間的報酬率分布，反推在此勝率下對應的漲跌幅。",
        )
        st.caption(
            f"依過去 {PRICE_TARGET_HOLD_DAYS} 個交易日的歷史報酬率分布估算，"
            "未考慮基本面或市場狀況，僅供參考，非投資建議。"
        )
        fwd_returns = close.pct_change(periods=PRICE_TARGET_HOLD_DAYS).dropna()
        ups, downs = fwd_returns[fwd_returns > 0], fwd_returns[fwd_returns < 0]
        up_move = np.percentile(ups, 100 - win_rate_pct) if not ups.empty else None
        down_move = np.percentile(downs, 100 - win_rate_pct) if not downs.empty else None

        col_buy, col_sell = st.columns(2)
        with col_buy:
            if down_move is not None:
                st.metric("建議買入價（逢低承接）", f"${latest * (1 + down_move):,.2f}",
                           f"{down_move * 100:.2f}%")
                st.caption(f"歷史下跌期間中，有 {win_rate_pct}% 的機率跌幅不超過此價位。")
            else:
                st.metric("建議買入價（逢低承接）", "資料不足")
        with col_sell:
            if up_move is not None:
                st.metric("建議賣出價（目標停利）", f"${latest * (1 + up_move):,.2f}",
                           f"{up_move * 100:.2f}%")
                st.caption(f"歷史上漲期間中，有 {win_rate_pct}% 的機率可達此漲幅。")
            else:
                st.metric("建議賣出價（目標停利）", "資料不足")

    st.divider()
    st.subheader(f"{primary} 基本面財務")
    fdf = dl.get_fundamentals_table([primary])
    if fdf.empty:
        st.warning("無法取得基本面資料。")
    else:
        display = fdf.copy()
        if "市值" in display:
            display["市值"] = display["市值"].apply(
                lambda v: f"${v / 1e9:,.1f}B" if pd.notnull(v) else None)
        for pct_col in ["營收成長率", "盈餘成長率", "淨利率", "ROE", "股息率"]:
            if pct_col in display:
                display[pct_col] = display[pct_col].apply(
                    lambda v: f"{v * 100:.2f}%" if pd.notnull(v) else None)
        st.dataframe(display, use_container_width=True)

    st.divider()
    news_date_label = news.recent_news_date_label()
    st.subheader(f"{primary} 相關新聞（{news_date_label}）")
    company_name = (
        fdf["公司名稱"].iloc[0] if not fdf.empty and "公司名稱" in fdf and pd.notnull(fdf["公司名稱"].iloc[0]) else None
    )
    news_items = news.get_recent_news(primary, company_name)
    if not news_items:
        st.info(f"暫無 {news_date_label} 的相關中文新聞。")
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
        period_label = st.selectbox("時間範圍", list(PERIOD_OPTIONS.keys()), index=3, key="period_tab2")
    period = PERIOD_OPTIONS[period_label]
    raw_compare = compare_input.strip()
    if raw_compare:
        if is_tw:
            compare_tickers = [universe.normalize_tw_ticker(t) for t in raw_compare.split(",") if t.strip()]
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
        normalized = close_df / close_df.iloc[0] * 100
        norm_fig = go.Figure()
        for t in normalized.columns:
            norm_fig.add_trace(go.Scatter(x=normalized.index, y=normalized[t], name=t))
        norm_fig.update_layout(height=400, title="累積報酬比較（基準=100）",
                                margin=dict(t=40, b=10))
        st.plotly_chart(norm_fig, use_container_width=True)

        corr = risk.correlation_matrix(close_df)
        heat_fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
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
            hist_fig.add_trace(go.Histogram(x=rets, name=t, opacity=0.6, nbinsx=60))
        hist_fig.update_layout(barmode="overlay", height=350,
                                xaxis_title="日報酬率 (%)", margin=dict(t=20, b=10))
        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.warning("無可用資料以計算風險指標。")

# ---------- Tab 3: Buy/sell recommendations ----------
with tab_reco:
    col_period3, col_rfr, col_topn = st.columns(3)
    with col_period3:
        period_label = st.selectbox("時間範圍", list(PERIOD_OPTIONS.keys()), index=3, key="period_tab3")
        period = PERIOD_OPTIONS[period_label]
    with col_rfr:
        risk_free_rate = st.number_input("無風險利率（年化，%）", value=4.0, step=0.1, key="rfr_tab3") / 100
    with col_topn:
        top_n = st.selectbox("建議買賣標的數量 (Top N)", [1, 5, 10, 15], index=1, key="topn_tab3")

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
        "買入價／賣出價以最新收盤價估算，目標區間為單純假設 3~5% 價格波動，"
        "未考慮基本面或市場狀況，僅供參考。"
    )
    if is_tw:
        reco_universe = universe.get_twse_tickers()
    else:
        reco_universe = sorted(set(universe.get_top_volume_tickers(30)) | set(universe.get_sp500_tickers()))
    with st.spinner(f"正在掃描 {len(reco_universe)} 檔標的計算評分，資料量較大可能需要數分鐘…"):
        reco_table = recommend.build_recommendation_table(reco_universe, period, risk_free_rate)
    if reco_table.empty:
        st.warning("無足夠資料產生建議，請確認時間範圍。")
    else:
        buy_df, sell_df = recommend.top_buy_sell(reco_table, top_n)
        buy_df = recommend.add_reason(recommend.add_price_targets(buy_df, "buy"), "buy")
        sell_df = recommend.add_reason(recommend.add_price_targets(sell_df, "sell"), "sell")

        def _format_reco(df: pd.DataFrame) -> pd.DataFrame:
            fmt = df.copy()
            for col in ["期間報酬率", "趨勢(價格/SMA50)"]:
                fmt[col] = fmt[col].apply(lambda v: f"{v * 100:.2f}%" if pd.notnull(v) else None)
            for col in ["Sharpe Ratio", "估值(1/預估PE)", "新聞情緒", "RSI (14)", "綜合評分"]:
                fmt[col] = fmt[col].apply(lambda v: f"{v:.2f}" if pd.notnull(v) else None)
            for col in ["建議買入價", "建議賣出價"]:
                if col in fmt:
                    fmt[col] = fmt[col].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else None)
            return fmt

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🟢 建議買入 Top {len(buy_df)}")
            st.dataframe(_format_reco(buy_df), use_container_width=True)
        with col2:
            st.markdown(f"#### 🔴 建議賣出 Top {len(sell_df)}")
            st.dataframe(_format_reco(sell_df), use_container_width=True)

        if len(buy_df) < top_n:
            st.info(
                f"目前範圍共 {len(reco_table)} 檔標的，為避免買入／賣出名單重複，"
                f"已各自裁切為 {len(buy_df)} 檔（最多取清單一半），而非選擇的 Top {top_n}。"
            )
