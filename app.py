"""US Stock Analyst Dashboard — Streamlit app."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from streamlit_local_storage import LocalStorage

from src import data_loader as dl
from src import fcn
from src import news
from src import recommend
from src import risk
from src import technical as ta
from src import universe

st.set_page_config(page_title="股市分析師看板", layout="wide", page_icon="📈")

# Dark-themed Plotly charts so they blend with the dark page (candlestick's
# explicit red/green colours are unaffected).
pio.templates.default = "plotly_dark"

# Cohesive dark-dashboard polish: tighter spacing, carded metrics/tabs, capped
# dropdown width, rounded tables. Pure presentation — no behaviour change.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px; }
      /* Dropdowns / number inputs: cap width so they don't stretch full-row */
      div[data-baseweb="select"], div[data-testid="stNumberInput"] { max-width: 360px; }
      div[data-baseweb="select"] > div { border-radius: 8px; border-color: #2a3342; }
      /* Tabs as carded pills with an accent on the active one */
      .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #232b38; }
      .stTabs [data-baseweb="tab"] {
        background: #141a23; border-radius: 10px 10px 0 0; padding: 8px 18px; font-weight: 600;
      }
      .stTabs [aria-selected="true"] { background: #1c2530; border-bottom: 3px solid #3da5ff; }
      /* Some mobile in-app browsers (LINE/FB webview) ignore the hidden
         attribute on inactive tab panels, stacking every tab's content on
         one page — force-hide them so only the active tab shows. */
      div[data-baseweb="tab-panel"][hidden] { display: none !important; }
      /* Metric cards */
      div[data-testid="stMetric"] {
        background: #161c26; border: 1px solid #232b38; border-radius: 14px; padding: 14px 18px;
      }
      div[data-testid="stMetricLabel"] p { opacity: .75; font-size: .85rem; }
      /* Rounded dataframes + radio row */
      div[data-testid="stDataFrame"] { border: 1px solid #232b38; border-radius: 12px; overflow: hidden; }
      div[role="radiogroup"] { gap: .4rem; }
      h3 { margin-top: .3rem; letter-spacing: .2px; }
      hr { margin: 1.1rem 0; border-color: #232b38; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# 買/賣切換刻意不持久化：每次開啟都回到「買」預設（_render_buy_sell_section
# 在這兩個 key 不存在時會把買=True/賣=False），同一 session 內仍可自由切換。
# reco_confirmed_/price_confirmed_ 是各分頁「確定」鈕的閘門旗標（含 _missing
# 提示狀態），也刻意不持久化——否則一旦按過確定，旗標會被存進 localStorage，
# 下次開啟就略過閘門、自動跑掉昂貴的掃描，違背「按確定才查詢」的設計。
# ticker_hist_pick_ 是「查詢紀錄」下拉（選取後把代號填入輸入框的觸發器），
# 持久化它只會讓下次開啟顯示一個過期的舊選取，故排除；紀錄本身
# （ticker_history_）仍照常持久化。
_PERSIST_EXCLUDE_PREFIXES = (
    "show_buy_", "show_sell_",
    "reco_confirmed_", "price_confirmed_", "compare_confirmed_", "fcn_confirmed_",
    "ticker_hist_pick_",
)
_local_storage = LocalStorage()
_saved_settings_raw = _local_storage.getItem(_SETTINGS_STORAGE_KEY)
try:
    _saved_settings = json.loads(_saved_settings_raw) if _saved_settings_raw else {}
except (TypeError, ValueError):
    _saved_settings = {}
for _k, _v in _saved_settings.items():
    if (_k not in st.session_state and _k not in _PERSIST_EXCLUDE_KEYS
            and not _k.startswith(_PERSIST_EXCLUDE_PREFIXES)):
        st.session_state[_k] = _v

PERIOD_OPTIONS = {
    "1個月": "1mo", "3個月": "3mo", "6個月": "6mo",
    "1年": "1y", "2年": "2y", "5年": "5y",
}
# The 買賣建議 tab offers extra short windows (1天～2週). yfinance's `period`
# can't express those (and "1d" returns a single daily bar — too few to derive
# a return), so each option pairs a safe fetch period with the number of recent
# trading days to slice for the scoring lookback. "lookback" None = use the full
# fetched window, matching PERIOD_OPTIONS. The fetched window is also what feeds
# the win-rate price-target distribution, so the short options keep a 1個月 fetch
# to retain enough forward-return samples for that.
# Each option also declares a horizon bucket (short/medium/long) so the
# composite-score factor weights auto-switch with the chosen window — short
# windows lean on momentum/news, long windows on valuation/risk-adjusted return
# (see recommend.FACTOR_WEIGHTS_BY_HORIZON).
RECO_PERIOD_OPTIONS = {
    "1天": {"fetch": "1mo", "lookback": 1, "horizon": "short"},
    "3天": {"fetch": "1mo", "lookback": 3, "horizon": "short"},
    "1週": {"fetch": "1mo", "lookback": 5, "horizon": "short"},
    "2週": {"fetch": "1mo", "lookback": 10, "horizon": "short"},
    "1個月": {"fetch": "1mo", "lookback": None, "horizon": "short"},
    "3個月": {"fetch": "3mo", "lookback": None, "horizon": "medium"},
    "6個月": {"fetch": "6mo", "lookback": None, "horizon": "medium"},
    "1年": {"fetch": "1y", "lookback": None, "horizon": "long"},
    "2年": {"fetch": "2y", "lookback": None, "horizon": "long"},
    "5年": {"fetch": "5y", "lookback": None, "horizon": "long"},
    "今年至今(YTD)": {"fetch": "ytd", "lookback": None, "horizon": "medium"},
}
_HORIZON_LABEL = {"short": "短期", "medium": "中期", "long": "長期"}

# Holding period (trading days) for the 買賣建議 tab's 預測準確機率 column and
# target prices. A preset dropdown plus a 自訂 (free-fill) option. The short
# end is consolidated into day-count buckets (1~5天/6~10天/11~15天) — each
# bucket runs the same formula at its upper-bound day count, since picking
# 1天 vs 3天 vs 5天 individually didn't change which formula applied anyway
# (朱家泓突破濾網 only ever distinguished ≤5 天 vs not). Past 15 天 the
# original calendar-period labels are kept as-is (1個月/3個月/...).
RECO_HOLD_OPTIONS = {
    "1~5天": 5, "6~10天": 10, "11~15天": 15,
    "1個月": 21, "3個月": 63, "6個月": 126, "1年": 252, "3年": 756, "5年": 1260,
}
_HOLD_CUSTOM_LABEL = "自訂天數…"

# 存股區 reuses RECO_PERIOD_OPTIONS/RECO_HOLD_OPTIONS but restricted to
# medium/long-horizon presets — a buy-and-hold view has no business offering
# 1天/1週 style short-term windows. No 自訂(custom) option either, since a
# custom day count could re-introduce a short hold by accident.
_HOLDING_PERIOD_OPTIONS = {k: v for k, v in RECO_PERIOD_OPTIONS.items() if v["horizon"] != "short"}
_HOLDING_HOLD_OPTIONS = {k: v for k, v in RECO_HOLD_OPTIONS.items() if v > 15}

# 目標積極度 → percentile of the favorable move used for the buy/sell target
# (中性=50 = median = the prior behaviour; 保守 closer/easier, 積極 farther).
RECO_AGGRESSIVENESS = {"保守": 30, "中性": 50, "積極": 70}
# Charts that render one trace/row per ticker (comparison overlay, correlation
# heatmap, distribution histogram) become unreadable and slow past this many
# tickers, so those views are capped — tables and the recommendation scan
# still use the full list.
MAX_CHART_TICKERS = 30
# Risk/statistics tab no longer exposes its own risk-free-rate control (that
# input now lives on the recommendation tab), so its Sharpe Ratio uses this
# fixed default instead.
DEFAULT_RISK_FREE_RATE = 0.04
# FCN tab: Monte Carlo path count (vectorized, so this stays fast even for a
# multi-year tenor or a several-asset worst-of basket).
FCN_N_SIMS = 8000
FCN_MAX_ASSETS = 5


def _confirm_gate(
    gate_key: str, signature: tuple, pressed: bool, *,
    missing: bool, info_msg: str, warning_msg: str = "",
) -> bool:
    """A 確定-button gate shared by every tab.

    Returns True only when the user has pressed 確定 on the *current* inputs
    and nothing required is blank. `signature` is a tuple of all gating input
    values; the confirmed snapshot is compared against it on every rerun, so
    changing any input re-closes the gate (the user must press 確定 again
    before the query/scan re-runs) — this is what stops a stale or
    half-edited form from auto-querying. `missing` flags whether a required
    input is still blank: pressing 確定 then shows `warning_msg` and the gate
    stays shut; otherwise `info_msg` is shown until 確定 is pressed.

    The caller renders the button itself (so it can sit in the right column)
    and passes its return value as `pressed`; the gate's info/warning render
    full-width here. Gate state keys all start with `gate_key`, which the
    callers prefix so _PERSIST_EXCLUDE_PREFIXES keeps them out of
    localStorage (a persisted "confirmed" flag would bypass the gate on the
    next visit).
    """
    snap_key = f"{gate_key}_snap"
    miss_key = f"{gate_key}_missing"
    if pressed:
        if missing:
            st.session_state[miss_key] = True
            st.session_state.pop(snap_key, None)
        else:
            st.session_state[miss_key] = False
            st.session_state[snap_key] = signature
    if not missing and st.session_state.get(snap_key) == signature:
        return True
    if missing and st.session_state.get(miss_key):
        st.warning(warning_msg)
    else:
        st.info(info_msg)
    return False


def _display_name(ticker: str) -> str:
    """Return "代碼(公司名稱)", falling back to the bare ticker if the name
    is unavailable (e.g. offline or an unrecognized symbol).

    TW tickers show the bare numeric code (no ".TW"/".TWO" suffix — that
    suffix is a Yahoo Finance implementation detail, not how 台股 codes are
    normally written/read) paired with the curated Chinese name (Yahoo's
    "shortName" for TWSE tickers comes back in English). US tickers keep
    the ticker as-is with the English shortName.
    """
    is_tw_ticker = ticker.endswith((".TW", ".TWO"))
    name = universe.get_tw_company_name(ticker) if is_tw_ticker else None
    if not name:
        info = dl.get_company_info(ticker)
        # Fall back to longName when shortName is missing so US tickers show
        # "代碼(公司名稱)" as consistently as TW does (TW uses a curated dict).
        name = info.get("shortName") or info.get("longName")
    code = ticker.split(".")[0] if is_tw_ticker else ticker
    return f"{code}({name})" if name else code


def _signal_rgb(side: str, frac: float) -> str:
    """`rgb(...)` colour for the merged table's buy/sell signal cell: green for
    buy, red for sell, shaded by `frac` in [0,1] (0 = lightest, 1 = deepest).
    `frac` is the table's min–max-normalized 預測準確機率, so the gradient always
    spans the full visible range."""
    f = min(max(frac, 0.0), 1.0)
    if side == "buy":  # light green -> deep green
        r, g, b = int(150 - 150 * f), int(210 - 110 * f), int(150 - 150 * f)
    else:              # light red -> deep red
        r, g, b = int(235 - 85 * f), int(120 - 120 * f), int(120 - 120 * f)
    return f"rgb({r},{g},{b})"


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


# No default selection: the picker no longer auto-resolves to a market on
# load (this used to default to 美股) — the user must actively pick
# 台股／美股, same spirit as excluding "market" from the localStorage
# round-trip below (avoid ever silently landing on a market the user didn't
# choose this visit). Each tab below gates its own query behind its own
# 確定 button instead of one global gate, so picking a market alone no
# longer blocks the whole page.
market = st.radio(
    "市場", ["美股", "台股"], index=None, horizontal=True, key="market",
    label_visibility="collapsed",
)
if market is None:
    st.info("請先選擇市場（美股／台股）。")
    st.stop()
is_tw = market == "台股"
currency = "NT$" if is_tw else "$"

# 多股比較與風險統計 sits to the right of FCN風險評估 (it feeds FCN inputs:
# 波動率/相關性), so the two related tabs are adjacent at the end.
tab_news, tab_overview, tab_reco, tab_stock_hold, tab_fcn, tab_compare_risk = st.tabs(
    ["🌍 市場焦點", "📈 個股資訊", "💡 買賣建議", "🏦 存股區", "📐 FCN風險評估", "🔗 多股比較與風險統計"]
)


def _is_english_title(title: str) -> bool:
    """No CJK characters → treat as English and machine-translate for display."""
    return not any("一" <= ch <= "鿿" for ch in title)


def _render_news_line(item: dict, translate: bool = False) -> None:
    """One headline as 'MM/DD [中文標題](link)（來源）', with the English
    original in a caption underneath when it was machine-translated."""
    title = news._strip_source_suffix(item["title"])
    src = item.get("source") or ""
    date_str = f"{item['published']:%m/%d}"
    if translate and _is_english_title(title):
        zh = news.translate_to_zh_tw(title)
        st.markdown(f"**{date_str}**　[{zh}]({item['link']})　`{src}`")
        if zh != title:
            st.caption(f"　原文：{title}")
    else:
        st.markdown(f"**{date_str}**　[{title}]({item['link']})　`{src}`")


def _us_flow_summary(vix, macro_df, sector_df) -> str | None:
    """Rule-based one-paragraph takeaway under the 美股資金流向 charts.
    Each clause is skipped when its data is missing; returns None if all are."""
    parts = []
    if vix is not None:
        mood = "高波動（市場恐慌）" if vix > 25 else "平靜" if vix < 15 else "中性"
        parts.append(f"VIX {vix:.1f}，市場情緒{mood}")
    if macro_df is not None and not macro_df.empty:
        r = {row["ticker"]: row["return_1d"] for _, row in macro_df.iterrows()}
        stocks = [v for v in (r.get("SPY"), r.get("QQQ")) if pd.notna(v)]
        havens = [v for v in (r.get("TLT"), r.get("GLD")) if pd.notna(v)]
        if stocks and havens:
            s_up, h_up = max(stocks) > 0, max(havens) > 0
            if s_up and not h_up:
                parts.append("資金偏向風險資產（股漲、債金弱），risk-on")
            elif h_up and not s_up:
                parts.append("資金流向避險資產（債/金漲、股弱），risk-off")
            elif s_up and h_up:
                parts.append("股與避險資產齊漲，資金充沛但方向分歧")
            else:
                parts.append("股與避險資產齊跌，觀望氣氛濃")
    if sector_df is not None and not sector_df.empty:
        sd = sector_df.dropna(subset=["return_1d"])
        if not sd.empty:
            n_up = int((sd["return_1d"] > 0).sum())
            best = sd.loc[sd["return_1d"].idxmax()]
            worst = sd.loc[sd["return_1d"].idxmin()]
            parts.append(
                f"11 大類股中 {n_up} 漲 {len(sd) - n_up} 跌，"
                f"最強 {best['sector']}（{best['return_1d']:+.2f}%）、"
                f"最弱 {worst['sector']}（{worst['return_1d']:+.2f}%）"
            )
    return "；".join(parts) + "。" if parts else None


def _tw_flow_summary(inst_data, ind_df) -> str | None:
    """Rule-based one-paragraph takeaway under the 台股資金流向 charts."""
    parts = []
    if inst_data:
        total = sum(d["net_bn"] for d in inst_data)
        direction = "買超" if total > 0 else "賣超"
        detail = "、".join(
            f"{d['name']}{'買超' if d['net_bn'] > 0 else '賣超'} {abs(d['net_bn']):,.1f} 億"
            for d in inst_data
        )
        parts.append(f"三大法人合計{direction} {abs(total):,.1f} 億元（{detail}）")
    if ind_df is not None and not ind_df.empty:
        n_up = int((ind_df["net_score"] > 0).sum())
        n_down = int((ind_df["net_score"] < 0).sum())
        best = ind_df.loc[ind_df["net_score"].idxmax()]
        worst = ind_df.loc[ind_df["net_score"].idxmin()]
        breadth = "普漲格局" if n_up >= len(ind_df) * 0.7 else "普跌格局" if n_down >= len(ind_df) * 0.7 else "漲跌互見"
        parts.append(
            f"{len(ind_df)} 個產業中 {n_up} 個偏漲、{n_down} 個偏跌，{breadth}，"
            f"最強 {best['industry']}（漲-跌 {best['net_score']:+d} 家）、"
            f"最弱 {worst['industry']}（{worst['net_score']:+d} 家）"
        )
    return "；".join(parts) + "。" if parts else None


# ---------- Tab 0: 市場焦點 (Dashboard + news) ----------
with tab_news:
    # Same pattern as the 價格技術指標 tab: charts default to no drag/pan so
    # mobile scrolling isn't trapped by them; tick to re-enable zoom/pan.
    _news_analysis_mode = st.checkbox(
        "📊 啟用圖表分析模式（可縮放、拖曳查看細節；行動裝置上頁面滑動會變得較不順手）",
        key="chart_analysis_mode_news",
    )

    # ── 美股資金流向 ──────────────────────────────────────────────
    st.subheader("🇺🇸 美股資金流向")
    with st.spinner("載入美股行情…"):
        _vix = dl.get_vix()
        _macro_df = dl.get_macro_flow_returns()
        _sector_df = dl.get_us_sector_returns()

    # Metric row: VIX + 5 macro assets (1d change)
    _cols_m = st.columns(6)
    with _cols_m[0]:
        _vix_str = f"{_vix:.1f}" if _vix is not None else "N/A"
        _vix_level = None if _vix is None else "⚠️ 高波動" if _vix > 25 else "😌 平靜" if _vix < 15 else "😐 中性"
        st.metric("😨 VIX 恐慌指數", _vix_str, delta=_vix_level, delta_color="off")
    for _mi, _mrow in _macro_df.iterrows():
        with _cols_m[int(_mi) + 1]:
            _d1 = _mrow["return_1d"]
            _d1_str = f"{_d1:+.2f}%" if not pd.isna(_d1) else None
            _lc = _mrow["last_close"]
            _lc_str = f"{_lc:,.2f}" if not pd.isna(_lc) else "N/A"
            st.metric(_mrow["label"], _lc_str, delta=_d1_str)

    # Two columns: sector bar | macro grouped bar
    _col_s, _col_mac = st.columns([3, 2])
    with _col_s:
        if not _sector_df.empty and _sector_df["return_1d"].notna().any():
            _sd = _sector_df.dropna(subset=["return_1d"]).sort_values("return_1d")
            _fig_s = go.Figure(go.Bar(
                x=_sd["return_1d"],
                y=_sd["sector"],
                orientation="h",
                marker=dict(
                    color=["#e66767" if r < 0 else "#3987e5" for r in _sd["return_1d"]],
                    line=dict(width=0),
                ),
                text=[f"{r:+.2f}%" for r in _sd["return_1d"]],
                textposition="outside",
                hovertemplate="%{y}（%{customdata}）: %{x:+.2f}%<extra></extra>",
                customdata=_sd["ticker"],
            ))
            _fig_s.update_layout(
                title=dict(text="11 大類股 ETF 當日漲跌", font=dict(size=13)),
                xaxis=dict(ticksuffix="%", zeroline=True, zerolinecolor="#383835",
                           zerolinewidth=2, showgrid=False),
                yaxis=dict(tickfont=dict(size=12), showgrid=False),
                height=370, margin=dict(l=0, r=60, t=38, b=8),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            _render_chart(_fig_s, _news_analysis_mode)
        else:
            st.info("無法取得類股 ETF 資料。")

    with _col_mac:
        if not _macro_df.empty and _macro_df["return_1d"].notna().any():
            _md = _macro_df.dropna(subset=["return_1d"])
            _fig_mac = go.Figure()
            _fig_mac.add_trace(go.Bar(
                name="1 日",
                x=_md["label"], y=_md["return_1d"],
                marker=dict(color="#3987e5", line=dict(width=0)),
                text=[f"{r:+.2f}%" for r in _md["return_1d"]],
                textposition="outside",
                hovertemplate="%{x} 1日: %{y:+.2f}%<extra></extra>",
            ))
            _fig_mac.add_trace(go.Bar(
                name="5 日",
                x=_md["label"], y=_md["return_5d"],
                marker=dict(color="#199e70", line=dict(width=0)),
                text=[f"{r:+.2f}%" if not pd.isna(r) else "" for r in _md["return_5d"]],
                textposition="outside",
                hovertemplate="%{x} 5日: %{y:+.2f}%<extra></extra>",
            ))
            _fig_mac.update_layout(
                title=dict(text="大類資產資金輪動（漲跌%）", font=dict(size=13)),
                barmode="group",
                yaxis=dict(ticksuffix="%", zeroline=True, zerolinecolor="#383835",
                           zerolinewidth=2, showgrid=False),
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                height=370, margin=dict(l=0, r=60, t=38, b=8),
                legend=dict(orientation="h", y=1.08, x=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            _render_chart(_fig_mac, _news_analysis_mode)
        else:
            st.info("無法取得大類資產資料。")

    _us_summary = _us_flow_summary(_vix, _macro_df, _sector_df)
    if _us_summary:
        st.info(f"📝 總結：{_us_summary}")

    st.divider()

    # ── 台股資金流向 ──────────────────────────────────────────────
    st.subheader("🇹🇼 台股資金流向")
    with st.spinner("載入台股法人與產業資料…"):
        _inst_data = dl.get_twse_institutional_summary()
        _ind_df = dl.get_tw_industry_flow()

    _col_inst, _col_ind = st.columns([1, 2])
    with _col_inst:
        if _inst_data:
            _inst_df = pd.DataFrame(_inst_data)
            _fig_inst = go.Figure(go.Bar(
                x=_inst_df["name"], y=_inst_df["net_bn"],
                marker=dict(
                    color=["#e66767" if v < 0 else "#3987e5" for v in _inst_df["net_bn"]],
                    line=dict(width=0),
                ),
                text=[f"{v:+.1f}億" for v in _inst_df["net_bn"]],
                textposition="outside",
                hovertemplate="%{x}: %{y:+.1f} 億<extra></extra>",
            ))
            _fig_inst.update_layout(
                title=dict(text="三大法人買賣超（今日，億元）", font=dict(size=13)),
                yaxis=dict(ticksuffix="億", zeroline=True, zerolinecolor="#383835",
                           zerolinewidth=2, showgrid=False),
                xaxis=dict(showgrid=False),
                height=320, margin=dict(l=0, r=10, t=38, b=8),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            _render_chart(_fig_inst, _news_analysis_mode)
        else:
            st.info("今日三大法人資料尚未更新（盤中或假日）。")

    with _col_ind:
        if not _ind_df.empty:
            _ind_s = _ind_df.sort_values("net_score")
            _fig_ind = go.Figure(go.Bar(
                x=_ind_s["net_score"],
                y=_ind_s["industry"],
                orientation="h",
                marker=dict(
                    color=["#e66767" if v < 0 else "#3987e5" for v in _ind_s["net_score"]],
                    line=dict(width=0),
                ),
                text=[f"{v:+d}" for v in _ind_s["net_score"]],
                textposition="outside",
                hovertemplate="%{y}<br>漲 %{customdata[0]} 跌 %{customdata[1]}<br>淨值=%{x}<extra></extra>",
                customdata=_ind_s[["count_up", "count_down"]].values,
            ))
            _fig_ind.update_layout(
                title=dict(text="各產業漲股數 − 跌股數", font=dict(size=13)),
                xaxis=dict(zeroline=True, zerolinecolor="#383835",
                           zerolinewidth=2, showgrid=False),
                yaxis=dict(tickfont=dict(size=11), showgrid=False),
                height=450, margin=dict(l=0, r=45, t=38, b=8),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            _render_chart(_fig_ind, _news_analysis_mode)
        else:
            st.info("今日產業資料尚未更新（盤中或假日）。")

    _tw_summary = _tw_flow_summary(_inst_data, _ind_df)
    if _tw_summary:
        st.info(f"📝 總結：{_tw_summary}")

    st.divider()

    # ── 新聞區塊 ──────────────────────────────────────────────────
    st.caption(
        "資料來源：Google News RSS（英文標題以機器翻譯轉為中文，僅供快速瀏覽，"
        "重要決策請點開原文確認）。快取 30 分鐘～6 小時自動更新。"
    )

    st.subheader("📅 美國本週重要紀事")
    st.caption("本週聯準會動態、CPI／就業數據、財報週等市場行事曆相關頭條。")
    with st.spinner("正在載入本週紀事…"):
        _week_items = news.get_us_week_ahead()
    if _week_items:
        for _n in _week_items:
            _render_news_line(_n, translate=True)
    else:
        st.info("目前無法取得本週紀事（來源無資料或網路異常）。")

    st.divider()
    st.subheader("📰 影響今日美股的新聞")
    st.caption("近兩天（涵蓋美台時差）影響大盤的頭條，已翻譯為中文並附原文。")
    with st.spinner("正在載入並翻譯今日美股新聞…"):
        _today_items = news.get_us_market_today()
    if _today_items:
        for _n in _today_items:
            _render_news_line(_n, translate=True)
    else:
        st.info("目前無法取得今日美股新聞（來源無資料或網路異常）。")

    st.divider()
    st.subheader(f"🌐 {dt.date.today().year} 年重要國際事件")
    st.caption(
        "今年以來的重大地緣政治／總經事件大事記（例如美伊衝突、關稅戰、聯準會利率決策），"
        "依主題分組、每個日期只留一則重點；新聞搜尋偏重近期，較早的事件為盡力涵蓋。"
    )
    _grouped_events = news.get_global_events_this_year()
    if _grouped_events:
        for _topic, _items in _grouped_events:
            with st.expander(f"**{_topic}**", expanded=True):
                for _n in _items:
                    _render_news_line(_n, translate=True)
    else:
        st.info("目前無法取得國際事件新聞（來源無資料或網路異常）。")

# ---------- Tab 1: Price, technical indicators & fundamentals (one ticker) ----------
with tab_overview:
    col_ticker, col_period, col_confirm1 = st.columns([2, 1, 1])
    with col_ticker:
        if is_tw:
            default_ticker = "2330"
            ticker_label = "股票代號（台股代碼，例如 2330、0050）"
        else:
            default_ticker = "AAPL"
            ticker_label = "股票代號"
        # Switching 市場 only builds the active market's ticker field, so the
        # other market's widget state is garbage-collected on switch. Passing
        # value=default while the localStorage restore also writes this key
        # triggers a default-vs-session-state conflict that can momentarily
        # desync the box from the header below. Instead seed the widget once
        # from a plain (never-GC'd) shadow key — no value= arg, no conflict —
        # so the field (and the header derived from it) stay consistent across
        # 台股⇄美股 切換時 still remember each market's last input.
        _pt_key = f"price_ticker_{'tw' if is_tw else 'us'}"
        _pt_shadow = f"price_ticker_value_{'tw' if is_tw else 'us'}"
        if _pt_key not in st.session_state:
            st.session_state[_pt_key] = st.session_state.get(_pt_shadow) or default_ticker
        # Plain text input (commits on blur, so 打完字直接按確定 works) plus a
        # separate 查詢紀錄 dropdown beside it: successfully queried codes are
        # remembered (localStorage round-trip), and picking one fills the text
        # field via on_change. An accept_new_options selectbox was tried here
        # first, but typed text only commits on Enter/click — clicking 確定
        # right after typing silently kept the previous code.
        _hist_key = f"ticker_history_{'tw' if is_tw else 'us'}"
        _pick_key = f"ticker_hist_pick_{'tw' if is_tw else 'us'}"

        def _fill_from_history():
            _sel = st.session_state.get(_pick_key)
            if _sel:
                st.session_state[_pt_key] = _sel

        _col_in, _col_hist = st.columns([3, 2])
        with _col_in:
            raw_primary = st.text_input(ticker_label, key=_pt_key).strip().upper() or default_ticker
        with _col_hist:
            st.selectbox(
                "查詢紀錄", st.session_state.get(_hist_key) or [],
                index=None, key=_pick_key, on_change=_fill_from_history,
                placeholder="選取查過的代號",
                help="查詢成功的代號會記錄在這裡；選取後自動填入左側輸入框，再按「確定」查詢。",
            )
        st.session_state[_pt_shadow] = raw_primary
    with col_period:
        period_label = st.selectbox(
            "時間範圍", list(PERIOD_OPTIONS.keys()), index=3, key=f"period_tab1_{'tw' if is_tw else 'us'}"
        )
    with col_confirm1:
        st.write("")
        # 股票代號與時間範圍都先暫存在 widget 裡，按下確定才把這一刻的值定格存進
        # session_state 並往下查詢/畫圖 — 避免每次打字/選單變動就立刻重新抓資料。
        _p1_confirm_key = f"price_confirmed_{'tw' if is_tw else 'us'}"
        if st.button("確定", key=f"price_confirm_btn_{'tw' if is_tw else 'us'}", use_container_width=True):
            st.session_state[_p1_confirm_key] = {
                "ticker": universe.resolve_tw_ticker(raw_primary) if is_tw else raw_primary,
                "period": PERIOD_OPTIONS[period_label],
            }

    if _p1_confirm_key not in st.session_state:
        st.info("請輸入股票代號與時間範圍，再按「確定」開始查詢。")
    else:
        primary = st.session_state[_p1_confirm_key]["ticker"]
        period = st.session_state[_p1_confirm_key]["period"]
        primary_label = _display_name(primary)
        # Plain markdown heading rather than st.subheader: the heading component
        # auto-generates an anchor id from its text and can fail to update the
        # rendered text on rerun (leaving a stale ticker like "TSLA" in the header
        # while the box/chart already show "AAPL"). Markdown has no such anchor.
        st.markdown(f"### {primary_label} 價格與技術指標")
        df = dl.get_price_history(primary, period=period)
        if df.empty:
            st.error(f"找不到 {primary} 的資料，請確認代號是否正確。")
        else:
            # Remember the validated code (data came back for it) so the
            # ticker selectbox can suggest it on later visits. TW codes are
            # stored bare (no .TW/.TWO — matching how they're typed); capped
            # so localStorage doesn't grow without bound.
            _hist_code = primary.split(".")[0] if is_tw else primary
            _hist = st.session_state.get(_hist_key) or []
            if _hist_code not in _hist:
                st.session_state[_hist_key] = sorted(set(_hist + [_hist_code]))[-50:]

            close = df["Close"]
            sma5, sma10, sma20 = ta.sma(close, 5), ta.sma(close, 10), ta.sma(close, 20)
            bb = ta.bollinger_bands(close)

            st.caption("提示：將滑鼠移到圖上（手機點一下 K 棒）即可看到當天開盤／最高／最低／收盤價，不需開啟下方分析模式。")
            # Overlay toggles: the full stack (SMA×3 + 布林×2 + 趨勢×3 + 支撐/壓力
            # ×4) buried the candles under 12+ legend items, so each line
            # family gets its own checkbox — off means not drawn at all (no
            # legend slot either). Defaults keep the most-read overlays only.
            _mk = "tw" if is_tw else "us"
            _cb_mode, _cb_sma, _cb_bb, _cb_trend, _cb_sr, _cb_pat = st.columns([2, 1, 1, 1, 1, 1])
            with _cb_mode:
                analysis_mode = st.checkbox(
                    "📊 啟用圖表分析模式（可縮放、拖曳查看細節；行動裝置上頁面滑動會變得較不順手）",
                    key=f"chart_analysis_mode_{_mk}",
                )
            # Seed the default-on toggles via session_state (not value=) so
            # the localStorage restore writing the same keys doesn't trigger
            # a default-vs-state conflict warning on every rerun.
            for _ov_key in (f"overlay_sma_{_mk}", f"overlay_sr_{_mk}", f"overlay_pattern_{_mk}"):
                if _ov_key not in st.session_state:
                    st.session_state[_ov_key] = True
            with _cb_sma:
                show_sma = st.checkbox("均線", key=f"overlay_sma_{_mk}")
            with _cb_bb:
                show_bb = st.checkbox("布林通道", key=f"overlay_bb_{_mk}")
            with _cb_trend:
                show_trend = st.checkbox("趨勢通道", key=f"overlay_trend_{_mk}")
            with _cb_sr:
                show_sr = st.checkbox("支撐/壓力", key=f"overlay_sr_{_mk}")
            with _cb_pat:
                show_pattern = st.checkbox("形態標註", key=f"overlay_pattern_{_mk}")

            # 形態辨識（K 棒 + 價格結構）— computed once here, drawn on the
            # chart when 形態標註 is ticked and always listed in the 形態辨識
            # section further down.
            candle_pats = ta.candlestick_patterns(df["Open"], df["High"], df["Low"], df["Close"])
            struct_pats = ta.chart_patterns(df["High"], df["Low"], close)

            # "三竹股市" look for TW stocks: 漲=red／跌=green candles (the reverse of
            # the US green-up/red-down convention). Chart background follows the dark
            # Plotly template (pio.templates.default) set at startup.
            up_color, down_color = ("#ff3333", "#00b300") if is_tw else ("#26c281", "#ff5b5b")
            legend_top = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name=primary_label,
                increasing_line_color=up_color, increasing_fillcolor=up_color,
                decreasing_line_color=down_color, decreasing_fillcolor=down_color,
            ))
            if show_sma:
                fig.add_trace(go.Scatter(x=df.index, y=sma5, name="SMA5", line=dict(width=1, color="#1f77b4")))
                fig.add_trace(go.Scatter(x=df.index, y=sma10, name="SMA10", line=dict(width=1, color="#ff7f0e")))
                fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20", line=dict(width=1, color="#9467bd")))
            if show_bb:
                # One legend entry for the pair (legendgroup): clicking it
                # toggles both bands at once.
                fig.add_trace(go.Scatter(x=df.index, y=bb["upper"], name="布林通道",
                                          legendgroup="bb",
                                          line=dict(width=1, dash="dot"), opacity=0.5))
                fig.add_trace(go.Scatter(x=df.index, y=bb["lower"], name="布林通道",
                                          legendgroup="bb", showlegend=False,
                                          line=dict(width=1, dash="dot"), opacity=0.5))
            if show_trend:
                # Linear regression trend channel — one legend entry for all 3 lines.
                lr = ta.linear_regression_channel(close)
                fig.add_trace(go.Scatter(x=df.index, y=lr["mid"], name="趨勢通道",
                                          legendgroup="trend",
                                          line=dict(color="#e67e22", width=1.5, dash="dash"), opacity=0.9))
                fig.add_trace(go.Scatter(x=df.index, y=lr["upper"], name="趨勢通道",
                                          legendgroup="trend", showlegend=False,
                                          line=dict(color="#e67e22", width=1, dash="dot"), opacity=0.75))
                fig.add_trace(go.Scatter(x=df.index, y=lr["lower"], name="趨勢通道",
                                          legendgroup="trend", showlegend=False,
                                          line=dict(color="#e67e22", width=1, dash="dot"), opacity=0.75))
            if show_sr:
                # Support / Resistance pivot levels, trimmed to the 2 nearest
                # to the current price each. One legend entry per side; the
                # price of each line sits as a small label at its right end
                # (the legend no longer enumerates values).
                _sup_levels, _res_levels = ta.support_resistance_levels(df["High"], df["Low"], close)
                _x0, _x1 = df.index[0], df.index[-1]
                for _si, lv in enumerate(_sup_levels[:2]):
                    fig.add_trace(go.Scatter(
                        x=[_x0, _x1], y=[lv, lv], mode="lines",
                        name="支撐", legendgroup="support", showlegend=_si == 0,
                        line=dict(color="limegreen", width=1, dash="dash"), opacity=0.75,
                    ))
                    fig.add_annotation(x=_x1, y=lv, text=f"{lv:.1f}", showarrow=False,
                                       xanchor="left", font=dict(size=10, color="limegreen"))
                for _ri, lv in enumerate(_res_levels[:2]):
                    fig.add_trace(go.Scatter(
                        x=[_x0, _x1], y=[lv, lv], mode="lines",
                        name="壓力", legendgroup="resistance", showlegend=_ri == 0,
                        line=dict(color="tomato", width=1, dash="dash"), opacity=0.75,
                    ))
                    fig.add_annotation(x=_x1, y=lv, text=f"{lv:.1f}", showarrow=False,
                                       xanchor="left", font=dict(size=10, color="tomato"))
            if show_pattern:
                # 價格結構形態：pivot 點連線＋名稱標籤＋頸線虛線
                for _spi, _sp in enumerate(struct_pats):
                    _px = [df.index[i] for i, _ in _sp["points"]]
                    _py = [p for _, p in _sp["points"]]
                    _sp_color = up_color if _sp["side"] == "bull" else down_color if _sp["side"] == "bear" else "#d4b106"
                    fig.add_trace(go.Scatter(
                        x=_px, y=_py, mode="lines+markers",
                        name="形態", legendgroup="pattern", showlegend=_spi == 0,
                        line=dict(color=_sp_color, width=2), marker=dict(size=7, color=_sp_color),
                        opacity=0.9, hovertext=_sp["name"], hoverinfo="text",
                    ))
                    fig.add_annotation(x=_px[-1], y=_py[-1], text=_sp["name"], showarrow=False,
                                       yshift=14 if _sp["side"] != "bull" else -14,
                                       font=dict(size=11, color=_sp_color))
                    if _sp.get("neckline"):
                        fig.add_trace(go.Scatter(
                            x=[_px[0], df.index[-1]], y=[_sp["neckline"]] * 2, mode="lines",
                            legendgroup="pattern", showlegend=False,
                            line=dict(color=_sp_color, width=1, dash="dot"), opacity=0.6,
                            hovertext=f"{_sp['name']} 頸線 {_sp['neckline']:.1f}", hoverinfo="text",
                        ))
                # K 棒形態：對應 K 棒上/下方小標籤（偏多在下方、偏空在上方）
                for _cp in candle_pats:
                    _ci = _cp["bar"]
                    _c_color = up_color if _cp["side"] == "bull" else down_color if _cp["side"] == "bear" else "#d4b106"
                    _below = _cp["side"] == "bull"
                    fig.add_annotation(
                        x=df.index[_ci],
                        y=float(df["Low"].iloc[_ci]) if _below else float(df["High"].iloc[_ci]),
                        text=("▲" if _below else "▼" if _cp["side"] == "bear" else "◆") + _cp["name"],
                        showarrow=False, yshift=-12 if _below else 12,
                        font=dict(size=10, color=_c_color),
                    )

            fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                               margin=dict(t=80, b=20, r=80), legend=legend_top)
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

            adx_df = ta.adx(df["High"], df["Low"], close)
            adx_fig = go.Figure()
            adx_fig.add_trace(go.Scatter(x=df.index, y=adx_df["adx"], name="ADX",
                                         line=dict(color="purple", width=2)))
            adx_fig.add_trace(go.Scatter(x=df.index, y=adx_df["plus_di"], name="+DI",
                                         line=dict(color="green")))
            adx_fig.add_trace(go.Scatter(x=df.index, y=adx_df["minus_di"], name="−DI",
                                         line=dict(color="red")))
            adx_fig.add_hline(y=25, line_dash="dash", line_color="purple",
                               annotation_text="25 趨勢明確", annotation_position="right")
            adx_fig.add_hline(y=20, line_dash="dot", line_color="gray",
                               annotation_text="20 盤整界線", annotation_position="right")
            adx_fig.update_layout(height=220, title="ADX (14) 趨勢強度", margin=dict(t=30, b=10))
            _render_chart(adx_fig, analysis_mode)

            # yfinance occasionally returns a still-open current trading day as
            # an all-NaN OHLC row (especially for non-US-market tickers); fall
            # back to the last actually-priced close so a transient incomplete
            # bar doesn't blank out 最新收盤價/建議買入賣出價 (same guard as
            # recommend.py's _yield_price).
            _valid_close = close.dropna()
            latest = _valid_close.iloc[-1] if not _valid_close.empty else float("nan")
            prev = _valid_close.iloc[-2] if len(_valid_close) > 1 else latest
            st.metric(f"{primary_label} 最新收盤價", f"{currency}{latest:,.2f}",
                       f"{(latest / prev - 1) * 100:.2f}%")

            # ETF: list constituent holdings (capped — a holdings table with
            # dozens/hundreds of rows, e.g. a bond ETF, would blow out the page
            # layout, so it's only shown as a short, glanceable list).
            _info_primary = dl.get_company_info(primary)
            if _info_primary.get("quoteType") == "ETF":
                holdings = dl.get_etf_top_holdings(primary)
                if not holdings.empty and len(holdings) <= 15:
                    st.markdown("##### 成份股")
                    hdf = holdings.reset_index()
                    hdf.columns = ["代號", "名稱", "權重"]
                    hdf["權重"] = (hdf["權重"] * 100).map(lambda v: f"{v:.2f}%")
                    st.dataframe(hdf, use_container_width=True, hide_index=True)
                else:
                    # yfinance has no holdings for many TW ETFs (e.g. active
                    # ETFs like 00997A) — fall back to MoneyDJ's 持股明細.
                    tw_holdings = dl.get_tw_etf_holdings(primary) if is_tw else pd.DataFrame()
                    if not tw_holdings.empty:
                        st.markdown("##### 成份股")
                        hdf = tw_holdings.head(15).copy()
                        hdf["權重"] = hdf["權重"].map(lambda v: f"{v:.2f}%")
                        st.dataframe(hdf, use_container_width=True, hide_index=True)
                        _cap = f"，僅列權重前 15 檔（共 {len(tw_holdings)} 檔）" if len(tw_holdings) > 15 else ""
                        st.caption(f"成份股資料來源：MoneyDJ{_cap}。")
                    else:
                        st.caption("暫無成份股資料。")

            st.markdown("##### 建議買入／賣出價格參考")
            col_h1, col_a1 = st.columns(2)
            with col_h1:
                # Same 持有天數 options as the 買賣建議 tab (RECO_HOLD_OPTIONS + 自訂).
                _pt_hold_key = f"price_target_horizon_{'tw' if is_tw else 'us'}"
                _pt_hold_opts = list(RECO_HOLD_OPTIONS.keys()) + [_HOLD_CUSTOM_LABEL]
                # Drop a stale persisted value from the old option set so the
                # selectbox doesn't raise "default value not in options".
                if st.session_state.get(_pt_hold_key) not in _pt_hold_opts:
                    st.session_state.pop(_pt_hold_key, None)
                hold_label1 = st.selectbox(
                    "持有天數(今日算起)", _pt_hold_opts, index=0, key=_pt_hold_key,
                )
                if hold_label1 == _HOLD_CUSTOM_LABEL:
                    hold_days = int(st.number_input(
                        "自訂持有天數（交易日）", min_value=1, max_value=1260, value=5, step=1,
                        key=f"price_target_hold_custom_{'tw' if is_tw else 'us'}",
                    ))
                    hold_disp1 = f"{hold_days} 個交易日"
                else:
                    hold_days = RECO_HOLD_OPTIONS[hold_label1]
                    hold_disp1 = hold_label1 if hold_label1 == f"{hold_days}天" else f"{hold_label1}（{hold_days} 交易日）"
            with col_a1:
                aggr_label = st.selectbox(
                    "目標積極度", list(RECO_AGGRESSIVENESS.keys()), index=1,
                    key=f"price_target_aggr_{'tw' if is_tw else 'us'}",
                    help="保守＝目標較近、較易達成（預測準確機率較高）；積極＝目標較遠。中性＝歷史中位幅度。",
                )
            aggr_pct = RECO_AGGRESSIVENESS[aggr_label]
            query_date = dt.date.today()
            target_date = query_date + dt.timedelta(days=round(hold_days * 7 / 5))  # trading→calendar
            st.caption(
                "統計期間皆是 1 年。計算邏輯與「買賣建議」分頁一致："
                "取價＝現價×(1＋歷史漲跌幅〔依目標積極度取百分位〕，並依技術訊號〔布林/SMA多頭/趨勢/形態辨識+動能〕微調)；"
                "預測準確機率＝路徑式回測（持有期內最高/最低觸及的實測比例）。"
                f"依過去 1 年、持有 {hold_disp1} 估算，"
                f"查詢日 {query_date.year}/{query_date.month}/{query_date.day} ~ "
                f"預測日 {target_date.year}/{target_date.month}/{target_date.day}，僅供參考，非投資建議。"
            )
            # Same logic as the 買賣建議 tab (median move pricing, technical nudge,
            # path-based touch rate), but on a fixed 1-year window, so 統計期間皆是 1 年.
            pt = dl.get_price_history(primary, period="1y")
            if pt.empty:
                pt = df  # chart data (non-empty here) as a safety net
            pt_close, pt_high, pt_low = pt["Close"], pt["High"], pt["Low"]
            fwd_returns = pt_close.pct_change(periods=hold_days).dropna()
            ups, downs = fwd_returns[fwd_returns > 0], fwd_returns[fwd_returns < 0]
            up_move = float(np.percentile(ups, aggr_pct)) if not ups.empty else None
            down_move = float(np.percentile(downs, 100 - aggr_pct)) if not downs.empty else None
            # Technical nudge (布林/SMA多頭/趨勢/形態辨識 + 動能), same as the 買賣建議 formula.
            _bias = recommend.technical_bias(pt_close, pt_high, pt_low,
                                             recommend.horizon_for_hold_days(hold_days),
                                             open_=pt["Open"] if "Open" in pt else None)
            if up_move is not None:
                up_move *= (1 + recommend.TECH_BIAS_BETA * _bias)
            if down_move is not None:
                down_move *= (1 - recommend.TECH_BIAS_BETA * _bias)

            col_buy, col_sell = st.columns(2)
            with col_buy:
                if down_move is not None:
                    acc = recommend.forward_touch_rate(pt_close, pt_low, hold_days, down_move, "down")
                    st.metric("建議買入價（逢低承接）", f"{currency}{latest * (1 + down_move):,.2f}",
                               f"{down_move * 100:.2f}%")
                    st.caption(f"預測準確機率：歷史 1 年中，持有 {hold_days} 個交易日內最低觸及此買入價的比例約 "
                               f"{acc:.0f}%。" if acc is not None else "預測準確機率：資料不足。")
                else:
                    st.metric("建議買入價（逢低承接）", "資料不足")
            with col_sell:
                if up_move is not None:
                    acc = recommend.forward_touch_rate(pt_close, pt_high, hold_days, up_move, "up")
                    st.metric("建議賣出價（目標停利）", f"{currency}{latest * (1 + up_move):,.2f}",
                               f"{up_move * 100:.2f}%")
                    st.caption(f"預測準確機率：歷史 1 年中，持有 {hold_days} 個交易日內最高觸及此賣出價的比例約 "
                               f"{acc:.0f}%。" if acc is not None else "預測準確機率：資料不足。")
                else:
                    st.metric("建議賣出價（目標停利）", "資料不足")

        st.divider()
        st.markdown(f"### {primary_label} 基本財務面與技術分析簡述")
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

        if not df.empty:
            st.markdown("##### 技術分析")
            tech_df, tech_conclusion = recommend.technical_analysis_brief(
                close, df["High"], df["Low"], recommend.horizon_for_hold_days(hold_days),
                open_=df["Open"] if "Open" in df else None)
            if tech_df.empty:
                st.info("資料不足，無法計算技術指標。")
            else:
                st.dataframe(tech_df, use_container_width=True, hide_index=True)
                st.caption(f"建議說明：{tech_conclusion}")

            st.markdown("##### 🕯️ 形態辨識")
            _side_label = {"bull": "偏多", "bear": "偏空", "neutral": "中性"}
            if not candle_pats and not struct_pats:
                st.info("近期未偵測到明顯的 K 棒形態或價格結構形態。")
            else:
                if struct_pats:
                    st.dataframe(pd.DataFrame([{
                        "形態": p["name"],
                        "多空": _side_label[p["side"]],
                        "頸線": f"{p['neckline']:.1f}" if p.get("neckline") else "—",
                        "說明": p["desc"],
                    } for p in struct_pats]), use_container_width=True, hide_index=True)
                if candle_pats:
                    st.dataframe(pd.DataFrame([{
                        "日期": df.index[p["bar"]].strftime("%m/%d"),
                        "K棒形態": p["name"],
                        "多空": _side_label[p["side"]],
                        "說明": p["desc"],
                    } for p in reversed(candle_pats)]), use_container_width=True, hide_index=True)
                st.caption(
                    "形態為規則式辨識（K 棒掃描近 10 個交易日；結構形態取 pivot 高低點比對 "
                    "W底/M頭、頭肩型、三角收斂），已同步標註在上方主圖（可用「形態標註」開關）。"
                    "形態僅供參考，需搭配量能與趨勢確認。"
                )

        st.divider()
        news_date_label = news.recent_news_date_label()
        st.markdown(f"### {primary_label} 相關新聞（{news_date_label}）")
        # TW tickers: prefer the curated Chinese name for the news query — yfinance's
        # "shortName" comes back in English for TWSE tickers (see _display_name),
        # and an English company name paired with a zh-TW Google News search
        # routinely returns zero matches.
        company_name = universe.get_tw_company_name(primary) if is_tw else None
        if not company_name:
            company_name = (
                fdf["公司名稱"].iloc[0] if not fdf.empty and "公司名稱" in fdf and pd.notnull(fdf["公司名稱"].iloc[0]) else None
            )
        # US tickers: zh-TW Google News rarely indexes small/mid-cap US names
        # (e.g. OKLO), so lead with a broad English-language search instead and
        # only fall back to the zh-TW search if that's empty. TW tickers keep
        # the zh-TW search as primary.
        if is_tw:
            news_items = news.get_recent_news(primary, company_name)
            extra_news = news.get_twse_news(primary, company_name) + news.get_mops_news(primary, company_name)
        else:
            news_items = news.get_recent_news_en(primary, company_name) or news.get_recent_news(primary, company_name)
            extra_news = news.get_reuters_news(primary, company_name) + news.get_sec_filings(primary)
        news_items = sorted(news_items + extra_news, key=lambda n: n["published"], reverse=True)
        if not news_items:
            st.info(f"暫無 {news_date_label} 的相關新聞。")
        else:
            for n in news_items:
                published_str = n["published"].strftime("%Y-%m-%d %H:%M UTC")
                # US headlines come back in English; show a best-effort zh-TW
                # translation alongside the original (machine translation, may
                # be imperfect — original title links through for verification).
                title = n["title"]
                if not is_tw:
                    translated = news.translate_to_zh_tw(title)
                    if translated and translated != title:
                        title = f"{translated}（{title}）"
                st.markdown(f"- [{title}]({n['link']})　_{n['source']}｜{published_str}_")

# ---------- Tab 2: Multi-stock comparison, correlation & risk stats ----------
with tab_compare_risk:
    col_compare, col_period2, col_confirm3 = st.columns([2, 1, 1])
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
    with col_confirm3:
        st.write("")
        _gate3 = f"compare_confirmed_{'tw' if is_tw else 'us'}"
        _pressed3 = st.button("確定", key=f"{_gate3}_btn", use_container_width=True)
    period = PERIOD_OPTIONS[period_label]
    # Gate the heavy scan behind 確定. Wrapped in `if` (not st.stop) because
    # every tab renders in the same script run — st.stop() here would also
    # blank out the tabs after this one.
    if _confirm_gate(
        _gate3, (compare_input, period_label), _pressed3, missing=False,
        info_msg="請輸入比較標的與時間範圍，再按「確定」開始比較。",
    ):
        raw_compare = compare_input.strip()
        if raw_compare:
            if is_tw:
                compare_tickers = [universe.resolve_tw_ticker(t) for t in raw_compare.split(",") if t.strip()]
            else:
                compare_tickers = [t.strip().upper() for t in raw_compare.split(",") if t.strip()]
        elif is_tw:
            compare_tickers = universe.get_twse_tickers()
            st.caption(f"已自動帶入台股觀察清單：上市成交值前200＋上市/上櫃成交量前20＋精選ETF（{len(compare_tickers)} 檔）。")
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
def _render_buy_sell_section(
    is_tw: bool, currency: str, tab_key: str,
    period_options: dict, default_period_label: str,
    hold_options: dict, default_hold_label: str,
    allow_zhu_gate: bool, header: str, show_formula_caption: bool = True,
    weight_table: dict | None = None, dividend_screen: bool = False,
    dividend_top_yield: int = 50, dividend_top_fill: int = 30,
    require_confirm: bool = False,
) -> None:
    """Shared body for 買賣建議 and 存股區: both scan the same universe with
    a cross-sectional multi-factor composite, but with deliberately different
    factor sets — 買賣建議 (capital gains) scores 9 factors with 配息穩定性
    weighted 0, while 存股區 (cash flow) passes
    recommend.FACTOR_WEIGHTS_HOLDING: 殖利率＋配息穩定性＋填息率 carry the
    largest combined weight and Sortino replaces Sharpe. They also offer a
    different subset of 統計期間/持有天數 presets (存股區 restricts to
    medium/long-horizon options since it's a buy-and-hold view) and a
    different widget key namespace (tab_key) so the two tabs' selections
    don't collide in session_state.

    require_confirm (買賣建議 + 存股區): all 4 selectors default blank
    (index=None) and a 確定 button gates the scan — nothing runs until the
    user explicitly fills 統計期間/Top N/持有天數/目標積極度 and presses 確定.
    """
    weight_table = weight_table or recommend.FACTOR_WEIGHTS_BY_HORIZON
    st.subheader(header)
    if require_confirm:
        col_period3, col_topn, col_hold, col_aggr, col_confirm2 = st.columns(5)
    else:
        col_period3, col_topn, col_hold, col_aggr = st.columns(4)
    with col_period3:
        period_keys = list(period_options.keys())
        period_label = st.selectbox(
            "統計期間", period_keys,
            index=None if require_confirm else period_keys.index(default_period_label),
            key=f"period_{tab_key}_{'tw' if is_tw else 'us'}",
            help="此處選的期間，就是「期間報酬率」涵蓋的區段：從最近一個交易日往回推算。"
                 "「今年至今(YTD)」則為今年 1 月 1 日至今。",
        )
    with col_topn:
        _topn_key = f"topn_{tab_key}_{'tw' if is_tw else 'us'}"
        # Drop a stale persisted value from the old selectbox widget (None or
        # out of the 1–30 range) so the number_input doesn't raise on restore.
        _topn_saved = st.session_state.get(_topn_key)
        if _topn_key in st.session_state and (
                not isinstance(_topn_saved, (int, float)) or not 1 <= _topn_saved <= 30):
            st.session_state.pop(_topn_key)
        top_n = int(st.number_input(
            "建議買賣標的數量 (Top N)", min_value=1, max_value=30, value=15, step=1,
            key=_topn_key,
        ))
        # 買/賣 are mutually exclusive (checking one unchecks the other) rather
        # than independent checkboxes, so the table always shows exactly one
        # side — 買 checked by default. Enforced via on_change since Streamlit
        # checkboxes don't support radio-style grouping natively.
        _buy_key = f"show_buy_{tab_key}_{'tw' if is_tw else 'us'}"
        _sell_key = f"show_sell_{tab_key}_{'tw' if is_tw else 'us'}"
        if _buy_key not in st.session_state and _sell_key not in st.session_state:
            st.session_state[_buy_key] = True
            st.session_state[_sell_key] = False

        def _on_buy_toggle():
            # on_change fires after st.session_state[_buy_key] already holds
            # the new (post-click) value.
            st.session_state[_sell_key] = not st.session_state[_buy_key]

        def _on_sell_toggle():
            st.session_state[_buy_key] = not st.session_state[_sell_key]

        _cb_buy, _cb_sell = st.columns(2)
        with _cb_buy:
            show_buy = st.checkbox("買", key=_buy_key, on_change=_on_buy_toggle)
        with _cb_sell:
            show_sell = st.checkbox("賣", key=_sell_key, on_change=_on_sell_toggle)
    with col_hold:
        hold_keys = list(hold_options.keys()) + ([_HOLD_CUSTOM_LABEL] if allow_zhu_gate else [])
        hold_label = st.selectbox(
            "持有天數", hold_keys,
            index=None if require_confirm else hold_keys.index(default_hold_label),
            key=f"hold_{tab_key}_{'tw' if is_tw else 'us'}",
            help="預測準確機率以此持有天數（交易日）計算，建議進場／賣出價也用同一持有期。",
        )
        if hold_label == _HOLD_CUSTOM_LABEL:
            hold_days = int(st.number_input(
                "自訂持有天數（交易日）", min_value=1, max_value=1260, value=5, step=1,
                key=f"hold_custom_{tab_key}_{'tw' if is_tw else 'us'}",
            ))
            hold_display = f"{hold_days} 個交易日"
        elif hold_label is not None:
            hold_days = hold_options[hold_label]
            # Surface the actual trading-day count for week/month/year labels so
            # the caption's 天數 is unambiguously the same as the holding period
            # used in the calculation (e.g. "1個月（21 交易日）"). Day-count labels
            # like "5天" already equal the count, so no suffix is added.
            # Half-width "~" (e.g. "1~5天") is swapped for full-width "～":
            # hold_display goes into st.caption markdown, where a pair of "~"
            # renders everything between them as strikethrough.
            hold_display = hold_label if hold_label == f"{hold_days}天" else f"{hold_label}（{hold_days} 交易日）"
            hold_display = hold_display.replace("~", "～")
        else:
            hold_days = hold_display = None
    with col_aggr:
        aggr_label3 = st.selectbox(
            "目標積極度", list(RECO_AGGRESSIVENESS.keys()),
            index=None if require_confirm else 1,
            key=f"aggr_{tab_key}_{'tw' if is_tw else 'us'}",
            help="保守＝買賣目標較近、較易達成（預測準確機率較高）；積極＝目標較遠。中性＝歷史中位幅度。",
        )
        reco_aggr = RECO_AGGRESSIVENESS.get(aggr_label3)

    if require_confirm:
        _gate_key = f"reco_confirmed_{tab_key}_{'tw' if is_tw else 'us'}"
        with col_confirm2:
            st.write("")
            _pressed = st.button("確定", key=f"{_gate_key}_btn", use_container_width=True)
        # Signature includes the 4 selectors plus 存股區's 殖利率/填息率 screen
        # inputs — changing any of them re-closes the gate so the scan only
        # re-runs after a fresh 確定 (not live on every edit).
        _sig = (period_label, top_n, hold_label, aggr_label3,
                dividend_screen, dividend_top_yield, dividend_top_fill)
        if not _confirm_gate(
            _gate_key, _sig, _pressed,
            missing=None in (period_label, hold_label, aggr_label3),
            info_msg="請選擇「統計期間」「持有天數」「目標積極度」並填「建議買賣標的數量」，再按「確定」開始查詢。",
            warning_msg="請完整選擇「統計期間」「持有天數」「目標積極度」，再按「確定」開始查詢。",
        ):
            return

    period_spec = period_options[period_label]
    period = period_spec["fetch"]
    reco_lookback = period_spec["lookback"]
    reco_horizon = period_spec["horizon"]
    reco_weights = weight_table[reco_horizon]

    # 因子占比，緊接在「統計期間」等控制項下方一列呈現。權重直接取自
    # weight_table（reco_weights），所以買賣建議(9因子)跟存股區(10因子，多了
    # 殖利率)即使因子數不同，這裡也會自動對齊，不需要兩份硬寫的清單。
    _FACTOR_LABELS_ALL = {
        "期間報酬率": "期間報酬率",
        "技術面": "技術面(動能+布林+SMA+型態)",
        "趨勢(價格/均線)": "價格趨勢",
        "Sharpe Ratio": "Sharpe",
        "Sortino": "Sortino(抗跌)",
        "填息率": "填息率",
        "估值(1/預估PE)": "估值",
        "基本面": "基本面",
        "籌碼": "籌碼面",
        "新聞情緒": "新聞情緒",
        "配息穩定性": "配息穩定性",
        "殖利率": "殖利率",
    }
    _FACTOR_DISPLAY = {k: _FACTOR_LABELS_ALL.get(k, k) for k in reco_weights}
    st.markdown(
        f"**綜合評分 ＝ 下列{len(_FACTOR_DISPLAY)}大因子加權（占比如下）**　"
        f"已依「統計期間」自動切換為 **{_HORIZON_LABEL[reco_horizon]}** 權重"
    )
    _fcols = st.columns(len(_FACTOR_DISPLAY))
    for _c, (_k, _label) in zip(_fcols, _FACTOR_DISPLAY.items()):
        _c.metric(_label, f"{reco_weights[_k] * 100:.0f}%")
    chip_src = "台股三大法人買賣超（近5日）" if is_tw else "資金流向 CMF（量價推估）"
    st.caption(
        "短期（≤1個月）重視價格動能、技術面與籌碼面、淡化估值與基本面；長期（≥1年）重視基本面、"
        "估值與風險調整報酬、淡化短線動能與消息；中期（3個月～半年、YTD）則居中平衡。"
        f"籌碼面資料來源：{chip_src}。"
    )

    if show_formula_caption:
        if is_tw:
            scope_desc = "「上市成交值前 200 大」「上市成交量前 20 大」「上櫃成交量前 20 大」「精選 ETF」的聯集"
        else:
            scope_desc = "「美股近期成交量前 30 大」「S&P 500 成分股」「Nasdaq-100 成分股」「道瓊 30 成分股」的聯集"
        st.caption(
            f"- **篩選範圍**：{scope_desc}\n"
            "- **評分方式**：上列九因子計算「組內相對排序（z 分數）」，僅反映目前範圍內標的相對高低，非投資建議\n"
            "- **基本面**：營收/盈餘成長率、淨利率、ROE；**技術面**：RSI/KDJ/MACD；**籌碼面**：台股三大法人、美股資金流 CMF\n"
            f"- **建議買入價**：買入＝現價逢低承接（−N日跌幅中位）、賣出＝現價逢高減碼（持有 {hold_display}）\n"
            f"- **建議賣出價／獲利%**：賣出價＝進場價×(1＋N日漲幅中位)；獲利%＝賣出/進場−1\n"
            f"- **預測準確機率**：歷史上 {hold_display} 內，股價「最高觸及建議賣出價」（買）／「最低觸及」（賣）的比例\n"
            "- **操作**：點各欄表頭可由大至小／小至大排序\n"
            + (
                (
                    f"- **短期進場濾網**：持有 {hold_display}（≤5 交易日）時，買入清單採朱家泓式突破訊號"
                    "（現價站上向上的MA20＋收盤同時突破MA5與前一日高點）做硬性篩選，沒訊號的標的不會入選買入清單"
                    "（回測 2026/02～06 美股 1/3/5 天持有：53.8%→60.4%、56.0%→56.5%、53.0%→53.4%；"
                    "6～10 天另外回測過，濾網沒有穩定效果甚至偶爾更差，故不套用，回歸純綜合評分排序）\n"
                    if hold_days <= 5 else ""
                )
                if allow_zhu_gate else ""
            )
            + "\n"
            f"**公式**（u＝{hold_display}上漲報酬〔依目標積極度取百分位，中性=中位數〕、d＝下跌報酬〔同〕〔d<0〕；"
            "並依技術訊號微調：u→u×(1+0.4×技術偏多度)、d→d×(1−0.4×技術偏多度)，"
            "技術偏多度由 布林/SMA多頭/趨勢/形態辨識+動能 依持有天數加權）：\n"
            "1. 建議買入價＝現價×(1+d)〔買〕／ 現價×(1+u)〔賣〕\n"
            "2. 建議賣出價＝現價×(1+u)〔買〕／ 現價×(1+d)〔賣〕\n"
            "3. 獲利%＝建議賣出價 / 建議買入價 − 1\n"
            "4. 預測準確機率＝歷史上 N 日內「最高價≥建議賣出價」〔買〕／「最低價≤建議賣出價」〔賣〕的比例"
        )
    if is_tw:
        reco_universe = universe.get_twse_tickers()
    else:
        reco_universe = sorted(
            set(universe.get_top_volume_tickers(30)) | set(universe.get_sp500_tickers())
            | set(universe.get_nasdaq100_tickers()) | set(universe.get_dow_tickers())
        )

    # 填息率 only exists for tickers that passed the screen — kept here (not
    # as a separate table) so it merges straight into the same buy/sell
    # table below instead of duplicating the screened list in its own
    # expander (the buy/sell list is already a scored subset of exactly
    # these tickers once the screen is on, so showing both was redundant).
    fill_rate_by_ticker = None
    if dividend_screen:
        with st.spinner(f"正在計算 {len(reco_universe)} 檔標的的殖利率／填息率…"):
            screen_df = recommend.dividend_fill_screen(
                reco_universe, top_yield=dividend_top_yield, top_fill=dividend_top_fill)
        if screen_df.empty:
            st.warning("目前範圍內沒有足夠的配息資料可供篩選，已停用此篩選、改用完整候選範圍。")
        else:
            st.caption(
                f"已套用篩選：配息殖利率前{dividend_top_yield}高 ∩ 填息率前{dividend_top_fill}高，"
                f"共 {len(screen_df)} 檔標的進入候選範圍（填息率會顯示在下方表格的「填息率」欄）。"
            )
            fill_rate_by_ticker = screen_df.set_index("代號")["填息率"]
            reco_universe = screen_df["代號"].tolist()

    with st.spinner(f"正在掃描 {len(reco_universe)} 檔標的計算評分，資料量較大可能需要數分鐘…"):
        reco_table = recommend.build_recommendation_table(
            reco_universe, period, DEFAULT_RISK_FREE_RATE,
            lookback_days=reco_lookback, weights=reco_weights, horizon=reco_horizon)
    if fill_rate_by_ticker is not None and not reco_table.empty:
        reco_table["填息率"] = reco_table.index.map(fill_rate_by_ticker)
    if reco_table.empty:
        st.warning("無足夠資料產生建議，請確認統計期間。")
        return

    # ≤5 交易日 only: gate buy picks to 朱家泓-style breakout triggers
    # (現價>上升MA20 且 收盤突破MA5+前日高點), not just highest score.
    # Backtested 2026-02~06 美股: lifts win rate at 1/3/5 天 (~54%→60% at
    # 1天); a separate 6~10 天 backtest found no reliable improvement
    # (sometimes worse), so the gate stops at 5 天. 存股區's hold options
    # are all >15 天 (allow_zhu_gate=False), so this never fires there.
    _zhu_gate = "_zhu_signal" if (allow_zhu_gate and hold_days <= 5) else None
    buy_df, sell_df = recommend.top_buy_sell(reco_table, top_n, require_signal_col=_zhu_gate)
    # 買/賣 checkboxes (mutually exclusive, 買 default) decide which side is
    # actually displayed — empty out whichever side is unchecked before the
    # (otherwise unchanged) price-target/reason/merge pipeline below.
    if not show_buy:
        buy_df = buy_df.iloc[0:0]
    if not show_sell:
        sell_df = sell_df.iloc[0:0]
    buy_df = recommend.add_reason(
        recommend.add_price_targets(buy_df, "buy", currency, hold_days,
                                    horizon=reco_horizon, aggressiveness=reco_aggr), "buy")
    sell_df = recommend.add_reason(
        recommend.add_price_targets(sell_df, "sell", currency, hold_days,
                                    horizon=reco_horizon, aggressiveness=reco_aggr), "sell")

    # 殖利率% is a raw fraction (e.g. 0.0523) like 期間報酬率/趨勢, not a
    # pre-z-scored composite — shown as an actual percentage since "看實際
    # 殖利率高低" is the point, a z-score wouldn't be legible here.
    # 填息率 (raw fraction too) exists when 存股區's weight table scores it,
    # or when the 殖利率+填息率篩選 was on (see fill_rate_by_ticker above) —
    # _format_reco/_column_config below guard for its absence.
    _PCT_COLS = ["期間報酬率", "趨勢(價格/均線)", "殖利率%", "填息率"]
    # 基本面/技術面/籌碼/配息穩定性 are 組內相對 z 分數（越高＝相對越強），同列以 2 位小數顯示。
    _PLAIN_COLS = ["Sharpe Ratio", "Sortino", "估值(1/預估PE)", "新聞情緒", "基本面", "技術面", "籌碼",
                   "配息穩定性", "RSI (14)", "綜合評分"]
    _PRICE_COLS = ["建議買入價", "建議賣出價"]
    _COL_ORDER = ["建議", "綜合評分", "殖利率%", "填息率", "期間報酬率", "技術面", "趨勢(價格/均線)", "Sharpe Ratio",
                  "Sortino", "估值(1/預估PE)", "基本面", "籌碼", "配息穩定性", "新聞情緒", "RSI (14)",
                  "建議買入價", "建議賣出價", "獲利%", "預測準確機率", "原因說明", "備註"]

    def _format_reco(df: pd.DataFrame) -> pd.DataFrame:
        fmt = df.copy()
        for col in _PCT_COLS:
            if col in fmt:
                fmt[col] = fmt[col] * 100
        fmt.index = [_display_name(t) for t in fmt.index]
        return fmt

    def _column_config(df: pd.DataFrame) -> dict:
        config = {"建議": st.column_config.TextColumn("建議", width="small", help="綠＝建議買入、紅＝建議賣出；顏色越深代表預測準確機率越高。")}
        # 獲利% and 預測準確機率 are already stored as percentages (not fractions),
        # so they only get the %% format, not the *100 in _format_reco.
        for col in _PCT_COLS + ["獲利%", "預測準確機率"]:
            if col in df:
                config[col] = st.column_config.NumberColumn(col, format="%.2f%%")
        for col in _PLAIN_COLS:
            if col in df:
                config[col] = st.column_config.NumberColumn(col, format="%.2f")
        if "綜合評分" in df:
            config["綜合評分"] = st.column_config.NumberColumn(
                "綜合評分", format="%.2f",
                help="多因子加權 z 分數（權重合計 100%）。為「組內相對分數」、以 0 為中位、"
                     "越高越好，無固定滿分；實務上多落在約 −2 ~ +2。",
            )
        for col in _PRICE_COLS:
            if col in df:
                config[col] = st.column_config.NumberColumn(col, format=f"{currency}%.2f")
        return config

    # Merge buy + sell into one table: unify the two side-specific price
    # columns to 進場價/目標價, prepend a 建議 dot, and keep buys (high score)
    # above sells. The dot is coloured green (buy) / red (sell) and shaded
    # by 預測準確機率 via a Styler.
    buy_u = buy_df.rename(columns={"建議買入價": "建議買入價", "目標賣出價": "建議賣出價"})
    sell_u = sell_df.rename(columns={"建議賣出價": "建議買入價", "逢低買回參考價": "建議賣出價"})
    buy_u["建議"] = "●"
    sell_u["建議"] = "●"
    merged = pd.concat([_format_reco(buy_u), _format_reco(sell_u)])
    if "備註" in merged.columns:
        merged["備註"] = merged["備註"].fillna("")
    merged = merged.reindex(columns=[c for c in _COL_ORDER if c in merged.columns])
    _sides = ["buy"] * len(buy_u) + ["sell"] * len(sell_u)
    _future = merged["預測準確機率"].tolist() if "預測準確機率" in merged else [None] * len(merged)
    # Min–max normalize 預測準確機率 across the table so the green/red gradient
    # always spans the full visible range (a fixed 0–40% scale made every dot
    # look the same dark shade when win rates clustered high).
    _vals = [w for w in _future if w is not None and pd.notna(w)]
    _fmin, _fmax = (min(_vals), max(_vals)) if _vals else (0.0, 1.0)
    _span = (_fmax - _fmin) or 1.0

    def _signal_styles(df: pd.DataFrame) -> pd.DataFrame:
        css = pd.DataFrame("", index=df.index, columns=df.columns)
        loc = df.columns.get_loc("建議")
        for i, (side, w) in enumerate(zip(_sides, _future)):
            frac = (w - _fmin) / _span if (w is not None and pd.notna(w)) else 0.0
            rgb = _signal_rgb(side, frac)
            # Fill the whole cell with the gradient colour; matching text
            # colour hides the ● glyph so the cell reads as a solid block.
            css.iloc[i, loc] = f"background-color: {rgb}; color: {rgb}"
        return css

    st.markdown(f"#### 建議買入（綠）{len(buy_df)} 檔 ／ 賣出（紅）{len(sell_df)} 檔")
    st.dataframe(
        merged.style.apply(_signal_styles, axis=None),
        use_container_width=True, column_config=_column_config(merged),
    )

    if show_buy and len(buy_df) < top_n:
        if _zhu_gate:
            st.info(
                f"買入清單僅 {len(buy_df)} 檔（非選擇的 Top {top_n}）：持有 {hold_display} 採用"
                "朱家泓式進場濾網（現價站上向上的MA20、且收盤同時突破MA5與前一日高點），"
                "目前範圍內符合此突破訊號的標的較少，沒訊號不勉強湊數。"
            )
        else:
            st.info(
                f"目前範圍共 {len(reco_table)} 檔標的，為避免買入／賣出名單重複，"
                f"已各自裁切為 {len(buy_df)} 檔（最多取清單一半），而非選擇的 Top {top_n}。"
            )


with tab_reco:
    _render_buy_sell_section(
        is_tw, currency, "tab3",
        RECO_PERIOD_OPTIONS, "1年",
        RECO_HOLD_OPTIONS, "1~5天",
        allow_zhu_gate=True, header="基金經理人觀點：建議買入 / 賣出",
        require_confirm=True,
    )

# ---------- Tab 4: 存股區 (long-term buy-and-hold view) ----------
with tab_stock_hold:
    st.caption(
        "存股區定位為**現金流（領息）**，與買賣建議（賺資本利得）採不同公式：配息三訊號合計權重最高——"
        "**殖利率（領多少）、配息穩定性（領得穩）、填息率（除息後60交易日內回補缺口的比例，貼息＝把本金"
        "配還給你，直接扣分）**；風險調整報酬改用 **Sortino（只懲罰下跌波動）** 取代 Sharpe；"
        "**基本面（營收/盈餘成長、ROE、淨利率）**維持高權重——配息的可持續性來自獲利；期間報酬率/技術面/"
        "籌碼/新聞情緒等短線訊號壓到接近零。「統計期間」與「持有天數」也只保留中期／長期選項，"
        "定位為長期存股／逢低布局參考，而非短線進出。"
    )
    col_div_chk, col_div_yield_n, col_div_fill_n = st.columns([2, 1, 1])
    with col_div_chk:
        _dividend_screen_on = st.checkbox(
            "套用「今年配息殖利率前N高 ∩ 填息率前N高」篩選",
            key=f"dividend_screen_{'tw' if is_tw else 'us'}",
            help="先把候選範圍縮小到近12個月殖利率前N高、且其中填息率（除息缺口60個交易日內回補的比例）前N高的"
                 "標的，再用下方九因子公式排序；不是另一套公式，是排序前的候選範圍篩選。",
        )
    with col_div_yield_n:
        _div_top_yield = int(st.number_input(
            "殖利率前N高", min_value=1, max_value=500, value=50, step=1,
            key=f"div_top_yield_{'tw' if is_tw else 'us'}", disabled=not _dividend_screen_on,
        ))
    with col_div_fill_n:
        _div_top_fill = int(st.number_input(
            "填息率前N高", min_value=1, max_value=_div_top_yield, value=min(30, _div_top_yield), step=1,
            key=f"div_top_fill_{'tw' if is_tw else 'us'}", disabled=not _dividend_screen_on,
        ))
    _render_buy_sell_section(
        is_tw, currency, "tab_hold",
        _HOLDING_PERIOD_OPTIONS, "1年",
        _HOLDING_HOLD_OPTIONS, "1年",
        allow_zhu_gate=False, header="長期存股觀點：建議買入 / 賣出",
        show_formula_caption=False, weight_table=recommend.FACTOR_WEIGHTS_HOLDING,
        dividend_top_yield=_div_top_yield, dividend_top_fill=_div_top_fill,
        dividend_screen=_dividend_screen_on, require_confirm=True,
    )

# ---------- Tab 5: FCN risk assessment ----------
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

        _fcn_gate = f"fcn_confirmed_{_mkt_suffix}"
        _fcn_pressed = st.button("確定", key=f"{_fcn_gate}_btn")
        # 任一條款參數變動都會讓閘門重新關閉，需再按「確定」才重新抓價＋模擬。
        _fcn_sig = (n_assets, fcn_raw_tickers, tenor_months, strike_pct,
                    ko_pct, ki_pct, coupon_rate, ki_style)
        if _confirm_gate(
            _fcn_gate, _fcn_sig, _fcn_pressed, missing=False,
            info_msg="請設定好所有參數，再按「確定」開始模擬。",
        ):
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
    and not k.startswith(_PERSIST_EXCLUDE_PREFIXES)
}
_local_storage.setItem(_SETTINGS_STORAGE_KEY, json.dumps(_settings_to_save), key="save_dashboard_settings")
