"""FCN (Fixed Coupon Note) risk assessment — Streamlit page 2."""
import pandas as pd
import streamlit as st

from src import data_loader as dl
from src import fcn
from src import universe

st.set_page_config(page_title="FCN 風險評估", layout="wide")

VOL_WINDOW_OPTIONS = {"3個月": "3mo", "6個月": "6mo", "1年": "1y", "2年": "2y"}
TENORS_MONTHS = [3, 6, 9, 12]
N_SIMS = 8000

st.title("📐 FCN 風險評估與條款試算")
st.caption(
    "本頁以蒙地卡羅模擬（幾何布朗運動，波動率取標的歷史日報酬年化）估算 FCN（Fixed Coupon "
    "Note，股權連結型定期配息票券）的合理年化收益率與下限價（KI）觸及機率，協助評估您持有"
    "或考慮買入的 FCN 風險。模型為簡化估算（單一標的、月配息、月觀察提前出場），"
    "未涉及實際發行商報價、信用風險或手續費，**僅供研究參考，非投資建議**。"
)

# ---------- Underlying & market data ----------
col_market, col_ticker, col_window = st.columns([1, 1.4, 1])
with col_market:
    market = st.radio("市場", ["美股", "台股"], horizontal=True, key="fcn_market")
is_tw = market == "台股"
with col_ticker:
    default_ticker = "2330" if is_tw else "AAPL"
    ticker_label = "標的股票代號" + ("（台股代碼，例如 2330）" if is_tw else "")
    raw_ticker = st.text_input(ticker_label, value=default_ticker, key="fcn_ticker").strip().upper() or default_ticker
    ticker = universe.resolve_tw_ticker(raw_ticker) if is_tw else raw_ticker
with col_window:
    window_label = st.selectbox("波動率估算窗口", list(VOL_WINDOW_OPTIONS.keys()), index=2, key="fcn_window")

currency = "NT$" if is_tw else "$"
df = dl.get_price_history(ticker, period=VOL_WINDOW_OPTIONS[window_label])
if df.empty:
    st.error(f"找不到 {ticker} 的價格資料，請確認代號是否正確。")
    st.stop()

vol_annual, drift_hist_annual = fcn.historical_vol_and_drift(df["Close"])
spot = float(df["Close"].iloc[-1])
info = dl.get_company_info(ticker)
dividend_yield = info.get("dividendYield") or 0.0
if dividend_yield > 0.5:  # yfinance has, at times, returned this as a percent rather than a fraction
    dividend_yield /= 100.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("最新收盤價", f"{currency}{spot:,.2f}")
m2.metric(f"年化歷史波動率（{window_label}）", f"{vol_annual * 100:.1f}%")
m3.metric(f"年化歷史漲跌幅（{window_label}）", f"{drift_hist_annual * 100:.1f}%")
m4.metric("股息率（估）", f"{dividend_yield * 100:.2f}%")

st.divider()

# ---------- Product structure & assumptions ----------
st.subheader("條款假設")
col_strike, col_autocall, col_rf = st.columns(3)
with col_strike:
    strike_pct = st.slider("執行價 Strike（% of 期初價）", 80, 110, 100, step=1, key="fcn_strike") / 100
with col_autocall:
    autocall_pct = st.slider("提前出場 Autocall（% of 期初價）", 80, 115, 100, step=1, key="fcn_autocall") / 100
with col_rf:
    risk_free_rate = st.number_input("無風險利率（年化 %）", min_value=0.0, max_value=10.0, value=4.0, step=0.25,
                                      key="fcn_rf") / 100

col_ki_range, col_ki_style, col_tol = st.columns(3)
with col_ki_range:
    ki_lo, ki_hi = st.slider("下限價 KI 掃描範圍（% of 期初價）", 40, 95, (60, 85), step=5, key="fcn_ki_range")
with col_ki_style:
    ki_style_label = st.radio(
        "KI 觀察方式", ["僅到期日判定（歐式，較常見）", "全程逐日觀察（美式，較嚴格）"],
        key="fcn_ki_style",
    )
    ki_style = "maturity" if ki_style_label.startswith("僅到期日") else "continuous"
with col_tol:
    risk_tolerance_pct = st.slider("可接受的本金虧損機率上限（%）", 5, 50, 20, step=5, key="fcn_tolerance")

drift_choice = st.radio(
    "風險評估的股價成長率假設",
    ["中性假設：預期報酬＝無風險利率，僅反映波動風險（較保守，預設）",
     f"延伸近期歷史走勢：年化 {drift_hist_annual * 100:.1f}%（可能過度樂觀或悲觀，僅供對照）"],
    key="fcn_drift_choice",
)
use_historical_drift = drift_choice.startswith("延伸近期歷史走勢")

ki_levels = [k / 100 for k in range(ki_lo, ki_hi + 1, 5) if k / 100 <= strike_pct]
if not ki_levels:
    st.warning("KI 掃描範圍須低於或等於執行價 Strike，請調整滑桿。")
    st.stop()

drift_for_risk = drift_hist_annual if use_historical_drift else (risk_free_rate - dividend_yield)
drift_risk_neutral = risk_free_rate - dividend_yield


@st.cache_data(show_spinner=False)
def _run(strike_pct, ki_pct, autocall_pct, tenor_months, vol_annual, drift_annual, risk_free_rate, ki_style, n_sims):
    stats = fcn.simulate_paths(
        strike_pct=strike_pct, ki_pct=ki_pct, autocall_pct=autocall_pct, tenor_months=tenor_months,
        vol_annual=vol_annual, drift_annual=drift_annual, risk_free_rate=risk_free_rate,
        ki_style=ki_style, n_sims=n_sims,
    )
    return stats


st.divider()
st.subheader("年化收益率與本金虧損機率（依 KI 與合約期間掃描）")
st.caption(
    "年化收益率＝在風險中性測度下，使票券折現價值等於票面（100%）的合理票息；"
    "本金虧損機率＝在您所選的風險評估假設下，未提前出場且到期觸及 KI 的機率（僅供風險參考，非報價）。"
)

with st.spinner("模擬中…"):
    coupon_table, breach_table, autocall_summary = {}, {}, {}
    for tenor in TENORS_MONTHS:
        rn_anchor = _run(strike_pct, ki_levels[0], autocall_pct, tenor, vol_annual, drift_risk_neutral,
                          risk_free_rate, ki_style, N_SIMS)
        autocall_summary[tenor] = {
            "提前出場機率": rn_anchor.prob_autocall,
            "平均出場月數": rn_anchor.avg_exit_month,
        }
        coupon_col, breach_col = {}, {}
        for ki in ki_levels:
            rn_stats = _run(strike_pct, ki, autocall_pct, tenor, vol_annual, drift_risk_neutral,
                             risk_free_rate, ki_style, N_SIMS)
            fair_coupon = fcn.fair_coupon_rate(rn_stats)
            risk_stats = (
                rn_stats if not use_historical_drift else
                _run(strike_pct, ki, autocall_pct, tenor, vol_annual, drift_for_risk,
                     risk_free_rate, ki_style, N_SIMS)
            )
            coupon_col[ki] = fair_coupon
            breach_col[ki] = risk_stats.prob_breach
        coupon_table[tenor] = coupon_col
        breach_table[tenor] = breach_col

coupon_df = pd.DataFrame(coupon_table)
breach_df = pd.DataFrame(breach_table)
coupon_df.index = [f"{k * 100:.0f}%" for k in coupon_df.index]
breach_df.index = [f"{k * 100:.0f}%" for k in breach_df.index]
coupon_df.columns = [f"{m}個月" for m in coupon_df.columns]
breach_df.columns = [f"{m}個月" for m in breach_df.columns]

col_coupon, col_breach = st.columns(2)
with col_coupon:
    st.markdown("##### 合理年化收益率")
    st.dataframe(coupon_df.style.format("{:.2%}"), use_container_width=True)
with col_breach:
    st.markdown("##### 本金虧損機率")
    st.dataframe(breach_df.style.format("{:.1%}"), use_container_width=True)

autocall_df = pd.DataFrame(autocall_summary).T
autocall_df.index = [f"{m}個月" for m in autocall_df.index]
st.markdown("##### 提前出場機率與平均出場月數（與 KI 無關，僅取決於合約期間／提前出場%）")
st.dataframe(
    autocall_df.style.format({"提前出場機率": "{:.1%}", "平均出場月數": "{:.1f}"}),
    use_container_width=True,
)

st.divider()

# ---------- Recommendation ----------
st.subheader("建議參數")
candidates = [
    (tenor, ki, coupon_table[tenor][ki], breach_table[tenor][ki])
    for tenor in TENORS_MONTHS for ki in ki_levels
    if breach_table[tenor][ki] <= risk_tolerance_pct / 100
]
if not candidates:
    st.warning("在目前的風險容忍度下，掃描範圍內沒有任何組合的本金虧損機率低於門檻，請放寬風險容忍度或調整 KI 範圍。")
else:
    best_tenor, best_ki, best_coupon, best_breach = max(candidates, key=lambda c: c[2])
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("建議合約期間", f"{best_tenor} 個月")
    r2.metric("執行價 Strike", f"{strike_pct * 100:.0f}%")
    r3.metric("下限價 KI", f"{best_ki * 100:.0f}%")
    r4.metric("提前出場 Autocall", f"{autocall_pct * 100:.0f}%")
    r5.metric("合理年化收益率", f"{best_coupon * 100:.2f}%", f"虧損機率 {best_breach * 100:.1f}%")
    st.caption(
        f"在「本金虧損機率 ≤ {risk_tolerance_pct}%」的限制下，掃描範圍內以此組合的年化收益率最高。"
        "若想要更高收益率，須承受更高的本金虧損機率（KI 設得更高）；"
        "若想要更低風險，年化收益率會相應降低（KI 設得更低）。"
    )

    st.markdown("##### 合約期間怎麼選？")
    per_tenor_best = {}
    for tenor in TENORS_MONTHS:
        ok = [(ki, coupon_table[tenor][ki]) for ki in ki_levels if breach_table[tenor][ki] <= risk_tolerance_pct / 100]
        if ok:
            ki_pick, coupon_pick = max(ok, key=lambda c: c[1])
            per_tenor_best[f"{tenor}個月"] = {
                "可用最高年化收益率": coupon_pick,
                "對應 KI": ki_pick,
                "本金虧損機率": breach_table[tenor][ki_pick],
                "提前出場機率": autocall_summary[tenor]["提前出場機率"],
                "平均出場月數": autocall_summary[tenor]["平均出場月數"],
            }
    if per_tenor_best:
        tenor_compare_df = pd.DataFrame(per_tenor_best).T
        st.dataframe(
            tenor_compare_df.style.format({
                "可用最高年化收益率": "{:.2%}", "對應 KI": "{:.0%}", "本金虧損機率": "{:.1%}",
                "提前出場機率": "{:.1%}", "平均出場月數": "{:.1f}",
            }),
            use_container_width=True,
        )
    st.caption(
        "一般而言，期間越長累積觀察次數越多、提前出場機率通常越高（資金可能更快收回），"
        "但到期觸及 KI 的機率也可能隨曝險時間拉長而上升；實際關係取決於標的波動率與您設定的"
        "提前出場／KI 門檻，請以上表的模擬結果為準，而非單純「越長越好」或「越短越好」。"
    )

st.divider()
st.caption(
    "模型限制：僅模擬單一標的、固定波動率（不含波動率微笑/期限結構）、不含股息再投資與交易成本、"
    "不含發行商信用風險與流動性折價，且歷史波動率／報酬率不保證代表未來。實際 FCN 報價請以發行商"
    "（券商/銀行）條款書為準。"
)
