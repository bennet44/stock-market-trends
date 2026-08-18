"""美股本週行事曆：財報日、總經數據發布日、四巫日等市場結構事件。

與 src/news.py 的差別：news.py 抓的是「新聞標題」（某件事被報導了），這裡給
的是「行事曆」（幾號會發生什麼），兩者在市場焦點分頁互補呈現。

資料來源分三類：
1. 純計算（免網路、永遠正確）——四巫日／月選擇權到期日／每週四初領失業金。
2. Nasdaq 財報行事曆 API——純 urllib + User-Agent 即可取得，不需金鑰。
   （yfinance 的 Ticker.calendar/get_earnings_dates 在部分環境走 curl_cffi
   會 SSL 失敗，故不採用；同 data_loader._chart_ohlcv 的處置理由。）
3. 內建總經時程表——BLS/BEA 官網對程式化請求回 403（含瀏覽器 UA 也擋），
   無法自動抓取，因此採人工維護的固定表，每年更新一次。
"""
from __future__ import annotations

import calendar as _calendar
import datetime as dt
import json
import ssl
import urllib.request

import streamlit as st

_NASDAQ_EARNINGS_URL = "https://api.nasdaq.com/api/calendar/earnings?date={date}"
_NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}


# ---------- 純計算型事件 ----------

# NYSE 全日休市日與提早收盤日（來源：nyse.com/markets/hours-calendars）。
# 每年需更新；未涵蓋的年份，休市相關事件與「遇假日順延」的調整都會自動略過。
_US_MARKET_HOLIDAYS: dict[str, str] = {
    "2026-01-01": "元旦",
    "2026-01-19": "馬丁路德金恩日",
    "2026-02-16": "總統日",
    "2026-04-03": "耶穌受難日",
    "2026-05-25": "陣亡將士紀念日",
    "2026-06-19": "六月節 (Juneteenth)",
    "2026-09-07": "勞動節",
    "2026-11-26": "感恩節",
    "2026-12-25": "耶誕節",
}
# 提早收盤（美東 13:00）——量能通常很淡，對當沖/結算有影響故一併列出。
_US_EARLY_CLOSES: dict[str, str] = {
    "2026-07-03": "國慶日順延（提早收盤 13:00）",
    "2026-11-27": "感恩節翌日（提早收盤 13:00）",
    "2026-12-24": "平安夜（提早收盤 13:00）",
}


def is_market_holiday(d: dt.date) -> bool:
    """該日是否為 NYSE 全日休市日。"""
    return d.isoformat() in _US_MARKET_HOLIDAYS


def _is_trading_day(d: dt.date) -> bool:
    return d.weekday() < _calendar.SATURDAY and not is_market_holiday(d)


def _nth_business_day(year: int, month: int, n: int) -> dt.date:
    """該月第 n 個營業日（跳過週末與休市日）。ISM 系列指標的發布慣例就是
    以營業日計（製造業第 1 個、服務業第 3 個）。"""
    d = dt.date(year, month, 1)
    count = 0
    while True:
        if _is_trading_day(d):
            count += 1
            if count == n:
                return d
        d += dt.timedelta(days=1)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> dt.date:
    """該月最後一個指定星期幾（0=週一）。諮商會消費者信心指數的慣例發布日
    是每月最後一個週二。"""
    last_day = _calendar.monthrange(year, month)[1]
    d = dt.date(year, month, last_day)
    while d.weekday() != weekday:
        d -= dt.timedelta(days=1)
    return d


def third_friday(year: int, month: int) -> dt.date:
    """該月的第三個週五（未考慮假日，純日曆定義）。多數情況即為選擇權
    到期日，但遇假日需改用 opex_date()。"""
    fridays = [
        dt.date(year, month, d)
        for d in _calendar.Calendar().itermonthdays(year, month)
        if d and dt.date(year, month, d).weekday() == _calendar.FRIDAY
    ]
    return fridays[2]


def opex_date(year: int, month: int) -> dt.date:
    """實際的選擇權到期日：原則上是第三個週五，但**該日若為休市日則提前到
    前一個交易日**（通常是週四）。

    這不是理論上的邊界情況——2026-06-19 同時是六月的第三個週五與六月節
    (Juneteenth) 休市日，若不調整就會把休市日標成四巫日。
    """
    d = third_friday(year, month)
    while not _is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def quad_witching_dates(year: int) -> list[dt.date]:
    """該年度四個四巫日：3/6/9/12 月的選擇權到期日（已做假日調整）。"""
    return [opex_date(year, m) for m in (3, 6, 9, 12)]


def _first_friday(year: int, month: int) -> dt.date:
    """該月第一個週五——非農就業報告 (NFP) 的慣例發布日。實際上 BLS 偶有
    調整（例如遇假日），所以呈現時標示為慣例日而非官方確認日。"""
    for d in range(1, 8):
        if dt.date(year, month, d).weekday() == _calendar.FRIDAY:
            return dt.date(year, month, d)
    raise ValueError(f"no Friday found in {year}-{month}")  # 不可能發生


# ---------- 內建總經時程表（需人工維護） ----------

# BLS/BEA 官網對程式化請求一律回 403（含瀏覽器 UA），無法自動抓取，因此
# 總經數據的「官方確定發布日」採人工維護的內建表，每年更新一次。
#
# 日期不是推估的——每一筆都經過實際查證：
#   CPI  三方獨立來源完全吻合（investing.com 個別指標頁的歷史紀錄、
#        usinflationcalculator、macroornoise），且 8/12 發布 7 月數據這點
#        另經網路搜尋佐證。
#   PPI  1~8 月取自 investing.com 歷史紀錄，9~12 月取自 BLS 官方時程。
#        注意 PPI 不是「CPI 隔日」的固定規律（3 月為 3/18、9 月反而早於
#        CPI 一天），所以不能用規則推導，必須逐筆查表。
#   PCE  取自 investing.com 歷史紀錄；目前僅確認到 8/26，9~12 月官方尚未
#        公布確定日，故先不列（寧可少列，也不放推估日期誤導看盤）。
#   FOMC 取自 federalreserve.gov 官方 FOMC 行事曆，並與該站
#        json/calendar.json 交叉比對一致。列的是「決策公布日」＝兩天會期的
#        第二天（利率結果與聲明在當天下午公布，才是行情發動點）。
#
# 官方時程來源（更新時請至此查證，勿憑印象填寫）：
#   CPI/PPI  https://www.bls.gov/schedule/news_release/
#   PCE/GDP  https://www.bea.gov/news/schedule
#   FOMC     https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
_MACRO_SCHEDULE_YEAR: int | None = 2026
_US_MACRO_SCHEDULE: dict[str, list[str]] = {
    "CPI 消費者物價指數": [
        "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10",
        "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
        "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10",
    ],
    "PPI 生產者物價指數": [
        "2026-01-14", "2026-01-30", "2026-02-27", "2026-03-18",
        "2026-04-14", "2026-05-13", "2026-06-11", "2026-07-15",
        "2026-08-13", "2026-09-10", "2026-10-15", "2026-11-13",
        "2026-12-15",
    ],
    # 僅列已確認的場次；9~12 月官方未公布前刻意留白。
    "PCE 物價指數（Fed 偏好指標）": [
        "2026-01-22", "2026-02-20", "2026-03-13", "2026-04-09",
        "2026-04-30", "2026-05-28", "2026-06-25", "2026-07-30",
        "2026-08-26",
    ],
    "FOMC 利率決策": [
        "2026-01-28", "2026-04-29", "2026-07-29", "2026-10-28",
    ],
    # 這四場同時發布經濟預測摘要(SEP)與點陣圖，市場關注度高於一般場次。
    "FOMC 利率決策（含經濟預測 SEP）": [
        "2026-03-18", "2026-06-17", "2026-09-16", "2026-12-09",
    ],
    # 消費占美國 GDP 約七成。2026 上半年發布日相當不規則（3/6、4/1、4/21），
    # 無法用「每月中旬」之類的規則推導，只能逐筆查表。9 月以後待官方公布。
    "零售銷售": [
        "2026-01-14", "2026-02-10", "2026-03-06", "2026-04-01",
        "2026-04-21", "2026-05-14", "2026-06-17", "2026-07-16",
        "2026-08-14",
    ],
}

# 內建表中，各指標已確認到哪個月份為止。用於在畫面上誠實揭露涵蓋範圍——
# 若不標示，使用者看到某週沒有零售銷售，會誤以為那週真的沒有這項數據，
# 而不是「官方還沒公布、我們也還沒補」。
_SCHEDULE_COVERAGE: dict[str, str] = {
    "CPI 消費者物價指數": "2026 全年",
    "PPI 生產者物價指數": "2026 全年",
    "FOMC 利率決策": "2026 全年",
    "PCE 物價指數（Fed 偏好指標）": "2026 年 8 月",
    "零售銷售": "2026 年 8 月",
}


def macro_schedule_ready(year: int) -> bool:
    """內建總經時程表是否涵蓋該年度。未涵蓋時呼叫端應顯示提示，而不是
    默默少列事件、讓使用者以為本週沒有重要數據。"""
    return _MACRO_SCHEDULE_YEAR == year and bool(_US_MACRO_SCHEDULE)


# ---------- 每週彙整 ----------

def get_week_events(start: dt.date, end: dt.date) -> list[dict]:
    """`start`~`end`（含）區間內的非財報事件，依日期排序。

    每筆為 {date, category, name, note}。category 用於畫面分組：
    「經濟數據」「市場結構」。計算型事件（四巫日／月選到期／初領失業金／
    非農）永遠會有；內建表型事件（CPI/PPI/PCE/FOMC/休市）僅在
    macro_schedule_ready() 為真時才會出現。
    """
    events: list[dict] = []
    # FOMC 會議紀要：決策日後三週公布，揭露委員討論細節，常引發二次行情。
    # 這是 Fed 的固定慣例（非推測），故由決策日推算而不另建表。
    _minutes = {
        dt.date.fromisoformat(d) + dt.timedelta(days=21)
        for key in ("FOMC 利率決策", "FOMC 利率決策（含經濟預測 SEP）")
        for d in _US_MACRO_SCHEDULE.get(key, [])
    }

    day = start
    while day <= end:
        # 市場結構：休市 / 提早收盤 / 選擇權到期
        if day.isoformat() in _US_MARKET_HOLIDAYS:
            events.append({
                "date": day, "category": "市場結構",
                "name": f"美股休市（{_US_MARKET_HOLIDAYS[day.isoformat()]}）",
                "note": "當日不交易",
            })
        if day.isoformat() in _US_EARLY_CLOSES:
            events.append({
                "date": day, "category": "市場結構",
                "name": _US_EARLY_CLOSES[day.isoformat()],
                "note": "量能通常極淡",
            })
        if day == opex_date(day.year, day.month):
            if day.month in (3, 6, 9, 12):
                events.append({
                    "date": day, "category": "市場結構", "name": "四巫日",
                    "note": "四種合約季度同日結算，量能與尾盤波動通常放大",
                })
            else:
                events.append({
                    "date": day, "category": "市場結構", "name": "月選擇權到期日",
                    "note": "月度結算日，量能略增",
                })

        # 經濟數據（依固定慣例推算；官方偶有調整，故標示為慣例日）
        if _is_trading_day(day) and day.weekday() == _calendar.THURSDAY:
            events.append({
                "date": day, "category": "經濟數據", "name": "初領失業金人數",
                "note": "每週四公布，勞動市場即時指標",
            })
        if day == _first_friday(day.year, day.month):
            events.append({
                "date": day, "category": "經濟數據", "name": "非農就業報告 (NFP)",
                "note": "慣例為每月第一個週五（官方偶有調整）",
            })
            adp = day - dt.timedelta(days=2)  # NFP 前的週三
            if start <= adp <= end:
                events.append({
                    "date": adp, "category": "經濟數據", "name": "ADP 民間就業",
                    "note": "非農前哨站，慣例為非農前的週三",
                })
        if day == _nth_business_day(day.year, day.month, 1):
            events.append({
                "date": day, "category": "經濟數據", "name": "ISM 製造業 PMI",
                "note": "慣例為每月第一個營業日",
            })
        if day == _nth_business_day(day.year, day.month, 3):
            events.append({
                "date": day, "category": "經濟數據", "name": "ISM 服務業 PMI",
                "note": "慣例為每月第三個營業日；服務業占美國經濟比重最高",
            })
        if day == _last_weekday_of_month(day.year, day.month, _calendar.TUESDAY):
            events.append({
                "date": day, "category": "經濟數據", "name": "消費者信心指數（諮商會）",
                "note": "慣例為每月最後一個週二",
            })
        if day in _minutes:
            events.append({
                "date": day, "category": "經濟數據", "name": "FOMC 會議紀要",
                "note": "決策日後三週公布，揭露委員討論細節",
            })
        day += dt.timedelta(days=1)

    # 經濟數據（內建表型）
    if macro_schedule_ready(start.year):
        for name, dates in _US_MACRO_SCHEDULE.items():
            for ds in dates:
                d = dt.date.fromisoformat(ds)
                if start <= d <= end:
                    category = "市場結構" if "休市" in name else "經濟數據"
                    events.append({"date": d, "category": category,
                                   "name": name, "note": ""})

    events.sort(key=lambda e: (e["date"], e["category"], e["name"]))
    return events


# ---------- 財報行事曆 ----------

_EARNINGS_TIME_LABEL = {
    "time-pre-market": "盤前",
    "time-after-hours": "盤後",
    "time-not-supplied": "未定",
}


def _parse_market_cap(raw: str | None) -> float:
    """Nasdaq 回傳的市值是 "$340,714,770,239" 字串；轉為 float 供排序。
    缺值/格式異常回 0.0（排到最後），不讓單一髒資料中斷整份清單。"""
    if not raw:
        return 0.0
    try:
        return float(str(raw).replace("$", "").replace(",", "").strip())
    except ValueError:
        return 0.0


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_earnings_week(start: dt.date, end: dt.date, top_n: int = 10) -> list[dict]:
    """`start`~`end` 期間公布財報的公司，依市值由大到小取前 `top_n`。

    Nasdaq 的行事曆 API 一次只能查一天，所以逐日查詢（一週約 5 次）；任何
    單日失敗（網路異常、當天無資料）只跳過該日，不影響其他日期——避免一天
    出問題就整個區塊空白。

    每筆：{date, symbol, name, time_label, market_cap, eps_forecast}。
    """
    rows: list[dict] = []
    day = start
    while day <= end:
        url = _NASDAQ_EARNINGS_URL.format(date=day.isoformat())
        try:
            req = urllib.request.Request(url, headers=_NASDAQ_HEADERS)
            payload = json.loads(urllib.request.urlopen(req, timeout=12).read())
            for r in (payload.get("data") or {}).get("rows") or []:
                if not r.get("symbol"):
                    continue
                rows.append({
                    "date": day,
                    "symbol": r["symbol"],
                    "name": (r.get("name") or "").strip(),
                    "time_label": _EARNINGS_TIME_LABEL.get(r.get("time"), "未定"),
                    "market_cap": _parse_market_cap(r.get("marketCap")),
                    "eps_forecast": (r.get("epsForecast") or "").strip(),
                })
        except Exception:
            pass  # 單日失敗不影響其他日期
        day += dt.timedelta(days=1)

    rows.sort(key=lambda r: r["market_cap"], reverse=True)
    return rows[:top_n]


def week_range(today: dt.date | None = None) -> tuple[dt.date, dt.date]:
    """本週的週一～週五（美股交易週）。

    週六/週日查詢時回傳**下一週**：該週的交易日都已結束，使用者週末在看的
    必然是「接下來要注意什麼」，回傳已過完的一週沒有意義。
    """
    today = today or dt.date.today()
    if today.weekday() >= _calendar.SATURDAY:  # 週末 → 推進到下週一
        monday = today + dt.timedelta(days=7 - today.weekday())
    else:
        monday = today - dt.timedelta(days=today.weekday())
    return monday, monday + dt.timedelta(days=4)


# ---------- 台股行事曆 ----------

# 證交所官方「市場開休市日期」——含國定假日休市、農曆年封關/開紅盤日。
# queryYear 用民國年（西元−1911）。這支有官方資料可抓，所以台股休市不像
# 美股總經時程那樣需要人工維護內建表。
_TWSE_HOLIDAY_URL = (
    "https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule"
    "?response=json&queryYear={roc}"
)
# 與 universe.py 同樣的理由：部分 Windows 環境對證交所憑證鏈驗證失敗，
# 這是唯讀的公開市場資料，沒有敏感內容。
_TW_SSL_CTX = ssl.create_default_context()
_TW_SSL_CTX.check_hostname = False
_TW_SSL_CTX.verify_mode = ssl.CERT_NONE

# 名稱含這些字樣的是「交易日提示」（封關前最後一天、年後開紅盤），不是休市。
_TW_TRADING_DAY_MARKERS = ("開始交易", "最後交易")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_tw_market_calendar(year: int) -> dict[str, tuple[str, str]]:
    """證交所公告的該年度開休市日期：{ISO 日期: (名稱, 類別)}。

    類別為「休市」或「交易日」——後者是農曆年封關/開紅盤這種**有交易**的
    提示日，不能當成休市處理（例如 2026-02-11 是春節前最後交易日）。
    任何失敗回 {}，呼叫端只會少列事件、不會壞掉。
    """
    try:
        url = _TWSE_HOLIDAY_URL.format(roc=year - 1911)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (stock-market-trends-app)"})
        payload = json.loads(urllib.request.urlopen(req, timeout=15, context=_TW_SSL_CTX).read())
    except Exception:
        return {}
    out: dict[str, tuple[str, str]] = {}
    for row in payload.get("data") or []:
        if len(row) < 2:
            continue
        day, name = str(row[0]).strip(), str(row[1]).strip()
        kind = "交易日" if any(m in name for m in _TW_TRADING_DAY_MARKERS) else "休市"
        out[day] = (name, kind)
    return out


def is_tw_market_holiday(d: dt.date) -> bool:
    """該日台股是否休市（不含一般週末）。"""
    entry = get_tw_market_calendar(d.year).get(d.isoformat())
    return bool(entry) and entry[1] == "休市"


def _is_tw_trading_day(d: dt.date) -> bool:
    return d.weekday() < _calendar.SATURDAY and not is_tw_market_holiday(d)


def taiex_settlement_date(year: int, month: int) -> dt.date:
    """台指期（月契約）結算日：該月第三個星期三，遇休市順延至下一個交易日。

    結算日當天現貨常出現結算相關的買賣壓與尾盤波動，是台股月度行事曆上最
    需要注意的一天，故單獨列出。
    """
    wednesdays = [
        dt.date(year, month, d)
        for d in _calendar.Calendar().itermonthdays(year, month)
        if d and dt.date(year, month, d).weekday() == _calendar.WEDNESDAY
    ]
    d = wednesdays[2]
    while not _is_tw_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def _monthly_revenue_deadline(year: int, month: int) -> dt.date:
    """月營收公布截止日：依規定為每月 10 日前公布上月營收，遇假日順延。
    營收是台股最頻繁的基本面事件，截止日前後常有個股表態。"""
    d = dt.date(year, month, 10)
    while not _is_tw_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def get_tw_week_events(start: dt.date, end: dt.date) -> list[dict]:
    """`start`~`end`（含）之間的台股行事曆事件，格式同 get_week_events。"""
    events: list[dict] = []
    cal = get_tw_market_calendar(start.year)
    if end.year != start.year:  # 跨年度的那一週
        cal = {**cal, **get_tw_market_calendar(end.year)}

    day = start
    while day <= end:
        entry = cal.get(day.isoformat())
        if entry:
            name, kind = entry
            events.append({
                "date": day, "category": "市場結構",
                "name": f"台股休市（{name}）" if kind == "休市" else name,
                "note": "當日不交易" if kind == "休市" else "有交易，農曆年前後的關鍵交易日",
            })
        if day == taiex_settlement_date(day.year, day.month):
            events.append({
                "date": day, "category": "市場結構", "name": "台指期結算日",
                "note": "月契約結算（第三個週三，遇休市順延），現貨尾盤波動常放大",
            })
        if day == _monthly_revenue_deadline(day.year, day.month):
            events.append({
                "date": day, "category": "基本面", "name": "上市櫃月營收公布截止",
                "note": "依規定每月 10 日前公布上月營收（遇假日順延）",
            })
        day += dt.timedelta(days=1)

    events.sort(key=lambda e: (e["date"], e["category"], e["name"]))
    return events
