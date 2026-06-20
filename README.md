# 美股／台股分析師看板

以資料分析師視角打造的美股、台股互動看板，使用 [Streamlit](https://streamlit.io/) + [yfinance](https://github.com/ranaroussi/yfinance) 免費資料。

## 功能

- **市場切換**：頁面最上方提供「美股／台股」切換鍵，切換後三個分頁的預設代號、可選股票範圍與「全部」選項皆會對應切換；台股範圍包含 ETF 及個股（清單為內建精選名單，代碼會自動補上 Yahoo Finance 所需的 `.TW` 後綴）。
- **價格、技術指標與基本面**（單一股票）：K 線圖（台股紅漲綠跌、美股綠漲紅跌，背景皆為白底）、SMA5/10/20 與布林通道（皆為輔助線、不額外填色，圖例置於圖表上方以放大繪圖區）、RSI、MACD、KD 隨機指標、成交量，與該股票的基本面財務（P/E、P/B、EPS、營收/盈餘成長率、ROE、淨利率、股息率），分頁內輸入股票代號（美股預設 AAPL，台股預設 2330）與時間範圍即可連動更新；可自行輸入「勝率 (%)」，系統會依過去 5 個交易日的歷史報酬率分布反推對應漲跌幅，估算建議買入價（逢低承接）與建議賣出價（目標停利），僅供參考非投資建議，頁面最下方附上該股票近 4 天（例如 6/17-6/20）的中文新聞標題（透過 Google News RSS 抓取）
- **多股比較、相關性與風險統計**：分頁內輸入逗號分隔的股票代號（美股預設 `AAPL, OKLO`，留空代表全部 S&P 500 成分股；台股預設 `2330, 0050`，留空代表全部台股觀察清單）與時間範圍，同時查看累積報酬比較、日報酬相關係數矩陣（熱力圖）、年化報酬率／年化波動率／Sharpe Ratio／最大回撤／VaR 與日報酬分布
- **買賣建議**：分頁內可設定時間範圍、勝率 (%) 與建議買賣標的數量 (Top N，可選 1/5/10/15，預設 5)。美股篩選範圍為「美股交易量前 30 大（觀察名單）」與「S&P 500 成分股」的聯集，台股篩選範圍為內建台股觀察清單（含 ETF 及個股），以基金經理人角度綜合期間報酬率、Sharpe Ratio（無風險利率採固定年化 4%）、價格趨勢、估值水準、近幾日中文新聞情緒（標題關鍵字判斷）計算評分，列出建議買入／賣出的前 N 項；買入／賣出價依設定勝率反推各標的歷史報酬率分布估算目標價，並列出對應的「獲利%」，表格各欄位皆可點選表頭排序，最後一欄附上「原因說明」標示該標的進入買入／賣出名單的主要驅動因子

## 安裝與執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

三個分頁的時間範圍、比較代號、勝率與建議買賣標的數量皆為各分頁內各自的欄位（彼此獨立）。美股模式下，「多股比較、相關性與風險統計」分頁的比較代號欄位留空則代表「全部」，會自動帶入 S&P 500 全部成分股（清單即時從 Wikipedia 抓取，若無網路則退回一份內建的知名成分股清單）；標的數量較多時，比較圖表會自動裁切前 30 檔以維持可讀性與效能，風險統計表則仍處理全部標的。台股模式下留空則代表內建台股觀察清單（含 ETF 及個股）。

## 部署到 Streamlit Community Cloud（免費，分享給他人）

部署後會得到一個公開網址（例如 `https://你的app名稱.streamlit.app`），任何人點連結即可使用，不需安裝任何東西。

1. 確認程式碼已推送到 GitHub（`app.py`、`requirements.txt`、`.streamlit/config.toml` 都在 repo 內）。
2. 前往 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 帳號登入並授權。
3. 點 **Create app** → **Deploy a public app from GitHub**。
4. 填入：
   - **Repository**：`你的帳號/0518test`
   - **Branch**：要部署的分支（例如 `main`）
   - **Main file path**：`app.py`
5. 點 **Deploy**，等待約 1～3 分鐘安裝相依套件並啟動。
6. 完成後即可分享網址。之後每次 push 到該分支，App 會自動重新部署為最新版。

> 本專案已內含 `.streamlit/config.toml`，部署時會自動套用主題與伺服器設定，無需額外調整。
> Community Cloud 的對外網路可正常連線 Yahoo Finance，因此線上版能抓取真實股價資料。

## 專案結構

```
app.py                      # Streamlit 主程式（頁面與圖表）
.streamlit/config.toml      # 主題與伺服器設定（部署時自動套用）
requirements.txt            # Python 相依套件
src/data_loader.py          # yfinance 資料存取（含快取）
src/technical.py            # 技術指標計算（SMA/EMA/RSI/MACD/Bollinger）
src/risk.py                 # 風險與統計指標計算
src/recommend.py            # 買賣建議綜合評分計算
src/news.py                 # 中文新聞抓取（Google News RSS）與關鍵字情緒分析
src/universe.py             # S&P 500/美股觀察名單與台股（個股+ETF）觀察清單
```
