# Power Lab｜家庭用電預測互動網站

參考 [bogerneal/1014 的 Startup Lab](https://github.com/bogerneal/1014) 的模型探索流程，另行設計家庭用電預測介面。程式為原創 HTML/CSS/JavaScript，不依賴 CDN 或前端建置框架。

## 本機使用

在 `hw_c` 根目錄執行：

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory power_forecast_site/dist
```

開啟 http://localhost:8765 。也可直接開啟 `dist/index.html`，所有預測計算在瀏覽器內完成。

## 互動功能

- 切換 Lasso + AR 與 Lasso + 多元線性回歸。
- 日期範圍篩選、重算範圍內指標、顯示 95% 預測區間與另一模型。
- 圖表日期游標、未來 1～30 天預測與 CSV 下載。
- 修改 Lasso 選中特徵的歷史日平均，立即計算下一天功率與新觀測預測區間。
- 兩種模型與前一天／前一週基準比較、係數與 Lasso 選擇資訊。
- 每日資料按年份、品質篩選及分頁；下載 CSV、Markdown 報告及情境 JSON。
- 手機版排版、鍵盤分頁操作、數字與日期驗證。

## 重建資料

先在上層完成兩種 Python 分析，再執行：

```powershell
.\.venv\Scripts\python.exe power_forecast_site/export_web_data.py
```

資料匯出器讀取既有結果，重建同一特徵集合的訓練／全資料模型，逐筆驗證其預測與原 CSV 一致，輸出 `dist/data.js` 與 `dist/data.json`，並複製分析報告與圖檔。依賴上層 `requirements.txt`。

網站為靜態應用，沒有 Python API，也沒有即時重新執行 Lasso 或上傳訓練資料。情境輸入只改變既有模型的線性預測；原始測試成績不會因使用者輸入而變動。

## 統計定義

- 目標：每日平均有功功率 kW。資料期間截止 2010-11-26，並非當前即時電表資料。
- 測試：2010-02-11～2010-11-26，289 日中 276 日有有效真值。日期篩選不重訓；不足兩日或真值零變異時 R² 顯示「—」。
- 情境模型：使用全部歷史資料重訓，係數與測試期模型不同；情境輸入不應被解讀為因果介入。
- 情境 AR 區間：常態近似，含殘差變異與係數共變異；情境 MLR 區間：OLS 的 t 分布新觀測區間。均未校正 Lasso 選擇。
- 未來 AR 使用遞迴誤差傳播（視係數固定）；未來 MLR 是 30 個不同步長的直接模型。逐日區間不是整條路徑同時覆蓋區間。
- 「誤差在 ±10% 內」為比例，不是分類 accuracy 或 100%−MAPE。

本網站使用新的 Sites 身分；參考 GitHub 專案與其原有網站均未修改。

## 驗證

已在 Chrome headless 通過 29 項互動與數值檢查，包括兩模型與 Python 的預測／區間一致性、日期篩選、缺失日、單日 R²、防呆、比較基準、資料分頁與鍵盤分頁操作。CSV、Markdown、JSON 三種下載成功；390px 手機寬度四個分頁均無整頁水平溢出，沒有瀏覽器執行錯誤。

本機伺服器執行時可用 `node qa.mjs <絕對輸出目錄>` 重跑，需 Node.js 22 以上與預設安裝路徑的 Google Chrome。QA 截圖與下載檔應放在網站原始碼目錄之外。
