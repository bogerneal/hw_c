# Power Lab｜家庭用電預測

以 **Lasso 特徵選擇**搭配 **AR 自迴歸**及**多元線性回歸（MLR）**，分析家庭用電資料並提供繁體中文互動網站。

公開網站：<https://bogerneal.github.io/hw_c/>，已成功部署至 GitHub Pages，無須登入。

GitHub 儲存庫：<https://github.com/bogerneal/hw_c>。

## 互動網站

- 切換 AR / MLR，檢視測試指標與兩模型比較。
- 篩選日期、查看預測曲線與 95% 預測區間。
- 修改歷史特徵，立即計算下一天的假設情境預測。
- 查看資料結束後 1～30 天預測，下載 CSV、Markdown 與 JSON。
- 支援每日資料篩選、分頁、手機版與鍵盤操作。

網站為靜態 HTML/CSS/JavaScript，無須伺服器端 Python，也沒有即時重新訓練。情境輸入只影響預測，不改變原始測試分數。

## 專案檔案

| 路徑 | 用途 |
|---|---|
| [0.py](0.py) | Lasso + AR 分析 |
| [results/prediction_report.md](results/prediction_report.md) | AR 指標、圖表與預測報告 |
| [multiple_linear_regression/0.py](multiple_linear_regression/0.py) | Lasso + MLR 分析 |
| [MLR 報告](multiple_linear_regression/results/prediction_report.md) | 多元線性回歸結果 |
| [dataset_description.md](dataset_description.md) | 資料集欄位、統計及限制 |
| [power_forecast_site/dist](power_forecast_site/dist) | 可直接發布的網站與資料 |
| [export_web_data.py](power_forecast_site/export_web_data.py) | 依 Python 結果重建網站資料 |
| [.github/workflows/pages.yml](.github/workflows/pages.yml) | GitHub Pages 自動部署 |

## 本機執行

建立 Python 3.12 虛擬環境並安裝依賴：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe 0.py
.\.venv\Scripts\python.exe multiple_linear_regression/0.py
.\.venv\Scripts\python.exe power_forecast_site/export_web_data.py
```

分析首次執行會由 Kaggle 下載資料；原始約 133 MB 的檔案、虛擬環境及下載工具不納入儲存庫。兩個分析程式也支援 `--data` 指定本機資料檔。

預覽網站：

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory power_forecast_site/dist
```

開啟 <http://localhost:8765>，或直接以瀏覽器開啟 `power_forecast_site/dist/index.html`。

## GitHub Pages 部署

儲存庫設為公開，Settings → Pages → Source 選擇 **GitHub Actions**。
推送 `power_forecast_site/dist/` 或部署工作流程至 `main` 時，Actions 會發布網站，也可手動執行工作流程。
只修改 Python 分析時，須先重跑分析與 `export_web_data.py`，再提交更新後的 `dist` 資料。
網站使用相對路徑，適用 `/hw_c/` 這類 GitHub Pages 專案路徑。

## 模型與資料

資料來自 [Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) 的 Individual Household Electric Power Consumption，原始來源為 [UCI](https://doi.org/10.24432/C58K54)，資料集授權 CC BY 4.0。
作者：Georges Hebrail、Alice Berard。用電資料為單一法國家庭的歷史記錄，截止 2010-11-26，並非目前即時電表資訊。

目標為每日平均有功功率（kW），不是每日累計電能（kWh）。測試期間採固定係數的一步預測，排除缺失真值，兩方法使用相同有效評分日期。情境模型以完整歷史重訓，與測試模型不同。

| 指標 | AR | MLR |
|---|---:|---:|
| MSE (kW²) | 0.05620 | 0.05586 |
| RMSE (kW) | 0.23707 | 0.23634 |
| MAE (kW) | 0.17568 | 0.17326 |
| R² | 0.39898 | 0.40267 |
| 誤差在 ±10% 內 (%) | 35.87 | 39.86 |

兩版候選特徵不同，不能將成績差異全部歸因於估計器。預測區間依賴誤差假設，未校正 Lasso 特徵選擇，也不保證適用其他家庭。

互動操作流程參考 [bogerneal/1014](https://github.com/bogerneal/1014)，本專案使用獨立實作。原有 Sites 私人網站保留，GitHub Pages 為另一個公開部署。
