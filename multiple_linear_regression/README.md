# Lasso + Multiple Linear Regression

以 Lasso 選擇各用電欄位的歷史特徵，再以 OLS 擬合多元線性回歸。
此資料夾為獨立版本，上層 `0.py` 與 AR 報告保持不變。

在 `hw_c` 目錄執行：

```powershell
.\.venv\Scripts\python.exe multiple_linear_regression\0.py
```

輸出：[預測報告](results/prediction_report.md)，包含評估指標、測試預測圖及未來 30 天預測圖（含 95% 預測區間）。

預設共用上層 `.tools/data` 資料快取。可使用 `--data` 指定本機原始檔、`--output` 指定輸出目錄。獨立環境先執行 `python -m pip install -r requirements.txt`。
