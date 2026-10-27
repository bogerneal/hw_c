"""Lasso lag selection and daily autoregressive power forecasting."""
from pathlib import Path
import argparse
import json
import kagglehub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = np.abs(actual - predicted)
    return {
        'MSE (kW²)': float(mean_squared_error(actual, predicted)),
        'RMSE (kW)': float(np.sqrt(mean_squared_error(actual, predicted))),
        'MAE (kW)': float(mean_absolute_error(actual, predicted)),
        'R²': float(r2_score(actual, predicted)),
        'MAPE (%)': float(np.mean(error / np.maximum(np.abs(actual), 1e-8)) * 100),
        'sMAPE (%)': float(np.mean(2 * error / np.maximum(np.abs(actual) + np.abs(predicted), 1e-8)) * 100),
        '誤差在 ±10% 內 (%)': float(np.mean(error <= .1 * np.abs(actual)) * 100),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    source = args.data
    cached = Path(__file__).resolve().parent / '.tools' / 'data' / 'household_power_consumption.txt'
    if source is None and cached.is_file():
        source = cached
    if source is None:
        directory = Path(kagglehub.dataset_download('uciml/electric-power-consumption-data-set'))
        source = next(directory.rglob('household_power_consumption.txt'))
    print(f'Reading {source}', flush=True)
    raw = pd.read_csv(source, sep=';', na_values=['?'], usecols=['Date', 'Time', 'Global_active_power'])
    times = pd.to_datetime(raw.Date + ' ' + raw.Time, format='%d/%m/%Y %H:%M:%S')
    minute = pd.Series(raw.Global_active_power.to_numpy(), index=times).sort_index()
    daily = minute.resample('D').agg(['mean', 'count'])
    observed = daily['mean'].where(daily['count'] >= 1296)
    observed = observed.loc[observed.first_valid_index():]
    y = observed.ffill().rename('power_kw')
    split = int(len(y) * .8)
    train = y.iloc[:split]
    features = pd.concat({f'lag_{k}': y.shift(k) for k in range(1, 31)}, axis=1)
    search = GridSearchCV(make_pipeline(StandardScaler(), Lasso(max_iter=50000)),
                          {'lasso__alpha': np.logspace(-4, 0, 50)},
                          cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error', n_jobs=1)
    search.fit(features.iloc[30:split], train.iloc[30:])
    coefs = search.best_estimator_.named_steps['lasso'].coef_
    lags = (np.flatnonzero(np.abs(coefs) > 1e-8) + 1).tolist()
    if not lags:
        raise RuntimeError('Lasso selected no lags; review CV before fitting an AR model.')
    print(f'Selected lags: {lags}', flush=True)
    model = AutoReg(train, lags=lags, trend='c', hold_back=30).fit()
    design = np.column_stack([np.ones(len(y) - split), features.iloc[split:][[f'lag_{k}' for k in lags]]])
    pred = design @ np.asarray(model.params)
    variance = model.sigma2 + np.einsum('ij,jk,ik->i', design, np.asarray(model.cov_params()), design)
    margin = norm.ppf(.975) * np.sqrt(variance)
    test = pd.DataFrame({'actual_kw': observed.iloc[split:], 'predicted_kw': pred,
                         'lower_95_kw': pred - margin, 'upper_95_kw': pred + margin,
                         'naive_1_kw': y.shift(1).iloc[split:], 'naive_7_kw': y.shift(7).iloc[split:]})
    scored = test.dropna(subset=['actual_kw'])
    scores = {name: metrics(scored.actual_kw, scored[column]) for name, column in
              [('Lasso + AR', 'predicted_kw'), ('前一天基準', 'naive_1_kw'), ('前一週同日基準', 'naive_7_kw')]}
    scores['Lasso + AR']['95% 區間覆蓋率 (%)'] = float(((scored.actual_kw >= scored.lower_95_kw) & (scored.actual_kw <= scored.upper_95_kw)).mean() * 100)
    scores['Lasso + AR']['平均區間寬度 (kW)'] = float((scored.upper_95_kw - scored.lower_95_kw).mean())
    full_model = AutoReg(y, lags=lags, trend='c', hold_back=30).fit()
    future = full_model.get_prediction(start=len(y), end=len(y) + 29).summary_frame(alpha=.05)
    future.columns = ['predicted_kw', 'prediction_se_kw', 'lower_95_kw', 'upper_95_kw']
    test.to_csv(out / 'test_predictions.csv', index_label='date')
    future.to_csv(out / 'future_predictions.csv', index_label='date')
    daily.assign(valid_day=daily['count'] >= 1296).to_csv(out / 'daily_data.csv', index_label='date')
    pd.DataFrame({'lag_days': range(1, 31), 'lasso_coefficient': coefs, 'selected': np.abs(coefs) > 1e-8}).to_csv(out / 'lasso_features.csv', index=False)
    pd.DataFrame(search.cv_results_).to_csv(out / 'lasso_cv.csv', index=False)
    (out / 'metrics.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
    for frame, history, filename, title in [
        (test, test.actual_kw, 'test_forecast.png', 'Held-out one-day-ahead forecasts'),
        (future, observed.iloc[-90:], 'future_forecast.png', 'Recursive 30-day forecast beyond dataset')]:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(history.index, history, color='#1f2937', linewidth=1, label='Observed')
        ax.plot(frame.index, frame.predicted_kw, color='#2563eb', linewidth=1, label='Lasso + AR')
        ax.fill_between(frame.index, frame.lower_95_kw, frame.upper_95_kw, color='#60a5fa', alpha=.3, label='95% prediction interval')
        ax.set(title=title, ylabel='Daily mean active power (kW)', xlabel='Date')
        ax.legend(loc='upper left', ncol=3)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
    rows = '\n'.join('| ' + key + ' | ' + ' | '.join(f'{value[key]:.4f}' if key in value else '—' for value in scores.values()) + ' |' for key in scores['Lasso + AR'])
    feature_rows = '\n'.join(f'| {k} | {coefs[k-1]:.6f} | {model.params.iloc[i+1]:.6f} |' for i, k in enumerate(lags))
    future_rows = '\n'.join(f'| {date:%Y-%m-%d} | {row.predicted_kw:.4f} | {row.lower_95_kw:.4f} | {row.upper_95_kw:.4f} |' for date, row in future.iterrows())
    better = scores['Lasso + AR']['RMSE (kW)'] < min(scores['前一天基準']['RMSE (kW)'], scores['前一週同日基準']['RMSE (kW)'])
    report = f'''# 家庭用電預測：Lasso 特徵選擇 + AR

## 資料與預測設定

來源：[Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)、[UCI 原始說明](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)。
原始 {len(raw):,} 筆，時間 {minute.index.min()}～{minute.index.max()}，有功功率缺失 {minute.isna().sum():,} 筆。

未指定目標與頻率，因此採用 Global_active_power 的每日平均（kW），不是每日用電量（kWh）。每日至少 1,296 個有效分鐘（完整日 90%）才視為有效觀測，其餘日以之前最後有效日補值作為輸入，不使用未來資訊；首個有效日前移除。
保留 {len(y)} 日，其中 {observed.isna().sum()} 日補值。補值日參與訓練，但不作為測試真值；這可能使訓練變異偏低。

- 訓練：{train.index.min():%Y-%m-%d}～{train.index.max():%Y-%m-%d}，{len(train)} 日，扣除 lag 後 {len(train)-30} 列。
- 測試：{test.index.min():%Y-%m-%d}～{test.index.max():%Y-%m-%d}，{len(test)} 日，實際評分 {len(scored)} 日。
- 依時間約 80% / 20% 切分，不隨機打散。

## Lasso 特徵選擇與 AR 模型

純 AR 使用目標本身的歷史值，候選特徵是 lag 1～30 天，不使用同日電流或分表讀值作為已知未來資料。
Lasso 使用 L1 正則化，訓練期內採 5 折擴展視窗交叉驗證，依 MSE 從 50 個 0.0001～1 的對數等距 alpha 中選擇。StandardScaler 在每折訓練部分獨立擬合，避免驗證資訊洩漏。

- 最佳 alpha：{search.best_params_['lasso__alpha']:.8f}。
- CV 平均 MSE：{-search.best_score_:.6f} kW²。
- 選中 {len(lags)} 個 lag：{lags}。
- 選完後以 statsmodels AutoReg 做無正則化最小平方法估計；模型為 ŷₜ = 截距 + Σ φₖ yₜ₋ₖ，截距 {model.params.iloc[0]:.6f}。

Lasso 係數對應標準化特徵；最終 AR 係數對應原始 kW 尺度。

| 落後天數 | Lasso 係數 | AR 係數 |
|---|---|---|
{feature_rows}

## 評估指標

測試為逐日一步預測：固定訓練所得係數，每天使用之前日期已觀測值或因果補值。不是在測試期起點一次預測整段未來。基準模型使用相同有效日期評分。

| 指標 | Lasso + AR | 前一天基準 | 前一週同日基準 |
|---|---|---|---|
{rows}

MSE、RMSE、MAE 越小越好；R² 越大越好，負值代表比測試集均值參考更差。
MAPE = 平均 |誤差/真值| × 100%；sMAPE = 平均 2|誤差|/(|真值|+|預測|) × 100%。
回歸沒有分類 accuracy，本報告將「準確率」定義為 |誤差| ≤ |真值| × 10% 的比例，不使用 100%−MAPE。
本次 AR 的 RMSE **{'優於' if better else '未優於'}**兩個基準中較佳者；Lasso 不保證預測一定改善。

## 測試圖與 95% 預測區間

![測試期預測](test_forecast.png)

區間針對新觀測值，並非平均值的信賴區間。單步區間使用 ŷ ± 1.96 √(殘差變異 + xᵀ Cov(β) x)，包含係數估計與觀測誤差，未納入特徵選擇的不確定性。
假設殘差近似常態、同方差且無自相關；實際資料未必符合，請以上表實測覆蓋率判讀，不保證達到 95%。未將負下界截斷。

## 資料結束後 30 天預測

保留訓練期選中的 lag，用所有歷史資料重新擬合 AR，遞迴預測 30 天。日期由資料終點推進，不代表目前日期的用電預測；無未來真值，不能計算未來準確率。
此處 AutoReg 多步區間包含創新誤差的遞迴傳播，但視係數為固定，不包含係數估計與特徵選擇不確定性，與前述單步區間方法不同。

![未來 30 天預測](future_forecast.png)

| 日期 | 預測 (kW) | 95% 下界 | 95% 上界 |
|---|---|---|---|
{future_rows}

## 重現與輸出檔

在專案目錄執行 `.venv/Scripts/python.exe 0.py`。其他 Python 環境先執行 `python -m pip install -r requirements.txt`，再執行 `python 0.py`。
優先使用專案 `.tools/data/household_power_consumption.txt` 快取；無快取時下載 Kaggle 資料，亦可指定 `--data 本機household_power_consumption.txt路徑`。

- [每日資料](daily_data.csv)
- [Lasso 係數](lasso_features.csv)、[交叉驗證結果](lasso_cv.csv)
- [測試預測](test_predictions.csv)、[未來預測](future_predictions.csv)、[評估 JSON](metrics.json)

限制：每日聚合無法保留日內尖峰；僅考慮 30 天內歷史值，未明確建模年季節性、天氣或假日；評估為單一時間切分，指標只描述此測試期間。
'''
    (out / 'prediction_report.md').write_text(report, encoding='utf-8')
    print(json.dumps(scores, ensure_ascii=False, indent=2), flush=True)
    print(f'Report: {out / "prediction_report.md"}', flush=True)


if __name__ == '__main__':
    main()
