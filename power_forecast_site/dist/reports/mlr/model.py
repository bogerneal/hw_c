"""Daily power forecasts using Lasso selection and multiple linear regression."""
from pathlib import Path
from datetime import timedelta
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = 'Global_active_power'
COLUMNS = [TARGET, 'Global_reactive_power', 'Voltage', 'Global_intensity',
           'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
LAGS = [1, 2, 3, 7, 14, 30]


def metrics(actual, predicted):
    a, p = np.asarray(actual), np.asarray(predicted)
    error = np.abs(a - p)
    return {
        'MSE (kW²)': float(mean_squared_error(a, p)),
        'RMSE (kW)': float(np.sqrt(mean_squared_error(a, p))),
        'MAE (kW)': float(mean_absolute_error(a, p)),
        'R²': float(r2_score(a, p)),
        'MAPE (%)': float(np.mean(error / np.maximum(np.abs(a), 1e-8)) * 100),
        'sMAPE (%)': float(np.mean(2 * error / np.maximum(np.abs(a) + np.abs(p), 1e-8)) * 100),
        '誤差在 ±10% 內 (%)': float(np.mean(error <= .1 * np.abs(a)) * 100),
    }


def prediction_frame(model, design):
    result = model.get_prediction(design).summary_frame(alpha=.05)
    return result[['mean', 'obs_ci_lower', 'obs_ci_upper']].rename(
        columns={'mean': 'predicted_kw', 'obs_ci_lower': 'lower_95_kw', 'obs_ci_upper': 'upper_95_kw'})


def plot(frame, history, path, title):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(history.index, history, color='#1f2937', linewidth=1, label='Observed')
    ax.plot(frame.index, frame.predicted_kw, color='#059669', linewidth=1, label='Lasso + Multiple Linear Regression')
    ax.fill_between(frame.index, frame.lower_95_kw, frame.upper_95_kw,
                    color='#34d399', alpha=.3, label='95% prediction interval')
    ax.set(title=title, ylabel='Daily mean active power (kW)', xlabel='Date')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    source = args.data or root / '.tools/data/household_power_consumption.txt'
    if args.data is None and not source.is_file():
        import kagglehub
        source = next(Path(kagglehub.dataset_download('uciml/electric-power-consumption-data-set')).rglob('household_power_consumption.txt'))
    print(f'Reading {source}', flush=True)
    raw = pd.read_csv(source, sep=';', na_values=['?'], usecols=['Date', 'Time'] + COLUMNS)
    times = pd.to_datetime(raw.Date + ' ' + raw.Time, format='%d/%m/%Y %H:%M:%S')
    minute = raw[COLUMNS].set_axis(times).sort_index()
    means, counts = minute.resample('D').mean(), minute.resample('D').count()
    observed_all = means.where(counts >= 1296)
    observed_all = observed_all.loc[observed_all[TARGET].first_valid_index():]
    daily = observed_all.ffill()
    observed, y = observed_all[TARGET], daily[TARGET]
    split = int(len(y) * .8)
    # Each target date t uses only measurements from t-1 or earlier.
    features = pd.DataFrame({f'{column}_lag_{lag}': daily[column].shift(lag)
                             for column in COLUMNS for lag in LAGS})
    X_train, y_train = features.iloc[30:split], y.iloc[30:split]
    if X_train.isna().any().any():
        raise ValueError('Insufficient past observations for input features.')
    search = GridSearchCV(make_pipeline(StandardScaler(), Lasso(max_iter=50000)),
                          {'lasso__alpha': np.logspace(-4, 0, 50)},
                          cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error', n_jobs=1)
    search.fit(X_train, y_train)
    coefs = search.best_estimator_.named_steps['lasso'].coef_
    selected = features.columns[np.abs(coefs) > 1e-8].tolist()
    if len(selected) < 2:
        raise RuntimeError('Fewer than two features selected; review selection before calling this MLR.')
    print(f'Selected {len(selected)} of {len(features.columns)} features', flush=True)
    design = sm.add_constant(features[selected], has_constant='add')
    model = sm.OLS(y_train, design.iloc[30:split]).fit()
    test = prediction_frame(model, design.iloc[split:])
    test['actual_kw'] = observed.iloc[split:]
    test['naive_1_kw'], test['naive_7_kw'] = y.shift(1).iloc[split:], y.shift(7).iloc[split:]
    scored = test.dropna(subset=['actual_kw'])
    scores = {name: metrics(scored.actual_kw, scored[column]) for name, column in
              [('Lasso + MLR', 'predicted_kw'), ('前一天基準', 'naive_1_kw'), ('前一週同日基準', 'naive_7_kw')]}
    scores['Lasso + MLR']['95% 區間覆蓋率 (%)'] = float(((scored.actual_kw >= scored.lower_95_kw) & (scored.actual_kw <= scored.upper_95_kw)).mean() * 100)
    scores['Lasso + MLR']['平均區間寬度 (kW)'] = float((scored.upper_95_kw - scored.lower_95_kw).mean())
    # Direct horizon models avoid assuming unknown future voltage/current values.
    last_date = y.index[-1].to_pydatetime()
    next_date = last_date + timedelta(days=1)
    next_inputs = pd.DataFrame({f'{column}_lag_{lag}': [daily[column].iloc[-lag]]
                                for column in COLUMNS for lag in LAGS}, index=[next_date])
    next_design = sm.add_constant(next_inputs[selected], has_constant='add')
    future_rows, horizon_params = [], []
    for horizon in range(1, 31):
        # X at t contains information through t-1; label is y[t+horizon-1].
        end = len(y) - horizon + 1
        labels = y.shift(-(horizon - 1)).iloc[30:end]
        fitted = sm.OLS(labels, design.iloc[30:end]).fit()
        row = prediction_frame(fitted, next_design).iloc[0].to_dict()
        row['date'] = last_date + timedelta(days=horizon)
        future_rows.append(row)
        horizon_params.append({'horizon_days': horizon, **fitted.params.to_dict()})
    future = pd.DataFrame(future_rows).set_index('date')
    test.to_csv(out / 'test_predictions.csv', index_label='date')
    future.to_csv(out / 'future_predictions.csv', index_label='date')
    pd.DataFrame(horizon_params).to_csv(out / 'future_model_coefficients.csv', index=False)
    pd.concat({'observed': observed_all, 'filled': daily, 'valid_minutes': counts.loc[daily.index]}, axis=1).to_csv(out / 'daily_data.csv', index_label='date')
    selection = pd.DataFrame({'feature': features.columns, 'lasso_coefficient': coefs,
                              'selected': np.abs(coefs) > 1e-8})
    selection.to_csv(out / 'lasso_features.csv', index=False)
    model.params.rename('coefficient').to_csv(out / 'mlr_coefficients.csv', index_label='feature')
    pd.DataFrame(search.cv_results_).to_csv(out / 'lasso_cv.csv', index=False)
    (out / 'metrics.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
    plot(test, test.actual_kw, out / 'test_forecast.png', 'Multiple Linear Regression: held-out one-day-ahead forecasts')
    plot(future, observed.iloc[-90:], out / 'future_forecast.png', 'Multiple Linear Regression: direct forecasts for the next 30 days')
    metric_rows = '\n'.join('| ' + key + ' | ' + ' | '.join(f'{s[key]:.4f}' if key in s else '—' for s in scores.values()) + ' |' for key in scores['Lasso + MLR'])
    coefficient_rows = '\n'.join(f'| {name} | {coefs[features.columns.get_loc(name)]:.6f} | {model.params[name]:.6f} |' for name in selected)
    forecast_rows = '\n'.join(f'| {date:%Y-%m-%d} | {row.predicted_kw:.4f} | {row.lower_95_kw:.4f} | {row.upper_95_kw:.4f} |' for date, row in future.iterrows())
    report = f'''# 家庭用電預測：Lasso + 多元線性回歸（MLR）

## 資料與切分

來源：[Kaggle 家庭用電資料](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)。共 {len(raw):,} 筆分鐘資料；{minute.index.min()}～{minute.index.max()}。
目標為 Global_active_power 每日平均（kW），不是每日用電量（kWh）。每欄每天須有至少 1,296 筆有效分鐘（90%），不足者以過去最後有效日向前填補，不使用未來值。首個有效目標日前移除。
保留 {len(y)} 日，目標補值 {observed.isna().sum()} 日；補值目標參與訓練，但不當作測試真值。

- 訓練：{y.index[0]:%Y-%m-%d}～{y.index[split-1]:%Y-%m-%d}，扣除 30 日暖機後 {len(y_train)} 列。
- 測試：{test.index[0]:%Y-%m-%d}～{test.index[-1]:%Y-%m-%d}，{len(test)} 日，其中 {len(scored)} 日可評分。
- 時間順序約 80% / 20%，與原 AR 版本相同日期切分，未隨機打散。

## Lasso 特徵選擇

候選欄位為有功功率、無功功率、電壓、電流及三個分表（{', '.join(COLUMNS)}）。對每欄建立 {LAGS} 天落後值，共 {len(features.columns)} 個特徵。每日聚合均取平均，分表仍是平均每分鐘讀值（Wh），不是每日累計。
只使用預測日前已知的歷史資料，不使用待預測當日的電流或電壓。目標歷史值也可作為其中一組解釋變數。
訓練期內以 5 折擴展視窗交叉驗證、MSE 選 alpha；50 個候選值為 0.0001～1 的對數等距值。每折獨立擬合 StandardScaler，再進行 Lasso（L1）。

- 最佳 alpha：{search.best_params_['lasso__alpha']:.8f}。
- CV 平均 MSE：{-search.best_score_:.6f} kW²（此為 Lasso 選擇階段，非最終 MLR 的 CV 分數）。
- 選出 {len(selected)} 個特徵。

## 多元線性回歸

使用選中特徵，以 statsmodels OLS 無正則化最小平方法重新估計多元線性回歸：ŷ = β₀ + β₁x₁ + … + βₚxₚ；截距為 {model.params['const']:.6f}。
此版本以多種用電變數作為解釋變數，未使用 AutoReg。Lasso 係數對應標準化特徵，MLR 係數對應各欄原始尺度，不能直接拿係數大小比較重要性。

| 特徵 | Lasso 係數 | MLR 係數 |
|---|---|---|
{coefficient_rows}

## 測試指標

固定訓練係數，逐日預測下一日；每次可使用之前日期已揭露的實測值或因果補值。這不是在測試期起點一次預測整段。基準使用相同有效評分日期。

| 指標 | Lasso + MLR | 前一天基準 | 前一週同日基準 |
|---|---|---|---|
{metric_rows}

MSE、RMSE、MAE 越小越好；R² 越高越好，負值代表比測試集均值參考更差。MAPE = 平均 |誤差/真值| × 100%；sMAPE = 平均 2|誤差|/(|真值|+|預測|) × 100%。
回歸沒有分類準確率，此處明確以「|誤差| ≤ |真值| × 10%」的比例表示準確率，不採 100%−MAPE。

![測試期預測與區間](test_forecast.png)

95% 預測區間使用 OLS 的 obs_ci_lower / obs_ci_upper，形式為 ŷ ± t₀.₉₇₅,df √(殘差變異 + xᵀ Cov(β) x)。包含新觀測誤差與係數估計誤差，不是平均值信賴區間。假設誤差常態、同方差且獨立；未涵蓋 Lasso 特徵選擇的不確定性，時間序列可能違反假設，應搭配實測覆蓋率判讀。負下界未截斷。

## 資料結束後 30 天預測

凍結訓練期選中的特徵，用全部歷史資料為各個預測步長 h=1～30 分別擬合 MLR：以日期 t 之前的特徵預測 t+h−1 的有功功率。每個模型只使用已知標籤的歷史配對，全部從資料終點的同一組已知歷史特徵出發。
這是直接多步預測，不需要假設未來電壓、電流與分表值。未來各步長未另做回測，測試指標只代表一步預測；這 30 天沒有真值，不能計算準確率。
每個區間是該步長 OLS 的邊際 95% 預測區間，不是整條 30 日路徑同時有 95% 覆蓋。日期由資料終點推進，並非目前日期。

![未來 30 天預測](future_forecast.png)

| 日期 | 預測 (kW) | 95% 下界 | 95% 上界 |
|---|---|---|---|
{forecast_rows}

## 重現與檔案

在 hw_c 目錄執行：`.venv/Scripts/python.exe multiple_linear_regression/0.py`。
共用上層虛擬環境及資料快取；獨立安裝可使用本資料夾的 requirements.txt。可指定 `--data` 與 `--output`。

- [Lasso 特徵](lasso_features.csv)、[CV 結果](lasso_cv.csv)、[MLR 係數](mlr_coefficients.csv)
- [測試預測](test_predictions.csv)、[未來預測](future_predictions.csv)、[各步長係數](future_model_coefficients.csv)
- [指標 JSON](metrics.json)、[每日資料](daily_data.csv)

限制：僅單一時間切分；每日聚合不保留日內尖峰，缺失補值可能低估變異；電流與有功功率等欄位可能共線，係數不代表因果關係。與 AR 版本相比，候選特徵也已改變，結果差異不能只歸因於估計器。
'''
    (out / 'prediction_report.md').write_text(report, encoding='utf-8')
    print(json.dumps(scores, ensure_ascii=False, indent=2), flush=True)
    print(f'Report: {out / "prediction_report.md"}', flush=True)


if __name__ == '__main__':
    main()
