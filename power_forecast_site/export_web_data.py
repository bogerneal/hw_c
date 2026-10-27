"""Rebuild browser assets from the two verified forecasting analyses.

Run from hw_c: .venv/Scripts/python.exe power_forecast_site/export_web_data.py
Requires existing results from both 0.py scripts. Does not modify those results.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import norm, t

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'dist'
OUT.mkdir(exist_ok=True)
raw = pd.read_csv(ROOT / 'multiple_linear_regression/results/daily_data.csv', header=[0, 1], index_col=0, parse_dates=True)
daily, observed = raw['filled'].asfreq('D'), raw['observed'].asfreq('D')
y = daily.Global_active_power
split = int(len(y) * .8)


def records(frame):
    frame = frame.copy()
    frame.index = frame.index.strftime('%Y-%m-%d')
    return json.loads(frame.rename_axis('date').reset_index().to_json(orient='records', double_precision=12))


models = {}
for key, directory in [('ar', ROOT / 'results'), ('mlr', ROOT / 'multiple_linear_regression/results')]:
    selection = pd.read_csv(directory / 'lasso_features.csv')
    chosen = selection[selection.selected]
    if key == 'ar':
        lags = chosen.lag_days.astype(int).tolist()
        names = [f'Global_active_power_lag_{lag}' for lag in lags]
        X = pd.DataFrame({name: y.shift(lag) for name, lag in zip(names, lags)})
        train = AutoReg(y.iloc[:split], lags=lags, hold_back=30, trend='c').fit()
        full = AutoReg(y, lags=lags, hold_back=30, trend='c').fit()
        residual_variance = full.sigma2
        critical = norm.ppf(.975)
        lasso = chosen.lasso_standardized_coefficient.tolist() if 'lasso_standardized_coefficient' in chosen else chosen.lasso_coefficient.tolist()
    else:
        names = chosen.feature.tolist()
        X = pd.DataFrame({name: daily[name.rsplit('_lag_', 1)[0]].shift(int(name.rsplit('_lag_', 1)[1])) for name in names})
        design = sm.add_constant(X, has_constant='add')
        train = sm.OLS(y.iloc[30:split], design.iloc[30:split]).fit()
        full = sm.OLS(y.iloc[30:], design.iloc[30:]).fit()
        residual_variance = full.mse_resid
        critical = t.ppf(.975, full.df_resid)
        lasso = chosen.lasso_coefficient.tolist()
    test = pd.read_csv(directory / 'test_predictions.csv', index_col=0, parse_dates=True)
    future = pd.read_csv(directory / 'future_predictions.csv', index_col=0, parse_dates=True)
    test_design = np.column_stack([np.ones(len(test)), X.iloc[split:].to_numpy()])
    np.testing.assert_allclose(test_design @ np.asarray(train.params), test.predicted_kw, atol=1e-10)
    inputs = [float(daily[name.rsplit('_lag_', 1)[0]].iloc[-int(name.rsplit('_lag_', 1)[1])]) for name in names]
    vec = np.array([1, *inputs])
    predicted = float(vec @ np.asarray(full.params))
    np.testing.assert_allclose(predicted, future.predicted_kw.iloc[0], atol=1e-10)
    covariance = np.asarray(full.cov_params())
    margin = float(critical * np.sqrt(residual_variance + vec @ covariance @ vec))
    cv = pd.read_csv(directory / 'lasso_cv.csv')
    best = cv.loc[cv.mean_test_score.idxmax()]
    features = []
    for i, name in enumerate(names):
        column, lag = name.rsplit('_lag_', 1)
        features.append({'name': name, 'column': column, 'lag': int(lag), 'default': inputs[i],
                         'min': float(X.iloc[30:split, i].min()), 'max': float(X.iloc[30:split, i].max()),
                         'trainCoefficient': float(train.params.iloc[i+1]), 'lassoCoefficient': float(lasso[i])})
    models[key] = {'name': 'Lasso + AR' if key == 'ar' else 'Lasso + MLR', 'features': features,
                    'candidateCount': len(selection), 'alpha': float(best.param_lasso__alpha),
                    'lassoCvMse': float(-best.mean_test_score),
                    'trainIntercept': float(train.params.iloc[0]),
                    'params': np.asarray(full.params).tolist(), 'covariance': covariance.tolist(),
                    'variance': float(residual_variance), 'critical': float(critical),
                    'scenarioReference': {'mean': predicted, 'lower': predicted-margin, 'upper': predicted+margin},
                    'metrics': json.loads((directory / 'metrics.json').read_text(encoding='utf-8')),
                    'test': records(test), 'future': records(future)}
    target = OUT / 'reports' / key
    target.mkdir(parents=True, exist_ok=True)
    for filename in ['prediction_report.md', 'test_predictions.csv', 'future_predictions.csv', 'metrics.json',
                     'test_forecast.png', 'future_forecast.png', 'daily_data.csv', 'lasso_features.csv', 'lasso_cv.csv']:
        shutil.copyfile(directory / filename, target / filename)
    if key == 'mlr':
        for filename in ['mlr_coefficients.csv', 'future_model_coefficients.csv']:
            shutil.copyfile(directory / filename, target / filename)
    shutil.copyfile(directory.parent / '0.py', target / 'model.py')

history = observed.copy()
history['valid_minutes'] = raw['valid_minutes'].Global_active_power
history['split'] = ['train' if i < split else 'test' for i in range(len(history))]
payload = {'models': models, 'history': records(history), 'meta': {
    'dataset': 'Individual Household Electric Power Consumption', 'rows': 2075259,
    'dailyRows': len(y), 'trainRows': split-30, 'testDays': len(y)-split, 'validTestDays': 276,
    'start': str(y.index[0].date()), 'end': str(y.index[-1].date()),
    'testStart': str(y.index[split].date()), 'forecastDate': str(future.index[0].date()),
    'lastDayImputed': bool(pd.isna(observed.Global_active_power.iloc[-1])),
    'source': 'https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set'}}
encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
(OUT / 'data.js').write_text('window.POWER_DATA = ' + encoded + ';\n', encoding='utf-8')
(OUT / 'data.json').write_text(encoded, encoding='utf-8')
description = (ROOT / 'dataset_description.md').read_text(encoding='utf-8')
description = description.replace('](.tools/data/household_power_consumption.txt)', '](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)')
description = description.replace('](0.py)', '](reports/ar/model.py)').replace('](multiple_linear_regression/0.py)', '](reports/mlr/model.py)')
description = description.replace('](results/prediction_report.md)', '](reports/ar/prediction_report.md)').replace('](multiple_linear_regression/results/prediction_report.md)', '](reports/mlr/prediction_report.md)')
(OUT / 'dataset_description.md').write_text(description, encoding='utf-8')
history.to_csv(OUT / 'daily_data.csv', index_label='date')
print('Exported two models; predictions verified against both original CSV results.', flush=True)
print(f'Browser payload: {len(encoded):,} characters', flush=True)
