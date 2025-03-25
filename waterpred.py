import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly,plot_cross_validation_metric
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import plotly.offline as py
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("christ_university_water_dataset_6months.csv")
df['Date'] = pd.to_datetime(df['Date'])


if df.isnull().sum().sum() > 0:
    print(f"Found {df.isnull().sum().sum()} missing values in the dataset")
    df = df.interpolate(method='time')

Q1 = df['Days Water Lasted'].quantile(0.25)
Q3 = df['Days Water Lasted'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Days Water Lasted'] < lower_bound) | (df['Days Water Lasted'] > upper_bound)]
if len(outliers) > 0:
    print(f"Found {len(outliers)} outliers in the dataset")
    print(outliers)

    for idx in outliers.index:
        date = df.loc[idx, 'Date']
        # Get 7 days before and after the outlier
        nearby = df[(df['Date'] >= date - pd.Timedelta(days=7)) &
                    (df['Date'] <= date + pd.Timedelta(days=7)) &
                    (~df.index.isin([idx]))]
        df.loc[idx, 'Days Water Lasted'] = nearby['Days Water Lasted'].median()
        print(f"Replaced outlier on {date.date()} with median value: {nearby['Days Water Lasted'].median()}")

df = df.sort_values('Date').set_index('Date')

train = df[df.index < '2025-03-01']
test = df[df.index >= '2025-03-01']

train_prophet = train.reset_index()[['Date', 'Days Water Lasted']].rename(
    columns={'Date': 'ds', 'Days Water Lasted': 'y'})
test_prophet = test.reset_index()[['Date', 'Days Water Lasted']].rename(
    columns={'Date': 'ds', 'Days Water Lasted': 'y'})

best_mae = float('inf')
best_params = {}

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative'],
    'weekly_seasonality': [True, False]
}

print("Performing hyperparameter tuning...")
for changepoint_prior_scale in param_grid['changepoint_prior_scale']:
    for seasonality_prior_scale in param_grid['seasonality_prior_scale']:
        for seasonality_mode in param_grid['seasonality_mode']:
            for weekly_seasonality in param_grid['weekly_seasonality']:
                model = Prophet(
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    seasonality_mode=seasonality_mode,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=False
                )

                model.add_country_holidays(country_name='IN')

                if not weekly_seasonality:
                    model.add_seasonality(name='weekly', period=7, fourier_order=3)

                model.fit(train_prophet)

                future = model.make_future_dataframe(periods=len(test))
                forecast = model.predict(future)

                # Extract test predictions
                preds = forecast[forecast['ds'].isin(test_prophet['ds'])][['ds', 'yhat']]
                preds = preds.merge(test_prophet, on='ds', how='inner')

                if len(preds) > 0:
                    mae = mean_absolute_error(preds['y'], preds['yhat'])

                    if mae < best_mae:
                        best_mae = mae
                        best_params = {
                            'changepoint_prior_scale': changepoint_prior_scale,
                            'seasonality_prior_scale': seasonality_prior_scale,
                            'seasonality_mode': seasonality_mode,
                            'weekly_seasonality': weekly_seasonality
                        }
                        print(f"New best MAE: {best_mae:.4f} with params: {best_params}")

print(f"\nBest parameters found: {best_params}")
print(f"Best MAE: {best_mae:.4f}")

final_model = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    seasonality_mode=best_params['seasonality_mode'],
    weekly_seasonality=best_params['weekly_seasonality'],
    daily_seasonality=False
)

final_model.add_country_holidays(country_name='IN')

if not best_params['weekly_seasonality']:
    final_model.add_seasonality(name='weekly', period=7, fourier_order=3)

final_model.fit(train_prophet)

print("\nPerforming cross-validation...")
df_cv = cross_validation(final_model, initial='120 days', period='7 days', horizon='30 days')
df_p = performance_metrics(df_cv)
print("\nCross-validation performance metrics:")
print(df_p[['horizon', 'mae', 'rmse', 'mape']].tail())

fig = plot_cross_validation_metric(df_cv, metric='mae')
plt.tight_layout()
plt.savefig("cross_validation_mae.png")
plt.show()

future = final_model.make_future_dataframe(periods=len(test))
forecast = final_model.predict(future)

test_predictions = forecast[forecast['ds'].isin(test_prophet['ds'])][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
test_predictions = test_predictions.merge(test_prophet, on='ds', how='inner')

mae = mean_absolute_error(test_predictions['y'], test_predictions['yhat'])
rmse = np.sqrt(mean_squared_error(test_predictions['y'], test_predictions['yhat']))
r2 = r2_score(test_predictions['y'], test_predictions['yhat'])
mape = np.mean(np.abs((test_predictions['y'] - test_predictions['yhat']) / test_predictions['y'])) * 100

print("\nTest Data Evaluation Metrics (March 2025):")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(test_predictions['ds'], test_predictions['y'], label='Actual', marker='o')
plt.plot(test_predictions['ds'], test_predictions['yhat'], label='Predicted', linestyle='--', marker='x')
plt.fill_between(test_predictions['ds'],
                 test_predictions['yhat_lower'],
                 test_predictions['yhat_upper'],
                 alpha=0.2,
                 label='95% Confidence Interval')
plt.title("Forecast vs Actuals (March 2025)")
plt.xlabel("Date")
plt.ylabel("Days Water Lasted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("forecast_vs_actuals.png")
plt.show()

fig = final_model.plot_components(forecast)
plt.tight_layout()
plt.savefig("prophet_components.png")
plt.show()

fig = plot_plotly(final_model, forecast)
py.plot(fig, filename='prophet_forecast.html', auto_open=True)

full_prophet = df.reset_index()[['Date', 'Days Water Lasted']].rename(columns={'Date': 'ds', 'Days Water Lasted': 'y'})
model_full = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    seasonality_mode=best_params['seasonality_mode'],
    weekly_seasonality=best_params['weekly_seasonality'],
    daily_seasonality=False
)
model_full.add_country_holidays(country_name='IN')

if not best_params['weekly_seasonality']:
    model_full.add_seasonality(name='weekly', period=7, fourier_order=3)

model_full.fit(full_prophet)

future_april = model_full.make_future_dataframe(periods=30, freq='D')
future_april = future_april[future_april['ds'] >= '2025-04-01']  # Filter to only include April
forecast_april = model_full.predict(future_april)

joblib.dump(model_full, "water_days_prophet_model.pkl")
forecast_april.to_csv("april_2025_water_forecast.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(forecast_april['ds'], forecast_april['yhat'], label='Forecast', color='blue')
plt.fill_between(forecast_april['ds'],
                 forecast_april['yhat_lower'],
                 forecast_april['yhat_upper'],
                 alpha=0.2,
                 color='blue',
                 label='95% Confidence Interval')
plt.title("April 2025 Water Usage Forecast")
plt.xlabel("Date")
plt.ylabel("Days Water Lasted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("april_2025_forecast.png")
plt.show()

print("\nApril 2025 Forecast Summary:")
print(f"Average predicted days water lasted: {forecast_april['yhat'].mean():.2f}")
print(
    f"Minimum predicted days: {forecast_april['yhat'].min():.2f} on {forecast_april.loc[forecast_april['yhat'].idxmin(), 'ds'].date()}")
print(
    f"Maximum predicted days: {forecast_april['yhat'].max():.2f} on {forecast_april.loc[forecast_april['yhat'].idxmax(), 'ds'].date()}")