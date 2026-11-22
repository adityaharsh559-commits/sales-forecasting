import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "retail_sales_daily.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    return df

def test_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print("--- Augmented Dickey-Fuller (ADF) Test ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    is_stationary = result[1] < 0.05
    print(f"Series is {'Stationary' if is_stationary else 'Non-Stationary (Differencing required)'}\n")
    return result

def plot_overview(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['sales'], color='#2b5c8f', alpha=0.6, label='Daily Sales Revenue ($)')
    plt.plot(df.index, df['sales'].rolling(window=30).mean(), color='#e63946', linewidth=2.2, label='30-Day Moving Average (Trend)')
    
    # Highlight promotional events
    promo_days = df[df['promotional_event'] == 1]
    plt.scatter(promo_days.index, promo_days['sales'], color='#f4a261', s=25, alpha=0.8, label='Promotional Event / Sale', zorder=5)
    
    plt.title("Retail Daily Sales Revenue Over Time (2021 - 2023)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Sales ($)", fontsize=11)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_sales_trend_and_seasonality.png"), dpi=300)
    plt.close()
    print("Saved 01_sales_trend_and_seasonality.png")

def plot_decomposition(df):
    decomposition = seasonal_decompose(df['sales'], model='additive', period=7)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(decomposition.observed, color='#1d3557')
    axes[0].set_ylabel('Observed', fontweight='bold')
    axes[0].set_title('Time Series Additive Decomposition (Period = 7 Days)', fontsize=14, fontweight='bold', pad=10)
    
    axes[1].plot(decomposition.trend, color='#457b9d')
    axes[1].set_ylabel('Trend', fontweight='bold')
    
    axes[2].plot(decomposition.seasonal, color='#2a9d8f')
    axes[2].set_ylabel('Seasonal (7d)', fontweight='bold')
    
    axes[3].scatter(decomposition.resid.index, decomposition.resid, color='#e76f51', alpha=0.5, s=10)
    axes[3].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[3].set_ylabel('Residual', fontweight='bold')
    axes[3].set_xlabel('Date', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_time_series_decomposition.png"), dpi=300)
    plt.close()
    print("Saved 02_time_series_decomposition.png")

def plot_promotional_impact(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Promo vs Non-Promo Sales
    sns.boxplot(x='promotional_event', y='sales', data=df, ax=axes[0], palette=['#a8dadc', '#e63946'])
    axes[0].set_xticklabels(['Regular Days', 'Promotional Days'])
    axes[0].set_title('Sales Distribution: Regular vs. Promotional Events', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Sales ($)', fontsize=11)
    
    # Calculate lift
    mean_reg = df[df['promotional_event'] == 0]['sales'].mean()
    mean_promo = df[df['promotional_event'] == 1]['sales'].mean()
    lift_pct = ((mean_promo - mean_reg) / mean_reg) * 100
    axes[0].text(0.5, 0.90, f"Avg Promo Lift: +{lift_pct:.1f}%", transform=axes[0].transAxes,
                 ha='center', fontsize=11, fontweight='bold', color='#1d3557',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1faee', edgecolor='#457b9d'))
    
    # 2. Day of Week Seasonality
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_sales = df.groupby(df.index.day_of_week)['sales'].mean()
    sns.barplot(x=dow_labels, y=dow_sales.values, ax=axes[1], palette='Blues_d')
    axes[1].set_title('Average Daily Sales by Day of Week', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Day of Week', fontsize=11)
    axes[1].set_ylabel('Average Sales ($)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_promotional_impact_analysis.png"), dpi=300)
    plt.close()
    print("Saved 03_promotional_impact_analysis.png")

def create_ml_lag_features(df):
    data = df.copy()
    data['day_of_week'] = data.index.day_of_week
    data['day_of_month'] = data.index.day
    data['month'] = data.index.month
    
    # Lags
    for lag in [1, 2, 7, 14, 21, 28]:
        data[f'lag_{lag}'] = data['sales'].shift(lag)
        
    # Rolling averages
    data['rolling_mean_7'] = data['sales'].shift(1).rolling(7).mean()
    data['rolling_mean_30'] = data['sales'].shift(1).rolling(30).mean()
    data['rolling_std_7'] = data['sales'].shift(1).rolling(7).std()
    
    data = data.dropna()
    return data

def train_and_evaluate_models(df):
    # Train / Test Split: Hold out final 90 days for forecasting evaluation
    test_days = 90
    train_df = df.iloc[:-test_days]
    test_df = df.iloc[-test_days:]
    print(f"Train period: {train_df.index.min().date()} to {train_df.index.max().date()} ({len(train_df)} days)")
    print(f"Test period:  {test_df.index.min().date()} to {test_df.index.max().date()} ({len(test_df)} days)\n")
    
    results = {}
    
    # --- 1. Baseline Model (Seasonal Naive: 7-day lag persistence) ---
    baseline_pred = train_df['sales'].iloc[-7:].tolist()
    # tile across test set
    n_repeats = int(np.ceil(test_days / 7))
    baseline_forecast = (baseline_pred * n_repeats)[:test_days]
    results['Baseline (Seasonal Naive)'] = {
        'forecast': np.array(baseline_forecast),
        'ci_lower': None,
        'ci_upper': None
    }
    
    # --- 2. Holt-Winters Exponential Smoothing ---
    print("Fitting Holt-Winters Exponential Smoothing...")
    hw_model = ExponentialSmoothing(
        train_df['sales'],
        trend='add',
        seasonal='add',
        seasonal_periods=7,
        initialization_method="estimated"
    ).fit()
    hw_forecast = hw_model.forecast(test_days)
    results['Holt-Winters'] = {
        'forecast': hw_forecast.values,
        'ci_lower': None,
        'ci_upper': None
    }
    
    # --- 3. SARIMAX with Exogenous Promotional & Calendar Regressors ---
    print("Fitting SARIMAX(1, 1, 1)x(1, 1, 1, 7) with Exogenous Features...")
    exog_cols = ['promotional_event', 'discount_pct', 'is_weekend']
    train_exog = train_df[exog_cols]
    test_exog = test_df[exog_cols]
    
    sarimax_model = SARIMAX(
        train_df['sales'],
        exog=train_exog,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    
    sarimax_pred_res = sarimax_model.get_forecast(steps=test_days, exog=test_exog)
    sarimax_forecast = sarimax_pred_res.predicted_mean
    sarimax_ci = sarimax_pred_res.conf_int(alpha=0.05)
    
    results['SARIMAX (with Exog Promos)'] = {
        'forecast': sarimax_forecast.values,
        'ci_lower': sarimax_ci.iloc[:, 0].values,
        'ci_upper': sarimax_ci.iloc[:, 1].values
    }
    
    # --- 4. Random Forest Regressor with Engineered Features ---
    print("Fitting Random Forest Regressor (ML Benchmark)...")
    ml_data = create_ml_lag_features(df)
    feature_cols = [c for c in ml_data.columns if c not in ['sales', 'store_traffic']]
    
    X_train = ml_data.loc[ml_data.index.isin(train_df.index), feature_cols]
    y_train = ml_data.loc[ml_data.index.isin(train_df.index), 'sales']
    X_test = ml_data.loc[ml_data.index.isin(test_df.index), feature_cols]
    y_test_ml = ml_data.loc[ml_data.index.isin(test_df.index), 'sales']
    
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    
    # Match dates with test_df
    rf_aligned = pd.Series(index=X_test.index, data=rf_preds).reindex(test_df.index).bfill().ffill()
    results['Random Forest Regressor'] = {
        'forecast': rf_aligned.values,
        'ci_lower': None,
        'ci_upper': None
    }
    
    # --- Evaluation Metrics ---
    metrics = []
    y_true = test_df['sales'].values
    
    for name, data in results.items():
        y_pred = data['forecast']
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        metrics.append({
            'Model': name,
            'MAE ($)': round(mae, 2),
            'RMSE ($)': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'R-Squared': round(r2, 4)
        })
        
    metrics_df = pd.DataFrame(metrics).sort_values('RMSE ($)')
    print("\n" + "="*60)
    print("MODEL PERFORMANCE EVALUATION SUMMARY (TEST SET: 90 DAYS)")
    print("="*60)
    print(metrics_df.to_string(index=False))
    print("="*60 + "\n")
    
    metrics_df.to_csv(os.path.join(BASE_DIR, "data", "evaluation_metrics.csv"), index=False)
    
    # --- Plot Forecast vs Actual ---
    plt.figure(figsize=(15, 7))
    plt.plot(test_df.index, y_true, label='Actual Sales (Ground Truth)', color='#111827', linewidth=2.5, marker='o', markersize=3)
    
    colors = {
        'SARIMAX (with Exog Promos)': '#2563eb',
        'Holt-Winters': '#10b981',
        'Random Forest Regressor': '#f59e0b',
        'Baseline (Seasonal Naive)': '#9ca3af'
    }
    
    for name, data in results.items():
        ls = '--' if 'Baseline' in name else '-'
        lw = 1.8 if 'SARIMAX' in name else 1.4
        plt.plot(test_df.index, data['forecast'], label=f"{name}", color=colors.get(name, 'purple'), linestyle=ls, linewidth=lw)
        
    # Plot confidence interval for SARIMAX
    sarimax_data = results['SARIMAX (with Exog Promos)']
    if sarimax_data['ci_lower'] is not None:
        plt.fill_between(test_df.index, sarimax_data['ci_lower'], sarimax_data['ci_upper'],
                         color='#2563eb', alpha=0.15, label='SARIMAX 95% Confidence Interval')
        
    plt.title("Forecast vs Actual Sales Comparison (90-Day Holdout Evaluation)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Sales ($)", fontsize=11)
    plt.legend(frameon=True, loc='upper left', facecolor='white', framealpha=0.95, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_forecast_vs_actual_comparison.png"), dpi=300)
    plt.close()
    print("Saved 04_forecast_vs_actual_comparison.png")
    
    return metrics_df

def run_pipeline():
    print("Starting Sales Forecasting Analytics Pipeline...\n")
    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} rows, columns: {list(df.columns)}")
    
    test_stationarity(df['sales'])
    plot_overview(df)
    plot_decomposition(df)
    plot_promotional_impact(df)
    train_and_evaluate_models(df)
    print("\nSales Forecasting Pipeline Completed Successfully!")

if __name__ == '__main__':
    run_pipeline()
