import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_retail_sales():
    np.random.seed(42)
    start_date = datetime(2021, 1, 1)
    n_days = 365 * 3 + 1  # 3 full years (2021-2023)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    df = pd.DataFrame({'date': dates})
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Base Trend: steady YoY growth of retail business
    time_steps = np.arange(n_days)
    trend = 1200 + (time_steps * 0.85)  # grows from ~1200 to ~2100
    
    # Weekly Seasonality (higher Friday - Sunday)
    # Mon: 0.85, Tue: 0.90, Wed: 0.95, Thu: 1.00, Fri: 1.15, Sat: 1.35, Sun: 1.25
    weekly_factors = np.array([0.85, 0.90, 0.95, 1.00, 1.18, 1.38, 1.25])
    weekly_seasonality = np.array([weekly_factors[dow] for dow in df['day_of_week']])
    
    # Annual Seasonality: Back to school (Aug-Sep) and Q4 holiday surge (Nov-Dec)
    annual_seasonality = 1.0 + 0.15 * np.sin(2 * np.pi * df['day_of_year'] / 365.25 - 1.5)
    # Extra boost in November and December
    holiday_boost = np.where(df['month'].isin([11, 12]), 1.25, 1.0)
    
    # Promotional campaigns (periodic flash sales, Black Friday, Prime week)
    promotions = np.zeros(n_days)
    discounts = np.zeros(n_days)
    
    for i, dt in enumerate(dates):
        # Black Friday / Cyber Monday week in late Nov
        if dt.month == 11 and 23 <= dt.day <= 29:
            promotions[i] = 1
            discounts[i] = 0.30
        # Holiday rush late Dec
        elif dt.month == 12 and 18 <= dt.day <= 24:
            promotions[i] = 1
            discounts[i] = 0.25
        # Mid-year Summer Mega Sale in July
        elif dt.month == 7 and 10 <= dt.day <= 16:
            promotions[i] = 1
            discounts[i] = 0.20
        # Random monthly weekend flash promotions
        elif df['is_weekend'].iloc[i] == 1 and np.random.rand() < 0.12:
            promotions[i] = 1
            discounts[i] = 0.15
            
    df['promotional_event'] = promotions.astype(int)
    df['discount_pct'] = (discounts * 100).round(1)
    
    # Promotional multiplier
    promo_lift = np.where(df['promotional_event'] == 1, 1.0 + (df['discount_pct'] / 100.0) * 1.6, 1.0)
    
    # Noise
    noise = np.random.normal(0, 45, n_days)
    
    # Calculate final sales revenue
    raw_sales = (trend * weekly_seasonality * annual_seasonality * holiday_boost * promo_lift) + noise
    df['sales'] = np.maximum(100, np.round(raw_sales, 2))
    
    # Store traffic (correlated with sales & promotions)
    traffic_noise = np.random.normal(0, 150, n_days)
    df['store_traffic'] = np.maximum(200, (df['sales'] * 1.8 + traffic_noise).astype(int))
    
    # Export clean CSV
    csv_path = "C:/Users/adity/.gemini/antigravity/scratch/sales-forecasting/data/retail_sales_daily.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} records saved to {csv_path}")
    print(df[['date', 'sales', 'promotional_event', 'discount_pct', 'store_traffic']].head())
    return df

if __name__ == '__main__':
    generate_retail_sales()
