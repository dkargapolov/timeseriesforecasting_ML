import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from timeforecastimng_ML.forecasting import *

# Generate synthetic time series data
np.random.seed(42)
date_rng = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
y_trend = np.linspace(50, 150, num=len(date_rng)) + np.random.normal(0, 10, size=len(date_rng))
y = y_trend + np.sin(np.arange(len(date_rng))) * 20

num_dates = list(range(1,len(date_rng)))

print(autocorelation(y_trend))
print(bGetting(y_trend, num_dates))

# Create a DataFrame
df = pd.DataFrame({'Date': date_rng, 'Value': y})
df.set_index('Date', inplace=True)

# Create time series features
def create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    #df['weekofyear'] = df.index.weekofyear
    return df

# Create features
df = create_features(df)

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Features and target
FEATURES = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', ]
TARGET = 'Value'

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# Train XGBoost model
reg = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')
reg.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = reg.predict(X_test)

# Calculate linear trendline
linear_trend_model = LinearRegression()
linear_trend_model.fit(np.arange(len(train)).reshape(-1, 1), train[TARGET])
linear_trendline = linear_trend_model.predict(np.arange(len(df)).reshape(-1, 1))

# Plot the actual data, XGBoost predictions, and linear trendline
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[TARGET], label='Actual Data')
plt.plot(test.index, y_pred, label='XGBoost Predictions', linestyle='--')
plt.plot(df.index, linear_trendline, label='Linear Trendline', linestyle='-.')
plt.legend()

# Add linear trendline formula to the plot
linear_coef, linear_intercept = linear_trend_model.coef_[0], linear_trend_model.intercept_
linear_formula = f'Linear Trendline: y = {linear_coef:.2f}x + {linear_intercept:.2f}'
plt.text(df.index[-1], linear_trendline[-1], linear_formula, verticalalignment='bottom', horizontalalignment='right')

plt.title('Time Series Forecasting with XGBoost and Linear Trendline')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

'''# Calculate RMSE for XGBoost predictions
rmse_xgboost = np.sqrt(mean_squared_error(test[TARGET], y_pred))
print(f'RMSE for XGBoost predictions: {rmse_xgboost:.2f}')
'''