import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('NFLX.csv')
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Convert Date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Handle missing values if any
df = df.fillna(method='ffill').fillna(method='bfill')

# Enhanced feature engineering
def create_technical_features(df):
    """Create comprehensive technical indicators and features"""
    df = df.copy()
    
    # Price-based features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
    
    # Exponential moving averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    # Volatility features
    df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
    
    # Momentum indicators
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volume features
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    # Lag features (past prices)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    
    # Time-based features
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    
    return df

# Create features
print("\nCreating technical features...")
df_features = create_technical_features(df)

# Drop rows with NaN values (from rolling calculations)
df_features = df_features.dropna()
print(f"Dataset shape after feature engineering: {df_features.shape}")

# Prepare features and target
def prepare_data(df):
    """Prepare feature matrix and target variable"""
    # Target variable
    target = df['Close'].values
    
    # Feature columns (exclude target and original OHLCV)
    feature_cols = [col for col in df.columns if col not in ['Close', 'Open', 'High', 'Low', 'Adj Close']]
    features = df[feature_cols].values
    
    return features, target, feature_cols

X, y, feature_names = prepare_data(df_features)
print(f"\nTotal samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {feature_names[:10]}... (showing first 10)")

# Time-series split: Use last 20% for validation
validation_size = int(len(X) * 0.2)
X_train = X[:-validation_size]
y_train = y[:-validation_size]
X_val = X[-validation_size:]
y_val = y[-validation_size:]

print(f"\nTraining samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")

# Use RobustScaler (better for outliers than StandardScaler)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Build Gradient Boosting model (generally better than Random Forest for time series)
print("\nTraining Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42,
    verbose=0
)

model.fit(X_train_scaled, y_train)
print("Training completed!")

# Make predictions
train_predictions = model.predict(X_train_scaled)
val_predictions = model.predict(X_val_scaled)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
train_mae = mean_absolute_error(y_train, train_predictions)
val_mae = mean_absolute_error(y_val, val_predictions)
train_r2 = r2_score(y_train, train_predictions)
val_r2 = r2_score(y_val, val_predictions)

# Calculate MAPE (Mean Absolute Percentage Error)
train_mape = np.mean(np.abs((y_train - train_predictions) / y_train)) * 100
val_mape = np.mean(np.abs((y_val - val_predictions) / y_val)) * 100

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Training RMSE: ${train_rmse:.2f}")
print(f"Validation RMSE: ${val_rmse:.2f}")
print(f"Training MAE: ${train_mae:.2f}")
print(f"Validation MAE: ${val_mae:.2f}")
print(f"Training MAPE: {train_mape:.2f}%")
print(f"Validation MAPE: {val_mape:.2f}%")
print(f"Training R² Score: {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"Validation R² Score: {val_r2:.4f} ({val_r2*100:.2f}%)")
print("="*60)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Retrain on all data for final model
print("\nRetraining on complete dataset for final predictions...")
X_all_scaled = scaler.fit_transform(X)
model.fit(X_all_scaled, y)
all_predictions = model.predict(X_all_scaled)
all_r2 = r2_score(y, all_predictions)
all_rmse = np.sqrt(mean_squared_error(y, all_predictions))
print(f"Complete dataset R² Score: {all_r2:.4f} ({all_r2*100:.2f}%)")
print(f"Complete dataset RMSE: ${all_rmse:.2f}")

# Future prediction (simplified - using last known features)
print("\nGenerating future predictions...")
last_features = X[-1].reshape(1, -1)
last_features_scaled = scaler.transform(last_features)
future_predictions = []

# Predict next 30 days (note: predictions become less reliable further out)
for i in range(30):
    next_pred = model.predict(last_features_scaled)[0]
    future_predictions.append(next_pred)
    # For simplicity, we'll use the same features (in practice, you'd update them)

# Create future dates (business days only)
last_date = df_features.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

print(f"\nNext 30 Business Days Predictions:")
for i, (date, price) in enumerate(zip(future_dates[:10], future_predictions[:10])):
    print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
print("...")
print(f"Note: Future predictions have uncertainty and should be used cautiously.")

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: Top Feature Importance
top_features = feature_importance.head(15)
axes[0, 0].barh(range(len(top_features)), top_features['importance'].values, color='steelblue', alpha=0.7)
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features['feature'].values, fontsize=8)
axes[0, 0].set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Importance')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Plot 2: Complete Dataset Predictions
all_dates = df_features.index
axes[0, 1].plot(all_dates, y, label='Actual Price', color='blue', alpha=0.6, linewidth=1.5)
axes[0, 1].plot(all_dates, all_predictions, label='Predicted Price', color='red', alpha=0.6, linewidth=1.5)
axes[0, 1].set_title(f'Complete Dataset Predictions (R²={all_r2:.4f}, RMSE=${all_rmse:.2f})', 
                      fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Stock Price ($)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Validation Set Performance
val_dates = df_features.index[-validation_size:]
axes[0, 2].plot(val_dates, y_val, label='Actual Price', color='blue', alpha=0.7, linewidth=2, marker='o', markersize=3)
axes[0, 2].plot(val_dates, val_predictions, label='Predicted Price', color='red', alpha=0.7, linewidth=2, marker='s', markersize=3)
axes[0, 2].set_title(f'Validation Set - R²={val_r2:.4f}, MAPE={val_mape:.2f}%', 
                      fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Date')
axes[0, 2].set_ylabel('Stock Price ($)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].tick_params(axis='x', rotation=45)

# Plot 4: Prediction Error Distribution
train_errors = y_train - train_predictions
val_errors = y_val - val_predictions
axes[1, 0].hist(train_errors, bins=50, alpha=0.6, label='Training Errors', color='blue', edgecolor='black')
axes[1, 0].hist(val_errors, bins=30, alpha=0.6, label='Validation Errors', color='red', edgecolor='black')
axes[1, 0].axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero Error')
axes[1, 0].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Prediction Error ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Actual vs Predicted Scatter
axes[1, 1].scatter(y_train, train_predictions, alpha=0.3, s=10, label='Training', color='blue')
axes[1, 1].scatter(y_val, val_predictions, alpha=0.6, s=20, label='Validation', color='red')
min_val = min(y.min(), all_predictions.min())
max_val = max(y.max(), all_predictions.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Actual Price ($)')
axes[1, 1].set_ylabel('Predicted Price ($)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Future Predictions
recent_history = 60
axes[1, 2].plot(df_features.index[-recent_history:], df_features['Close'].values[-recent_history:], 
                label='Historical Price', color='blue', linewidth=2)
axes[1, 2].plot(future_dates, future_predictions, label='Future Predictions (30 Days)', 
                color='green', linewidth=2, linestyle='--', marker='o', markersize=4)
axes[1, 2].axvline(x=df_features.index[-1], color='red', linestyle=':', linewidth=2, label='Today')
# Add confidence interval (simplified)
std_dev = val_rmse
axes[1, 2].fill_between(future_dates, 
                         np.array(future_predictions) - std_dev, 
                         np.array(future_predictions) + std_dev, 
                         alpha=0.2, color='green', label=f'±1 RMSE (${std_dev:.2f})')
axes[1, 2].set_title('Next 30 Days Price Prediction', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Date')
axes[1, 2].set_ylabel('Stock Price ($)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('netflix_stock_prediction.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'netflix_stock_prediction.png'")
plt.show()

print("\n" + "="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"✓ Model: Gradient Boosting Regressor")
print(f"✓ Features: {X.shape[1]} technical indicators")
print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Validation R²: {val_r2:.4f}")
print(f"✓ Validation RMSE: ${val_rmse:.2f}")
print(f"✓ Validation MAPE: {val_mape:.2f}%")
print("="*60)
