import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Netflix Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #E50914;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📈 Netflix Stock Prediction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Model Configuration")

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv('NFLX.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

# Feature engineering function
@st.cache_data
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
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    
    # Time-based features
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    
    return df

# Prepare data function
def prepare_data(df):
    """Prepare feature matrix and target variable"""
    target = df['Close'].values
    feature_cols = [col for col in df.columns if col not in ['Close', 'Open', 'High', 'Low', 'Adj Close']]
    features = df[feature_cols].values
    return features, target, feature_cols

# Train model function
@st.cache_resource
def train_model(X_train, y_train, n_estimators, learning_rate, max_depth):
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    return model

# Load and process data
with st.spinner("Loading data..."):
    df = load_data()
    df_features = create_technical_features(df)
    df_features = df_features.dropna()

# Sidebar parameters
st.sidebar.subheader("Model Hyperparameters")
n_estimators = st.sidebar.slider("Number of Estimators", 100, 500, 300, 50)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.2, 0.05, 0.01)
max_depth = st.sidebar.slider("Max Depth", 3, 10, 5, 1)
validation_split = st.sidebar.slider("Validation Split (%)", 10, 30, 20, 5) / 100

st.sidebar.subheader("Prediction Settings")
prediction_days = st.sidebar.slider("Future Prediction Days", 7, 60, 30, 1)

# Prepare data
X, y, feature_names = prepare_data(df_features)
validation_size = int(len(X) * validation_split)
X_train = X[:-validation_size]
y_train = y[:-validation_size]
X_val = X[-validation_size:]
y_val = y[-validation_size:]

# Scale data
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train model
with st.spinner("Training model..."):
    model = train_model(X_train_scaled, y_train, n_estimators, learning_rate, max_depth)

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
train_mape = np.mean(np.abs((y_train - train_predictions) / y_train)) * 100
val_mape = np.mean(np.abs((y_val - val_predictions) / y_val)) * 100

# Main dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Predictions", "🔍 Feature Analysis", "📉 Technical Indicators"])

with tab1:
    st.header("Model Performance Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Validation R² Score", f"{val_r2:.4f}", f"{(val_r2-train_r2):.4f}")
    with col2:
        st.metric("Validation RMSE", f"${val_rmse:.2f}", f"${(val_rmse-train_rmse):.2f}")
    with col3:
        st.metric("Validation MAE", f"${val_mae:.2f}", f"${(val_mae-train_mae):.2f}")
    with col4:
        st.metric("Validation MAPE", f"{val_mape:.2f}%", f"{(val_mape-train_mape):.2f}%")
    
    st.markdown("---")
    
    # Performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training vs Validation Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE ($)', 'MAE ($)', 'MAPE (%)', 'R² Score'],
            'Training': [train_rmse, train_mae, train_mape, train_r2],
            'Validation': [val_rmse, val_mae, val_mape, val_r2]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Dataset Information")
        info_df = pd.DataFrame({
            'Property': ['Total Samples', 'Training Samples', 'Validation Samples', 'Features', 'Date Range'],
            'Value': [
                len(X),
                len(X_train),
                len(X_val),
                X.shape[1],
                f"{df_features.index[0].strftime('%Y-%m-%d')} to {df_features.index[-1].strftime('%Y-%m-%d')}"
            ]
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Actual vs Predicted scatter plot
    st.subheader("Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_train, train_predictions, alpha=0.3, s=20, label='Training', color='blue')
    ax.scatter(y_val, val_predictions, alpha=0.6, s=30, label='Validation', color='red')
    min_val = min(y.min(), train_predictions.min(), val_predictions.min())
    max_val = max(y.max(), train_predictions.max(), val_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Price ($)', fontsize=12)
    ax.set_ylabel('Predicted Price ($)', fontsize=12)
    ax.set_title('Actual vs Predicted Stock Prices', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.header("Stock Price Predictions")
    
    # Historical predictions
    st.subheader("Historical Price Predictions")
    
    # Retrain on all data
    X_all_scaled = scaler.fit_transform(X)
    model.fit(X_all_scaled, y)
    all_predictions = model.predict(X_all_scaled)
    
    # Plot historical
    fig, ax = plt.subplots(figsize=(14, 6))
    all_dates = df_features.index
    ax.plot(all_dates, y, label='Actual Price', color='blue', linewidth=2, alpha=0.7)
    ax.plot(all_dates, all_predictions, label='Predicted Price', color='red', linewidth=2, alpha=0.7, linestyle='--')
    ax.fill_between(all_dates, y, all_predictions, alpha=0.2, color='gray')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.set_title('Netflix Stock Price: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Future predictions
    st.subheader(f"Future Price Predictions ({prediction_days} Days)")
    
    last_features = X[-1].reshape(1, -1)
    last_features_scaled = scaler.transform(last_features)
    future_predictions = []
    
    for i in range(prediction_days):
        next_pred = model.predict(last_features_scaled)[0]
        future_predictions.append(next_pred)
    
    last_date = df_features.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='B')
    
    # Plot future predictions
    fig, ax = plt.subplots(figsize=(14, 6))
    recent_history = 90
    ax.plot(df_features.index[-recent_history:], df_features['Close'].values[-recent_history:], 
            label='Historical Price', color='blue', linewidth=2)
    ax.plot(future_dates, future_predictions, label=f'Future Predictions ({prediction_days} Days)', 
            color='green', linewidth=2, linestyle='--', marker='o', markersize=4)
    ax.axvline(x=df_features.index[-1], color='red', linestyle=':', linewidth=2, label='Today')
    
    # Confidence interval
    std_dev = val_rmse
    ax.fill_between(future_dates, 
                     np.array(future_predictions) - std_dev, 
                     np.array(future_predictions) + std_dev, 
                     alpha=0.2, color='green', label=f'±1 RMSE (${std_dev:.2f})')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.set_title(f'Netflix Stock Price Forecast - Next {prediction_days} Business Days', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Future predictions table
    st.subheader("Detailed Future Predictions")
    future_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        'Predicted Price': [f"${p:.2f}" for p in future_predictions],
        'Lower Bound': [f"${p-std_dev:.2f}" for p in future_predictions],
        'Upper Bound': [f"${p+std_dev:.2f}" for p in future_predictions]
    })
    st.dataframe(future_df, use_container_width=True, hide_index=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${df_features['Close'].iloc[-1]:.2f}")
    with col2:
        predicted_change = future_predictions[-1] - df_features['Close'].iloc[-1]
        st.metric(f"Predicted Price (Day {prediction_days})", f"${future_predictions[-1]:.2f}", 
                  f"${predicted_change:.2f}")
    with col3:
        pct_change = (predicted_change / df_features['Close'].iloc[-1]) * 100
        st.metric("Expected Change", f"{pct_change:.2f}%")

with tab3:
    st.header("Feature Importance Analysis")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Top 20 Most Important Features")
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance.head(20)
        ax.barh(range(len(top_features)), top_features['Importance'].values, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'].values, fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top 10 Features")
        top_10 = feature_importance.head(10)[['Feature', 'Importance']]
        top_10['Importance'] = top_10['Importance'].apply(lambda x: f"{x:.4f}")
        st.dataframe(top_10, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Error distribution
    st.subheader("Prediction Error Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        train_errors = y_train - train_predictions
        val_errors = y_val - val_predictions
        ax.hist(train_errors, bins=50, alpha=0.6, label='Training Errors', color='blue', edgecolor='black')
        ax.hist(val_errors, bins=30, alpha=0.6, label='Validation Errors', color='red', edgecolor='black')
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        val_dates = df_features.index[-validation_size:]
        ax.plot(val_dates, val_errors, color='red', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2)
        ax.fill_between(val_dates, 0, val_errors, alpha=0.3, color='red')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Prediction Error ($)', fontsize=12)
        ax.set_title('Validation Error Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

with tab4:
    st.header("Technical Indicators")
    
    # Date range selector
    st.subheader("Select Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df_features.index[-180])
    with col2:
        end_date = st.date_input("End Date", df_features.index[-1])
    
    # Filter data
    mask = (df_features.index >= pd.to_datetime(start_date)) & (df_features.index <= pd.to_datetime(end_date))
    df_filtered = df_features[mask]
    
    # Price and Moving Averages
    st.subheader("Price with Moving Averages")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_filtered.index, df_filtered['Close'], label='Close Price', color='black', linewidth=2)
    ax.plot(df_filtered.index, df_filtered['MA_5'], label='MA 5', color='blue', linewidth=1.5, alpha=0.7)
    ax.plot(df_filtered.index, df_filtered['MA_20'], label='MA 20', color='orange', linewidth=1.5, alpha=0.7)
    ax.plot(df_filtered.index, df_filtered['MA_50'], label='MA 50', color='red', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title('Stock Price with Moving Averages', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Bollinger Bands
    st.subheader("Bollinger Bands")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_filtered.index, df_filtered['Close'], label='Close Price', color='black', linewidth=2)
    ax.plot(df_filtered.index, df_filtered['BB_upper'], label='Upper Band', color='red', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.plot(df_filtered.index, df_filtered['BB_middle'], label='Middle Band', color='blue', linewidth=1.5, alpha=0.7)
    ax.plot(df_filtered.index, df_filtered['BB_lower'], label='Lower Band', color='green', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.fill_between(df_filtered.index, df_filtered['BB_lower'], df_filtered['BB_upper'], alpha=0.1, color='gray')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title('Bollinger Bands', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # RSI and MACD
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RSI (Relative Strength Index)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_filtered.index, df_filtered['RSI'], label='RSI', color='purple', linewidth=2)
        ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5, label='Overbought (70)')
        ax.axhline(y=30, color='green', linestyle='--', linewidth=1.5, label='Oversold (30)')
        ax.fill_between(df_filtered.index, 30, 70, alpha=0.1, color='gray')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('RSI', fontsize=12)
        ax.set_title('RSI Indicator', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("MACD")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_filtered.index, df_filtered['MACD'], label='MACD', color='blue', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.fill_between(df_filtered.index, 0, df_filtered['MACD'], 
                        where=(df_filtered['MACD'] >= 0), alpha=0.3, color='green', label='Positive')
        ax.fill_between(df_filtered.index, 0, df_filtered['MACD'], 
                        where=(df_filtered['MACD'] < 0), alpha=0.3, color='red', label='Negative')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('MACD', fontsize=12)
        ax.set_title('MACD Indicator', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Volume Analysis
    st.subheader("Volume Analysis")
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['green' if df_filtered['Close'].iloc[i] >= df_filtered['Open'].iloc[i] else 'red' 
              for i in range(len(df_filtered))]
    ax.bar(df_filtered.index, df_filtered['Volume'], color=colors, alpha=0.6, width=1)
    ax.plot(df_filtered.index, df_filtered['Volume_MA_20'], label='Volume MA 20', color='blue', linewidth=2)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volume', fontsize=12)
    ax.set_title('Trading Volume with 20-Day Moving Average', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Netflix Stock Prediction Dashboard | Powered by Gradient Boosting ML Model</p>
        <p>⚠️ Disclaimer: This is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)
