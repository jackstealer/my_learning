# Netflix Stock Prediction Dashboard 📈

An interactive machine learning dashboard for predicting Netflix stock prices using advanced technical indicators and Gradient Boosting algorithms.

## Features

### 📊 Overview Tab

- Real-time model performance metrics (R², RMSE, MAE, MAPE)
- Training vs validation comparison
- Dataset information and statistics
- Actual vs predicted price scatter plots

### 📈 Predictions Tab

- Historical price predictions with actual comparison
- Future price forecasts (7-60 days configurable)
- Confidence intervals based on validation RMSE
- Detailed prediction tables with upper/lower bounds

### 🔍 Feature Analysis Tab

- Top 20 feature importance visualization
- Prediction error distribution analysis
- Error trends over time
- Comprehensive feature ranking

### 📉 Technical Indicators Tab

- Moving Averages (5, 20, 50-day)
- Bollinger Bands
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Volume analysis with trends
- Interactive date range selection

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit Dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Run the Python Script (for batch processing):

```bash
python netflix_stock_prediction.py
```

## Model Details

### Algorithm

- **Gradient Boosting Regressor** - Superior performance for time series prediction

### Features (40+ technical indicators)

- **Price Features**: Daily returns, price ranges, ratios
- **Moving Averages**: 5, 10, 20, 50-day MAs with ratios
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Volatility Measures**: 5-day and 20-day volatility
- **Momentum Indicators**: ROC, momentum calculations
- **Volume Analysis**: Volume ratios and moving averages
- **Lag Features**: 1, 2, 3, 5, 10-day historical prices
- **Time Features**: Day of week, month, quarter

### Data Preprocessing

- Missing value handling (forward/backward fill)
- RobustScaler for outlier-resistant normalization
- Time-series aware train/validation split

### Performance Metrics

- R² Score (coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

## Dashboard Configuration

### Sidebar Controls

- **Number of Estimators**: 100-500 (default: 300)
- **Learning Rate**: 0.01-0.2 (default: 0.05)
- **Max Depth**: 3-10 (default: 5)
- **Validation Split**: 10-30% (default: 20%)
- **Future Prediction Days**: 7-60 days (default: 30)

## File Structure

```
├── app.py                          # Streamlit dashboard
├── netflix_stock_prediction.py     # Standalone ML script
├── NFLX.csv                        # Netflix stock data
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Technical Stack

- **Python 3.8+**
- **Streamlit**: Interactive web dashboard
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization

## Disclaimer

⚠️ **This tool is for educational and research purposes only.**

This is NOT financial advice. Stock market predictions are inherently uncertain. Always consult with qualified financial advisors before making investment decisions.

## Future Enhancements

- [ ] LSTM/GRU neural network models
- [ ] Real-time data fetching via API
- [ ] Multiple stock comparison
- [ ] Portfolio optimization
- [ ] Sentiment analysis integration
- [ ] Model ensemble methods

## License

MIT License - Feel free to use and modify for your projects.
