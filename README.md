# Netflix Stock Prediction Dashboard 📈

An interactive ML pipeline dashboard built with Streamlit for predicting Netflix stock prices.

## Features

### 1. Data Source
- Upload custom CSV files
- Fetch real-time Netflix stock data from Yahoo Finance
- Support for historical data analysis

### 2. Data & EDA
- Dataset summary statistics
- Correlation heatmap visualization
- Time series plots
- Feature distribution analysis
- Interactive data exploration

### 3. Cleaning & Engineering
- Handle missing values (delete, impute with mean/median)
- Outlier removal using IQR method
- Time-based feature engineering (Year, Month, Day, DayOfWeek)
- Technical indicators (Moving Averages, Volatility, Price Changes)

### 4. Feature Selection
- All features selection
- Variance threshold filtering
- Correlation-based selection
- Manual feature selection
- Feature statistics preview

### 5. Model Training
- Multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting
- Configurable train/test split
- K-Fold Cross Validation (k=5)
- Model-specific hyperparameter tuning

### 6. Performance Evaluation
- Comprehensive metrics (R², RMSE, MAE)
- Cross-validation stability visualization
- Predicted vs Actual scatter plots
- Residual analysis
- Feature importance (for tree-based models)
- Export predictions to CSV

### 7. Predictions
- Single prediction with custom inputs
- Batch predictions from CSV upload
- Confidence intervals for predictions
- Download prediction results

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the dashboard:
```bash
streamlit run netflix_stock_dashboard.py
```

2. Open your browser at `http://localhost:8501`

3. Follow the workflow:
   - Upload CSV or fetch Netflix data
   - Explore data in EDA tab
   - Clean and engineer features
   - Select relevant features
   - Train your model
   - Evaluate performance
   - Make predictions

## Data Format

If uploading CSV, ensure it contains:
- Numeric columns for features
- A target column (e.g., 'Close' price)
- Optional: 'Date' column for time series analysis

Example columns for stock data:
- Date, Open, High, Low, Close, Volume

## Tips

- Use technical indicators for better stock predictions
- Try different models and compare performance
- Use correlation-based feature selection to reduce overfitting
- Check residual plots to validate model assumptions
- Random Forest and Gradient Boosting often perform well for stock prediction

## Technologies

- **Streamlit**: Interactive web dashboard
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning models
- **Plotly**: Interactive visualizations
- **yfinance**: Stock data fetching
- **Seaborn/Matplotlib**: Statistical visualizations
