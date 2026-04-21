# 🌍 Air Quality Index (AQI) Prediction Project

## Overview

A complete machine learning solution for predicting Air Quality Index (AQI) with an interactive Streamlit dashboard for real-time predictions and data exploration.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python air_quality_prediction.py
```

### 3. Launch Dashboard

```bash
python run_dashboard.py
# Or directly: streamlit run app.py
```

**Dashboard URL**: http://localhost:8501

## 🎯 Key Features

### Machine Learning Pipeline

- ✅ **95.14% accuracy** on test set
- ✅ **11 algorithms compared** (Linear, Tree, Ensemble, SVM, KNN)
- ✅ **20+ engineered features** from 6 input parameters
- ✅ **Advanced preprocessing** (outlier handling, feature selection)
- ✅ **Cross-validation** and hyperparameter tuning
- ✅ **Best Model**: Lasso Regression

### Interactive Dashboard (5 Pages)

- 🏠 **Home**: Overview with key metrics and visualizations
- 📊 **Data Explorer**: Interactive data analysis and exploration
- 🔮 **Prediction**: Real-time AQI predictions with health recommendations
- 📈 **Model Performance**: Detailed metrics and feature importance
- ℹ️ **About**: Complete methodology documentation

## 📊 Model Performance

- **Test R² Score**: 0.9514 (95.14% accuracy)
- **Mean Absolute Error**: 3.68 AQI points
- **Root Mean Squared Error**: 4.61
- **Cross-Validation R²**: 0.9222 ± 0.0134

## 🎨 Dashboard Features

### Real-time Predictions

Enter air quality parameters and get instant AQI predictions:

- **Input**: PM2.5, PM10, NO₂, CO, Temperature, Humidity
- **Output**: AQI value with color-coded category
- **Health Recommendations**: Personalized advice based on AQI level

### Interactive Visualizations

- Correlation heatmaps
- Distribution plots
- Scatter plot analysis
- Feature importance charts
- Model comparison metrics

## 🎯 AQI Categories

| Range   | Category              | Color | Health Impact               |
| ------- | --------------------- | ----- | --------------------------- |
| 0-50    | Good                  | 🟢    | Air quality is satisfactory |
| 51-100  | Moderate              | 🟡    | Acceptable for most people  |
| 101-150 | Unhealthy (Sensitive) | 🟠    | Sensitive groups affected   |
| 151-200 | Unhealthy             | 🔴    | Everyone may be affected    |
| 201-300 | Very Unhealthy        | 🟣    | Health alert                |
| 301+    | Hazardous             | 🟤    | Emergency conditions        |

## 💡 Example Predictions

### Good Air Quality 🟢

```
Input:  PM2.5=20, PM10=35, NO₂=25, CO=0.8, Temp=22°C, Humidity=55%
Output: AQI ≈ 30 (Good)
Advice: Enjoy outdoor activities!
```

### Unhealthy Air Quality 🔴

```
Input:  PM2.5=110, PM10=175, NO₂=65, CO=1.8, Temp=28°C, Humidity=70%
Output: AQI ≈ 120 (Unhealthy)
Advice: Reduce outdoor activities
```

## 🛠️ Technology Stack

- **ML Framework**: Scikit-learn
- **Dashboard**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib

## 📁 Project Structure

```
air_quality_prediction/
├── air_quality_prediction.py    # ML training pipeline
├── app.py                       # Streamlit dashboard
├── run_dashboard.py             # Dashboard launcher
├── 12_air_quality.csv           # Dataset (1000 samples)
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── QUICKSTART.md               # Quick start guide
├── aqi_prediction_model.pkl    # Trained model
├── selected_features.pkl       # Feature list
└── outputs/                    # Generated visualizations
    ├── 01_distributions.png
    ├── 02_boxplots.png
    ├── 03_correlation_heatmap.png
    ├── 04_pairplot_high_corr.png
    └── 05_model_diagnostics.png
```

## 🔬 Methodology

### Data Processing

1. **Data Cleaning**: Handle missing values and outliers
2. **Feature Engineering**: Create 20+ domain-specific features
3. **Feature Selection**: Remove redundant and low-correlation features
4. **Preprocessing**: StandardScaler for normalization

### Model Development

1. **Model Comparison**: Test 11 different algorithms
2. **Cross-Validation**: 5-fold CV for robust evaluation
3. **Hyperparameter Tuning**: RandomizedSearchCV optimization
4. **Model Selection**: Choose best performing model (Lasso Regression)

### Feature Engineering Examples

- `pm_avg` = (PM2.5 + PM10) / 2
- `pollution_humidity` = PM2.5 × Humidity
- `pm25_squared` = PM2.5²
- `log_pm25` = log(1 + PM2.5)

## 🎓 Educational Value

### Skills Demonstrated

- Complete ML pipeline development
- Advanced feature engineering
- Interactive dashboard creation
- Data visualization and analysis
- Model evaluation and selection
- Production-ready code structure

### Best Practices

- No data leakage (proper train-test split)
- Cross-validation for model selection
- Comprehensive documentation
- Modular and reusable code
- Professional UI/UX design

## 🚀 Deployment

### Local Development

```bash
git clone https://github.com/jackstealer/my_learning.git
cd air_quality_prediction
pip install -r requirements.txt
python air_quality_prediction.py
python run_dashboard.py
```

### Cloud Deployment

The dashboard can be deployed to:

- Streamlit Cloud
- Heroku
- AWS/GCP/Azure
- Docker containers

## 📞 Support

For questions or issues:

1. Check [QUICKSTART.md](QUICKSTART.md) for common problems
2. Review the dashboard's About page for methodology
3. Ensure all dependencies are installed correctly

## 📄 License

MIT License - Free to use and modify

## 🙏 Acknowledgments

- **EPA**: For AQI standards and guidelines
- **Scikit-learn**: ML framework
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualizations

---

**Built with ❤️ for environmental awareness and public health**

_Last Updated: 2024 | Version: 1.0.0_
