# 🚀 Quick Start Guide

## Get Started in 3 Steps

### Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

This installs:

- pandas, numpy (data processing)
- scikit-learn (machine learning)
- matplotlib, seaborn, plotly (visualization)
- streamlit (dashboard)
- scipy, joblib (utilities)

### Step 2: Train the Model (3-5 minutes)

```bash
python air_quality_prediction.py
```

**What happens:**

- ✅ Loads and cleans data
- ✅ Engineers 20+ features
- ✅ Trains 11 different models
- ✅ Selects best model (Random Forest/Gradient Boosting)
- ✅ Saves model as `aqi_prediction_model.pkl`
- ✅ Generates 5 visualization files

**Output files created:**

- `aqi_prediction_model.pkl` (trained model)
- `selected_features.pkl` (feature list)
- `01_distributions.png` through `05_model_diagnostics.png`

### Step 3: Launch Dashboard (instant)

```bash
python run_dashboard.py
```

**Dashboard opens at:** `http://localhost:8501`

---

## Using the Dashboard

### 1. Home Page 🏠

- View key metrics and statistics
- Explore AQI distribution
- See correlation heatmap

### 2. Data Explorer 📊

- Browse the dataset
- Create custom scatter plots
- Analyze feature distributions

### 3. Make Predictions 🔮

**Example: Predict Good Air Quality**

1. Navigate to **🔮 Prediction** page
2. Enter these values:
   - PM2.5: `20`
   - PM10: `35`
   - NO₂: `25`
   - CO: `0.8`
   - Temperature: `22`
   - Humidity: `55`
3. Click **🚀 Predict AQI**
4. See result: **AQI ~30 (Good)** 🟢

**Example: Predict Unhealthy Air Quality**

1. Enter these values:
   - PM2.5: `110`
   - PM10: `175`
   - NO₂: `65`
   - CO: `1.8`
   - Temperature: `28`
   - Humidity: `70`
2. Click **🚀 Predict AQI**
3. See result: **AQI ~120 (Unhealthy)** 🔴

### 4. View Model Performance 📈

- Check R² score, MAE, RMSE
- See feature importance
- Compare different models

---

## Common Commands

### Start Dashboard

```bash
python run_dashboard.py
```

### Train Model

```bash
python air_quality_prediction.py
```

### Install Packages

```bash
pip install -r requirements.txt
```

### Use Different Port

```bash
streamlit run app.py --server.port 8502
```

### Stop Dashboard

Press `Ctrl + C` in terminal

---

## Troubleshooting

### ❌ "Model not found"

**Solution:** Train the model first

```bash
python air_quality_prediction.py
```

### ❌ "Module not found"

**Solution:** Install dependencies

```bash
pip install -r requirements.txt
```

### ❌ "Dataset not found"

**Solution:** Ensure `12_air_quality.csv` is in the same folder

### ❌ "Port already in use"

**Solution:** Use different port

```bash
streamlit run app.py --server.port 8502
```

### ❌ Browser doesn't open

**Solution:** Manually go to `http://localhost:8501`

---

## Understanding AQI Values

| AQI Range | Category                | Color     | Health Impact               |
| --------- | ----------------------- | --------- | --------------------------- |
| 0-50      | Good                    | 🟢 Green  | Air quality is satisfactory |
| 51-100    | Moderate                | 🟡 Yellow | Acceptable for most people  |
| 101-150   | Unhealthy for Sensitive | 🟠 Orange | Sensitive groups affected   |
| 151-200   | Unhealthy               | 🔴 Red    | Everyone may be affected    |
| 201-300   | Very Unhealthy          | 🟣 Purple | Health alert                |
| 301+      | Hazardous               | 🟤 Maroon | Emergency conditions        |

---

## Next Steps

### Explore the Data

1. Go to **📊 Data Explorer**
2. Try different scatter plot combinations
3. Download the dataset

### Make Multiple Predictions

1. Go to **🔮 Prediction**
2. Try different input combinations
3. Compare results

### Understand the Model

1. Go to **📈 Model Performance**
2. Check feature importance
3. Review cross-validation scores

### Read Documentation

- [README.md](README.md) - Complete project documentation
- [DASHBOARD_README.md](DASHBOARD_README.md) - Dashboard guide

---

## Tips & Tricks

### Dashboard Navigation

- Use **sidebar** to switch between pages
- **Hover** over charts for details
- **Zoom** by scrolling on charts
- **Download** charts by clicking camera icon

### Best Practices

- ✅ Train model before using dashboard
- ✅ Use realistic input values
- ✅ Check health recommendations
- ✅ Compare multiple predictions

### Performance

- Dashboard caches data automatically
- Model loads once and stays in memory
- Refresh page with `R` key
- Clear cache with `C` key

---

## Example Workflow

### Scenario: Analyze Air Quality

1. **Start Dashboard**

   ```bash
   python run_dashboard.py
   ```

2. **Explore Data** (Home page)
   - Check average AQI: ~57
   - See PM2.5 correlation: 0.95
   - View category distribution

3. **Make Prediction** (Prediction page)
   - Input current air quality readings
   - Get AQI prediction
   - Read health recommendations

4. **Analyze Model** (Performance page)
   - Check model accuracy: R² = 0.93
   - See top features: PM2.5, PM10
   - Review cross-validation: 0.92 ± 0.01

5. **Export Results**
   - Download dataset (Data Explorer)
   - Save charts (click camera icon)
   - Screenshot predictions

---

## Need Help?

### Documentation

- 📖 [README.md](README.md) - Main documentation
- 📖 [DASHBOARD_README.md](DASHBOARD_README.md) - Dashboard guide
- 📖 [QUICKSTART.md](QUICKSTART.md) - This file

### Resources

- Streamlit docs: https://docs.streamlit.io
- Scikit-learn docs: https://scikit-learn.org
- AQI information: https://www.airnow.gov/aqi/

### Common Issues

- Check troubleshooting section above
- Ensure all files are in same directory
- Verify Python version (3.8+)

---

**Ready to start? Run:**

```bash
python run_dashboard.py
```

**Happy predicting! 🌍**
