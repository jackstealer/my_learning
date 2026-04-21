Optimized **produtin-grad**air pollutants and enironmental fctos. The pipeline ncludes advanced preprcessing, feature engineering, comprehensive model comparison, and rigorovldation.

## Key Improvements Ove BasicAproaches

### 1. **Rbust Data Ceaning**

- Muti-method olier detection (IQR + Z-score)
- Outlier cpping isead of removal (preerves data)
- Logical consistency checks (PM10 ≥ PM2.5)
  -Proper hling ofphysically impossible valus

### 2. **Advanced Feature Engineerig**

- Domain-knowledge based features (20+ engineered features)
- Polynomial features for non-linear relationships
- Interaction terms between pollutants and entors
- Log transformaions f kewed distributions
- Categorical binning for continuous variables

### 3 **Comprehensive Model Comparison**

- 11 different algorithms tested
- Proper cross-validation strategy (5-fold)
- Stratified train-test split for balanced representation- Multiple evaluation metrics (R², MAE, RMSE, MAPE)
  #4. **Rigorous Validation**
- No data leakage (scaling after split)
- Cross-validation on trining se only
- Residual anlysi and diagnostic plots
- Overfitting dection## Dataset
  original Original | Feature | Description | Unit ||--------|-------------|------|
  || | μg/m³ || | | μg/m³ || | |b|| | | || | |° || | Relative | %|| |) | -|

### Engineered Features (Examples)

- `pm_ag`, `pm_diff`, `pm_rtio`, `pm_sum`
- `pollution_humidity`, `pollution_temp`
- `no2_co_ato`, `no2_co_interaction`
- `pm25_squred`, `pm10_squared`
- `log_pm25`, `og_pm10`
- Temprature and humidity categories (one-hot encoded Complete (13)Data ing& Explorionniialdationa QulityAsssmetnomesand nconitncieAdvanced Dat Cengandle msin vlue anderrr 4.**Outlie Dect& Trent** - Multi-method outlier hndling5Exploratory AalyssStsticalnaysvisualization6Advanced ngineeg20+ doma-spcific7Rmovow-ion and rundant8Satfd sp (80/20)11alothmswthcs-validaionHypeprameter TuRdomizdSachCV ptizatinFinal Comprehensive on test setModl DignoscsResulanalyss and dgnosc pltsMode Perssece Save trainedmden feuresh

# Clone or download te project -r requirements.txt

````

### Requirements
- Python 3.8+
- >= 1.5.0
- >= 1.23.0
- scikit-learn >= 1.2.0
- >= 3.6.0
- >= 0.12.0
-py >= 1.10.0
 jobib >= 1.2.0

The script will:1. Process the data with detailed logging2.Generat 5 viaizaion file3. Train and compare 11 models4. Perform hyperparameter tuning
5. Save the best model as `aqi_prediction_model.pkl`
6. Display comprehensive performance metrics

sed

| Model | Type | Key Charactertics ||-------|------|---------------------||| Linear | Baseline, interpretable |
| idgeRegression| Linear | L2 regularization |
| Lasso Regression | Linear | L regularizationfeature selection |
| lasticNet|Linear | L1 + L2 regularization |||Tree-based| Non-linearinterpretable|| |Ensemble|Bagging robust |
| Extra Trees | Ensemble |ore randomization |
| Gradient Boosting | Ensemble | Boosting, high performance |
| daBoost | nsemble | Adaptive boosting |
| SVR | Kernel-based | Non-linear, powerful |
| KNN | Instance-based |Non-parametric| ExpectedResuts

### (Typical) **BestModel: Random Forest or Gdient Boostg
- **Test3 - 0.6MAE 3.5 -45TeMSE45 - 5.5Test P6% - %CV ² (5-fold)091 - 0.94FaureImortanc (Top 5)
1. PM2.5 (stongest redictor)
2. PM10
3. PM veg
4. PM2.5 squared
5. Polluion-humidity intaction

## Output File| File | Description |
|-----|-------------|
|01dirbutions.png` | Histogras of ll feaue||02boxlos.png| Boxplots after outlier treatment ||03_correlatoheta.png` | Feature correation matrix |
| `04_pairplot_high_corr.png` | Pairplot of highly corrlated feature |
| `05model_diagnotics.ng` | 6-pane dagnosicplots ||aq_predictioodel.k` | Traind model (erialized) |
| `selectedfeatures.pk` | Fture names orprediction |Modl Dignosics

Thdianostc plots iclud:
1. **Actual vs Prdicted** - Model accuacy vsualizatio2. **Residual Plot** - hck for ptrns (shoulb rndom)
3. **Residual Disribtion** - Should be appoximatly normal4. **Q-Q Plot** - Test normality of residuals5. **Prediction Error Distribution**  Errornitude analysis
6.**Featur Impotnc** - Topcntributing eatures

## UsingtheTrieodel

```python
import joblibimport pandas asd

# Load model and features
odel = joblib.load('aqipreicton_model.pkl')
eatures =joblb.load('selected_eatures.pkl')

# Prpa ew data (must have samfaurs)
ew_data=pd.DataFrme({
   'pm5': [500],
    'pm10': [8.0],   'no2_pb': [40.0],
    'c_mgm3': [1.2],
    'empc': [25.0],
    ''[600]
})

#Engineerfeatres (sae as tranng)# ...(apy same featre engineerg ss)

#redict
prediction=modl.redict(new_data[fres])
print(f"Predicted AQI: {pdiction[0]:.2f}")
```1.** aredminantpredico**(>2. **Featureeingsigniicnly impovefrmance** (+3-5% R²)
3. **Ensemblethsout liner modls**(,GradenBoos)
4. **Modelgenerlzs well** (low orfittinggap < 00)
5.**esidualsare apprximately ormal**(good model fi)

## BPractice Implmened
✅**N ataleaka** - Scalig aft trn-ttsplit
✅ **Stratifid samping** - Baancedargetdribuion
✅ **C**- 5-fld CV on taining et only  ✅**Mlp metric** - R², MAE, RMSE, MAPE  ✅**Oulie hdlin**Cappng inead of emoval
✅ **Featre selec**- Remveredundnt  ✅**Hyerarameter tuin**RandomizedSearchCV
✅ **Resida analyss** - Modliagnostic chcks
✅ **Model persisne** - Save for producuse

## Lmitation & Fture Work

### Current LmitsLimited t 1000 sampls (smal daset)
- N tmporl feaures (tie-series alysis)
No spatial fs(loatin-based)
- No extenal data (weathe forcss, traffc)

### PtetialIprovemens
1. **Deep Leanng** - Neural networks for comple patterns2. **Time Series** LSTM/GRU for empor denencs
3 **Ensemble Stacki** Combine multiplems
4.**Featue Selcton** - Reursive feature elimination
5. **Calibra** -Probbility alibration for ncetint
6.**Exlainability** - SHAP vaues fr interpretabiliy              # Dataset       # Main pipeline script                        # This filet                 # Dependencies
├── aqi_prediction_model.pkl        # Trained model (generated)
├── selected_features.pkl           # Feature list (generaed)                         # Visualizations (generated)01_dtribuin02_s03_ap.png
    ├── 04_pirlot_high_corr05_model_dignostics.png
````

## License

MIT Liense - Feel free to use and modify

## Author

Generaed from air qality anysi notebook

## Refeences

- Scikit-larn ocumentaton: https://siki-larn.org
- Air Quality Inex: https://wwwairow.ov/aqi/- Feature Engineering: https://www.kaggle.com/learn/feature-engineering

---

## 🎨 Interactive Dashboard

### Quick Start

1. **Train the model** (first time only):

   ```bash
   python air_quality_prediction.py
   ```

2. **Launch the dashboard**:

   ```bash
   python run_dashboard.py
   ```

   Or directly:

   ```bash
   streamlit run app.py
   ```

3. **Access the dashboard**: Opens automatically at `http://localhost:8501`

### Dashboard Features

#### 🏠 Home Page

- Real-time key metrics (total records, average AQI, max/min values)
- Interactive AQI distribution histogram
- AQI category breakdown pie chart
- Feature correlation heatmap
- Automated insights and analysis

#### 📊 Data Explorer

- Browse complete dataset with pagination
- Download data as CSV
- Statistical summaries for all features
- Interactive scatter plots (choose any X/Y variables)
- Feature trend visualization
- Multi-feature box plot comparisons

#### 🔮 Prediction

- **Real-time AQI predictions** with user-friendly input form
- Input parameters:
  - PM2.5 (0-500 μg/m³)
  - PM10 (0-600 μg/m³)
  - NO₂ (0-200 ppb)
  - CO (0-10 mg/m³)
  - Temperature (-20 to 50°C)
  - Humidity (0-100%)
- **Color-coded results** based on AQI category
- **Health recommendations** tailored to predicted AQI
- **Interactive gauge chart** showing AQI level
- Input summary with all parameters

#### 📈 Model Performance

- Comprehensive performance metrics (R², MAE, RMSE, MAPE)
- Feature importance visualization (top 15 features)
- Model comparison across 11 algorithms
- Cross-validation results with 5-fold CV
- Detailed metric tables

#### ℹ️ About

- Complete methodology documentation
- AQI category reference table
- Technology stack information
- System status and diagnostics

### Dashboard Screenshots

The dashboard provides:

- **Interactive visualizations** powered by Plotly
- **Responsive design** that works on all screen sizes
- **Real-time updates** as you interact with controls
- **Export capabilities** for charts and data
- **Professional UI** with custom styling

### Example Predictions

**Good Air Quality Example:**

- PM2.5: 20, PM10: 35, NO₂: 25, CO: 0.8, Temp: 22°C, Humidity: 55%
- **Result**: AQI ~30 (Good) - Green indicator

**Unhealthy Air Quality Example:**

- PM2.5: 110, PM10: 175, NO₂: 65, CO: 1.8, Temp: 28°C, Humidity: 70%
- **Result**: AQI ~120 (Unhealthy) - Red indicator

### Dashboard Requirements

Additional packages for dashboard:

```bash
pip install streamlit>=1.28.0 plotly>=5.17.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### Troubleshooting Dashboard

**Model not loaded warning:**

```bash
# Train the model first
python air_quality_prediction.py
```

**Port already in use:**

```bash
streamlit run app.py --server.port 8502
```

**Browser doesn't open:**

- Manually navigate to `http://localhost:8501`

For detailed dashboard documentation, see [DASHBOARD_README.md](DASHBOARD_README.md)

---

## Complete Project Files

```
📁 Air Quality Prediction Project
├── 📄 12_air_quality.csv              # Dataset (1000 samples)
├── 🐍 air_quality_prediction.py       # ML training pipeline
├── 🎨 app.py                           # Streamlit dashboard
├── 🚀 run_dashboard.py                 # Dashboard launcher
├── 📖 README.md                        # Main documentation
├── 📖 DASHBOARD_README.md              # Dashboard guide
├── 📋 requirements.txt                 # Python dependencies
├── 🤖 aqi_prediction_model.pkl        # Trained model (generated)
├── 📊 selected_features.pkl           # Feature list (generated)
└── 📁 outputs/                         # Visualizations (generated)
    ├── 01_distributions.png
    ├── 02_boxplots.png
    ├── 03_correlation_heatmap.png
    ├── 04_pairplot_high_corr.png
    └── 05_model_diagnostics.png
```

## Technology Stack

### Machine Learning

- **Scikit-learn**: Model training and evaluation
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **SciPy**: Statistical analysis

### Visualization

- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts (dashboard)

### Dashboard

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Joblib**: Model persistence

## Quick Reference

### Train Model

```bash
python air_quality_prediction.py
```

### Run Dashboard

```bash
python run_dashboard.py
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Make Prediction (Python)

```python
import joblib
import pandas as pd

model = joblib.load('aqi_prediction_model.pkl')
# ... prepare data with feature engineering
prediction = model.predict(data)
```
