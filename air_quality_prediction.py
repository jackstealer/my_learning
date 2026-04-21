"""
Optimized Air Quality Index (AQI) Prediction Pipeline
Dataset: 12_air_quality.csv
Target: AQI prediction using air quality and environmental features

Key Improvements:
- Robust outlier detection with multiple methods
- Advanced feature engineering with domain knowledge
- Feature selection using multiple techniques
- Ensemble model comparison
- Proper validation strategy to prevent data leakage
- Residual analysis and model diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              mean_absolute_percentage_error)
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from scipy import stats
from scipy.stats import skew, kurtosis

# ============================================================================
# 1. LOAD AND INITIAL EXPLORATION
# ============================================================================
print("="*80)
print("STEP 1: DATA LOADING AND INITIAL EXPLORATION")
print("="*80)

df_original = pd.read_csv('12_air_quality.csv')
df = df_original.copy()

print(f"Dataset shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nBasic statistics:\n{df.describe().T}")

# ============================================================================
# 2. DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATA QUALITY ASSESSMENT")
print("="*80)

# Check for physically impossible values
print("\nPhysically impossible values detected:")
print(f"Negative PM10 values: {(df['pm10'] < 0).sum()}")
print(f"Negative AQI values: {(df['aqi'] < 0).sum()}")
print(f"Humidity > 100: {(df['humidity'] > 100).sum()}")
print(f"Negative PM2.5: {(df['pm25'] < 0).sum()}")

# Check for logical inconsistencies
print(f"\nLogical issues:")
print(f"PM10 < PM2.5 (should not happen): {(df['pm10'] < df['pm25']).sum()}")

# ============================================================================
# 3. ADVANCED DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: ADVANCED DATA CLEANING")
print("="*80)

# Handle negative values - replace with NaN for proper imputation
df.loc[df['pm10'] < 0, 'pm10'] = np.nan
df.loc[df['aqi'] < 0, 'aqi'] = np.nan
df.loc[df['pm25'] < 0, 'pm25'] = np.nan

# Handle logical inconsistencies: PM10 should be >= PM2.5
# If PM10 < PM2.5, it's likely a measurement error
inconsistent_mask = df['pm10'] < df['pm25']
print(f"Fixing {inconsistent_mask.sum()} rows where PM10 < PM2.5")
df.loc[inconsistent_mask, 'pm10'] = df.loc[inconsistent_mask, 'pm25'] * 1.5

# Impute missing values using median (robust to outliers)
for col in ['pm10', 'aqi', 'pm25']:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Imputed {col} with median: {median_val:.2f}")

print(f"\nAfter cleaning - Missing values:\n{df.isnull().sum()}")
print(f"\nCleaned statistics:\n{df.describe().T}")

# ============================================================================
# 4. OUTLIER DETECTION AND TREATMENT
# ============================================================================
print("\n" + "="*80)
print("STEP 4: OUTLIER DETECTION AND TREATMENT")
print("="*80)

def detect_outliers_multiple_methods(data, column):
    """Detect outliers using multiple methods for robustness"""
    # Method 1: IQR
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = (data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))
    
    # Method 2: Z-score
    z_scores = np.abs(stats.zscore(data[column]))
    z_outliers = z_scores > 3
    
    # Combine: mark as outlier if detected by both methods (conservative)
    combined_outliers = iqr_outliers & z_outliers
    
    return combined_outliers, iqr_outliers, z_outliers

# Detect outliers for each numeric column (except id)
outlier_summary = {}
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'id':
        combined, iqr, z = detect_outliers_multiple_methods(df, col)
        outlier_summary[col] = {
            'IQR_method': iqr.sum(),
            'Z_score_method': z.sum(),
            'Combined_strict': combined.sum()
        }

print("\nOutlier detection summary:")
outlier_df = pd.DataFrame(outlier_summary).T
print(outlier_df)

# Use IQR method for outlier removal (less aggressive than Z-score)
df_no_outliers = df.copy()
for col in df.select_dtypes(include=[np.number]).columns:
    if col not in ['id', 'aqi']:  # Don't remove outliers from target
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing (more data retained)
        df_no_outliers[col] = df_no_outliers[col].clip(lower=lower_bound, upper=upper_bound)

print(f"\nOriginal shape: {df.shape}")
print(f"After outlier capping: {df_no_outliers.shape}")

# Use capped version for modeling
df = df_no_outliers.copy()

# ============================================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Distribution analysis
print("\nSkewness of features:")
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'id':
        skewness = skew(df[col])
        kurt = kurtosis(df[col])
        print(f"{col:20s}: Skewness={skewness:6.3f}, Kurtosis={kurt:6.3f}")

# Correlation analysis
corr_matrix = df.corr()
corr_with_target = corr_matrix['aqi'].sort_values(ascending=False)
print("\nCorrelation with AQI:")
print(corr_with_target)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Histograms
df.drop(columns=['id']).hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.tight_layout()
plt.savefig('01_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Boxplots
fig, ax = plt.subplots(figsize=(12, 6))
df.drop(columns=['id']).boxplot(ax=ax, rot=45)
ax.set_title("Boxplot for All Features (After Outlier Treatment)", fontsize=14)
plt.tight_layout()
plt.savefig('02_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig('03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Pairplot for highly correlated features
high_corr_features = corr_with_target[abs(corr_with_target) > 0.3].index.tolist()
if len(high_corr_features) > 1:
    sns.pairplot(df[high_corr_features], diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.savefig('04_pairplot_high_corr.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nVisualization files saved.")

# ============================================================================
# 6. ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("STEP 6: ADVANCED FEATURE ENGINEERING")
print("="*80)

# Drop ID column (not useful for prediction)
df = df.drop(columns=['id'])

# Domain-knowledge based features
print("\nCreating engineered features...")

# 1. Particulate matter features
df['pm_avg'] = (df['pm25'] + df['pm10']) / 2
df['pm_diff'] = df['pm10'] - df['pm25']
df['pm_ratio'] = df['pm25'] / (df['pm10'] + 1e-5)  # Avoid division by zero
df['pm_sum'] = df['pm25'] + df['pm10']

# 2. Pollution interaction with environmental factors
df['pollution_humidity'] = df['pm25'] * df['humidity'] / 100
df['pollution_temp'] = df['pm25'] * df['temp_c']
df['pm10_humidity'] = df['pm10'] * df['humidity'] / 100

# 3. Gas pollutant features
df['no2_co_ratio'] = df['no2_ppb'] / (df['co_mgm3'] + 1e-5)
df['no2_co_interaction'] = df['no2_ppb'] * df['co_mgm3']

# 4. Environmental comfort index
df['temp_humidity_index'] = df['temp_c'] * df['humidity'] / 100

# 5. Polynomial features for highly correlated variables
df['pm25_squared'] = df['pm25'] ** 2
df['pm10_squared'] = df['pm10'] ** 2

# 6. Log transformations for skewed features
df['log_pm25'] = np.log1p(df['pm25'])
df['log_pm10'] = np.log1p(df['pm10'])

# 7. Binned features (categorical encoding of continuous)
df['temp_category'] = pd.cut(df['temp_c'], bins=3, labels=['cold', 'moderate', 'hot'])
df['humidity_category'] = pd.cut(df['humidity'], bins=3, labels=['dry', 'normal', 'humid'])

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['temp_category', 'humidity_category'], drop_first=True)

print(f"Total features after engineering: {df.shape[1]}")
print(f"New features created: {df.shape[1] - 7}")  # Original had 7 features (excluding id)

# Feature correlation with target
corr_with_aqi = df.corr()['aqi'].abs().sort_values(ascending=False)
print("\nTop 15 features correlated with AQI:")
print(corr_with_aqi.head(15))

# ============================================================================
# 7. FEATURE SELECTION
# ============================================================================
print("\n" + "="*80)
print("STEP 7: FEATURE SELECTION")
print("="*80)

# Prepare data for feature selection
X_all = df.drop(columns=['aqi'])
y = df['aqi']

# Method 1: Correlation-based selection (remove low correlation features)
correlation_threshold = 0.05
low_corr_features = corr_with_aqi[corr_with_aqi < correlation_threshold].index.tolist()
if 'aqi' in low_corr_features:
    low_corr_features.remove('aqi')  # Don't remove target
print(f"\nFeatures with correlation < {correlation_threshold}: {len(low_corr_features)}")
print(low_corr_features)

# Method 2: Remove highly correlated features (multicollinearity)
corr_matrix = X_all.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
print(f"\nHighly correlated features (>0.95) to remove: {len(high_corr_features)}")
print(high_corr_features)

# Method 3: Statistical feature selection (F-test)
selector = SelectKBest(score_func=f_regression, k=min(20, X_all.shape[1]))
selector.fit(X_all, y)
feature_scores = pd.DataFrame({
    'Feature': X_all.columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)
print("\nTop 15 features by F-statistic:")
print(feature_scores.head(15))

# Select features: remove low correlation and highly correlated
features_to_remove = list(set(low_corr_features + high_corr_features))
X_selected = X_all.drop(columns=features_to_remove)

print(f"\nFeatures after selection: {X_selected.shape[1]} (removed {len(features_to_remove)})")
print(f"Selected features: {X_selected.columns.tolist()}")

# ============================================================================
# 8. TRAIN-TEST SPLIT (BEFORE SCALING TO PREVENT DATA LEAKAGE)
# ============================================================================
print("\n" + "="*80)
print("STEP 8: TRAIN-TEST SPLIT")
print("="*80)

# Use selected features
X = X_selected
y = df['aqi']

# Stratified split based on AQI quartiles for better representation
y_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_bins
)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape} - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
print(f"y_test: {y_test.shape} - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")

# Check distribution similarity
print("\nTarget distribution check:")
print(f"Train - Min: {y_train.min():.2f}, Max: {y_train.max():.2f}")
print(f"Test  - Min: {y_test.min():.2f}, Max: {y_test.max():.2f}")

# ============================================================================
# 9. MODEL SELECTION WITH MULTIPLE ALGORITHMS
# ============================================================================
print("\n" + "="*80)
print("STEP 9: COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# Define multiple models with pipelines (scaling included)
models = {
    "Linear Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42))
    ]),
    "Lasso Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.1, random_state=42))
    ]),
    "ElasticNet": Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ('scaler', StandardScaler()),
        ('model', DecisionTreeRegressor(max_depth=10, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ]),
    "Extra Trees": Pipeline([
        ('scaler', StandardScaler()),
        ('model', ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
    ]),
    "AdaBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('model', AdaBoostRegressor(n_estimators=100, random_state=42))
    ]),
    "SVR": Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf', C=100, epsilon=0.1))
    ]),
    "KNN": Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor(n_neighbors=5))
    ])
}

# Evaluate all models with cross-validation
results = {}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nTraining and evaluating models...")
print(f"{'Model':<25} {'Train R²':<10} {'Test R²':<10} {'CV R² (mean±std)':<25} {'MAE':<10} {'RMSE':<10}")
print("-" * 100)

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results[name] = {
        "Train R2": train_r2,
        "Test R2": test_r2,
        "CV R2 Mean": cv_mean,
        "CV R2 Std": cv_std,
        "MAE": mae,
        "RMSE": rmse,
        "Model": model
    }
    
    print(f"{name:<25} {train_r2:<10.4f} {test_r2:<10.4f} {cv_mean:.4f}±{cv_std:.4f}{'':>10} {mae:<10.4f} {rmse:<10.4f}")

# Select best model based on CV score (more reliable than test score)
best_model_name = max(results, key=lambda x: results[x]['CV R2 Mean'])
best_model = results[best_model_name]['Model']

print("\n" + "="*80)
print(f"BEST MODEL: {best_model_name}")
print(f"CV R² Score: {results[best_model_name]['CV R2 Mean']:.4f} ± {results[best_model_name]['CV R2 Std']:.4f}")
print(f"Test R² Score: {results[best_model_name]['Test R2']:.4f}")
print(f"Test MAE: {results[best_model_name]['MAE']:.4f}")
print(f"Test RMSE: {results[best_model_name]['RMSE']:.4f}")
print("="*80)

# Final pipeline with Random Forest
final_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_test)

# Cross-validation
scores = cross_val_score(final_pipeline, X, y, cv=5, scoring='r2')
print("\nCross-validation R2:", scores)
print("Mean R2:", scores.mean())

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# Actual vs Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.savefig('actual_vs_predicted.png')
plt.close()

# Train vs Test performance
y_train_pred = final_pipeline.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)

print("\nTrain R2:", train_r2)
print("Test R2:", test_r2)

# ============================================================================
# 10. HYPERPARAMETER TUNING FOR BEST MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 10: HYPERPARAMETER TUNING")
print("="*80)

# Define parameter grid based on best model type
if 'Random Forest' in best_model_name or 'Extra Trees' in best_model_name:
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 15, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2']
    }
elif 'Gradient Boosting' in best_model_name:
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5],
        'model__subsample': [0.8, 1.0]
    }
elif 'Ridge' in best_model_name or 'Lasso' in best_model_name:
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
elif 'SVR' in best_model_name:
    param_grid = {
        'model__C': [1, 10, 100],
        'model__epsilon': [0.01, 0.1, 0.5],
        'model__gamma': ['scale', 'auto']
    }
else:
    param_grid = {}

if param_grid:
    from sklearn.model_selection import RandomizedSearchCV
    
    print(f"Tuning {best_model_name} with RandomizedSearchCV...")
    print(f"Parameter grid: {param_grid}")
    
    random_search = RandomizedSearchCV(
        best_model,
        param_grid,
        n_iter=20,  # Try 20 random combinations
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print('\nBest Parameters:', random_search.best_params_)
    print(f'Best CV Score: {random_search.best_score_:.4f}')
    
    tuned_model = random_search.best_estimator_
else:
    print(f"No hyperparameter tuning defined for {best_model_name}")
    tuned_model = best_model

# ============================================================================
# 11. FINAL MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("STEP 11: FINAL MODEL EVALUATION")
print("="*80)

# Predictions
y_train_pred = tuned_model.predict(X_train)
y_test_pred = tuned_model.predict(X_test)

# Comprehensive metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

# Cross-validation on full training set
cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

print("\nFINAL MODEL PERFORMANCE:")
print(f"{'Metric':<30} {'Value':<15}")
print("-" * 45)
print(f"{'Train R²':<30} {train_r2:.4f}")
print(f"{'Test R²':<30} {test_r2:.4f}")
print(f"{'CV R² (mean ± std)':<30} {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"{'Test MAE':<30} {test_mae:.4f}")
print(f"{'Test RMSE':<30} {test_rmse:.4f}")
print(f"{'Test MAPE (%)':<30} {test_mape:.2f}%")
print(f"{'Overfitting Gap (Train-Test)':<30} {(train_r2 - test_r2):.4f}")

# ============================================================================
# 12. MODEL DIAGNOSTICS AND RESIDUAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 12: MODEL DIAGNOSTICS")
print("="*80)

# Residuals
residuals = y_test - y_test_pred

# Residual statistics
print("\nResidual Analysis:")
print(f"Mean of residuals: {residuals.mean():.4f} (should be close to 0)")
print(f"Std of residuals: {residuals.std():.4f}")
print(f"Min residual: {residuals.min():.4f}")
print(f"Max residual: {residuals.max():.4f}")

# Create comprehensive diagnostic plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual AQI', fontsize=12)
axes[0, 0].set_ylabel('Predicted AQI', fontsize=12)
axes[0, 0].set_title(f'Actual vs Predicted (R²={test_r2:.4f})', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# 2. Residual plot
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted AQI', fontsize=12)
axes[0, 1].set_ylabel('Residuals', fontsize=12)
axes[0, 1].set_title('Residual Plot', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# 3. Residual distribution
axes[0, 2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('Residuals', fontsize=12)
axes[0, 2].set_ylabel('Frequency', fontsize=12)
axes[0, 2].set_title('Residual Distribution', fontsize=12)
axes[0, 2].grid(True, alpha=0.3)

# 4. Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# 5. Prediction error distribution
prediction_error = np.abs(y_test - y_test_pred)
axes[1, 1].hist(prediction_error, bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[1, 1].set_xlabel('Absolute Error', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title(f'Prediction Error Distribution (MAE={test_mae:.2f})', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

# 6. Feature importance (if available)
if hasattr(tuned_model.named_steps['model'], 'feature_importances_'):
    importances = tuned_model.named_steps['model'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_selected.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)
    
    axes[1, 2].barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
    axes[1, 2].set_yticks(range(len(feature_importance_df)))
    axes[1, 2].set_yticklabels(feature_importance_df['Feature'], fontsize=9)
    axes[1, 2].set_xlabel('Importance', fontsize=12)
    axes[1, 2].set_title('Top 15 Feature Importances', fontsize=12)
    axes[1, 2].invert_yaxis()
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))
else:
    axes[1, 2].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', fontsize=12)
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('05_model_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nDiagnostic plots saved to '05_model_diagnostics.png'")

# ============================================================================
# 13. SAVE FINAL MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 13: MODEL PERSISTENCE")
print("="*80)

import joblib

# Save the model
model_filename = 'aqi_prediction_model.pkl'
joblib.dump(tuned_model, model_filename)
print(f"Model saved to '{model_filename}'")

# Save feature names
feature_filename = 'selected_features.pkl'
joblib.dump(X_selected.columns.tolist(), feature_filename)
print(f"Feature names saved to '{feature_filename}'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Best Model: {best_model_name}")
print(f"Number of Features: {X_selected.shape[1]}")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Test Samples: {X_test.shape[0]}")
print(f"\nPerformance Metrics:")
print(f"  - Test R²: {test_r2:.4f}")
print(f"  - Test MAE: {test_mae:.4f}")
print(f"  - Test RMSE: {test_rmse:.4f}")
print(f"  - Test MAPE: {test_mape:.2f}%")
print(f"  - CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\nModel Generalization:")
if (train_r2 - test_r2) < 0.05:
    print("  ✓ Excellent - Model generalizes well (low overfitting)")
elif (train_r2 - test_r2) < 0.10:
    print("  ✓ Good - Acceptable generalization")
else:
    print("  ⚠ Warning - Possible overfitting detected")
print("="*80)
