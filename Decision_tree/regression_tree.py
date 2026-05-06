import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# Load Boston dataset
try:
    from sklearn.datasets import load_boston
    boston = load_boston()
except ImportError:
    # For newer sklearn versions, use fetch_openml
    from sklearn.datasets import fetch_openml
    boston_data = fetch_openml(name='boston', version=1, as_frame=False, parser='auto')
    boston = type('obj', (object,), {
        'data': boston_data.data,
        'target': boston_data.target,
        'feature_names': boston_data.feature_names
    })()

# Create DataFrame
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df['MEDV'] = boston.target

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Prepare features and target
X = df.iloc[:,0:13]
y = df.iloc[:,13]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Decision Tree Regressor...")
# Train model
rt = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
rt.fit(X_train, y_train)

# Make predictions
y_pred = rt.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
print(f"\nR² Score: {r2:.4f}")

print("\n=== Hyperparameter Tuning ===")
# Hyperparameter tuning
param_grid = {
    'max_depth': [2, 4, 8, 10, None],
    'criterion': ['squared_error', 'absolute_error'],
    'max_features': [0.25, 0.5, 1.0],
    'min_samples_split': [2, 5, 10]
}

reg = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, cv=5)
reg.fit(X_train, y_train)

print(f"\nBest Score: {reg.best_score_:.4f}")
print(f"Best Parameters: {reg.best_params_}")

print("\n=== Feature Importance ===")
for importance, name in sorted(zip(rt.feature_importances_, X_train.columns), reverse=True):
    print(f"{name:10s}: {importance:.4f}")
