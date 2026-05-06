import graphviz.backend as be
from sklearn.datasets import *
from sklearn import tree
from dtreeviz import dtreeviz, explain_prediction_path, rtreeviz_univar, rtreeviz_bivar_3D
from IPython.display import Image, display_svg, SVG
import numpy as np

print("=" * 60)
print("1. Classification Tree")
print("=" * 60)

clas = tree.DecisionTreeClassifier()  
iris = load_iris()

X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(clas, feature_names=iris.feature_names, class_names=["setosa", "versicolor", "virginica"], filled=True)
plt.savefig("classification_tree.png")
print("Saved classification tree to classification_tree.png")
plt.close()

print("\n" + "=" * 60)
print("2. Regression Tree")
print("=" * 60)

regr = tree.DecisionTreeRegressor(max_depth=1)
try:
    from sklearn.datasets import load_boston
    boston = load_boston()
except ImportError:
    from sklearn.datasets import fetch_openml
    boston_data = fetch_openml(name='boston', version=1, as_frame=False, parser='auto')
    boston = type('obj', (object,), {
        'data': boston_data.data,
        'target': boston_data.target,
        'feature_names': boston_data.feature_names
    })()

X_train = boston.data
y_train = boston.target
regr.fit(X_train, y_train)

print(f"Regression tree trained with max_depth=1")
print(f"Training score: {regr.score(X_train, y_train):.4f}")

print("\n" + "=" * 60)
print("3. Prediction Path Example")
print("=" * 60)

clas = tree.DecisionTreeClassifier()  
iris = load_iris()

X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)

X = iris.data[np.random.randint(0, len(iris.data)),:]

print(f"Sample: {X}")
print(f"\nPrediction path (plain english):")
print(explain_prediction_path(clas, X, feature_names=iris.feature_names, explanation_type="plain_english"))

print(f"\nPrediction path (sklearn default):")
print(explain_prediction_path(clas, X, feature_names=iris.feature_names, explanation_type="sklearn_default"))

print("\n" + "=" * 60)
print("4. Univariate Regression Tree Visualization (Cars)")
print("=" * 60)

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df_cars = pd.read_csv("cars.csv")
X, y = df_cars[['WGT']], df_cars['MPG']

dt = DecisionTreeRegressor(max_depth=3, criterion="absolute_error")
dt.fit(X, y)

print(f"Trained regression tree on cars data")
print(f"Features: Weight (WGT)")
print(f"Target: MPG")
print(f"Training score: {dt.score(X, y):.4f}")

fig = plt.figure(figsize=(12, 6))
ax = fig.gca()
rtreeviz_univar(dt, X, y, 'WGT', 'MPG', ax=ax)
plt.title("Decision Tree Regression: MPG vs Weight")
plt.savefig("cars_univariate.png", dpi=150, bbox_inches='tight')
print("Saved univariate visualization to cars_univariate.png")
plt.show()

print("\n" + "=" * 60)
print("5. Bivariate 3D Regression Tree Visualization (Cars)")
print("=" * 60)

from mpl_toolkits.mplot3d import Axes3D

X = df_cars[['WGT','ENG']]
y = df_cars['MPG']

dt = DecisionTreeRegressor(max_depth=3, criterion="absolute_error")
dt.fit(X, y)

print(f"Trained regression tree on cars data")
print(f"Features: Weight (WGT) and Engine/Horsepower (ENG)")
print(f"Target: MPG")
print(f"Training score: {dt.score(X, y):.4f}")

figsize = (10, 8)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='3d')

try:
    t = rtreeviz_bivar_3D(dt,
                          X, y,
                          feature_names=['Vehicle Weight', 'Horse Power'],
                          target_name='MPG',
                          fontsize=14,
                          elev=20,
                          azim=25,
                          dist=8.2,
                          show={'splits','title'},
                          ax=ax)
    plt.savefig("cars_bivariate_3d.png", dpi=150, bbox_inches='tight')
    print("Saved 3D bivariate visualization to cars_bivariate_3d.png")
    plt.show()
except Exception as e:
    print(f"Could not create 3D visualization: {e}")
    print("Creating alternative 2D scatter plot instead...")
    plt.close()
    
    # Alternative visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X['WGT'], X['ENG'], c=y, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Vehicle Weight (WGT)')
    ax.set_ylabel('Horse Power (ENG)')
    ax.set_title('MPG by Weight and Horsepower')
    plt.colorbar(scatter, label='MPG')
    plt.savefig("cars_bivariate_2d.png", dpi=150, bbox_inches='tight')
    print("Saved 2D scatter plot to cars_bivariate_2d.png")
    plt.show()

print("\n" + "=" * 60)
print("Script completed successfully!")
print("=" * 60)
