"""
COM7022 – Machine Learning
Lesson 3: Regression using Titanic Dataset
Student Name: Farid Negahbani
Student ID: 24154844

This script demonstrates:
1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. LASSO Regression

Dataset: Titanic dataset from seaborn
Target variable: fare
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------
# Create output folder
# -----------------------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------
# 1. Load Titanic dataset
# -----------------------------------
df = sns.load_dataset('titanic')

print("=" * 60)
print("Original Dataset Preview")
print("=" * 60)
print(df.head())

# -----------------------------------
# 2. Data Cleaning and Preparation
# -----------------------------------
df = df[['fare', 'age', 'pclass', 'sibsp', 'parch', 'sex', 'embarked']].copy()

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['fare'] = df['fare'].fillna(df['fare'].median())

df = df.drop_duplicates()

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = np.where(df['family_size'] == 1, 1, 0)

print("\n" + "=" * 60)
print("Cleaned Dataset Preview")
print("=" * 60)
print(df.head())

print("\nDataset Shape:", df.shape)

# -----------------------------------
# 3. Correlation analysis
# -----------------------------------
print("\n" + "=" * 60)
print("Correlation Matrix")
print("=" * 60)
print(df.corr(numeric_only=True))

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix - Titanic Regression Features")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------------
# 4. Simple Linear Regression
# Predict fare from age
# -----------------------------------
print("\n" + "=" * 60)
print("1. SIMPLE LINEAR REGRESSION")
print("=" * 60)

X_slr = df[['age']]
y_slr = df['fare']

X_train_slr, X_test_slr, y_train_slr, y_test_slr = train_test_split(
    X_slr, y_slr, test_size=0.2, random_state=42
)

slr_model = LinearRegression()
slr_model.fit(X_train_slr, y_train_slr)

y_pred_slr = slr_model.predict(X_test_slr)

slr_mse = mean_squared_error(y_test_slr, y_pred_slr)
slr_r2 = r2_score(y_test_slr, y_pred_slr)

print(f"Coefficient (m): {slr_model.coef_[0]:.4f}")
print(f"Intercept (c): {slr_model.intercept_:.4f}")
print(f"MSE: {slr_mse:.4f}")
print(f"R²: {slr_r2:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(X_test_slr, y_test_slr, color='blue', alpha=0.6, label='Actual Fare')
plt.scatter(X_test_slr, y_pred_slr, color='red', alpha=0.6, label='Predicted Fare')
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Simple Linear Regression: Age vs Fare")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "simple_linear_regression.png"), dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------------
# 5. Multiple Linear Regression
# -----------------------------------
print("\n" + "=" * 60)
print("2. MULTIPLE LINEAR REGRESSION")
print("=" * 60)

features_mlr = ['age', 'pclass', 'sibsp', 'parch', 'sex', 'embarked', 'family_size', 'is_alone']
X_mlr = df[features_mlr]
y_mlr = df['fare']

X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(
    X_mlr, y_mlr, test_size=0.2, random_state=42
)

mlr_model = LinearRegression()
mlr_model.fit(X_train_mlr, y_train_mlr)

y_pred_mlr = mlr_model.predict(X_test_mlr)

mlr_mse = mean_squared_error(y_test_mlr, y_pred_mlr)
mlr_r2 = r2_score(y_test_mlr, y_pred_mlr)

print("Coefficients:")
for feature, coef in zip(features_mlr, mlr_model.coef_):
    print(f"{feature}: {coef:.4f}")

print(f"Intercept: {mlr_model.intercept_:.4f}")
print(f"MSE: {mlr_mse:.4f}")
print(f"R²: {mlr_r2:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test_mlr, y_pred_mlr, alpha=0.7, color='green')
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Multiple Linear Regression: Actual vs Predicted Fare")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "multiple_linear_regression.png"), dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------------
# 6. Polynomial Regression
# -----------------------------------
print("\n" + "=" * 60)
print("3. POLYNOMIAL REGRESSION")
print("=" * 60)

degrees = [2, 3]

for degree in degrees:
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])

    poly_model.fit(X_train_slr, y_train_slr)
    y_pred_poly = poly_model.predict(X_test_slr)

    poly_mse = mean_squared_error(y_test_slr, y_pred_poly)
    poly_r2 = r2_score(y_test_slr, y_pred_poly)

    print(f"Degree {degree} -> MSE: {poly_mse:.4f}, R²: {poly_r2:.4f}")

poly_vis_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

poly_vis_model.fit(X_slr, y_slr)

X_range = np.linspace(X_slr.min(), X_slr.max(), 300).reshape(-1, 1)
y_range = poly_vis_model.predict(X_range)

plt.figure(figsize=(8, 5))
plt.scatter(X_slr, y_slr, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_range, y_range, color='red', linewidth=2, label='Polynomial Fit (Degree 2)')
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Polynomial Regression: Age vs Fare")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "polynomial_regression.png"), dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------------
# 7. LASSO Regression
# -----------------------------------
print("\n" + "=" * 60)
print("4. LASSO REGRESSION")
print("=" * 60)

X_lasso = df[features_mlr]
y_lasso = df['fare']

X_train_lasso, X_test_lasso, y_train_lasso, y_test_lasso = train_test_split(
    X_lasso, y_lasso, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_lasso)
X_test_scaled = scaler.transform(X_test_lasso)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train_lasso)

y_pred_lasso = lasso_model.predict(X_test_scaled)

lasso_mse = mean_squared_error(y_test_lasso, y_pred_lasso)
lasso_r2 = r2_score(y_test_lasso, y_pred_lasso)

print(f"MSE: {lasso_mse:.4f}")
print(f"R²: {lasso_r2:.4f}")

print("\nLASSO Coefficients:")
for feature, coef in zip(features_mlr, lasso_model.coef_):
    print(f"{feature}: {coef:.4f}")

important_features = [feature for feature, coef in zip(features_mlr, lasso_model.coef_) if abs(coef) > 1e-4]

print("\nImportant features selected by LASSO:")
print(important_features)

# save LASSO coefficients as bar chart
plt.figure(figsize=(10, 5))
plt.bar(features_mlr, lasso_model.coef_, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("LASSO Regression Coefficients")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "lasso_coefficients.png"), dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------------
# 8. Model comparison summary
# -----------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"Simple Linear Regression R²:   {slr_r2:.4f}")
print(f"Multiple Linear Regression R²: {mlr_r2:.4f}")
print(f"LASSO Regression R²:           {lasso_r2:.4f}")

print(f"\nAll output images saved in folder: {output_dir}")
print("Project completed successfully.")