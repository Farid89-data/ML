"""
COM7022 – Machine Learning - Arden University
Student Name: Farid Negahbani
Student ID: 24154844
Data Preparation for Titanic Dataset
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# -----------------------------
# 1. Load Titanic dataset
# -----------------------------
df = sns.load_dataset('titanic')

# -----------------------------
# 2. Understanding the dataset
# -----------------------------
print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset info:")
print(df.info())

print("\nData types:")
print(df.dtypes)

print("\nData type of age:", df['age'].dtype)
print("Data type of sex:", df['sex'].dtype)

# -----------------------------
# 3. Separate numerical and categorical columns
# -----------------------------
numerical_cols = df.select_dtypes(include=['int64', 'float64'])
print("\nNumerical columns:")
print(numerical_cols.head())

categorical_cols = df.select_dtypes(include=['object', 'category', 'bool'])
print("\nCategorical columns:")
print(categorical_cols.head())

# -----------------------------
# 4. Missing value analysis
# -----------------------------
print("\nMissing values count:")
print(df.isnull().sum())

print("\nCheck null values:")
print(df.isnull().head())

print("\nCheck not null values:")
print(df.notnull().head())

# -----------------------------
# 5. Data Cleaning
# -----------------------------
# Fill missing numerical values
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())

# Fill missing categorical values
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

# Drop column with many missing values
df = df.drop(columns=['deck'])

# Remove duplicates
print("\nDuplicates before:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates after:", df.duplicated().sum())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -----------------------------
# 6. Data Integration
# -----------------------------
extra_data = pd.DataFrame({
    'class': ['First', 'Second', 'Third'],
    'class_code': [1, 2, 3]
})

print("\nExtra Data:")
print(extra_data)

df = pd.merge(df, extra_data, on='class', how='left')

print("\nAfter merge:")
print(df.head())

# -----------------------------
# 7. Data Reduction
# -----------------------------
df_reduced = df.drop(columns=['alive', 'who'], errors='ignore')
print("\nReduced dataset:")
print(df_reduced.head())

# Sampling example
sample_df = df_reduced.sample(n=10, random_state=42)
print("\nSample rows:")
print(sample_df)

# -----------------------------
# 8. Data Transformation
# -----------------------------
# Create new features
df['family_size'] = df['sibsp'] + df['parch'] + 1
print("\nFamily size:")
print(df[['sibsp', 'parch', 'family_size']].head())

df['is_alone'] = np.where(df['family_size'] == 1, 1, 0)
print("\nIs alone feature:")
print(df[['family_size', 'is_alone']].head())

# Encode categorical variables
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("\nEncoded values:")
print(df[['sex', 'embarked']].head())

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['embark_town'], drop_first=True)
print("\nOne-hot encoded:")
print(df_encoded.head())

# -----------------------------
# 9. Feature Scaling
# -----------------------------
# Min-Max Scaling
scaler = MinMaxScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

print("\nMinMax scaled values:")
print(df[['age', 'fare']].head())

# Standard Scaling
scaler_std = StandardScaler()
df[['age', 'fare']] = scaler_std.fit_transform(df[['age', 'fare']])

print("\nStandard scaled values:")
print(df[['age', 'fare']].head())

# -----------------------------
# 10. Final dataset info
# -----------------------------
print("\nFinal dataset info:")
print(df.info())
