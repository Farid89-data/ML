"""
COM7022 – Machine Learning - Arden University
Student Name: Farid Negahbani
Student ID: 24154844
Data Preparation for Titanic Dataset
"""

import pandas as pd
import numpy as np
import seaborn as sns

# Load Titanic dataset from seaborn
df = sns.load_dataset('titanic')

# Display first 5 rows
print(df.head())

# Shape of dataset
print("Shape of dataset:", df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# General information
print("\nDataset info:")
print(df.info())

# Data types of all attributes
print(df.dtypes)

print("Data type of age:", df['age'].dtype)
print("Data type of sex:", df['sex'].dtype)

# Select numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64'])
print("Numerical columns:")
print(numerical_cols.head())

# Select categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool'])
print("\nCategorical columns:")
print(categorical_cols.head())

# Check missing values
print(df.isnull().sum())
print(df.isnull().head())
print(df.notnull().head())
# Fill missing age with median
df['age'] = df['age'].fillna(df['age'].median())

# Fill missing fare with median if needed
df['fare'] = df['fare'].fillna(df['fare'].median())
# Fill missing embarked with mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Fill missing embark_town with mode
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
df = df.drop(columns=['deck'])
print("Duplicates before:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after:", df.duplicated().sum())
print(df.isnull().sum())
print(df.info())
extra_data = pd.DataFrame({
    'class': ['First', 'Second', 'Third'],
    'class_code': [1, 2, 3]
})

print(extra_data)
df = pd.merge(df, extra_data, on='class', how='left')

print(df.head())
top_rows = df.head(3)
bottom_rows = df.tail(3)

combined_rows = pd.concat([top_rows, bottom_rows], axis=0)
print(combined_rows)
df_reduced = df.drop(columns=['alive', 'who'], errors='ignore')
print(df_reduced.head())
sample_df = df_reduced.sample(n=10, random_state=42)
print(sample_df)
selected_features = df_reduced[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
print(selected_features.head())
df['family_size'] = df['sibsp'] + df['parch'] + 1
print(df[['sibsp', 'parch', 'family_size']].head())
df['is_alone'] = np.where(df['family_size'] == 1, 1, 0)
print(df[['family_size', 'is_alone']].head())
# Convert sex to numeric
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Convert embarked to numeric
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print(df[['sex', 'embarked']].head())
df_encoded = pd.get_dummies(df, columns=['embark_town'], drop_first=True)
print(df_encoded.head())
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

print(df[['age', 'fare']].head())
from sklearn.preprocessing import StandardScaler

scaler_std = StandardScaler()

df[['age', 'fare']] = scaler_std.fit_transform(df[['age', 'fare']])

print(df[['age', 'fare']].head())
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load Titanic dataset
df = sns.load_dataset('titanic')

# -----------------------------
# 1. Understanding the dataset
# -----------------------------
print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------
# 2. Data cleaning
# -----------------------------
# Fill missing numerical values
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())

# Fill missing categorical values
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

# Drop column with too many missing values
df = df.drop(columns=['deck'])

# Remove duplicates
df = df.drop_duplicates()

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -----------------------------
# 3. Data integration
# -----------------------------
extra_data = pd.DataFrame({
    'class': ['First', 'Second', 'Third'],
    'class_code': [1, 2, 3]
})

df = pd.merge(df, extra_data, on='class', how='left')

# -----------------------------
# 4. Data reduction
# -----------------------------
df = df.drop(columns=['alive', 'who'], errors='ignore')

# -----------------------------
# 5. Data transformation
# -----------------------------
# Create new features
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = np.where(df['family_size'] == 1, 1, 0)

# Encode categorical values
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Min-max scaling for age and fare
scaler = MinMaxScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

print("\nPrepared dataset:")
print(df.head())

print("\nFinal info:")
print(df.info())