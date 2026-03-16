"""
COM7022 – Machine Learning - Arden University
Student Name: Farid Negahbani
Student ID: 24154844
Data Preparation for Titanic Dataset
"""
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
