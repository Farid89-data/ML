# 🚢 Titanic Dataset – Lesson 3 Regression Analysis

**Course:** COM7022 – Machine Learning  
**University:** Arden University  
**Student:** Farid Negahbani  
**Student ID:** 24154844  

---

## 📋 Overview

This project applies regression techniques from **Lesson 3** to the Titanic dataset using Python.

Although the Titanic dataset is typically used for classification tasks, this project reformulates it as a **regression problem** by predicting passenger **fare**.

### 🔍 Techniques Used

- Simple Linear Regression  
- Multiple Linear Regression  
- Polynomial Regression  
- LASSO Regression  

### 🧠 Key Components

- Data cleaning and preprocessing  
- Correlation analysis  
- Feature engineering  
- Model evaluation using **MSE** and **R²**  
- Saving visual outputs  

---

## 🎯 Project Objective

The main objective is to enhance the Titanic dataset analysis by applying regression models and comparing their performance.

- **Target Variable:** `fare`

---

## 📁 Project Structure

```text
titanic-regression-project/
│
├── Data_Preparation.py
├── titanic_regression_lesson3.py
├── README.md
└── output/
    ├── correlation_matrix.png
    ├── simple_linear_regression.png
    ├── multiple_linear_regression.png
    ├── polynomial_regression.png
    └── lasso_coefficients.png
'''# 🚢 Titanic Dataset – Lesson 3 Regression Analysis

**Course:** COM7022 – Machine Learning  
**University:** Arden University  
**Student:** Farid Negahbani  
**Student ID:** 24154844  

---

## 📋 Overview

This project applies regression techniques from **Lesson 3** to the Titanic dataset using Python.

Although the Titanic dataset is typically used for classification tasks, this project reformulates it as a **regression problem** by predicting passenger **fare**.

### 🔍 Techniques Used

- Simple Linear Regression  
- Multiple Linear Regression  
- Polynomial Regression  
- LASSO Regression  

### 🧠 Key Components

- Data cleaning and preprocessing  
- Correlation analysis  
- Feature engineering  
- Model evaluation using **MSE** and **R²**  
- Saving visual outputs  

---

## 🎯 Project Objective

The main objective is to enhance the Titanic dataset analysis by applying regression models and comparing their performance.

- **Target Variable:** `fare`

---

## 📁 Project Structure

titanic-regression-project/
│
├── Data_Preparation.py
├── titanic_regression_lesson3.py
├── README.md
└── output/
    ├── correlation_matrix.png
    ├── simple_linear_regression.png
    ├── multiple_linear_regression.png
    ├── polynomial_regression.png
    └── lasso_coefficients.png

---

## ⚙️ Requirements

Python 3.8+
pandas
numpy
seaborn
matplotlib
scikit-learn

Install dependencies:

pip install pandas numpy seaborn matplotlib scikit-learn

---

## 📦 Dataset

The dataset is loaded directly using `seaborn`:

import seaborn as sns
df = sns.load_dataset('titanic')

No external download is required.

---

## 🧹 Data Preparation

### ✔️ Selected Features

- `fare` (target)
- `age`
- `pclass`
- `sibsp`
- `parch`
- `sex`
- `embarked`

### ✔️ Data Cleaning

- Filled missing values:
  - `age` → median  
  - `fare` → median  
  - `embarked` → mode  
- Removed duplicates  

### ✔️ Encoding

- `sex`: male → 0, female → 1  
- `embarked`: S → 0, C → 1, Q → 2  

### ✔️ Feature Engineering

- `family_size = sibsp + parch + 1`  
- `is_alone = 1` if alone, else `0`  

### 📊 Final Dataset Shape

(758, 9)

---

## 📊 Correlation Analysis

### 🔑 Key Insights

- Strong negative correlation:
  - `fare` vs `pclass` → -0.5569
- Weak correlation:
  - `fare` vs `age` → 0.0921
- Strong relationships:
  - `family_size` with `sibsp` and `parch`
  - `is_alone` negatively with `family_size`

---

## 🔄 Regression Models

### 1️⃣ Simple Linear Regression

- Predictor: `age`

MSE: 2217.4961  
R2: -0.0045  

Conclusion:
- Very poor performance  
- `age` is not useful alone  

---

### 2️⃣ Multiple Linear Regression

MSE: 1084.2115  
R2: 0.5089  

Conclusion:
- Significant improvement  
- Explains ~51% of variance  

---

### 3️⃣ Polynomial Regression

Degree 2 → R2: -0.0054  
Degree 3 → R2: -0.0259  

Conclusion:
- No improvement  

---

### 4️⃣ LASSO Regression

MSE: 1084.4644  
R2: 0.5088  

Conclusion:
- Similar to multiple regression  
- Performs feature selection  

---

## 📈 Model Comparison

Simple Linear Regression:   R2 = -0.0045  
Multiple Linear Regression: R2 = 0.5089  
LASSO Regression:           R2 = 0.5088  

---

## 🏆 Best Model

Multiple Linear Regression

---

## 🖼️ Output Visualisations

- correlation_matrix.png  
- simple_linear_regression.png  
- multiple_linear_regression.png  
- polynomial_regression.png  
- lasso_coefficients.png  

---

## 🚀 How to Run

python titanic_regression_lesson3.py

---

## 📌 Key Findings

- `age` alone is not a good predictor  
- Combining features improves accuracy  
- `pclass` is highly influential  
- LASSO helps feature selection  

---

## 📚 Conclusion

This project demonstrates the importance of:
- Feature selection  
- Combining variables  
- Choosing appropriate models  
# 🚢 Titanic Dataset – Lesson 3 Regression Analysis

**Course:** COM7022 – Machine Learning  
**University:** Arden University  
**Student:** Farid Negahbani  
**Student ID:** 24154844  

---

## 📋 Overview

This project applies regression techniques from **Lesson 3** to the Titanic dataset using Python.

Although the Titanic dataset is typically used for classification tasks, this project reformulates it as a **regression problem** by predicting passenger **fare**.

### 🔍 Techniques Used

- Simple Linear Regression  
- Multiple Linear Regression  
- Polynomial Regression  
- LASSO Regression  

### 🧠 Key Components

- Data cleaning and preprocessing  
- Correlation analysis  
- Feature engineering  
- Model evaluation using **MSE** and **R²**  
- Saving visual outputs  

---

## 🎯 Project Objective

The main objective is to enhance the Titanic dataset analysis by applying regression models and comparing their performance.

- **Target Variable:** `fare`

---

## 📁 Project Structure

titanic-regression-project/
│
├── Data_Preparation.py
├── titanic_regression_lesson3.py
├── README.md
└── output/
    ├── correlation_matrix.png
    ├── simple_linear_regression.png
    ├── multiple_linear_regression.png
    ├── polynomial_regression.png
    └── lasso_coefficients.png

---

## ⚙️ Requirements

Python 3.8+
pandas
numpy
seaborn
matplotlib
scikit-learn

Install dependencies:

pip install pandas numpy seaborn matplotlib scikit-learn

---

## 📦 Dataset

The dataset is loaded directly using `seaborn`:

import seaborn as sns
df = sns.load_dataset('titanic')

No external download is required.

---

## 🧹 Data Preparation

### ✔️ Selected Features

- `fare` (target)
- `age`
- `pclass`
- `sibsp`
- `parch`
- `sex`
- `embarked`

### ✔️ Data Cleaning

- Filled missing values:
  - `age` → median  
  - `fare` → median  
  - `embarked` → mode  
- Removed duplicates  

### ✔️ Encoding

- `sex`: male → 0, female → 1  
- `embarked`: S → 0, C → 1, Q → 2  

### ✔️ Feature Engineering

- `family_size = sibsp + parch + 1`  
- `is_alone = 1` if alone, else `0`  

### 📊 Final Dataset Shape

(758, 9)

---

## 📊 Correlation Analysis

### 🔑 Key Insights

- Strong negative correlation:
  - `fare` vs `pclass` → -0.5569
- Weak correlation:
  - `fare` vs `age` → 0.0921
- Strong relationships:
  - `family_size` with `sibsp` and `parch`
  - `is_alone` negatively with `family_size`

---

## 🔄 Regression Models

### 1️⃣ Simple Linear Regression

- Predictor: `age`

MSE: 2217.4961  
R2: -0.0045  

Conclusion:
- Very poor performance  
- `age` is not useful alone  

---

### 2️⃣ Multiple Linear Regression

MSE: 1084.2115  
R2: 0.5089  

Conclusion:
- Significant improvement  
- Explains ~51% of variance  

---

### 3️⃣ Polynomial Regression

Degree 2 → R2: -0.0054  
Degree 3 → R2: -0.0259  

Conclusion:
- No improvement  

---

### 4️⃣ LASSO Regression

MSE: 1084.4644  
R2: 0.5088  

Conclusion:
- Similar to multiple regression  
- Performs feature selection  

---

## 📈 Model Comparison

Simple Linear Regression:   R2 = -0.0045  
Multiple Linear Regression: R2 = 0.5089  
LASSO Regression:           R2 = 0.5088  

---

## 🏆 Best Model

Multiple Linear Regression

---

## 🖼️ Output Visualisations

- correlation_matrix.png  
- simple_linear_regression.png  
- multiple_linear_regression.png  
- polynomial_regression.png  
- lasso_coefficients.png  

---

## 🚀 How to Run

python titanic_regression_lesson3.py

---

## 📌 Key Findings

- `age` alone is not a good predictor  
- Combining features improves accuracy  
- `pclass` is highly influential  
- LASSO helps feature selection  

---

## 📚 Conclusion

This project demonstrates the importance of:
- Feature selection  
- Combining variables  
- Choosing appropriate models  

---

## ⚙️ Requirements

Python 3.8+
pandas
numpy
seaborn
matplotlib
scikit-learn

Install dependencies:

pip install pandas numpy seaborn matplotlib scikit-learn

---

## 📦 Dataset

The dataset is loaded directly using `seaborn`:

import seaborn as sns
df = sns.load_dataset('titanic')

No external download is required.

---

## 🧹 Data Preparation

### ✔️ Selected Features

- `fare` (target)
- `age`
- `pclass`
- `sibsp`
- `parch`
- `sex`
- `embarked`

### ✔️ Data Cleaning

- Filled missing values:
  - `age` → median  
  - `fare` → median  
  - `embarked` → mode  
- Removed duplicates  

### ✔️ Encoding

- `sex`: male → 0, female → 1  
- `embarked`: S → 0, C → 1, Q → 2  

### ✔️ Feature Engineering

- `family_size = sibsp + parch + 1`  
- `is_alone = 1` if alone, else `0`  

### 📊 Final Dataset Shape

(758, 9)

---

## 📊 Correlation Analysis

### 🔑 Key Insights

- Strong negative correlation:
  - `fare` vs `pclass` → -0.5569
- Weak correlation:
  - `fare` vs `age` → 0.0921
- Strong relationships:
  - `family_size` with `sibsp` and `parch`
  - `is_alone` negatively with `family_size`

---

## 🔄 Regression Models

### 1️⃣ Simple Linear Regression

- Predictor: `age`

MSE: 2217.4961  
R2: -0.0045  

Conclusion:
- Very poor performance  
- `age` is not useful alone  

---

### 2️⃣ Multiple Linear Regression

MSE: 1084.2115  
R2: 0.5089  

Conclusion:
- Significant improvement  
- Explains ~51% of variance  

---

### 3️⃣ Polynomial Regression

Degree 2 → R2: -0.0054  
Degree 3 → R2: -0.0259  

Conclusion:
- No improvement  

---

### 4️⃣ LASSO Regression

MSE: 1084.4644  
R2: 0.5088  

Conclusion:
- Similar to multiple regression  
- Performs feature selection  

---

## 📈 Model Comparison

Simple Linear Regression:   R2 = -0.0045  
Multiple Linear Regression: R2 = 0.5089  
LASSO Regression:           R2 = 0.5088  

---

## 🏆 Best Model

Multiple Linear Regression

---

## 🖼️ Output Visualisations

- correlation_matrix.png  
- simple_linear_regression.png  
- multiple_linear_regression.png  
- polynomial_regression.png  
- lasso_coefficients.png  

---

## 🚀 How to Run

python titanic_regression_lesson3.py

---

## 📌 Key Findings

- `age` alone is not a good predictor  
- Combining features improves accuracy  
- `pclass` is highly influential  
- LASSO helps feature selection  

---

## 📚 Conclusion

This project demonstrates the importance of:
- Feature selection  
- Combining variables  
- Choosing appropriate models  
