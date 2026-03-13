# 🚢 Titanic Dataset – Data Preparation

**Course:** COM7022 – Machine Learning | Arden University  
**Student:** Farid Negahbani  
**Student ID:** 24154844

---

## 📋 Overview

This project performs end-to-end **data preparation** on the classic Titanic dataset using Python. It covers all major stages of the data preprocessing pipeline: understanding, cleaning, integration, reduction, and transformation — preparing the data for downstream machine learning tasks.

---

## 📁 Project Structure

```
titanic-data-preparation/
│
├── data_preparation.py       # Full data preparation pipeline (clean version)
├── data_preparation_dev.py   # Exploratory/development version with step-by-step prints
└── README.md
```

---

## ⚙️ Requirements

- Python 3.7+
- pandas
- numpy
- seaborn
- scikit-learn

Install dependencies with:

```bash
pip install pandas numpy seaborn scikit-learn
```

---

## 🚀 How to Run

```bash
python data_preparation.py
```

---

## 🔄 Pipeline Steps

### 1. 📊 Understanding the Dataset
- Load the Titanic dataset from `seaborn`
- Inspect shape, column names, data types, and missing values
- Separate numerical and categorical columns

### 2. 🧹 Data Cleaning
| Issue | Solution |
|---|---|
| Missing `age` values | Filled with **median** |
| Missing `fare` values | Filled with **median** |
| Missing `embarked` values | Filled with **mode** |
| Missing `embark_town` values | Filled with **mode** |
| `deck` column (too many nulls) | **Dropped** |
| Duplicate rows | **Removed** |

### 3. 🔗 Data Integration
- Created a supplementary `class_code` lookup table (`First=1`, `Second=2`, `Third=3`)
- Merged it into the main dataframe on the `class` column using a **left join**

### 4. ✂️ Data Reduction
- Dropped redundant columns: `alive`, `who`
- Sampled 10 random rows for verification (`random_state=42`)
- Selected key features: `survived`, `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`

### 5. 🔧 Data Transformation
- **Feature engineering:**
  - `family_size` = `sibsp` + `parch` + 1
  - `is_alone` = 1 if travelling alone, else 0
- **Encoding:**
  - `sex`: `male → 0`, `female → 1`
  - `embarked`: `S → 0`, `C → 1`, `Q → 2`
  - `embark_town`: One-hot encoded with `get_dummies` (drop first)
- **Scaling:**
  - `age` and `fare` scaled using **Min-Max Normalization** (final output)

---

## 📦 Dataset

The Titanic dataset is loaded directly via `seaborn`:

```python
import seaborn as sns
df = sns.load_dataset('titanic')
```

No external file download is required.

---

## 📝 Notes

- `StandardScaler` was explored during development but **MinMaxScaler** was chosen for the final pipeline
- The `deck` column was dropped due to ~77% missing values
- All transformations are applied in-place on the main dataframe `df`
