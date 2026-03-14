# 📡 Telco Customer Churn Prediction

A machine learning project to predict whether a telecom customer is likely to churn, built with Random Forest and deployed via Streamlit.

🔗 **[Live Demo](https://telco-churn-prediction-fj2vhyj55dvcjo8karokro.streamlit.app/)**

---

## 📌 Overview

Customer churn is one of the biggest challenges for telecom companies. This project builds an ML pipeline to predict churn based on customer demographics, services, and billing data — helping companies take proactive retention steps.

---

## 📂 Dataset

The dataset includes:
- Customer demographics (gender, senior citizen, partner, dependents)
- Account info (tenure, contract type, payment method)
- Services subscribed (internet, streaming, security, tech support)
- Billing details (monthly charges, total charges)
- Target: `Churn` (Yes/No)

---

## 🔄 Project Workflow

### 1. Data Preprocessing
- Handled missing values and corrected data types
- Removed unnecessary columns
- Prepared cleaned dataset

### 2. Exploratory Data Analysis (EDA)
Key findings:
- Month-to-month contracts → highest churn rate
- Longer tenure → lower churn
- Higher monthly charges → higher churn probability
- Tech support & online security → reduce churn

### 3. Feature Engineering
- Binary features manually encoded (Yes/No → 1/0)
- `OrdinalEncoder` for Contract
- `OneHotEncoder` for remaining categorical features
- scikit-learn `Pipeline` for end-to-end preprocessing + modeling

### 4. Models Trained

| Model | Accuracy |
|---|---|
| Logistic Regression | ~78% |
| Decision Tree | ~72% |
| Random Forest | ~80% ✅ |
| XGBoost | ~76% |
| LightGBM | ~78% |

### 5. Final Model
**Random Forest** with hyperparameter tuning (`n_estimators=300`, `max_depth=12`) — **~80% accuracy**

---

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `LightGBM` `Matplotlib` `Seaborn` `Streamlit`

---

## 📁 Project Structure
```
telco-churn-prediction/
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_training.ipynb
│
├── app.py
├── churn_model.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Improvements
- Handle class imbalance with SMOTE
- Deeper hyperparameter tuning
- Try additional algorithms (CatBoost, etc.)