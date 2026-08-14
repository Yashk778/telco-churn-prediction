# 📡 Telco Customer Churn Prediction

An end-to-end machine learning project for predicting whether a telecom customer is likely to churn.

The project covers the complete Data Science workflow — from data cleaning and exploratory data analysis to feature engineering, model comparison, cross-validation, hyperparameter tuning, classification threshold optimization, final evaluation, and Streamlit deployment.

🔗 **[🚀 Live Demo](https://telco-churn-prediction-fj2vhyj55dvcjo8karokro.streamlit.app/)**

---

## 📌 Problem Statement

Customer churn is a major challenge for telecom companies. Identifying customers who are likely to leave can help businesses take proactive retention actions.

The goal of this project is to build a machine learning model that predicts customer churn based on:

- Customer demographics
- Account information
- Services subscribed
- Contract details
- Payment method
- Monthly charges
- Total charges
- Customer tenure

Because the dataset is imbalanced, the project focuses not only on accuracy but also on **Precision, Recall, F1-score, ROC-AUC, and PR-AUC**.

---

## 📂 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains information about:

### Customer Demographics
- Gender
- Senior Citizen
- Partner
- Dependents

### Account Information
- Tenure
- Contract
- Paperless Billing
- Payment Method

### Services
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### Billing
- Monthly Charges
- Total Charges

### Target

`Churn`

- `Yes` → Customer churned
- `No` → Customer stayed

---

# 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Preprocessing Pipeline
     ↓
Baseline Model
     ↓
Model Evaluation
     ↓
Multicollinearity Analysis
     ↓
Feature Removal Experiment
     ↓
Model Comparison
     ↓
Cross-Validation
     ↓
ROC-AUC & PR-AUC
     ↓
Promising Model Selection
     ↓
Hyperparameter Tuning
     ↓
Threshold Optimization
     ↓
Final Model Evaluation
     ↓
Streamlit Deployment
```

---

# 🧹 1. Data Cleaning

The dataset was inspected for:

- Missing values
- Duplicate records
- Incorrect data types
- Unnecessary columns
- Numerical and categorical features

### `TotalCharges`

`TotalCharges` was initially stored as an object instead of a numerical feature.

It was converted to numeric values and missing values created during conversion were handled appropriately.

The customer ID column was also excluded because it does not provide useful predictive information.

---

# 📊 2. Exploratory Data Analysis

EDA was performed to understand the structure of the dataset and identify patterns related to customer churn.

The analysis included:

- Numerical feature distributions
- Categorical feature distributions
- Target class distribution
- Numerical feature relationships
- Numerical features vs churn
- Categorical features vs churn
- Correlation analysis
- Multicollinearity investigation

## Key Findings

Some important patterns observed during EDA were:

- **Month-to-month customers show substantially higher churn**
- Customers with **shorter tenure are more likely to churn**
- Higher **monthly charges** are associated with higher churn
- Customers without services such as **Online Security** and **Tech Support** show higher churn
- Contract type, internet service, and payment method show noticeable differences in churn behavior
- `tenure` and `TotalCharges` have a strong positive correlation

---

# ⚙️ 3. Feature Engineering

Two domain-based features were created.

## `household_type`

This feature combines `Partner` and `Dependents`:

| Partner | Dependents | household_type |
|---|---|---|
| No | No | Single |
| Yes | No | Couple |
| No | Yes | Single Parent |
| Yes | Yes | Family |

## `long_term_customer`

This feature is derived from customer tenure:

```text
tenure >= 24 months → Yes
tenure < 24 months  → No
```

These engineered features are automatically generated from the user's inputs in the deployed Streamlit application.

---

# 🔀 4. Train/Test Split

The dataset was divided into training and testing sets using an **80/20 split**.

Stratification was used to maintain a similar churn class distribution in both sets.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

---

# 🛠️ 5. Data Preprocessing

A `ColumnTransformer` and scikit-learn `Pipeline` were used to create an end-to-end preprocessing and modeling workflow.

### Encoding

- `OrdinalEncoder` for binary categorical features
- `OrdinalEncoder` for the ordered `Contract` feature
- `OneHotEncoder` for nominal categorical features

### Scaling

`StandardScaler` was used for numerical features in the Logistic Regression pipeline.

Tree-based models were evaluated without unnecessary numerical scaling.

Using a pipeline ensures that preprocessing is consistently applied during both training and prediction.

---

# 🧪 6. Baseline Model

Logistic Regression was initially used as the baseline model.

The baseline was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Because the churn class is imbalanced, **Churn-class Recall and F1-score** were given particular attention.

---

# 🔎 7. Multicollinearity Analysis

Correlation analysis identified a strong relationship between:

- `tenure`
- `TotalCharges`

Variance Inflation Factor (VIF) was then used to investigate multicollinearity.

Rather than automatically removing highly correlated features, feature-removal experiments were performed.

The effect of removing:

- `tenure`
- `TotalCharges`
- both features

was compared using model performance.

The experiments showed that removing these features did not provide an improvement sufficient to justify removing them, so the original features were retained.

---

# 🤖 8. Model Comparison

Multiple classification algorithms were evaluated using a common preprocessing and evaluation workflow.

The models tested were:

- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Logistic Regression

### Initial Test Performance

| Model | Churn F1 | ROC-AUC |
|---|---:|---:|
| Decision Tree | 0.49 | 0.649 |
| Random Forest | 0.55 | 0.817 |
| Gradient Boosting | 0.59 | 0.844 |
| XGBoost | 0.54 | 0.821 |
| Logistic Regression | 0.61 | 0.843 |

The results showed that Logistic Regression and Gradient Boosting were the most promising candidates for further evaluation and optimization.

---

# 🔁 9. Cross-Validation

To obtain a more reliable estimate of model performance, **5-fold Stratified Cross-Validation** was performed.

F1-score was used as the primary cross-validation metric because the churn class is imbalanced and a balance between precision and recall is important.

### Mean Cross-Validation F1

| Model | Mean F1 |
|---|---:|
| Decision Tree | 0.492 |
| Random Forest | 0.549 |
| Gradient Boosting | 0.588 |
| XGBoost | 0.547 |
| Logistic Regression | 0.596 |

Logistic Regression achieved the highest mean cross-validation F1-score among the evaluated models.

---

# 📈 10. ROC-AUC Analysis

ROC-AUC was used to evaluate how well each model distinguishes between churn and non-churn customers across different classification thresholds.

Results:

| Model | ROC-AUC |
|---|---:|
| Decision Tree | 0.649 |
| Random Forest | 0.817 |
| Gradient Boosting | 0.844 |
| XGBoost | 0.821 |
| Logistic Regression | 0.842 |

Gradient Boosting and Logistic Regression demonstrated strong ranking performance.

---

# 📉 11. Precision-Recall AUC

Since the churn class is imbalanced, the Precision-Recall curve provides additional insight into the trade-off between precision and recall for the positive class.

### PR-AUC

| Model | PR-AUC |
|---|---:|
| Decision Tree | 0.365 |
| Random Forest | 0.609 |
| Gradient Boosting | 0.663 |
| XGBoost | 0.608 |
| Logistic Regression | 0.634 |

Gradient Boosting achieved the highest PR-AUC, while Logistic Regression also showed strong performance.

Therefore, both models were retained as promising candidates for further optimization.

---

# 🎛️ 12. Hyperparameter Tuning

The promising models were optimized using **RandomizedSearchCV with 5-fold cross-validation**.

F1-score was used as the optimization metric.

## Logistic Regression

Best parameters:

```text
C = 100
solver = lbfgs
```

Best CV F1:

```text
≈ 0.600
```

## Gradient Boosting

Best parameters:

```text
n_estimators = 50
learning_rate = 0.2
max_depth = 3
min_samples_split = 20
min_samples_leaf = 4
```

Best CV F1:

```text
≈ 0.588
```

Hyperparameter tuning produced only a small improvement, showing that changing model parameters alone did not substantially improve the models.

---

# 🎚️ 13. Classification Threshold Optimization

The default classification threshold of `0.50` was investigated because the project places importance on identifying customers who are likely to churn.

Instead of selecting the threshold using the test set, **out-of-fold probability predictions from the training data** were used to evaluate different thresholds.

The threshold that produced the best cross-validation F1-score was selected.

### Selected Thresholds

| Model | Optimal Threshold |
|---|---:|
| Logistic Regression | **0.32** |
| Gradient Boosting | **0.34** |

Lowering the threshold increases the number of customers classified as potential churners, improving recall at the cost of some precision.

---

# 🏆 14. Final Model

After comparing the models, performing cross-validation, hyperparameter tuning, and threshold optimization, **Logistic Regression** was selected as the final deployed model.

The final classification threshold is:

```text
0.32
```

---

# 📊 15. Final Test Performance

The final Logistic Regression pipeline was evaluated on the test set using the selected threshold.

| Metric | Score |
|---|---:|
| Accuracy | **0.76** |
| Churn Precision | **0.53** |
| Churn Recall | **0.72** |
| Churn F1-score | **0.61** |
| ROC-AUC | **0.841** |
| PR-AUC | **0.625** |

### Confusion Matrix

```text
                 Predicted
              No Churn   Churn

Actual
No Churn          794      241
Churn             103      271
```

The final model correctly identified **271 out of 374 actual churn customers**, resulting in approximately **72% recall** for the churn class.

Compared with the default threshold, threshold optimization substantially improved churn recall while maintaining a similar F1-score.

---

# 🚀 16. Streamlit Deployment

The final model is deployed using Streamlit.

The saved model contains the complete preprocessing and Logistic Regression pipeline.

The application performs the following steps:

```text
User Input
    ↓
Feature Engineering
    ↓
Preprocessing Pipeline
    ↓
Logistic Regression
    ↓
Churn Probability
    ↓
Threshold = 0.32
    ↓
Churn / No Churn
```

The application automatically creates:

- `household_type`
- `long_term_customer`

from the user-provided information before passing the data to the model.

### 🌐 Live Demo

🔗 **[Launch the Telco Customer Churn Prediction App](https://telco-churn-prediction-fj2vhyj55dvcjo8karokro.streamlit.app/)**

---

# 🛠️ Tech Stack

### Programming & Data

- Python
- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

### Visualization

- Matplotlib
- Seaborn

### Deployment

- Streamlit
- Pickle

### Development

- Jupyter Notebook
- Git
- GitHub

---

# 📁 Project Structure

```text
telco-churn-prediction/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
│
├── app.py
├── Churn_Predictor_Model.pkl
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Yashk778/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 💡 Key Learnings

This project provided practical experience with:

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Handling imbalanced classification
- Encoding categorical variables
- Feature scaling
- Building reusable ML pipelines
- Multicollinearity and VIF
- Model comparison
- Cross-validation
- Precision, Recall and F1-score
- ROC-AUC and Precision-Recall AUC
- Hyperparameter tuning
- Classification threshold optimization
- Model serialization
- Streamlit deployment

---

# 🔮 Future Improvements

Potential future improvements include:

- Exploring additional domain-specific features
- Cost-sensitive learning based on business requirements
- Probability calibration
- Further investigation of class imbalance
- Model monitoring after deployment
- Testing the model on newer customer data
- Incorporating business-specific churn costs into threshold selection

---

## 👨‍💻 Author

**Yash Kamod**

Computer Science & Design

🔗 **[GitHub](https://github.com/Yashk778)**

---

⭐ If you found this project useful, feel free to explore the repository and try the live demo!