Telco Customer Churn Prediction
Overview

Customer churn is one of the biggest challenges faced by telecom companies. When customers stop using a company's services and switch to another provider, the company loses revenue and must spend additional resources to attract new customers.

The goal of this project is to build a machine learning model that can predict whether a telecom customer is likely to churn based on their service usage, account details, and billing information.

By predicting churn in advance, telecom companies can take proactive steps such as offering better plans or incentives to retain customers.

Dataset

The dataset used in this project contains customer information from a telecom company. It includes features such as:

Customer demographics (gender, senior citizen status, etc.)

Account information (tenure, contract type, payment method)

Services subscribed (internet service, streaming services, security services)

Billing details (monthly charges and total charges)

Customer churn status

The target variable in this project is Churn, which indicates whether the customer left the company.

Project Workflow

The project was divided into multiple stages:

1. Data Preprocessing

Loaded the dataset and inspected its structure

Handled missing values

Corrected data types

Removed unnecessary columns

Prepared the dataset for analysis

2. Exploratory Data Analysis (EDA)

Exploratory analysis was performed to understand how different features affect customer churn.

Some key observations from the data include:

Customers with month-to-month contracts have the highest churn rate.

Customers with longer tenure tend to stay with the company.

Higher monthly charges are associated with higher churn.

Services like online security and tech support reduce churn probability.

3. Feature Engineering and Encoding

To prepare the data for machine learning:

Binary categorical features were converted into numeric form.

Categorical variables were encoded using OneHotEncoder and OrdinalEncoder.

A scikit-learn pipeline was used to combine preprocessing and modeling steps.

4. Model Training

Multiple machine learning models were trained and compared:

Logistic Regression

Decision Tree

Random Forest

XGBoost

LightGBM

Each model was evaluated based on its performance on the test dataset.

5. Final Model Selection

After comparing the models, Random Forest provided the best performance with an accuracy of approximately 80%, so it was selected as the final model.

Model Evaluation

The model was evaluated using:

Accuracy Score

Classification Report

Confusion Matrix

The final Random Forest model achieved around 80% accuracy on the test dataset.

Technologies Used

The following tools and libraries were used in this project:

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

XGBoost

LightGBM

Jupyter Notebook

Project Structure
telco-churn-prediction
│
├── notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_training.ipynb
│
├── churn_model.pkl
├── README.md

Future Improvements

Possible improvements for this project include:

Performing deeper hyperparameter tuning

Addressing class imbalance using techniques like SMOTE

Deploying the model using a web application (Streamlit or Flask)

Testing additional machine learning algorithms

Conclusion

This project demonstrates a complete machine learning workflow, starting from data preprocessing and exploratory analysis to model training and evaluation. The Random Forest model was able to predict customer churn with approximately 80% accuracy, showing that machine learning can be effectively used to identify customers who are at risk of leaving.