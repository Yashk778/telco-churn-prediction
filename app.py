import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load trained pipeline
# ---------------------------

model = pickle.load(open("Churnp_redictor_Model.pkl", "rb"))

st.title("Telco Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

# ---------------------------
# Customer Inputs
# ---------------------------

gender_input = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender_input == "Male" else 0

SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

partner_input = st.selectbox("Partner", ["Yes", "No"])
Partner = 1 if partner_input == "Yes" else 0

dependents_input = st.selectbox("Dependents", ["Yes", "No"])
Dependents = 1 if dependents_input == "Yes" else 0

tenure = st.slider("Tenure (months)", 0, 72)

phone_input = st.selectbox("Phone Service", ["Yes", "No"])
PhoneService = 1 if phone_input == "Yes" else 0

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

# Sent as string — pipeline handles encoding
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        'gender': [int(gender)],
        'SeniorCitizen': [int(SeniorCitizen)],
        'Partner': [int(Partner)],
        'Dependents': [int(Dependents)],
        'tenure': [int(tenure)],
        'PhoneService': [int(PhoneService)],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],  # string, not int
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [float(MonthlyCharges)],
        'TotalCharges': [float(TotalCharges)]
    })

    # Force numeric types only for manually encoded columns
    input_data = input_data.astype({
        "gender": "int64",
        "SeniorCitizen": "int64",
        "Partner": "int64",
        "Dependents": "int64",
        "tenure": "int64",
        "PhoneService": "int64",
        "MonthlyCharges": "float64",
        "TotalCharges": "float64"
    })

    # Ensure all object columns are string to avoid isnan type error
    for col in input_data.select_dtypes(include='object').columns:
        input_data[col] = input_data[col].astype(str)

    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"Customer likely to churn ⚠️ (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"Customer likely to stay ✅ (Churn Risk: {probability*100:.2f}%)")