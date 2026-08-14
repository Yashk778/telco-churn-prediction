import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load final trained pipeline
# ---------------------------

model = pickle.load(
    open("Churn_Predictor_Model.pkl", "rb")
)

# Selected during threshold tuning
FINAL_THRESHOLD = 0.32

st.title("Telco Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")


# ---------------------------
# Customer Inputs
# ---------------------------

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure (months)",
    0,
    72
)

PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

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

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=0.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=0.0
)


# ---------------------------
# Feature Engineering
# ---------------------------

def create_household_type(partner, dependents):

    if partner == "Yes" and dependents == "No":
        return "Couple"

    if partner == "No" and dependents == "Yes":
        return "Single Parent"

    if partner == "Yes" and dependents == "Yes":
        return "Family"

    return "Single"


household_type = create_household_type(
    Partner,
    Dependents
)

# Same rule used during training:
# tenure >= 24 → Yes
long_term_customer = (
    "Yes" if tenure >= 24 else "No"
)


# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [int(SeniorCitizen)],
        "tenure": [int(tenure)],
        "PhoneService": [PhoneService],

        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],

        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],

        "MonthlyCharges": [float(MonthlyCharges)],
        "TotalCharges": [float(TotalCharges)],

        # Engineered features used during training
        "household_type": [household_type],
        "long_term_customer": [long_term_customer]
    })


    # ---------------------------
    # Get churn probability
    # ---------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]


    # Apply selected threshold
    prediction = (
        1 if probability >= FINAL_THRESHOLD else 0
    )


    # ---------------------------
    # Display Result
    # ---------------------------

    if prediction == 1:

        st.error(
            f"Customer likely to churn ⚠️ "
            f"(Churn Probability: {probability * 100:.2f}%)"
        )

    else:

        st.success(
            f"Customer likely to stay ✅ "
            f"(Churn Probability: {probability * 100:.2f}%)"
        )