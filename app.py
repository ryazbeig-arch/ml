
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# App title
st.title("📊 Telecom Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the telecom service.")

# Load trained model
@st.cache_resource
def load_model():
    # Change this filename to your actual model filename
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.info("Check that your model file exists in the project folder.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Customer Information")

tenure = st.sidebar.slider(
    "Tenure (months)", 0, 72, 12
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)", 0.0, 150.0, 70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges ($)",
    min_value=0.0,
    max_value=10000.0,
    value=840.0
)

# Prediction
if st.button("Predict Churn", type="primary"):

    # Feature order must match model training
    features = pd.DataFrame(
        [[tenure, monthly_charges, total_charges]],
        columns=[
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
    )

    try:
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("⚠️ Prediction: Customer is likely to churn.")
        else:
            st.success("✅ Prediction: Customer is likely to stay.")

        # Show probability if supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            st.write(
                f"Churn probability: {probabilities[1] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.warning(
            "Check that the input features and their order match "
            "the features used during model training."
        )
