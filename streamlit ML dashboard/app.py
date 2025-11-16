import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# -------------------------------
# Load models
# -------------------------------
rf_model = joblib.load("models/random_forest.pkl")
log_model = joblib.load("models/logistic_reg.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="ML Live Prediction Dashboard", layout="wide")

st.title("📊 Machine Learning Live Prediction Dashboard")
st.write("Interactively test ML models and see real-time predictions!")

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.header("User Input")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Random Forest", "Logistic Regression")
)

# Example dataset features
age = st.sidebar.slider("Age", 18, 60, 25)
salary = st.sidebar.number_input("Salary", 10000, 200000, 50000)
balance = st.sidebar.number_input("Bank Balance", 0, 300000, 50000)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)

features = np.array([[age, salary, balance, credit_score]])

# Scale input
scaled_features = scaler.transform(features)

if model_choice == "Random Forest":
    selected_model = rf_model
else:
    selected_model = log_model

# --------------------------------
# Live Prediction Section
# --------------------------------
st.header("🔮 Live Model Prediction")

if st.button("Predict"):
    with st.spinner("Running prediction..."):
        time.sleep(1)
        pred = selected_model.predict(scaled_features)[0]
        prob = selected_model.predict_proba(scaled_features)[0][1]

    st.success(f"Prediction: **{pred}**")
    st.metric(label="Probability", value=f"{prob:.2f}")

    # Live graphs
    st.subheader("📈 Prediction Confidence")
    st.progress(int(prob * 100))

# --------------------------------
# Batch Prediction
# --------------------------------
st.header("📥 Batch Prediction (Upload CSV)")

uploaded = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Uploaded Data:", df)

    scaled = scaler.transform(df)

    preds = selected_model.predict(scaled)
    df["prediction"] = preds

    st.write("✔ Predictions:")
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("Download Results", csv, "predictions.csv")

