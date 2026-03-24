import streamlit as st
import numpy as np
import joblib

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
model = joblib.load("credit_risk_xgb.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("💳 Credit Risk Scoring Dashboard")
st.markdown("### Intelligent Loan Risk Assessment System")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.sidebar.header("📋 Applicant Details")

income = st.sidebar.slider("Income (Rs.)", 50000, 500000, 200000)
credit_amount = st.sidebar.slider("Credit Amount (Rs.)", 50000, 1000000, 400000)
annuity = st.sidebar.slider("Annuity (Rs.)", 5000, 50000, 20000)
age_days = st.sidebar.slider("Age (days)", -25000, -7000, -10000)
emp_days = st.sidebar.slider("Employment (days)", -10000, 0, -2000)

# -----------------------------
# CONVERT VALUES
# -----------------------------
age_years = abs(age_days) // 365
emp_years = abs(emp_days) // 365

# -----------------------------
# FEATURE CREATION
# -----------------------------
data = {
    "AMT_INCOME_TOTAL": income,
    "AMT_CREDIT": credit_amount,
    "AMT_ANNUITY": annuity,
    "DAYS_BIRTH": age_days,
    "DAYS_EMPLOYED": emp_days,
    "CREDIT_INCOME_RATIO": credit_amount / (income + 1),
    "ANNUITY_INCOME_RATIO": annuity / (income + 1),
    "EMPLOYMENT_STABILITY": emp_years / (age_years + 1),
}

# Convert to array aligned with training
input_data = np.zeros(len(feature_order))

for i, col in enumerate(feature_order):
    if col in data:
        input_data[i] = data[col]

input_scaled = scaler.transform([input_data])

# -----------------------------
# PREDICTION
# -----------------------------
if st.sidebar.button("Generate Prediction"):

    prob = model.predict_proba(input_scaled)[0][1]
    score = int(700 - prob * 300)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Prediction Result")
        st.metric("Default Risk %", f"{prob*100:.2f}%")
        st.metric("Credit Score", score)

        if prob > 0.6:
            st.error("❌ Loan Rejected (High Risk)")
        elif prob > 0.4:
            st.warning("⚠️ Medium Risk")
        else:
            st.success("✅ Loan Approved")

    with col2:
        st.subheader("📄 Applicant Summary")
        st.write(f"💰 Income: Rs. {income}")
        st.write(f"🏦 Credit Amount: Rs. {credit_amount}")
        st.write(f"📉 Annuity: Rs. {annuity}")
        st.write(f"🎂 Age: {age_years} years")
        st.write(f"💼 Employment: {emp_years} years")

    # -----------------------------
    # RISK INSIGHTS
    # -----------------------------
    st.subheader("🧠 Risk Insights")

    insights = []

    if credit_amount > income * 2:
        insights.append("High loan amount compared to income")

    if emp_years < 3:
        insights.append("Short employment history")

    if annuity > income * 0.3:
        insights.append("High EMI burden")

    if not insights:
        insights.append("Stable financial profile")

    for i in insights:
        st.write("⚠️", i)
