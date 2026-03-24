import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import time

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("credit_risk_xgb.pkl")
feature_order = joblib.load("feature_order.pkl")

# =========================
# TITLE
# =========================
st.title("💳 Credit Risk Scoring Dashboard")
st.markdown("### Intelligent Loan Risk Assessment System")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("📋 Applicant Details")

income = st.sidebar.slider("Income (Rs.)", 50000, 500000, 200000)
credit = st.sidebar.slider("Credit Amount (Rs.)", 50000, 1000000, 400000)
annuity = st.sidebar.slider("Annuity (Rs.)", 5000, 50000, 20000)

age_days = st.sidebar.slider("Age (days)", -25000, -7000, -10000)
employment_days = st.sidebar.slider("Employment (days)", -10000, 0, -2000)

# =========================
# CONVERT TO YEARS ✅ FIX
# =========================
age_years = abs(age_days) // 365
employment_years = abs(employment_days) // 365

# =========================
# CREATE FEATURE VECTOR (SAFE WAY)
# =========================
data = {
    "AMT_INCOME_TOTAL": income,
    "AMT_CREDIT": credit,
    "AMT_ANNUITY": annuity,
    "DAYS_BIRTH": age_days,
    "DAYS_EMPLOYED": employment_days,
}

X = np.zeros(len(feature_order))

for i, col in enumerate(feature_order):
    if col in data:
        X[i] = data[col]

X = X.reshape(1, -1)

# =========================
# BUTTON
# =========================
if st.sidebar.button("Generate Prediction"):

    with st.spinner("Analyzing applicant profile..."):
        time.sleep(1)

        try:
            prob = model.predict_proba(X)[0][1]

            # Score
            score = int(300 + (1 - prob) * 600)

            # Risk
            if prob < 0.2:
                risk = "Low"
            elif prob < 0.5:
                risk = "Medium"
            else:
                risk = "High"

            risk_percent = prob * 100
            safe_percent = (1 - prob) * 100

            # =========================
            # METRICS
            # =========================
            col1, col2, col3 = st.columns(3)

            col1.metric("Default Risk %", f"{risk_percent:.2f}%")
            col2.metric("Credit Score", score)
            col3.metric("Risk Level", risk)

            # =========================
            # DECISION
            # =========================
            if risk == "High":
                st.error("❌ Loan Rejected")
            elif risk == "Medium":
                st.warning("⚠️ Manual Review Required")
            else:
                st.success("✅ Loan Approved")

            # =========================
            # GAUGE
            # =========================
            st.subheader("📊 Risk Gauge")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                title={'text': "Default Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "red"}
                    ],
                }
            ))

            st.plotly_chart(fig_gauge, use_container_width=True)

            # =========================
            # TABS
            # =========================
            tab1, tab2, tab3 = st.tabs(["📈 Analysis", "📊 Charts", "📄 Report"])

            # ---------- ANALYSIS ----------
            with tab1:
                st.subheader("Applicant Summary")

                st.write(f"💰 Income: Rs. {income}")
                st.write(f"🏦 Credit Amount: Rs. {credit}")
                st.write(f"📉 Annuity: Rs. {annuity}")
                st.write(f"🎂 Age: {age_years} years")  # ✅ FIX
                st.write(f"💼 Employment: {employment_years} years")  # ✅ FIX

                st.subheader("🧠 Risk Insights")

                insights = []

                if income < 100000:
                    insights.append("Low income may increase risk")

                if credit > income * 2:
                    insights.append("High loan compared to income")

                if employment_years < 3:
                    insights.append("Short employment history")

                if annuity > income * 0.3:
                    insights.append("High EMI burden")

                if not insights:
                    insights.append("Financial profile looks stable")

                for i in insights:
                    st.write("⚠️", i)

            # ---------- CHARTS ----------
            with tab2:

                colA, colB = st.columns(2)

                with colA:
                    fig_bar = go.Figure(data=[
                        go.Bar(x=["Default Risk"], y=[risk_percent]),
                        go.Bar(x=["Safe"], y=[safe_percent])
                    ])
                    st.plotly_chart(fig_bar, use_container_width=True)

                with colB:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=["Default Risk", "Safe"],
                        values=[risk_percent, safe_percent]
                    )])
                    st.plotly_chart(fig_pie, use_container_width=True)

            # ---------- REPORT ----------
            with tab3:

                report = f"""
CREDIT RISK REPORT

Default Probability: {risk_percent:.2f}%
Credit Score: {score}
Risk Category: {risk}

Income: Rs. {income}
Credit Amount: Rs. {credit}
Annuity: Rs. {annuity}

Age: {age_years} years
Employment: {employment_years} years
"""

                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name="credit_report.txt"
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")
