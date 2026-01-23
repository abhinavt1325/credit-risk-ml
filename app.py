import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("models/credit_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide", page_icon="💳")

# Header
st.markdown("## 💳 Credit Risk Prediction Dashboard")
st.markdown("This system estimates the probability of loan default using machine learning.")

st.divider()

# Layout: Input (left) | Output (right)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧾 Borrower Details")

    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=5000)
    term = st.selectbox("Loan Term (months)", [36, 60])
    int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=40.0, value=12.5)
    emp_length = st.slider("Employment Length (years)", 0, 10, 3)
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=60000)
    dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=50.0, value=15.0)
    fico_avg = st.number_input("FICO Score", min_value=300, max_value=850, value=700)

    predict_btn = st.button("🔍 Analyze Credit Risk")

with col2:
    st.markdown("### 📊 Risk Assessment")

    if predict_btn:
        # Feature engineering
        loan_to_income = loan_amnt / annual_inc
        log_annual_inc = np.log1p(annual_inc)

        X = np.array([[loan_amnt, term, int_rate, emp_length, annual_inc, dti, fico_avg, loan_to_income, log_annual_inc]])
        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0, 1]

        # Risk level
        if prob < 0.2:
            risk_label = "Low Risk"
            color = "green"
            decision = "Approve Loan"
        elif prob < 0.4:
            risk_label = "Medium Risk"
            color = "orange"
            decision = "Manual Review Required"
        else:
            risk_label = "High Risk"
            color = "red"
            decision = "Reject Loan"

        st.metric("Probability of Default", f"{prob:.2f}")
        st.markdown(f"### Risk Level: :{color}[{risk_label}]")
        st.markdown(f"**Suggested Decision:** {decision}")

        # Gauge chart
        fig, ax = plt.subplots()
        ax.barh(["Risk"], [prob], color=color)
        ax.set_xlim(0, 1)
        ax.set_title("Default Risk Gauge")
        st.pyplot(fig)

        # Explainability section
        st.markdown("### 🔍 Key Risk Drivers")

        factors = []
        if loan_to_income > 0.3:
            factors.append("• High loan-to-income ratio")
        if fico_avg < 650:
            factors.append("• Low credit score")
        if dti > 20:
            factors.append("• High debt-to-income ratio")
        if int_rate > 15:
            factors.append("• High interest rate")
        if emp_length < 2:
            factors.append("• Short employment history")

        if factors:
            for f in factors:
                st.write(f)
        else:
            st.write("• Borrower financial profile appears stable.")

st.divider()

# Footer info
st.markdown("""
### ℹ️ About This System
- Built using Logistic Regression for credit risk modeling.
- Evaluated using ROC-AUC and recall to handle class imbalance.
- Designed to simulate real-world banking decision systems.

👨‍💻 Developed as an end-to-end Machine Learning project with explainable AI and cloud-ready deployment.
""")
