# 💳 Credit Risk & Loan Default Prediction

## 📌 Overview
An end-to-end credit risk prediction system built using machine learning to estimate the probability of loan default from borrower financial data.  
The project emphasizes interpretability, risk ranking, and business relevance, which are critical in financial applications.

---

## 🏦 Problem Statement
Predict whether a borrower is likely to default on a loan, framed as a binary classification problem with focus on identifying risky borrowers.

---

## 📊 Dataset
- Source: Lending Club (Accepted Loans)
- Records Used: ~300,000

> Dataset not included due to size constraints  
> Download: https://www.kaggle.com/datasets/wordsforthewise/lending-club

---

## 🧠 Approach
- Selected finance-relevant features such as loan amount, interest rate, income, DTI, FICO score, and employment length
- Performed domain-aware data preprocessing and feature engineering
- Trained a Logistic Regression model with class imbalance handling
- Evaluated performance using ROC-AUC and recall instead of accuracy
- Interpreted model coefficients for explainability
- Manually validated high-risk vs low-risk borrower rankings

---

## 📈 Results
- ROC-AUC: ~0.70
- Recall (Defaulters): ~63%

> Accuracy was not over-optimized, as missing defaulters is more costly in credit risk scenarios.

---

## 🌐 Streamlit Web App
An interactive Streamlit dashboard was built to simulate real-world credit risk assessment.  
The application provides:
- Real-time default probability prediction
- Risk classification (Low / Medium / High)
- Decision recommendations (Approve / Review / Reject)
- Key risk driver explanations
Live Demo: https://credit-risk-prediction-mlproject.streamlit.app/
---

## ☁️ Cloud Integration
- Exported trained model and scaler using joblib
- Performed inference using Google Cloud Shell to demonstrate cloud readiness

---

## 🛠️ Tech Stack
- Programming Language: Python  
- Data Analysis & Processing: Pandas, NumPy  
- Machine Learning: Scikit-learn (Logistic Regression)  
- Model Evaluation: ROC-AUC, Recall, Precision  
- Visualization & UI: Streamlit  
- Version Control: Git, GitHub  
- Cloud Platform: Google Cloud (Cloud Shell)  

---

## 🎯 Key Takeaways
- Credit risk modeling is primarily a ranking problem
- Recall and ROC-AUC are more meaningful than accuracy in finance
- Explainability is essential for real-world ML systems
- Simple, interpretable models are effective in regulated domains

---

## 🚀 Future Improvements
- Threshold optimization based on business cost
- Advanced explainability using SHAP
- API-based model deployment

---

## 👨‍💻 Author
Abhinav Thakur
