Project Title

Credit Risk & Loan Default Prediction using Machine Learning

Overview

This project builds an end-to-end credit risk prediction system to estimate the probability of loan default using real-world Lending Club data. The objective is to identify risky borrowers while maintaining interpretability and business relevance.

Dataset

Source: Lending Club (Accepted Loans)

Size used: ~300,000 records

Domain: Finance / Credit Risk

Problem Statement

Predict whether a borrower will default on a loan based on financial and credit attributes.

Approach

Selected finance-relevant features (loan amount, interest rate, income, DTI, FICO, employment length)

Performed domain-aware data preprocessing and missing value handling

Engineered key risk indicators such as loan-to-income ratio and average FICO score

Trained a Logistic Regression model with class imbalance handling

Evaluated using recall and ROC-AUC (more suitable than accuracy for credit risk)

Manually validated high-risk vs low-risk borrower rankings

Exported the trained model and performed inference on Google Cloud

Model & Evaluation

Model: Logistic Regression

ROC-AUC: ~0.70

Recall (Defaulters): ~63%

Focused on explainability and risk ranking rather than raw accuracy

Cloud Integration

Exported trained model and scaler using joblib

Ran inference in Google Cloud Shell to demonstrate deployment readiness

Key Learnings

Credit risk modeling is a ranking problem, not just classification

Recall and ROC-AUC are more important than accuracy in finance

Model explainability is crucial for real-world decision-making

## Dataset
The dataset used in this project is the Lending Club Accepted Loans dataset.
Due to size constraints, the dataset is not included in this repository.

You can download it from:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

## Web Application (Streamlit)
A Streamlit-based dashboard was developed to simulate real-world credit risk assessment.  
Users can input borrower details and receive default probability, risk classification, and decision recommendations.

The UI also highlights key risk drivers to improve model interpretability.

