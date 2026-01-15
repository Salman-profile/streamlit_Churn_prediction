import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.page_link("Churn_Prediction_App.py", label="Home", icon="🏠")

col1, col2 = st.columns([0.2,0.8])
with col2:
        st.title("👩‍💻 Customer Churn Prediction")

# -----------------------
# Load models
# -----------------------
nn1 = load_model("models/nn1.h5")
nn2 = load_model("models/nn2.h5")
xgb = joblib.load("models/xgb.pkl")

scaler1 = joblib.load("models/scaler1.pkl")
scaler2 = joblib.load("models/scaler2.pkl")

# -----------------------
# Feature sets
# -----------------------
F_NN1 = ['gender','SeniorCitizen','tenure','PhoneService','InternetService','TechSupport','StreamingTV','Contract','MonthlyCharges']
F_NN2 = ['MonthsInService','MonthlyMinutes','BlockedCalls','DroppedCalls','CustomerCareCalls','RetentionCalls','IncomeGroup', 'CreditRating','CurrentEquipmentDays','HandsetRefurbished']
F_XGB = ['Account Length','Day Mins','VMail Message','Day Charge','Night Mins','Night Charge','Intl Mins','Intl Charge','CustServ Calls']

# -----------------------
# User input
# -----------------------
st.sidebar.header("Customer Information")

user = {
    "gender": st.sidebar.selectbox("Gender", [1, 0]),
    "SeniorCitizen": st.sidebar.selectbox("SeniorCitizen", [1, 0]),
    "tenure": st.sidebar.number_input("Tenure"),
    "PhoneService": st.sidebar.selectbox("PhoneService", [0, 1]),
    "InternetService": st.sidebar.selectbox("InternetService", [0, 1, 2]),
    "TechSupport": st.sidebar.selectbox("TechSupport", [0, 1, 2]),
    "StreamingTV": st.sidebar.selectbox("StreamingTV", [0, 1, 2]),
    "Contract": st.sidebar.selectbox("Contract", [0, 1, 2]),
    "MonthlyCharges": st.sidebar.number_input("MonthlyCharges"),
    "MonthsInService": st.sidebar.number_input("MonthsInService"),
    "MonthlyMinutes": st.sidebar.number_input("MonthlyMinutes"),
    "BlockedCalls": st.sidebar.number_input("BlockedCalls"),
    "DroppedCalls": st.sidebar.number_input("DroppedCalls"),
    "CustomerCareCalls": st.sidebar.number_input("CustomerCareCalls"),
    "RetentionCalls": st.sidebar.number_input("RetentionCalls"),
    "IncomeGroup": st.sidebar.selectbox("IncomeGroup", [0,1,2,3,4,5,6,7,8,9]),
    "CreditRating": st.sidebar.selectbox("CreditRating", [0,1,2,3,4,5,6]),
    "CurrentEquipmentDays": st.sidebar.number_input("CurrentEquipmentDays"),
    "HandsetRefurbished": st.sidebar.selectbox("HandsetRefurbished", [0,1]),
    "Account Length": st.sidebar.number_input("Account Length"),
    "Day Mins": st.sidebar.number_input("Day Mins"),
    "Day Charge": st.sidebar.number_input("Day Charge"),
    "Night Mins": st.sidebar.number_input("Night Mins"),
    "Night Charge": st.sidebar.number_input("Night Charge"),
    "Intl Mins": st.sidebar.number_input("Intl Mins"),
    "Intl Charge": st.sidebar.number_input("Intl Charge"),
    "CustServ Calls": st.sidebar.number_input("CustServ Calls"),
    "VMail Message": st.sidebar.number_input("VMail Message")

}

df = pd.DataFrame([user])

# -----------------------
# Predict
# -----------------------
if st.button("Predict Churn"):

    p1 = nn1.predict(scaler1.transform(df[F_NN1]))[0][0]
    p2 = nn2.predict(scaler2.transform(df[F_NN2]))[0][0]
    p3 = xgb.predict_proba(df[F_XGB])[0][1]

### Ensemeble Mean Calculation
    ensemble_mean = (p1 + p2 + p3) / 3

### Ensemble Weights by Performance
    total_perf = p1 + p2 + p3
    w1, w2, w3 = p1/total_perf, p2/total_perf, p3/total_perf
    ensemble_w_perf = w1*p1 + w2*p2 + w3*p3

### Ensemble Weights by Size
    total_size = 7032 + 49752 + 101174
    w1, w2, w3 = 7032/total_size, 49752/total_size, 101174/total_size
    ensemble_w_size = w1*p1 + w2*p2 + w3*p3


    col1, col2 = st.columns([0.2,0.8])
    with col2:
        st.subheader("📚 Individual Model Probabilities")

    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Neural Network 1", f"{p1:.2f}")
    with col2:
        st.metric("Neural Network 2", f"{p2:.2f}")
    with col3:
        st.metric("XGBoost", f"{p3:.2f}")


    col1, col2 = st.columns([0.2,0.8])
    with col2:
        st.subheader("⚙️ Ensemble Model Probabilities")

    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⏳ Simple Mean", f"{ensemble_mean:.2f}")
    with col2:
        st.metric("⚖️ Weigth by Performance", f"{ensemble_w_perf:.2f}")
    with col3:
        st.metric("🧮 Weight by Dataset Size", f"{ensemble_w_size:.2f}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🚨 Churn Likely" if ensemble_mean > 0.5 else "✅ No Churn Risk")
    with col2:
        st.success("🚨 Churn Likely" if ensemble_w_perf > 0.5 else "✅ No Churn Risk")
    with col3:
        st.success("🚨 Churn Likely" if ensemble_w_size > 0.5 else "✅ No Churn Risk")
