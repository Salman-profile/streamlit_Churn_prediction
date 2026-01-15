import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(
    page_title="Churn-Prediction",
    layout="wide"
)

st.page_link("pages/Make_Prediction.py", label="Make Prediction", icon="🔍")

st.title("📊🔎 Ensemble Model Evaluation (Neural-Network / XGBOOST)")

# --------------------------------------------------
# Load models
# --------------------------------------------------
nn1 = load_model("models/nn1.h5")
nn2 = load_model("models/nn2.h5")

scaler1 = joblib.load("models/scaler1.pkl")
scaler2 = joblib.load("models/scaler2.pkl")

xgb = joblib.load("models/xgb.pkl")

# --------------------------------------------------
# Upload datasets
# --------------------------------------------------
st.sidebar.header("Upload-Datasets")

d1 = st.sidebar.file_uploader("Dataset-1 Test (NN-1)", type="csv")
d2 = st.sidebar.file_uploader("Dataset-2 Test (NN-2)", type="csv")
d3 = st.sidebar.file_uploader("Dataset-3 Test (XGBoost)", type="csv")

if d1 and d2 and d3:

    D1 = pd.read_csv(d1)
    D2 = pd.read_csv(d2)
    D3 = pd.read_csv(d3)

    X1, y1 = D1.drop("Churn", axis=1), D1["Churn"]
    X2, y2 = D2.drop("Churn", axis=1), D2["Churn"]
    X3, y3 = D3.drop("Churn", axis=1), D3["Churn"]

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------
    p1 = nn1.predict(scaler1.transform(X1)).ravel()
    p2 = nn2.predict(scaler2.transform(X2)).ravel()
    p3 = xgb.predict_proba(X3)[:, 1]

    y1_pred = (p1 > 0.5).astype(int)
    y2_pred = (p2 > 0.5).astype(int)
    y3_pred = (p3 > 0.5).astype(int)

    # --------------------------------------------------
    # Individual Performance
    # --------------------------------------------------
    acc1 = accuracy_score(y1, y1_pred)
    acc2 = accuracy_score(y2, y2_pred)
    acc3 = accuracy_score(y3, y3_pred)

    auc1 = roc_auc_score(y1, p1)
    auc2 = roc_auc_score(y2, p2)
    auc3 = roc_auc_score(y3, p3)

    mean_ensemble_acc = np.mean([acc1, acc2, acc3])
    mean_ensemble_auc = np.mean([auc1, auc2, auc3])

    
    total_acc = acc1 + acc2 + acc3
    w1, w2, w3 = acc1/total_acc, acc2/total_acc, acc3/total_acc
    wp_ensemble_acc = w1*acc1 + w2*acc2 + w3*acc3
    
    total_auc = auc1 + auc2 + auc3
    w1, w2, w3 = auc1/total_auc, auc2/total_auc, auc3/total_auc
    wp_ensemble_auc = w1*auc1 + w2*auc2 + w3*auc3

    n1, n2, n3 = len(D1), len(D2), len(D3)
    w1 = n1 / (n1 + n2 + n3)
    w2 = n2 / (n1 + n2 + n3)
    w3 = n3 / (n1 + n2 + n3)

    ws_ensemble_acc = w1*acc1 + w2*acc2 + w3*acc3
    ws_ensemble_auc = w1*auc1 + w2*auc2 + w3*auc3
    
    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------
    results = pd.DataFrame({
        "Model": ["Neural Network 1", "Neural Network 2", "XGBoost"],
        "Accuracy": [acc1, acc2, acc3],
        "AUC": [auc1, auc2, auc3]
    })
    #results['Accuracy'] = (results['Accuracy']*100).round(2).astype(str) + ' %'
    #results['AUC'] = (results['AUC']*100).round(2).astype(str) + ' %'

    col1, col2 = st.columns(2)
    with col1:
         st.subheader("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📌 Accuracy""")
    with col2:
         st.subheader("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📈 Area Under Curve""")
         

    col1, col2 , col3 , col4 , col5 , col6 = st.columns(6)
    with col1:
            col1.metric("Neural-Network 1", f"{acc1:.2%}")
    with col2:
            col2.metric("Neural-Network 2", f"{acc2:.2%}")
    with col3:
            col3.metric("XGBOOST", f"{acc3:.2%}")
    with col4:
            col4.metric("Neural-Network 1", f"{auc1:.2%}")
    with col5:
            col5.metric("Neural-Network 2", f"{auc2:.2%}")
    with col6:
            col6.metric("XGBOOST", f"{auc3:.2%}")


    col1, col2 = st.columns(2)
    with col1:
         st.subheader(" ⏳ Ensemble By Mean Performance")
         #st.subheader("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📌 Individual Models ACC""")
    with col2:
          st.subheader(" ⚖️ Ensemble By Weighted Performance ")
         #st.subheader("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📌 Individual Models AUC""")
       
    
    col1, col2 , col3 , col4 = st.columns(4)
    col1.metric("Ensemble Accuracy", f"{mean_ensemble_acc:.2%}")
    col2.metric("Ensemble AUC", f"{mean_ensemble_auc:.2%}")
    col3.metric("Ensemble Accuracy", f"{wp_ensemble_acc:.2%}")
    col4.metric("Ensemble AUC", f"{wp_ensemble_auc:.2%}")


    st.subheader(" 🧮 Ensemble By Dataset Size")
    col1, col2 , col3 , col4 = st.columns(4)
    col1.metric("Ensemble Accuracy", f"{ws_ensemble_acc:.2%}")
    col2.metric("Ensemble AUC", f"{ws_ensemble_auc:.2%}")



else:
    st.info(" 👈 Please upload all three test datasets for Performance Evluation.")

