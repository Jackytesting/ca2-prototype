# This file must work standing ALONE on Streamlit Cloud's server (no Colab, no Drive).
# That's why it re-loads the data and re-trains the model itself when the app starts.

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# 👉 PASTE your final Task 1 + Task 3 values here as PLAIN VALUES (not Colab variables):
DATA_URL = "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
TASK = "classification"
FEATURES = ["tenure", "MonthlyCharges"]
RANDOM_STATE = 42   # a public demo — does not need to match your own Student ID

@st.cache_data   # avoids re-training every time someone moves a slider
def load_and_train():
    df = pd.read_csv(DATA_URL)
    X_all = df[FEATURES]
    y_all = (df[TARGET] == "Yes").astype(int) if (TASK == "classification" and df[TARGET].dtype == "object") else df[TARGET]
    strat = y_all if TASK == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE) if TASK == "classification" else LinearRegression()
    model.fit(X_train, y_train)
    return df, model

df, model = load_and_train()

st.title("CA2 Prototype")
st.write(f"Predicts **{TARGET}**. Backed by a real trained {type(model).__name__} — not a fake rule.")

# 👉 PASTE/adapt your final Task 4 AI-Copilot Streamlit code below
#    (it can reuse df / model / FEATURES / TASK / TARGET defined above):

inputs = {}
for col in FEATURES:
    if df[col].dtype == "object":
        inputs[col] = st.selectbox(col, sorted(df[col].dropna().unique().tolist()))
    else:
        inputs[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

if st.button("Predict"):
    row = pd.DataFrame([inputs])
    pred = model.predict(row)[0]
    if TASK == "classification":
        label = "Yes" if pred == 1 else "No"
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)[0][1]
            st.success(f"Prediction: {label}  (probability: {proba:.1%})")
        else:
            st.success(f"Prediction: {label}")
    else:
        st.success(f"Predicted {TARGET}: {pred:.2f}")
