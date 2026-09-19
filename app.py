# High-score exemplar app — self-contained (re-loads data + re-trains on Streamlit Cloud's fresh server).
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_URL = "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42

@st.cache_data
def load_and_train():
    df = pd.read_csv(DATA_URL)
    y = (df["Churn"] == "Yes").astype(int)
    contract_ohe = pd.get_dummies(df["Contract"], prefix="Contract")
    internet_ohe = pd.get_dummies(df["InternetService"], prefix="Internet")
    X = pd.concat([df[["tenure", "MonthlyCharges"]], contract_ohe, internet_ohe], axis=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    return df, model, list(X.columns)

df, model, feature_cols = load_and_train()

st.title("CA2 Prototype — High Score Exemplar")
st.write("Predicts **Churn**. Backed by a real trained RandomForestClassifier (4 business features).")
st.caption("⚠️ Remember: this model's Recall is only ~46% — it misses over half of real churners. "
           "Higher accuracy is not automatically a 'good enough' model.")

tenure = st.slider("tenure", 0, 72, 12)
monthly = st.slider("MonthlyCharges", 18.0, 119.0, 70.0)
contract = st.selectbox("Contract", sorted(df["Contract"].unique().tolist()))
internet = st.selectbox("InternetService", sorted(df["InternetService"].unique().tolist()))

if st.button("Predict"):
    row = pd.DataFrame([{"tenure": tenure, "MonthlyCharges": monthly}])
    for c in feature_cols:
        if c.startswith("Contract_"):
            row[c] = 1 if c == f"Contract_{contract}" else 0
        elif c.startswith("Internet_"):
            row[c] = 1 if c == f"Internet_{internet}" else 0
    row = row[feature_cols]
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]
    label = "Yes" if pred == 1 else "No"
    st.success(f"Prediction: {label}  (probability: {proba:.1%})")
