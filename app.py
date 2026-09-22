# This file must work standing ALONE on Streamlit Cloud (no Colab, no Drive).
# Upload Lab04_hk_car_price.csv in the SAME GitHub folder as this file.

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# CHANGE THIS LINE. Use your own column names. Do not leave the words PASTE / HERE.
# Example: ["Manufacture_Year", "Mileage_km", "Horsepower_PS", "Brand"]
# Do not add Displacement_cc. Empty engine cc will crash training.
FEATURES = ['Manufacture_Year', 'Mileage_km', 'Horsepower_PS', 'Brand']
RANDOM_STATE = 42   # public demo — does not need to match your Student ID

@st.cache_data
def load_and_train():
    df = pd.read_csv("Lab04_hk_car_price.csv")
    X_all = df[FEATURES].copy()
    y_all = df["Price_HKD"]

    # Brand is text. Split it into number columns before fit().
    text_cols = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(X_all[c])]
    if text_cols:
        X_all = pd.get_dummies(X_all, columns=text_cols, drop_first=True)
        X_all = X_all.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return df, model, list(model.feature_names_in_)

df, model, model_columns = load_and_train()

st.title("HK Used Car Price Estimator — CA2 Prototype")
st.write("Predicts **resale price (HKD)** from real Hong Kong Motor City transactions. This is a quote ballpark — not an official valuation form.")

# Sliders and brand menus are built from FEATURES. Do not delete this loop.
inputs = {}
for col in FEATURES:
    if not pd.api.types.is_numeric_dtype(df[col]):
        inputs[col] = st.selectbox(col, sorted(df[col].dropna().unique().tolist()))
    else:
        inputs[col] = st.slider(
            col, float(df[col].min()), float(df[col].max()), float(df[col].mean())
        )

if st.button("Estimate Price"):
    row = pd.DataFrame([inputs])
    text_cols = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if text_cols:
        row = pd.get_dummies(row, columns=text_cols)
    row = row.reindex(columns=model_columns, fill_value=0)
    price = model.predict(row)[0]
    st.success(f"Estimated price: HK${price:,.0f}")
