import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["Car_Age_At_Sale", "Mileage_km", "Horsepower_PS"]
RANDOM_STATE = 42

@st.cache_data
def load_and_train():
    df = pd.read_csv("Lab04_hk_car_price.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["Price_HKD"], test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return df, model

df, model = load_and_train()
st.title("HK Used Car Price Estimator")
st.caption("Trained on real Hong Kong Motor City transactions — not a hardcoded guess.")

age = st.slider("Car age at sale (years)", 0, 30, 6)
km = st.slider("Mileage (km)", 0, 200000, 40000, step=1000)
hp = st.slider("Horsepower (PS)", 50, 800, 200)

if st.button("Estimate Price"):
    row = pd.DataFrame([{
        "Car_Age_At_Sale": age,
        "Mileage_km": km,
        "Horsepower_PS": hp,
    }])[FEATURES]
    price = model.predict(row)[0]
    st.success(f"Estimated price: HK${price:,.0f}")
