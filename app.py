import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --------------------------
# Train a dummy ML model
# --------------------------
def train_model():
    np.random.seed(42)
    data = pd.DataFrame({
        "distance": np.random.randint(50, 1000, 500),
        "weight": np.random.randint(10, 500, 500),
        "traffic_index": np.random.randint(1, 10, 500),
        "warehouse_load": np.random.randint(10, 100, 500),
        "weather": np.random.choice([0, 1, 2], 500),  # 0=Clear,1=Rainy,2=Snowy
        "planned_days": np.random.randint(1, 7, 500)
    })
    data["delay"] = (
        0.005 * data["distance"] +
        0.01 * data["weight"] +
        0.3 * data["traffic_index"] +
        0.2 * data["warehouse_load"] / 100 +
        data["weather"] * 0.5 +
        np.random.normal(0, 0.5, 500)
    )
    X = data.drop(columns=["delay"])
    y = data["delay"]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

model = train_model()

# --------------------------
# Streamlit App
# --------------------------
def main():
    st.title("📦 Predict Shipment Delay")

    col1, col2 = st.columns(2)

    with col1:
        distance = st.number_input("Distance (km)", value=450)
        traffic = st.number_input("Traffic Index", value=7)
        planned = st.text_input("Planned Delivery", "3 days")

    with col2:
        weight = st.number_input("Weight (kg)", value=120)
        warehouse = st.number_input("Warehouse Load", value=87)
        weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Snowy"], index=1)

    weather_map = {"Clear": 0, "Rainy": 1, "Snowy": 2}

    if st.button("Predict Delay"):
        X_new = pd.DataFrame([{
            "distance": distance,
            "weight": weight,
            "traffic_index": traffic,
            "warehouse_load": warehouse,
            "weather": weather_map[weather],
            "planned_days": int(planned.split()[0])
        }])
        prediction = model.predict(X_new)[0]
        st.success(f"📦 Delay: {prediction:.1f} days")

    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Choose file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        preds = model.predict(df)
        df["Predicted Delay"] = preds
        st.dataframe(df)
        st.download_button("Export CSV", df.to_csv(index=False), "results.csv")

    st.subheader("📊 Visual Insights")
    st.info("Graphs like Actual vs Predicted & Feature Importance can be added here.")

if __name__ == "__main__":
    main()
