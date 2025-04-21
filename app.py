import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

data = {
    "NO2": [30, 40, 35, 60, 45],  # Nitrogen Dioxide (ppm)
    "PM2.5": [50, 40, 30, 70, 65],  # Particulate Matter 2.5 (µg/m³)
    "CO": [0.2, 0.3, 0.25, 0.5, 0.4],  # Carbon Monoxide (ppm)
    "SO2": [5, 6, 4, 8, 7],  # Sulfur Dioxide (ppb)
    "O3": [100, 150, 120, 200, 180],  # Ozone (ppb)
    "AQI": [80, 90, 85, 120, 110]  # Air Quality Index (Target Variable)
}

df = pd.DataFrame(data)

X = df[["NO2", "PM2.5", "CO", "SO2", "O3"]]  # Pollutant levels
y = df["AQI"]  # AQI values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()

# Input layer (5 features) and hidden layers
model.add(Dense(units=64, activation='relu', input_dim=5))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=0)

st.set_page_config(page_title="AQI Prediction 🌿", page_icon="🌍", layout="wide")
st.title("🌱 **Air Quality Index (AQI) Prediction** 🌱")
st.markdown(
    "Welcome to the **Air Quality Index (AQI) Prediction** app! 🌍🌿"
    " This app predicts the AQI based on the levels of common pollutants in the air. 🚗💨"
)

st.sidebar.header("📝 **Enter Pollutant Levels**:")
no2 = st.sidebar.number_input("NO2 (Nitrogen Dioxide) [ppm] 🌫️", min_value=0.0, value=30.0, step=0.1)
pm25 = st.sidebar.number_input("PM2.5 (Particulate Matter 2.5) [µg/m³] 💨", min_value=0, value=50)
co = st.sidebar.number_input("CO (Carbon Monoxide) [ppm] 🚗", min_value=0.0, value=0.2)
so2 = st.sidebar.number_input("SO2 (Sulfur Dioxide) [ppb] ⚡", min_value=0, value=5)
o3 = st.sidebar.number_input("O3 (Ozone) [ppb] 🌤️", min_value=0, value=100)

if st.sidebar.button("🔮 Predict AQI"):
    input_data = np.array([[no2, pm25, co, so2, o3]])
    input_scaled = scaler.transform(input_data)  # Scale the input

    predicted_aqi = model.predict(input_scaled)  # Make prediction

    st.subheader("🔮 **Your AQI Prediction:**")
    st.markdown(f"### The predicted **Air Quality Index (AQI)** is: ")
    st.write(f"### **{predicted_aqi[0][0]:.2f}** 🌟")
    
    if predicted_aqi[0][0] <= 50:
        st.write("🌿 **Good** - Air quality is considered satisfactory, and air pollution poses little or no risk.")
    elif 51 <= predicted_aqi[0][0] <= 100:
        st.write("🟡 **Moderate** - Air quality is acceptable; however, some pollutants may have a mild impact on sensitive groups.")
    elif 101 <= predicted_aqi[0][0] <= 150:
        st.write("🟠 **Unhealthy for Sensitive Groups** - People with respiratory or heart conditions may experience mild symptoms.")
    elif 151 <= predicted_aqi[0][0] <= 200:
        st.write("🔴 **Unhealthy** - Everyone may experience some symptoms of health effects. Avoid prolonged exposure.")
    else:
        st.write("⚠️ **Very Unhealthy** - Health alert: everyone may experience more serious health effects.")
    
    st.markdown("### 💡 **Tips for Better Air Quality:**")
    st.write("1. Use air purifiers to improve indoor air quality. 🏠💨")
    st.write("2. Limit outdoor activities during high pollution periods. 🚶‍♂️🌆")
    st.write("3. Plant trees and greenery to help reduce air pollution. 🌳🌿")

st.markdown(
    "### 🌍 **What is AQI?**"
    "\nThe **Air Quality Index (AQI)** is a measure used to communicate how polluted the air currently is or how polluted it is forecast to become."
    "\nThe AQI is calculated based on the levels of common pollutants such as NO2, PM2.5, CO, O3, and SO2."
    "\nThis tool helps you predict the AQI and take necessary precautions to protect your health. 🚶‍♀️💡"
)
