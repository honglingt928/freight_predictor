import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv('synthetic_freight_rates.csv')

# Separate features and target
X = data.drop('Rate_USD', axis=1)
y = data['Rate_USD']

# Preprocessor
categorical_cols = ['Origin', 'Destination', 'Mode', 'Season', 'Carrier_Tier']
numerical_cols = ['Distance_miles', 'Fuel_Index']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model.fit(X, y)

# Streamlit UI
st.title("🚚 Smart Freight Rate Predictor - Hong Ling POC")

st.sidebar.header("Input Freight Details")

origin = st.sidebar.selectbox("Origin", data['Origin'].unique())
destination = st.sidebar.selectbox("Destination", data['Destination'].unique())
mode = st.sidebar.selectbox("Mode", data['Mode'].unique())
distance = st.sidebar.number_input("Distance (miles)", min_value=50, max_value=10000, value=1000)
season = st.sidebar.selectbox("Season", data['Season'].unique())
fuel_index = st.sidebar.slider("Fuel Index", 0.8, 1.3, 1.0, 0.01)
carrier_tier = st.sidebar.selectbox("Carrier Tier", data['Carrier_Tier'].unique())

# Prepare input for prediction
input_df = pd.DataFrame({
    'Origin': [origin],
    'Destination': [destination],
    'Mode': [mode],
    'Distance_miles': [distance],
    'Season': [season],
    'Fuel_Index': [fuel_index],
    'Carrier_Tier': [carrier_tier]
})

# Predict rate
predicted_rate = model.predict(input_df)[0]

st.subheader("💰 Predicted Freight Rate")
st.success(f"${predicted_rate:,.2f}")

st.subheader("Input Details")
st.write(input_df)

# -------------------------------------------------
# NEW FEATURE 1: Distance Efficiency Calculator
# -------------------------------------------------

st.subheader("📏 Distance Efficiency Calculator")

cost_per_mile = predicted_rate / distance

col1, col2 = st.columns(2)

with col1:
    st.metric("Distance (Miles)", f"{distance:,}")

with col2:
    st.metric("Cost Per Mile", f"${cost_per_mile:.2f}")

st.info(
    "Cost per mile helps logistics teams compare efficiency across shipping lanes."
)

# -------------------------------------------------
# NEW FEATURE 2: Season + Fuel Scenario Simulator
# -------------------------------------------------

st.subheader("🧠 Scenario Simulator: Season & Fuel Impact")

sim_season = st.selectbox(
    "Simulated Season",
    data['Season'].unique(),
    key="sim_season"
)

sim_fuel = st.slider(
    "Simulated Fuel Index",
    0.8,
    1.5,
    1.1,
    0.01,
    key="sim_fuel"
)

scenario_df = pd.DataFrame({
    'Origin': [origin],
    'Destination': [destination],
    'Mode': [mode],
    'Distance_miles': [distance],
    'Season': [sim_season],
    'Fuel_Index': [sim_fuel],
    'Carrier_Tier': [carrier_tier]
})

scenario_rate = model.predict(scenario_df)[0]

st.subheader("📊 Scenario Prediction")

col3, col4 = st.columns(2)

with col3:
    st.metric("Current Prediction", f"${predicted_rate:,.2f}")

with col4:
    st.metric("Scenario Prediction", f"${scenario_rate:,.2f}")

difference = scenario_rate - predicted_rate

if difference > 0:
    st.warning(f"Estimated rate increase: ${difference:,.2f}")
else:
    st.success(f"Estimated rate decrease: ${abs(difference):,.2f}")
