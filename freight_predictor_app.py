import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# -------- Page config for mobile --------
st.set_page_config(page_title="Smart Freight Rate Predictor", layout="centered")

# -------- Load and cache dataset --------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("synthetic_freight_rates.csv")
    except FileNotFoundError:
        # fallback synthetic dataset if CSV not found
        return pd.DataFrame({
            "Origin": ["Shanghai", "Los Angeles", "Hamburg"],
            "Destination": ["Los Angeles", "Shanghai", "Singapore"],
            "Mode": ["Ocean", "Air", "Rail"],
            "Distance_miles": [7000, 6000, 5000],
            "Season": ["Winter", "Summer", "Spring"],
            "Fuel_Index": [1.0, 0.95, 1.05],
            "Carrier_Tier": ["A", "B", "C"],
            "Rate_USD": [1200, 1100, 900]
        })

data = load_data()

# -------- Separate features and target --------
X = data.drop('Rate_USD', axis=1)
y = data['Rate_USD']

# -------- Preprocessor & Model pipeline --------
categorical_cols = ['Origin', 'Destination', 'Mode', 'Season', 'Carrier_Tier']
numerical_cols = ['Distance_miles', 'Fuel_Index']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

@st.cache_data
def train_model(X, y):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    return model

model = train_model(X, y)

# -------- Streamlit UI --------
st.title("🚚 Smart Freight Rate Predictor")
st.markdown("Predict freight rates based on shipment details.")

# -------- Sidebar inputs --------
st.sidebar.header("Input Freight Details")

origin = st.sidebar.selectbox("Origin", data['Origin'].unique())
destination = st.sidebar.selectbox("Destination", data['Destination'].unique())
mode = st.sidebar.selectbox("Mode", data['Mode'].unique())
distance = st.sidebar.number_input("Distance (miles)", min_value=50, max_value=10000, value=1000)
season = st.sidebar.selectbox("Season", data['Season'].unique())
fuel_index = st.sidebar.slider("Fuel Index", 0.8, 1.3, 1.0, 0.01)
carrier_tier = st.sidebar.selectbox("Carrier Tier", data['Carrier_Tier'].unique())

# -------- Prepare input for prediction --------
input_df = pd.DataFrame({
    'Origin': [origin],
    'Destination': [destination],
    'Mode': [mode],
    'Distance_miles': [distance],
    'Season': [season],
    'Fuel_Index': [fuel_index],
    'Carrier_Tier': [carrier_tier]
})

# -------- Predict --------
predicted_rate = model.predict(input_df)[0]

st.subheader("Predicted Freight Rate (USD)")
st.success(f"${predicted_rate:,.2f}")

# -------- Input details in expandable section --------
with st.expander("View Input Details"):
    st.dataframe(input_df, use_container_width=True)

# -------- Dataset preview in expandable section --------
with st.expander("View Dataset"):
    st.dataframe(data, use_container_width=True)

st.markdown("---")
st.info("App optimized for mobile: caching data & model, responsive tables, and expandable sections.")
