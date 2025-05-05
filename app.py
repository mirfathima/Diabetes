import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the diabetes dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

# Train-test split (we'll just train once)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏥 Diabetes Progression Predictor")
st.write("Enter the following patient details to predict diabetes progression:")

# Create input sliders for each feature
def user_input_features():
    inputs = {}
    for feature in diabetes.feature_names:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        default_val = float(X[feature].mean())
        inputs[feature] = st.slider(f"{feature}", min_val, max_val, default_val)
    return pd.DataFrame([inputs])

# Get user input
input_df = user_input_features()

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted diabetes progression score: {prediction:.2f}")
