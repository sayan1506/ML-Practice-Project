import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =====================================
# Load Model and Scaler
# =====================================
with open("personality_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =====================================
# Feature Columns (MUST MATCH TRAINING ORDER)
# =====================================
feature_columns = [
    'social_energy',
    'alone_time_preference',
    'talkativeness',
    'deep_reflection',
    'group_comfort',
    'party_liking',
    'listening_skill',
    'empathy',
    'organization',
    'leadership',
    'risk_taking',
    'public_speaking_comfort',
    'curiosity',
    'routine_preference',
    'excitement_seeking',
    'friendliness',
    'planning',
    'spontaneity',
    'adventurousness',
    'reading_habit',
    'sports_interest',
    'online_social_usage',
    'travel_desire',
    'gadget_usage',
    'work_style_collaborative',
    'decision_speed'
]

# =====================================
# Label Mapping (0 / 1 → TEXT)
# =====================================
label_map = {
    0: "Introvert",
    1: "Ambivert",
    2: "Extrovert"
}

# =====================================
# Streamlit Page Config
# =====================================
st.set_page_config(
    page_title="Personality Type Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Personality Type Predictor")
st.write("Adjust the sliders and predict personality type using Machine Learning.")

st.divider()

# =====================================
# User Input Section
# =====================================
user_input = {}

for feature in feature_columns:
    user_input[feature] = st.slider(
        label=feature.replace("_", " ").title(),
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )

input_df = pd.DataFrame([user_input])

# =====================================
# Scale Input
# =====================================
input_scaled = scaler.transform(input_df)

# =====================================
# Prediction Section
# =====================================
if st.button("🚀 Predict Personality"):
    # Predict class
    prediction_num = model.predict(input_scaled)[0]
    prediction_label = label_map[prediction_num]

    # Predict probabilities
    probabilities = model.predict_proba(input_scaled)[0]

    # Display Result
    st.success(f"### 🎯 Predicted Personality: **{prediction_label}**")

    # Confidence Chart
    proba_df = pd.DataFrame(
        probabilities.reshape(1, -1),
        columns=[label_map[i] for i in model.classes_]
    )

    st.subheader("📊 Prediction Confidence")
    st.bar_chart(proba_df.T)

# =====================================
# Footer
# =====================================
st.divider()
st.caption("Built with Streamlit • Logistic Regression • Scikit-Learn")
