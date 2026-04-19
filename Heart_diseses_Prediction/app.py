import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="centered")

# Tailwind CDN
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Load model
model = joblib.load('KNN_heart_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('heart_columns.pkl')

# Header Card
st.markdown("""
<div class="bg-red-500 p-6 rounded-2xl shadow-lg text-white text-center">
    <h1 class="text-3xl font-bold">❤️ Heart Stroke Prediction</h1>
    <p class="mt-2">AI-powered health risk detection system</p>
</div>
<br>
""", unsafe_allow_html=True)

# Input Section Card
st.markdown('<div class="bg-white p-6 rounded-2xl shadow-md">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease History", [0, 1])

with col2:
    avg_glucose_level = st.number_input("Avg Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

st.markdown('</div><br>', unsafe_allow_html=True)

# Convert input
input_data = pd.DataFrame({
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'gender': [gender],
    'smoking_status': [smoking_status]
})

# Encoding
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=expected_columns, fill_value=0)

# Scaling
input_scaled = scaler.transform(input_data)

# Button
if st.button("🔍 Predict Risk"):
    with st.spinner("Analyzing..."):
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0][1]

    # Result Card
    if prediction[0] == 1:
        st.markdown(f"""
        <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-xl shadow">
            <h2 class="text-xl font-bold">⚠️ High Risk of Heart Stroke</h2>
            <p class="mt-2">Risk Probability: {prob:.2f}</p>
            <p class="mt-2">Please consult a doctor immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 rounded-xl shadow">
            <h2 class="text-xl font-bold">✅ Low Risk of Heart Stroke</h2>
            <p class="mt-2">Risk Probability: {prob:.2f}</p>
            <p class="mt-2">Maintain a healthy lifestyle.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<br>
<div class="text-center text-gray-500">
    Developed by Divesh Kumar 🚀
</div>
""", unsafe_allow_html=True)