import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# PAGE CONFIG 
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

#LOAD MODEL 
model = load_model("diabetes_prediction_model.keras")
scaler = joblib.load("scaler.pkl")

# SIDEBAR 
st.sidebar.title("🩺 Diabetes Prediction")

st.sidebar.markdown("---")

st.sidebar.header("📋 About Project")
st.sidebar.write("""
This web application predicts whether a patient is likely
to have diabetes using an Artificial Neural Network (ANN)
built with TensorFlow.
""")

st.sidebar.markdown("---")

st.sidebar.header("⚙️ Technologies Used")
st.sidebar.write("""
✅ Python

✅ TensorFlow

✅ ANN

✅ Streamlit

✅ Scikit-learn
""")

st.sidebar.markdown("---")

st.sidebar.info("Enter patient details and click **Predict**.")

st.sidebar.markdown("---")
st.sidebar.success("Developed by Bala Murugan")

# TITLE 
st.title("🩺 Diabetes Prediction System")
st.write("### Enter Patient Details")

#INPUTS 
col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

    gender_mapping = {
        "Female": 0,
        "Male": 1,
        "Other": 2
    }

    gender = gender_mapping[gender]

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

with col2:

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1]
    )

    smoking = st.selectbox(
        "Smoking History",
        [
            "No Info",
            "Current",
            "Former",
            "Never",
            "Ever",
            "Not Current"
        ]
    )

    smoking_mapping = {
        "No Info": 0,
        "Current": 1,
        "Former": 2,
        "Never": 3,
        "Ever": 4,
        "Not Current": 5
    }

    smoking = smoking_mapping[smoking]

    hba1c = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.0
    )

    glucose = st.number_input(
        "Blood Glucose Level",
        min_value=50.0,
        max_value=300.0,
        value=100.0
    )

st.markdown("---")

# PREDICT
if st.button("🔍 Predict"):

    data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose]
    })

    data = scaler.transform(data)

    prediction = model.predict(data)

    probability = prediction[0][0] * 100

    st.markdown("## 📊 Prediction Result")

    if prediction[0][0] >= 0.5:
        st.error("⚠️ Patient is likely to have Diabetes")
        st.write(f"**Risk Score : {probability:.2f}%**")
        st.warning("Please consult a healthcare professional.")
    else:
        st.success("✅ Patient is unlikely to have Diabetes")
        st.write(f"**Risk Score : {(100-probability):.2f}%**")
        st.info("Maintain a healthy lifestyle.")

st.markdown("---")

st.caption("Developed using TensorFlow | ANN | Streamlit")