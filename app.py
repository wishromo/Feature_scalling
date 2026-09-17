
import streamlit as st
import pandas as pd
import joblib

# Load saved pipeline and label encoder
model = joblib.load("promotion_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Employee Promotion Prediction")

st.write("Enter employee details below:")

age = st.number_input("Age", min_value=18, max_value=70, value=30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

city = st.selectbox(
    "City",
    ["Rawalpindi", "Islamabad", "Lahore", "Karachi", "Peshawar"]
)

education = st.selectbox(
    "Education Level",
    ["Matric", "Intermediate", "Bachelor", "Master", "PhD"]
)

employment = st.selectbox(
    "Employment Type",
    ["Full-time", "Part-time", "Freelance", "Unemployed"]
)

experience = st.number_input(
    "Experience Years",
    min_value=0.0,
    max_value=40.0,
    value=5.0
)

income = st.number_input(
    "Monthly Income PKR",
    min_value=0.0,
    value=50000.0
)

work_hours = st.number_input(
    "Work Hours Per Week",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

satisfaction = st.selectbox(
    "Satisfaction Level",
    ["Low", "Medium", "High"]
)

certification = st.selectbox(
    "Has Certification",
    ["No", "Yes"]
)

performance = st.number_input(
    "Performance Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

if st.button("Predict Promotion"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "City": [city],
        "Education_Level": [education],
        "Employment_Type": [employment],
        "Experience_Years": [experience],
        "Monthly_Income_PKR": [income],
        "Work_Hours_Per_Week": [work_hours],
        "Satisfaction_Level": [satisfaction],
        "Has_Certification": [certification],
        "Performance_Score": [performance]
    })

    prediction = model.predict(input_data)[0]

    prediction_label = label_encoder.inverse_transform([prediction])[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction_label == "Yes":
        st.success("Employee is likely to be promoted.")
    else:
        st.warning("Employee is not likely to be promoted.")

    st.write(f"Promotion Probability: {probability:.2%}")
