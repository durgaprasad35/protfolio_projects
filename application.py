import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# -------------------------------
# Load Model & Preprocessor
# -------------------------------
def load_model():
    model_path = os.path.join("artifacts", "model.pkl")
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


model, preprocessor = load_model()

# -------------------------------
# UI Title
# -------------------------------
st.title("🎓 Student Dropout Prediction")

st.write("Enter student details below:")

# -------------------------------
# INPUT FIELDS
# -------------------------------

Age = st.number_input("Age", min_value=10, max_value=60, value=20)

Family_Income = st.number_input("Family Income", value=50000.0)

Study_Hours_per_Day = st.number_input("Study Hours per Day", value=4.0)

Attendance_Rate = st.number_input("Attendance Rate (%)", value=75.0)

Assignment_Delay_Days = st.number_input("Assignment Delay Days", value=2)

Travel_Time_Minutes = st.number_input("Travel Time (Minutes)", value=30)

Stress_Index = st.number_input("Stress Index", value=5.0)

GPA = st.number_input("GPA", value=7.0)

Semester_GPA = st.number_input("Semester GPA", value=7.0)

CGPA = st.number_input("CGPA", value=7.0)

Gender = st.selectbox("Gender", ["male", "female"])

Internet_Access = st.selectbox("Internet Access", ["yes", "no"])

Part_Time_Job = st.selectbox("Part Time Job", ["yes", "no"])

Scholarship = st.selectbox("Scholarship", ["yes", "no"])

Semester = st.number_input("Semester", min_value=1, max_value=8, value=1)

Department = st.text_input("Department", "CSE")

Parental_Education = st.text_input("Parental Education", "degree")

# -------------------------------
# PREDICTION BUTTON
# -------------------------------

if st.button("Predict Dropout"):

    input_data = pd.DataFrame({
        "Age": [Age],
        "Family_Income": [Family_Income],
        "Study_Hours_per_Day": [Study_Hours_per_Day],
        "Attendance_Rate": [Attendance_Rate],
        "Assignment_Delay_Days": [Assignment_Delay_Days],
        "Travel_Time_Minutes": [Travel_Time_Minutes],
        "Stress_Index": [Stress_Index],
        "GPA": [GPA],
        "Semester_GPA": [Semester_GPA],
        "CGPA": [CGPA],
        "Gender": [Gender],
        "Internet_Access": [Internet_Access],
        "Part_Time_Job": [Part_Time_Job],
        "Scholarship": [Scholarship],
        "Semester": [Semester],
        "Department": [Department],
        "Parental_Education": [Parental_Education]
    })

    # Preprocess
    data_scaled = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(data_scaled)

    result = "⚠️ Dropout" if prediction[0] == 1 else "✅ No Dropout"

    st.subheader(f"Prediction: {result}")