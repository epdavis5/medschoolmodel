
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("med_school_model.pkl")

# App title
st.title("Med School Enrollment Predictor 🚀")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Ensure data format is correct
    st.write("Preview of uploaded data:", data.head())

    # Predict using the model
    probabilities = model.predict_proba(data)[:, 1]  # Get probability of enrollment
data["Enrollment Probability (%)"] = (probabilities * 100).round(2)  # Convert to percentage

    # Show results
    st.write("Predictions:", data)

    # Allow download of predictions
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

