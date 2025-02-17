
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

# Define categorical columns (these must match the training data)
categorical_cols = ['fap_yes_no', 'sex', 'URM', 'citizenship_country_code', 
                    'perm_state', 'residency_state_code', 'race_full_desc', 
                    'hispanic_ethnicity_yes_no', 'first_generation_yes_no', 'Rural']

# Ensure all categorical columns are label-encoded before prediction
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].astype(str)  # Convert to string to avoid dtype errors
        data[col] = label_encoders[col].transform(data[col])  # Apply label encoding

    # Predict using the model
st.write("Data format being used for prediction:", data.dtypes)
probabilities = model.predict_proba(data)[:, 1]  # Get probability of enrollment
data["Enrollment Probability (%)"] = (probabilities * 100).round(2)  # Convert to percentage

    # Show results
st.write("Predictions:", data)

    # Allow download of predictions
csv = data.to_csv(index=False).encode("utf-8")
st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

