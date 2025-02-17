import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("med_school_model.pkl")

# Load label encoders
label_encoders = joblib.load("label_encoders.pkl")

# App title
st.title("🎓 Med School Enrollment Predictor 🚀")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Excel or CSV file", type=["csv", "xlsx"])

# Initialize `data` variable to avoid undefined errors
data = None  

if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Ensure file is not empty
        if data.empty:
            st.error("❌ The uploaded file is empty. Please upload a valid dataset.")
            data = None  # Reset data to prevent further processing
        else:
            st.write("📊 Preview of uploaded data:")
            st.write(data.head())  # Show first few rows
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        data = None

# Only proceed if data is successfully loaded
if data is not None:
    # Define categorical columns (must match training data)
    categorical_cols = ['fap_yes_no', 'sex', 'URM', 'citizenship_country_code', 
                        'perm_state', 'residency_state_code', 'race_full_desc', 
                        'hispanic_ethnicity_yes_no', 'first_generation_yes_no', 'Rural']

    # Ensure categorical columns are encoded before prediction
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)  # Convert to string to avoid dtyp
