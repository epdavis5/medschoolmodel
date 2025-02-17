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

if uploaded_file:
    # Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Show preview of uploaded data
st.write("📊 Preview of uploaded data:", data.head())

    # Define categorical columns (must match training data)
categorical_cols = ['fap_yes_no', 'sex', 'URM', 'citizenship_country_code', 
                        'perm_state', 'residency_state_code', 'race_full_desc', 
                        'hispanic_ethnicity_yes_no', 'first_generation_yes_no', 'Rural']

    # Ensure categorical columns are encoded before prediction
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)  # Convert to string to avoid dtype errors
            if col in label_encoders:
                data[col] = label_encoders[col].transform(data[col])  # Apply label encoding

    # Debugging: Show data types before prediction
st.write("🛠️ Data format before prediction:", data.dtypes)

    # Predict probabilities instead of binary 0/1
probabilities = model.predict_proba(data)[:, 1]  # Get probability of enrollment
data["Enrollment Probability (%)"] = (probabilities * 100).round(2)  # Convert to %

    # Show predictions
st.write("🔮 Predictions:", data)

    # Allow user to download the predictions
csv = data.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
