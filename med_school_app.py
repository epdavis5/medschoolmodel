import streamlit as st
import pandas as pd
import joblib
import numpy as np

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
            data[col] = data[col].astype(str)  # Convert to string to avoid dtype errors
            
            if col in label_encoders:
                # Handle unseen labels by assigning "Unknown" (or most frequent label)
                known_classes = list(label_encoders[col].classes_)  # Get known labels
                data[col] = data[col].apply(lambda x: x if x in known_classes else "Unknown")  # Replace unknowns
                
                # Update label encoder to include "Unknown"
                if "Unknown" not in known_classes:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, "Unknown")
                
                # Apply label encoding
                data[col] = label_encoders[col].transform(data[col])
            else:
                st.warning(f"⚠️ Warning: No encoder found for column {col}, skipping encoding.")

    # Debugging: Show data types before prediction
    st.write("🛠️ Data format before prediction:")
    st.write(data.dtypes)

    # Ensure all necessary columns exist
    expected_features = model.feature_names_in_  # Get feature names used during training
    missing_features = [col for col in expected_features if col not in data.columns]
    if missing_features:
        st.error(f"❌ The following required columns are missing: {missing_features}")
    else:
        # Predict probabilities instead of binary 0/1
        try:
            probabilities = model.predict_proba(data)[:, 1]  # Get probability of enrollment
            data["Enrollment Probability (%)"] = (probabilities * 100).round(2)  # Convert to %

            # Show predictions
            st.write("🔮 Predictions:")
            st.write(data)

            # Allow user to download the predictions
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
