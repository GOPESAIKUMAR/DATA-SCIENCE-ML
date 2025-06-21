import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("xgb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("HR Promotion Predictor")

uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

threshold = st.slider("Select Prediction Threshold", 0.0, 1.0, 0.25, 0.01)

if uploaded_file:
    try:
        test = pd.read_csv(uploaded_file)
        employee_id = test["employee_id"]
        X_test = test.drop(columns=["employee_id", "region"], errors="ignore")
        X_test_transformed = preprocessor.transform(X_test)

        probs = model.predict_proba(X_test_transformed)[:, 1]
        predictions = (probs >= threshold).astype(int)

        test["is_promoted"] = predictions
        st.write("### Prediction Results", test[["employee_id", "is_promoted"]])

        csv = test[["employee_id", "is_promoted"]].to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions as CSV", csv, "submission.csv", "text/csv")
    except Exception as e:
        st.error(f"An error occurred: {e}")
