import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# Load trained model and scaler
# -------------------------------
model = load_model("loan_default_model.h5")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏦 Loan Repayment Prediction App")

st.write("Enter applicant details below to check the **likelihood of loan repayment**.")

# -------------------------------
# Input fields
# -------------------------------
ApplicantIncome = st.number_input("Applicant Income", value=5000.0)
CoapplicantIncome = st.number_input("Coapplicant Income", value=1500.0)
LoanAmount = st.number_input("Loan Amount (in thousands)", value=128.0)
Loan_Amount_Term = st.number_input("Loan Term (in days)", value=360.0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])

Gender_Male = 1 if st.selectbox("Gender", ["Male", "Female"]) == "Male" else 0
Married_Yes = 1 if st.selectbox("Married", ["Yes", "No"]) == "Yes" else 0

Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Dependents_1 = 1 if Dependents == "1" else 0
Dependents_2 = 1 if Dependents == "2" else 0
Dependents_3plus = 1 if Dependents == "3+" else 0

Education_NotGraduate = 1 if st.selectbox("Education", ["Graduate", "Not Graduate"]) == "Not Graduate" else 0
Self_Employed_Yes = 1 if st.selectbox("Self Employed", ["Yes", "No"]) == "Yes" else 0

Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
Property_Area_Urban = 1 if Property_Area == "Urban" else 0

# -------------------------------
# Prepare input data
# -------------------------------
input_data = np.array([[
    float(ApplicantIncome),
    float(CoapplicantIncome),
    float(LoanAmount),
    float(Loan_Amount_Term),
    float(Credit_History),
    float(Gender_Male),
    float(Married_Yes),
    float(Dependents_1),
    float(Dependents_2),
    float(Dependents_3plus),
    float(Education_NotGraduate),
    float(Self_Employed_Yes),
    float(Property_Area_Semiurban),
    float(Property_Area_Urban)
]])

# Scale only the numeric columns
numeric_indices = [0, 1, 2, 3, 4]
input_scaled = input_data.copy()
input_scaled[:, numeric_indices] = scaler.transform(input_data[:, numeric_indices])

# -------------------------------
# Predict button
# -------------------------------
if st.button("🔮 Predict Loan Repayment"):
    prob_repay = float(model.predict(input_scaled)[0][0])
    prob_default = 1 - prob_repay  # Flip for readability

    st.write(f"### 💡 Probability of Repayment: `{prob_repay:.2f}`")
    st.write(f"### ⚠️ Probability of Default: `{prob_default:.2f}`")

    if prob_repay >= 0.5:
        st.success("✅ Applicant is **likely to repay** the loan.")
    else:
        st.error("❌ Applicant is **likely to default** on the loan.")

    # Optional debug info
    st.caption(f"Raw Input: {list(input_data[0])}")
