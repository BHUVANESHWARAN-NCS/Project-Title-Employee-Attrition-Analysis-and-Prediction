
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('D:/Data science/Projects/Employee-Attrition/random_forest_model.pkl')
scaler = joblib.load('D:/Data science/Projects/Employee-Attrition/scaler.pkl')

st.title("🧠 Employee Attrition Prediction App")
st.markdown("Predict whether an employee will leave the company based on their profile.")

# Form input
with st.form("prediction_form"):
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    job_satisfaction = st.selectbox("Job Satisfaction (1 = Low, 4 = Very High)", [1, 2, 3, 4])
    over_time = st.selectbox("OverTime (0 = No, 1 = Yes)", [0, 1])
    distance_from_home = st.slider("Distance From Home", 1, 30, 10)
    work_life_balance = st.selectbox("Work-Life Balance (1 = Bad, 4 = Excellent)", [1, 2, 3, 4])
    
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([[age, distance_from_home, monthly_income, years_at_company,
                                    job_satisfaction, over_time, work_life_balance]],
                                  columns=['Age', 'DistanceFromHome', 'MonthlyIncome', 'YearsAtCompany',
                                           'JobSatisfaction', 'OverTime', 'WorkLifeBalance'])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ The employee is likely to leave. (Risk Score: {prob:.2f})")
        else:
            st.success(f"✅ The employee is likely to stay. (Risk Score: {prob:.2f})")
