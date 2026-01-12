import streamlit as st
from src.train import train_models
from src.evaluate import evaluate

st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="centered"
)

st.title("Employee Attrition Prediction Dashboard")
st.write(
    "This application predicts the likelihood of an employee leaving the company "
    "based on HR-related factors."
)

# Train models (runs once on app load)
log_model, rf_model, scaler, X, y, features = train_models()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Employee Details")

age = st.sidebar.slider("Age", 18, 60, 30)

income = st.sidebar.slider(
    "Monthly Income (₹)",
    min_value=20000,
    max_value=200000,
    value=50000,
    step=5000
)

job_sat = st.sidebar.slider(
    "Job Satisfaction (1 = Low, 4 = High)", 1, 4, 3
)

years = st.sidebar.slider("Years at Company", 0, 40, 5)

wlb = st.sidebar.slider(
    "Work Life Balance (1 = Poor, 4 = Excellent)", 1, 4, 3
)

overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Random Forest"]
)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Attrition"):
    # Normalize income to match training scale
    normalized_income = income / 10

    input_data = [[
        age,
        normalized_income,
        job_sat,
        years,
        wlb,
        1 if overtime == "Yes" else 0,
        1 if gender == "Male" else 0
    ]]

    input_scaled = scaler.transform(input_data)

    model = log_model if model_choice == "Logistic Regression" else rf_model

    # Probability of attrition (class = 1)
    proba = model.predict_proba(input_scaled)[0][1]
    attrition_percent = round(proba * 100, 2)

    st.subheader("Prediction Result")

    st.metric(
        label="Attrition Probability",
        value=f"{attrition_percent}%"
    )

    st.progress(attrition_percent / 100)

    if attrition_percent > 70:
        st.markdown("🔴 **High Risk Zone**")
        st.error("⚠️ High Risk of Attrition")
    elif attrition_percent > 40:
        st.markdown("🟠 **Medium Risk Zone**")
        st.warning("⚠️ Moderate Risk of Attrition")
    else:
        st.markdown("🟢 **Low Risk Zone**")
        st.success("✅ Low Risk of Attrition")

# ---------------- EVALUATION SECTION ----------------
st.markdown("---")
st.subheader("Model Evaluation (On Training Dataset)")

st.info(
    "These evaluation metrics are calculated on the historical dataset "
    "and do not change with individual employee inputs."
)

if st.button("Show Evaluation Metrics"):
    model = log_model if model_choice == "Logistic Regression" else rf_model
    metrics = evaluate(model, X, y)

    st.write("**Accuracy:**", metrics["Accuracy"])
    st.write("**Precision:**", metrics["Precision"])
    st.write("**Recall:**", metrics["Recall"])
    st.write("**Confusion Matrix:**")
    st.write(metrics["Confusion Matrix"])
