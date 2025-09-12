import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ----------------------------
# Load Models & Data
# ----------------------------
best_model = joblib.load("best_model_stack.pkl")  # Stacking
log_reg = joblib.load("log_reg.pkl")
svm_model = joblib.load("svm_best.pkl")
rf_model = joblib.load("rf.pkl")
xgb_model = joblib.load("xgb.pkl")

scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

models = {
    "Stacking": best_model,
    "Logistic Regression": log_reg,
    "SVM": svm_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# ----------------------------
# Prediction Function
# ----------------------------
def predict_diabetes(input_data, model):
    df = pd.DataFrame([input_data])
    
    # Feature Engineering
    df['AgeGroup'] = pd.cut(df['Age'], bins=[20,30,40,50,60,100], labels=['20s','30s','40s','50s','60+'])
    df['BMICategory'] = pd.cut(df['BMI'], bins=[0,18.5,24.9,29.9,100], labels=['Underweight','Normal','Overweight','Obese'])
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Age_DPF"] = df["Age"] * df["DiabetesPedigreeFunction"]
    df["PregnantBinary"] = df["Pregnancies"].apply(lambda x: 1 if x>0 else 0)
    
    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)
    
    # Add missing columns
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0
            
    # Arrange columns
    df = df[selected_features]
    
    # Scale
    df_scaled = scaler.transform(df)
    
    # Prediction
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]
    
    return {"prediction": "Diabetic" if pred==1 else "Non-Diabetic",
            "probability": prob,
            "input_scaled": df_scaled}

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Diabetes Prediction App with SHAP & Visualization")

# Sidebar
st.sidebar.header("Patient Data Input")
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))

Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
Glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, value=120)
BloodPressure = st.sidebar.number_input("BloodPressure", min_value=0, max_value=200, value=70)
SkinThickness = st.sidebar.number_input("SkinThickness", min_value=0, max_value=100, value=20)
Insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
BMI = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DPF = st.sidebar.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5)
Age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)

input_data = {
    "Pregnancies": Pregnancies,
    "Glucose": Glucose,
    "BloodPressure": BloodPressure,
    "SkinThickness": SkinThickness,
    "Insulin": Insulin,
    "BMI": BMI,
    "DiabetesPedigreeFunction": DPF,
    "Age": Age
}

if st.button("Predict"):
    model = models[selected_model]
    result = predict_diabetes(input_data, model)
    
    st.subheader(f"Model Used: {selected_model}")
    st.write(f"Prediction: **{result['prediction']}**")
    st.write(f"Probability of Diabetes: **{result['probability']*100:.2f}%**")
    
    # ----------------------------
    # Glucose vs BMI Plot
    # ----------------------------
    st.subheader("Glucose vs BMI Visualization")
    fig, ax = plt.subplots()
    np.random.seed(42)
    bg_glucose = np.random.normal(120, 30, 100)
    bg_bmi = np.random.normal(25, 5, 100)
    ax.scatter(bg_glucose, bg_bmi, alpha=0.2, label="Training data")
    ax.scatter(input_data['Glucose'], input_data['BMI'], color='red', s=100, label='Patient')
    ax.set_xlabel("Glucose")
    ax.set_ylabel("BMI")
    ax.set_title("Patient on Glucose vs BMI")
    ax.legend()
    st.pyplot(fig)
    plt.clf()
    
    # ----------------------------
    # SHAP Feature Importance
    # ----------------------------

    # ----------------------------
# SHAP Feature Importance (XGBoost & RandomForest only)
# ----------------------------
st.subheader("Feature Contribution (SHAP Values)")
st.text("Red = positive impact, Blue = negative impact")

if selected_model in ["XGBoost"]:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(result['input_scaled'])
        
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray) and expected_value.shape[0] > 1:
            ev = expected_value[1]
        else:
            ev = expected_value
        
        shap.initjs()
        shap.force_plot(ev, shap_values.values, result['input_scaled'], matplotlib=True, show=False)
        st.pyplot(plt.gcf())
        
    except Exception as e:
        st.warning(f"SHAP plot could not be generated. Error: {e}")
else:
    st.info("SHAP plot is only available for Tree-based models (XGBoost).")

