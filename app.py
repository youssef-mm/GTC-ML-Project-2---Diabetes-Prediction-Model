from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ----------------------------
# 1. Page Config & Theme
# ----------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Diabetes Prediction System with Explainable AI (SHAP)")
st.markdown(
    "Predict diabetes risk using Machine Learning models (Stacking, Random Forest, XGBoost, SVM, Logistic Regression)."
)

BASE_DIR = Path(__file__).resolve().parent


# ----------------------------
# 2. Cached Model & Scaler Loading
# ----------------------------
@st.cache_resource
def load_models_and_artifacts():
    best_model = joblib.load(BASE_DIR / "best_model_stack.pkl")  # Stacking
    log_reg = joblib.load(BASE_DIR / "log_reg.pkl")
    svm_model = joblib.load(BASE_DIR / "svm_best.pkl")
    rf_model = joblib.load(BASE_DIR / "rf.pkl")
    xgb_model = joblib.load(BASE_DIR / "xgb.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    selected_features = joblib.load(BASE_DIR / "selected_features.pkl")

    models_dict = {
        "Stacking (Best Performance)": best_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "Support Vector Machine (SVM)": svm_model,
        "Logistic Regression": log_reg,
    }
    return models_dict, scaler, selected_features


models, scaler, selected_features = load_models_and_artifacts()


# ----------------------------
# 3. Prediction Function
# ----------------------------
def predict_diabetes(input_data, model):
    df = pd.DataFrame([input_data])

    # Feature Engineering
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[20, 30, 40, 50, 60, 100],
        labels=["20s", "30s", "40s", "50s", "60+"],
    )
    df["BMICategory"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Age_DPF"] = df["Age"] * df["DiabetesPedigreeFunction"]
    df["PregnantBinary"] = df["Pregnancies"].apply(lambda x: 1 if x > 0 else 0)

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

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_scaled)[0][1]
    else:
        prob = float(pred)

    return {
        "prediction": "Diabetic" if pred == 1 else "Non-Diabetic",
        "probability": prob,
        "input_scaled": df_scaled,
    }


# ----------------------------
# 4. Sidebar Inputs
# ----------------------------
st.sidebar.header("📋 Patient Clinical Inputs")
selected_model_name = st.sidebar.selectbox(
    "Choose ML Model:", list(models.keys())
)

Pregnancies = st.sidebar.number_input(
    "Pregnancies", min_value=0, max_value=20, value=1
)
Glucose = st.sidebar.number_input(
    "Glucose Level (mg/dL)", min_value=0, max_value=300, value=120
)
BloodPressure = st.sidebar.number_input(
    "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=72
)
SkinThickness = st.sidebar.number_input(
    "Skin Thickness (mm)", min_value=0, max_value=100, value=23
)
Insulin = st.sidebar.number_input(
    "Insulin Level (mu U/ml)", min_value=0, max_value=900, value=85
)
BMI = st.sidebar.number_input(
    "BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=28.1, step=0.1
)
DPF = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.47,
    step=0.01,
)
Age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=35)

input_data = {
    "Pregnancies": Pregnancies,
    "Glucose": Glucose,
    "BloodPressure": BloodPressure,
    "SkinThickness": SkinThickness,
    "Insulin": Insulin,
    "BMI": BMI,
    "DiabetesPedigreeFunction": DPF,
    "Age": Age,
}

predict_btn = st.sidebar.button("Run Assessment 🚀", use_container_width=True)

# ----------------------------
# 5. Dashboard Output
# ----------------------------
if predict_btn:
    model = models[selected_model_name]
    result = predict_diabetes(input_data, model)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("🎯 Assessment Summary")
        prob_pct = result["probability"] * 100

        if result["prediction"] == "Diabetic":
            st.error(
                f"⚠️ **Result:** High Risk ({result['prediction']})\n\n**Confidence:** {prob_pct:.1f}%"
            )
        else:
            st.success(
                f"✅ **Result:** Low Risk ({result['prediction']})\n\n**Confidence:** {100 - prob_pct:.1f}%"
            )

        st.metric(
            label="Diabetes Probability",
            value=f"{prob_pct:.1f}%",
            delta="Elevated" if prob_pct >= 50 else "Normal",
            delta_color="inverse" if prob_pct >= 50 else "normal",
        )

        st.markdown(f"**Model Utilized:** `{selected_model_name}`")

    with col2:
        st.subheader("📊 Patient Position on Glucose vs BMI")
        fig, ax = plt.subplots(figsize=(6, 4))
        np.random.seed(42)
        bg_glucose = np.random.normal(120, 30, 150)
        bg_bmi = np.random.normal(27, 6, 150)
        ax.scatter(
            bg_glucose,
            bg_bmi,
            alpha=0.25,
            color="#90CAF9",
            label="Population Baseline",
        )
        ax.scatter(
            input_data["Glucose"],
            input_data["BMI"],
            color="#E53935",
            s=120,
            edgecolors="black",
            label="Current Patient",
        )
        ax.set_xlabel("Glucose (mg/dL)")
        ax.set_ylabel("BMI")
        ax.legend()
        st.pyplot(fig)
        plt.clf()

    # ----------------------------
    # 6. SHAP Explainability
    # ----------------------------
    st.markdown("---")
    st.subheader("🔍 Explainable AI (SHAP Feature Contributions)")

    # Only run SHAP for XGBoost and Random Forest where tree shap applies
    if "XGBoost" in selected_model_name or "Random Forest" in selected_model_name:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(result["input_scaled"])

            expected_value = explainer.expected_value
            if isinstance(expected_value, (np.ndarray, list)) and len(expected_value) > 1:
                ev = expected_value[1]
                vals = (
                    shap_values.values[:, :, 1]
                    if len(shap_values.values.shape) == 3
                    else shap_values.values
                )
            else:
                ev = (
                    expected_value[0]
                    if isinstance(expected_value, (np.ndarray, list))
                    else expected_value
                )
                vals = shap_values.values

            st.text("Visualizing how patient features influence the risk score:")
            fig_shap, ax_shap = plt.subplots(figsize=(8, 3))
            shap.plots.bar(
                shap.Explanation(
                    values=vals[0],
                    data=result["input_scaled"][0],
                    feature_names=selected_features,
                ),
                show=False,
            )
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.info(
                f"Feature contribution visualization note: {e}"
            )
    else:
        st.info(
            "💡 SHAP bar explanations are available when selecting **XGBoost** or **Random Forest** from the sidebar."
        )

else:
    st.info(
        "👈 Enter the patient data in the sidebar and click **'Run Assessment 🚀'** to generate predictions."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
        ### 🧪 Features Evaluated:
        - **Glucose:** Plasma glucose concentration.
        - **Blood Pressure:** Diastolic blood pressure (mm Hg).
        - **BMI:** Body Mass Index (weight in kg / height in m²).
        - **Diabetes Pedigree Function:** Genetic influence score.
        - **Age & Insulin:** Clinical demographic and metabolic indicators.
        """
        )
    with col_b:
        st.markdown(
            """
        ### 🧠 Models Benchmarked:
        1. **Stacking Ensemble:** Meta-classifier combining multiple algorithms for top accuracy.
        2. **XGBoost:** Gradient boosted trees with SHAP interpretability.
        3. **Random Forest:** Ensemble of bagging decision trees.
        4. **Support Vector Classifier (SVM)**
        5. **Logistic Regression**
        """
        )
