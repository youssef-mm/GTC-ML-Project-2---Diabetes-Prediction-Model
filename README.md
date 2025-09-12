# GTC-ML-Project-2---Diabetes-Prediction-Model

# 🩺 Diabetes Prediction Model

## 📌 Overview
This project aims to build a **Machine Learning pipeline** to predict whether a patient is diabetic or not, based on medical and lifestyle features.  
We applied data preprocessing, feature selection, multiple machine learning models, hyperparameter tuning, and explainability techniques (**SHAP** & **LIME**).

---

## 📂 Dataset
- Features include: `Glucose`, `BMI`, `Age`, `Blood Pressure`, `Insulin`, `Skin Thickness`, etc.  
- Target: **Outcome** (0 = Non-Diabetic, 1 = Diabetic)

---

## ⚙️ Pipeline Steps
1. **Data Preprocessing**
   - Handling missing values with `SimpleImputer`
   - Balancing data using **SMOTE**
   - Feature scaling using `StandardScaler`

2. **Feature Selection**
   - Using `SelectKBest (f_classif)` to pick top features

3. **Model Training**
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
   - Stacking Classifier (Ensemble)

4. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Cross-validation with `StratifiedKFold`

5. **Explainability**
   - **SHAP** plots for global & local interpretability
   - **LIME** explanations for individual predictions

6. **Model Saving**
   - Trained models saved using `joblib` (`best_model_stack.pkl`, `log_reg.pkl`, `svm.pkl`, etc.)

---

## 📊 Results
- **Best Model:** Stacking Classifier  
- Achieved highest ROC-AUC and F1-score compared to individual models.  
- SHAP & LIME highlighted the importance of **Glucose**, **BMI**, and **Age** in prediction.

---

## 🚀 How to Run
### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
```
## 3️⃣ Run Google Colab Notebook
```
 Google Colab notebook
```
4️⃣ Run Streamlit App (Optional)
```
streamlit run app.py
```
## 🛠️ Tech Stack
Python (Pandas, NumPy, Matplotlib, Seaborn)
Scikit-learn
Imbalanced-learn (SMOTE)
SHAP, LIME
Streamlit (for deployment)
