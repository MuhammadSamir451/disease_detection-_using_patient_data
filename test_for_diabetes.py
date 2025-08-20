import joblib
import numpy as np
import pandas as pd

# ---------- Load Logistic Regression Model and Scaler ----------
model_path = r"c:\Users\hp\OneDrive\Desktop\Projects\Project 1 (Disease Prediction Using Patient Data)\models\diabetes_model_logistic_regression.pkl"
scaler_path = r"c:\Users\hp\OneDrive\Desktop\Projects\Project 1 (Disease Prediction Using Patient Data)\models\scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ---------- Define User Input ----------
# IMPORTANT: Keys must match your dataset's columns EXACTLY
user_input = {
    "HighBP": 1,
    "HighChol": 0,
    "CholCheck": 1,
    "BMI": 28,
    "Smoker": 1,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 3,
    "MentHlth": 2,
    "PhysHlth": 1,
    "DiffWalk": 0,
    "Sex": 1,   # 1 = Male, 0 = Female
    "Age": 9,
    "Education": 4,
    "Income": 3
}

# ---------- Convert to DataFrame with correct column order ----------
X_columns = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

user_df = pd.DataFrame([user_input], columns=X_columns)

# ---------- Scale input ----------
user_scaled = scaler.transform(user_df)

# ---------- Prediction ----------
prediction = model.predict(user_scaled)[0]
probability = model.predict_proba(user_scaled)[0][1]

if prediction == 1:
    print(f"⚠️ Patient is predicted to have Diabetes (Probability: {probability:.2f})")
else:
    print(f"✅ Patient is predicted as Non-Diabetic (Probability: {probability:.2f})")
