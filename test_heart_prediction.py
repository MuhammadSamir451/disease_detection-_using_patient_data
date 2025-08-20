import joblib
import numpy as np
import os

# Paths
MODEL_PATH = r"c:\Users\hp\OneDrive\Desktop\Projects\Project 1 (Disease Prediction Using Patient Data)\models\heartDisease_model_random_forest.pkl"
SCALER_PATH = r"c:\Users\hp\OneDrive\Desktop\Projects\Project 1 (Disease Prediction Using Patient Data)\models\heart_scaler.pkl"

# Load model & scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature order must match training:
# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
feature_names = [
    "Age", "Sex (1=male, 0=female)", "Chest Pain Type (0-3)", "Resting BP", "Cholesterol",
    "Fasting Blood Sugar (1 > 120mg/dl)", "Resting ECG (0-2)", "Max Heart Rate",
    "Exercise Angina (1=yes)", "ST Depression (oldpeak)", "Slope (0-2)",
    "Number of Vessels (0-3)", "Thal (1=normal, 2=fixed defect, 3=reversible defect)"
]

# Take user input
user_data = []
print("\n--- Heart Disease Prediction ---\n")
for feature in feature_names:
    val = float(input(f"Enter {feature}: "))
    user_data.append(val)

# Convert to numpy and scale
user_array = np.array(user_data).reshape(1, -1)
user_array_scaled = scaler.transform(user_array)

# Predict
prediction = model.predict(user_array_scaled)[0]
probability = model.predict_proba(user_array_scaled)[0][1]

# Output
if prediction == 1:
    print(f"\n❌ Patient is predicted to have **Heart Disease** (Probability: {probability:.2f})")
else:
    print(f"\n✅ Patient is predicted as **No Heart Disease** (Probability: {probability:.2f})")
