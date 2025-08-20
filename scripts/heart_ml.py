import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset (no header in UCI file)
df = pd.read_csv(
    r"C:\Users\hp\OneDrive\Desktop\Projects\Project 1 (Disease Prediction Using Patient Data)\data\processed.cleveland.data",
    header=None
)

# Assign column names from UCI documentation
col_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df.columns = col_names

# Replace "?" with NaN and convert all to numeric
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric, errors="coerce")

# Fill missing values with column mean
df = df.fillna(df.mean())

# Convert target into binary (0 = no disease, 1 = disease present)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Features & target
X = df.drop("target", axis=1)
y = df["target"]

# Scale only features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# EDA: Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Heart Disease Dataset")
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy:", acc_log)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", acc_rf)

# Save the best model
save_dir = r"c:\Users\hp\OneDrive\Desktop\Projects\Project 1 (Disease Prediction Using Patient Data)\models"
os.makedirs(save_dir, exist_ok=True)

if acc_rf >= acc_log:
    best_model = rf_model
    best_name = "random_forest"
    best_acc = acc_rf
else:
    best_model = log_model
    best_name = "logistic_regression"
    best_acc = acc_log

model_path = os.path.join(save_dir, f"heartDisease_model_{best_name}.pkl")
scaler_path = os.path.join(save_dir, "heart_scaler.pkl")

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✅ Best Model: {best_name.replace('_', ' ').title()}")
print(f"✅ Accuracy: {best_acc:.4f}")
print(f"✅ Model saved at: {model_path}")
print(f"✅ Scaler saved at: {scaler_path}")
