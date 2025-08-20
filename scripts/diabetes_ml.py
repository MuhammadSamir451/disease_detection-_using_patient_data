import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Dataset
file_path = r"C:\Users\hp\OneDrive\Desktop\Projects\Project 1 (Disease Prediction Using Patient Data)\data\diabetes_binary_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_path)

print("Dataset Preview:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())

# 2. Split features and target
X = df.drop("Diabetes_binary", axis=1)   # features
y = df["Diabetes_binary"]               # target

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. ------------------------Train Models------------------------
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # RF does not need scaling

# 6. ------------------------Evaluation------------------------
# Logistic Regression Performance
y_pred_log = log_reg.predict(X_test_scaled)
acc_log = accuracy_score(y_test, y_pred_log)
print("\nLogistic Regression Report:\n", classification_report(y_test, y_pred_log))
print("Accuracy (LogReg):", acc_log)

# Random Forest Performance
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Report:\n", classification_report(y_test, y_pred_rf))
print("Accuracy (Random Forest):", acc_rf)

# 7. Save Best Model & Scaler
save_dir = r"c:\Users\hp\OneDrive\Desktop\Projects\Project 1 (Disease Prediction Using Patient Data)\models"
os.makedirs(save_dir, exist_ok=True)

if acc_rf >= acc_log:
    best_model = rf
    best_name = "random_forest"
    best_acc = acc_rf
else:
    best_model = log_reg
    best_name = "logistic_regression"
    best_acc = acc_log

model_path = os.path.join(save_dir, f"diabetes_model_{best_name}.pkl")
scaler_path = os.path.join(save_dir, "scaler.pkl")

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✅ Best Model: {best_name.replace('_', ' ').title()}")
print(f"✅ Accuracy: {best_acc:.4f}")
print(f"✅ Model saved at: {model_path}")
print(f"✅ Scaler saved at: {scaler_path}")
