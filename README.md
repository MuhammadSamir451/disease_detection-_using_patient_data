# 🧑‍⚕️ Disease Prediction Using Patient Data

This project predicts **Diabetes** and **Heart Disease** using **Machine Learning models**.  
It was built as my first ML project after returning from my village, and serves as a foundation for applying AI to healthcare problems.  

---

## 📌 Project Overview
- Implemented **data preprocessing** (handling missing values, scaling, encoding).  
- Trained multiple ML models (**Logistic Regression**, **Random Forest**) for both diseases.  
- Compared models and saved the **best-performing model** for each disease.  
- Added **user input prediction scripts** to test with patient data.  
- Visualized model insights with **heatmaps and accuracy plots**.  

---

## 📊 Datasets
- **Diabetes Dataset** – PIMA Indian Diabetes dataset (CSV format).  
- **Heart Disease Dataset** – UCI Heart Disease dataset (`processed.cleveland.data`).  

Both datasets were preprocessed:  
- Missing values filled with column means.  
- Features scaled using **MinMaxScaler**.  
- Target variable converted to **binary classification**.  

---

## ⚙️ Workflow
1. **Data Cleaning & Preprocessing**  
   - Replaced `"?"` with `NaN`.  
   - Applied scaling for continuous values.  
   - Converted categorical/text columns using encoders.  

2. **Exploratory Data Analysis (EDA)**  
   - Generated **heatmaps** to analyze feature correlations.  
   - Plotted **model accuracy comparison** for Logistic Regression vs Random Forest.  

3. **Model Training**  
   - Logistic Regression  
   - Random Forest Classifier  
   - Selected the best model based on accuracy score.  

4. **Saving Models**  
   - Trained models saved with `joblib` into the `models/` directory.  
   - Separate scalers saved for consistent prediction.  

5. **Prediction Script**  
   - User inputs patient data through CLI.  
   - Script scales inputs and predicts **Disease / No Disease** with probability score.  

---

## 📈 Results

### Diabetes Prediction
- **Best Model:** Logistic Regression  
- **Accuracy:** ~0.82  
- Example Prediction:  
