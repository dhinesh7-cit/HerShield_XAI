# HerShield_XAI 💡🩺

**Maternal Health Risk Classification with SHAP-based Explainable AI**

HerShield_XAI is a machine learning project focused on predicting maternal health risks based on key health indicators. This project incorporates SHAP (SHapley Additive exPlanations) to provide interpretable insights into how each feature impacts the model’s predictions, promoting transparency and trust in healthcare AI applications.

---

## 🧠 Objective

To classify maternal health risk levels — **Low**, **Mid**, or **High** — using supervised machine learning, and to explain model decisions using SHAP visualizations.

---

## 📊 Dataset

- **Source:** [Kaggle - Maternal Health Risk Data](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data)

### Features:
- Age
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Blood Sugar
- Body Temperature
- Heart Rate

### Target:
- Risk Level (`low`, `mid`, `high`)

---

## 🧪 Models Used

- Logistic Regression
- Random Forest
- XGBoost

The models were evaluated using metrics like:
- Accuracy
- Precision
- Recall
- F1-Score

---

## 🔍 Explainable AI (XAI) with SHAP

We used SHAP to:
- Identify the most influential features on predictions
- Provide global and individual prediction explanations
- Generate intuitive visualizations like force plots, summary plots, and dependence plots

---

## 📁 Project Structure

HerShield_XAI/
│
├── dataset/                 # Contains the dataset file(s)
│   └── maternal_health_risk.csv
│
├── main.py                  # Main Python script for training, evaluation, and SHAP analysis
└── README.md                # Project overview and instructions

---

## ✨ Features

- ✅ **Maternal Risk Classification** using ML models (Logistic Regression, Random Forest, XGBoost)
- 🧹 **Data Preprocessing**: Cleansing, encoding, normalization
- 📊 **Model Evaluation**: Accuracy, Precision, Recall, F1-Score
- 🔍 **Explainable AI with SHAP & LIME**:
  - SHAP: Global feature importance, force plots, summary plots
  - LIME: Local interpretable explanations for individual predictions
- 📜 **Visualizations**: Easy-to-understand insights into model behavior
- 🧠 **Single-Script Execution** via `main.py`
- ⚡ **Lightweight & Reproducible**: Minimal setup, quick execution

---
