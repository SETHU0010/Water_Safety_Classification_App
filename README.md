# 💧 Water Safety Classification App

A machine learning-powered web application using **Streamlit** to predict whether a water sample is **Safe** or **Unsafe** based on its physicochemical properties.

---

## 🧩 Problem Statement

Contaminated drinking water leads to serious health issues and environmental risks. Traditional lab testing is often expensive, time-consuming, and inaccessible in remote areas. This project addresses the need for a **quick, data-driven classification system** to determine water safety using measurable water quality parameters.

---

## 🎯 Aim

To build an intelligent, user-friendly system that classifies water as safe or unsafe based on key quality indicators using machine learning and provides instant insights through a web interface.

---

## ✅ Features

- 📁 Upload water quality datasets (CSV format)
- ⚙️ Automated preprocessing and handling of missing values
- ⚖️ Handles class imbalance using SMOTE
- 🌲 Random Forest Classifier with GridSearchCV
- 📊 Displays classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- 🔍 Real-time predictions for user-input values
- 💾 Saves model and scaler for reuse

---

## ⚡ Advantages

- Fast, automated analysis without lab dependency
- Easy-to-use interface for non-technical users
- Supports batch prediction from uploaded files
- Scalable for real-time or large-scale deployments
- Model is reusable and updateable

---

## ⚠️ Limitations

- Requires accurate and complete input data
- Performance depends on quality and diversity of training data
- May not capture region-specific contaminants not in the dataset

---

## 💻 Tech Stack

- **Language:** Python 3.10+
- **Libraries:** Pandas, NumPy, Scikit-learn, imbalanced-learn, Streamlit, Seaborn, Matplotlib, Joblib

