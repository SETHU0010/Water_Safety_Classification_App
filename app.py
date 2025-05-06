# water_safety_app.py

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# 1. Title
st.title("Water Safety Classification App")
st.markdown("Upload water quality data to predict whether it's **Safe** or **Unsafe** for consumption.")

# 2. Data Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Dataset")
    st.dataframe(df.head())

    # 3. Data Cleaning
    try:
        df['ammonia'] = pd.to_numeric(df['ammonia'], errors='coerce')
    except:
        st.warning("No 'ammonia' column found")

    df.dropna(inplace=True)

    # 4. Feature and Target
    X = df.drop('is_safe', axis=1)
    y = df['is_safe']

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Handle Imbalanced Data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    # 8. Model Training with Hyperparameter Tuning
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_resampled, y_resampled)
    best_model = grid_search.best_estimator_

    # 9. Model Evaluation
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])

    st.subheader("Model Evaluation Metrics")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**Precision:** {prec:.2f}")
    st.write(f"**Recall:** {rec:.2f}")
    st.write(f"**F1-Score:** {f1:.2f}")
    st.write(f"**ROC-AUC:** {roc:.2f}")

    # 10. Save Model
    joblib.dump(best_model, "water_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # 11. Predict New Sample
    st.subheader("Real-time Prediction")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"Enter {col} value", value=0.0)

    if st.button("Predict Water Safety"):
        user_df = pd.DataFrame([input_data])
        user_scaled = scaler.transform(user_df)
        prediction = best_model.predict(user_scaled)
        st.write("✅ **Safe Water**" if prediction[0] == 1 else "⚠️ **Unsafe Water**")
