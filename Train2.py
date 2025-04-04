import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

# ✅ 1. โหลดข้อมูล
df = pd.read_csv('final_fake_news.csv')

# ✅ 2. บันทึก dataset เข้า MLflow
# mlflow.log_artifact("final_fake_news.csv")

# ✅ 3. เตรียมข้อมูล
X = df['text'].astype(str)
y = df['label']

# ตั้งค่า MLflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Fake News Detection")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

# Feature Selection
selector = SelectKBest(score_func=chi2, k=1000)
X_selected = selector.fit_transform(X_tfidf, y)

# ✅ 4. Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ✅ 5. Train Logistic Regression with Cross-validation
log_reg = LogisticRegression(max_iter=1000)
log_scores = cross_val_score(log_reg, X_selected, y, cv=kf, scoring='accuracy')
log_mean_acc = np.mean(log_scores)
log_reg.fit(X_selected, y)
y_pred_log = log_reg.predict(X_selected)

# Compute additional metrics
log_precision = precision_score(y, y_pred_log)
log_recall = recall_score(y, y_pred_log)
log_f1 = f1_score(y, y_pred_log)

with mlflow.start_run():
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("cv_accuracy", log_mean_acc)
    mlflow.log_metric("precision", log_precision)
    mlflow.log_metric("recall", log_recall)
    mlflow.log_metric("f1_score", log_f1)
    
    mlflow.sklearn.log_model(log_reg, "Logistic_Regression_Model")

    log_model_path = "logistic_regression_model.pkl"
    with open(log_model_path, "wb") as f:
        pickle.dump(log_reg, f)
    mlflow.log_artifact(log_model_path)

print(f"Logistic Regression CV Accuracy: {log_mean_acc}")

# ✅ 6. Train Random Forest with Cross-validation
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf_model, X_selected, y, cv=kf, scoring='accuracy')
rf_mean_acc = np.mean(rf_scores)

rf_model.fit(X_selected, y)
y_pred_rf = rf_model.predict(X_selected)

rf_precision = precision_score(y, y_pred_rf)
rf_recall = recall_score(y, y_pred_rf)
rf_f1 = f1_score(y, y_pred_rf)

with mlflow.start_run():
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("cv_accuracy", rf_mean_acc)
    mlflow.log_metric("precision", rf_precision)
    mlflow.log_metric("recall", rf_recall)
    mlflow.log_metric("f1_score", rf_f1)
    
    mlflow.sklearn.log_model(rf_model, "Random_Forest_Model")

    rf_model_path = "random_forest_model.pkl"
    with open(rf_model_path, "wb") as f:
        pickle.dump(rf_model, f)
    mlflow.log_artifact(rf_model_path)

print(f"Random Forest CV Accuracy: {rf_mean_acc}")

# ✅ 7. Compare Models and Save Best Model
best_model = log_reg if log_mean_acc > rf_mean_acc else rf_model
best_model_name = "Logistic Regression" if log_mean_acc > rf_mean_acc else "Random Forest"
best_accuracy = max(log_mean_acc, rf_mean_acc)

# ✅ บันทึกโมเดลที่ดีที่สุด
best_model_path = "best_model.pkl"
with open(best_model_path, "wb") as f:
    pickle.dump(best_model, f)

with mlflow.start_run():
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_artifact(best_model_path)

print(f"Best Model: {best_model_name} (Accuracy: {best_accuracy})")