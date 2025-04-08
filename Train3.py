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

# ✅ MLflow setup - MOVED THIS SECTION UP
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Fake News Detection 3")

# ✅ Load data
df = pd.read_csv('final_fake_news.csv')

X = df['text'].astype(str)
y = df['label']

# ✅ Split data
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf_trainval = vectorizer.fit_transform(X_trainval)
X_tfidf_test = vectorizer.transform(X_test)

# Feature Selection
selector = SelectKBest(score_func=chi2, k=1000)
X_selected_trainval = selector.fit_transform(X_tfidf_trainval, y_trainval)
X_selected_test = selector.transform(X_tfidf_test)

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ========== Logistic Regression ==========
log_reg = LogisticRegression(max_iter=1000)
log_scores = cross_val_score(log_reg, X_selected_trainval, y_trainval, cv=kf, scoring='accuracy')
log_mean_acc = np.mean(log_scores)
log_reg.fit(X_selected_trainval, y_trainval)

# Predict & Evaluate on Test Set
y_pred_log = log_reg.predict(X_selected_test)
log_precision = precision_score(y_test, y_pred_log)
log_recall = recall_score(y_test, y_pred_log)
log_f1 = f1_score(y_test, y_pred_log)
log_acc = accuracy_score(y_test, y_pred_log)

with mlflow.start_run(run_name="Logistic Regression"):
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("cv_accuracy", log_mean_acc)
    mlflow.log_metric("test_accuracy", log_acc)
    mlflow.log_metric("precision", log_precision)
    mlflow.log_metric("recall", log_recall)
    mlflow.log_metric("f1_score", log_f1)
    mlflow.sklearn.log_model(log_reg, "Logistic_Regression_Model")
    
    # Only log the dataset in one of the runs
    mlflow.log_artifact("final_fake_news.csv")

    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump(log_reg, f)
    mlflow.log_artifact("logistic_regression_model.pkl")

print(f"Logistic Regression Test Accuracy: {log_acc}")

# ========== Random Forest ==========
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf_model, X_selected_trainval, y_trainval, cv=kf, scoring='accuracy')
rf_mean_acc = np.mean(rf_scores)
rf_model.fit(X_selected_trainval, y_trainval)

y_pred_rf = rf_model.predict(X_selected_test)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_acc = accuracy_score(y_test, y_pred_rf)

with mlflow.start_run(run_name="Random Forest"):
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("cv_accuracy", rf_mean_acc)
    mlflow.log_metric("test_accuracy", rf_acc)
    mlflow.log_metric("precision", rf_precision)
    mlflow.log_metric("recall", rf_recall)
    mlflow.log_metric("f1_score", rf_f1)
    mlflow.sklearn.log_model(rf_model, "Random_Forest_Model")

    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    mlflow.log_artifact("random_forest_model.pkl")

print(f"Random Forest Test Accuracy: {rf_acc}")

# ========== Save Best Model ==========
if log_acc > rf_acc:
    best_model = log_reg
    best_model_name = "Logistic Regression"
    best_accuracy = log_acc
else:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_accuracy = rf_acc

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
    
# บันทึก vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# บันทึก selector
with open("selector.pkl", "wb") as f:
    pickle.dump(selector, f)

with mlflow.start_run(run_name="Model Comparison"):
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_artifact("best_model.pkl")
    mlflow.log_artifact("vectorizer.pkl") 
    mlflow.log_artifact("selector.pkl")

print(f"Best Model: {best_model_name} (Test Accuracy: {best_accuracy})")