import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB  # Changed from SVM to Naive Bayes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from xgboost import XGBClassifier


# ✅ MLflow setup - MOVED THIS SECTION UP

if(os.environ.get("ENVIRONMENT") == "PRODUCTION"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_URL"])
else:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Fake News Detection")

# ✅ Load data
df = pd.read_csv('final_fake_news_v2.csv')

X = df['text'].astype(str)
y = df['label']

# ✅ Split data
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf_trainval = vectorizer.fit_transform(X_trainval)
X_tfidf_test = vectorizer.transform(X_test)
with mlflow.start_run(run_name="Feature Engineering"):
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("ngram_range", vectorizer.ngram_range)
    mlflow.log_param("max_features", vectorizer.max_features)

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
    mlflow.log_param("solver", log_reg.solver)
    mlflow.log_param("penalty", log_reg.penalty)
    mlflow.log_param("C", log_reg.C)
    mlflow.log_metric("cv_accuracy", log_mean_acc)
    mlflow.log_metric("test_accuracy", log_acc)
    mlflow.log_metric("precision", log_precision)
    mlflow.log_metric("recall", log_recall)
    mlflow.log_metric("f1_score", log_f1)
    mlflow.sklearn.log_model(log_reg, "Logistic_Regression_Model")
    
    # Only log the dataset in one of the runs
    mlflow.log_artifact("final_fake_news_v2.csv")

    with open("logistic_regression_model_8.pkl", "wb") as f:
        pickle.dump(log_reg, f)
    mlflow.log_artifact("logistic_regression_model_8.pkl")

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
    mlflow.log_param("max_depth", rf_model.max_depth)
    mlflow.log_metric("cv_accuracy", rf_mean_acc)
    mlflow.log_metric("test_accuracy", rf_acc)
    mlflow.log_metric("precision", rf_precision)
    mlflow.log_metric("recall", rf_recall)
    mlflow.log_metric("f1_score", rf_f1)
    mlflow.sklearn.log_model(rf_model, "Random_Forest_Model")

    with open("random_forest_model_8.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    mlflow.log_artifact("random_forest_model_8.pkl")

print(f"Random Forest Test Accuracy: {rf_acc}")

#========== XGBoost ==========
xgb_model = XGBClassifier(n_estimators=300,use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_scores = cross_val_score(xgb_model, X_selected_trainval, y_trainval, cv=kf, scoring='accuracy')
xgb_mean_acc = np.mean(xgb_scores)

xgb_model.fit(X_selected_trainval, y_trainval)
y_pred_xgb = xgb_model.predict(X_selected_test)

xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

with mlflow.start_run(run_name="XGBoost"):
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("use_label_encoder", False)
    mlflow.log_param("eval_metric", 'logloss')
    mlflow.log_metric("cv_accuracy", xgb_mean_acc)
    mlflow.log_metric("test_accuracy", xgb_acc)
    mlflow.log_metric("precision", xgb_precision)
    mlflow.log_metric("recall", xgb_recall)
    mlflow.log_metric("f1_score", xgb_f1)
    mlflow.sklearn.log_model(xgb_model, "XGBoost_Model")

    with open("xgboost_model_8.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    mlflow.log_artifact("xgboost_model_8.pkl")

print(f"XGBoost Test Accuracy: {xgb_acc}")

# ========== Naive Bayes ========== 
# Replaced SVM with Naive Bayes
nb_model = MultinomialNB(alpha=1.0)  # alpha is the smoothing parameter
nb_scores = cross_val_score(nb_model, X_selected_trainval, y_trainval, cv=kf, scoring='accuracy')
nb_mean_acc = np.mean(nb_scores)
nb_model.fit(X_selected_trainval, y_trainval)

y_pred_nb = nb_model.predict(X_selected_test)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)
nb_acc = accuracy_score(y_test, y_pred_nb)

with mlflow.start_run(run_name="Naive Bayes"):
    mlflow.log_param("model_type", "Naive Bayes")
    mlflow.log_param("alpha", 1.0)
    mlflow.log_metric("cv_accuracy", nb_mean_acc)
    mlflow.log_metric("test_accuracy", nb_acc)
    mlflow.log_metric("precision", nb_precision)
    mlflow.log_metric("recall", nb_recall)
    mlflow.log_metric("f1_score", nb_f1)
    mlflow.sklearn.log_model(nb_model, "Naive_Bayes_Model")

    with open("naive_bayes_model_8.pkl", "wb") as f:
        pickle.dump(nb_model, f)
    mlflow.log_artifact("naive_bayes_model_8.pkl")

print(f"Naive Bayes Test Accuracy: {nb_acc}")

#========== Select Best Model ==========
# Include all models in the comparison
accuracies = {
    "Logistic Regression": log_acc,
    "Random Forest": rf_acc,
    "XGBoost": xgb_acc,
    "Naive Bayes": nb_acc  # Changed from SVM to Naive Bayes
}

best_model_name = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model_name]

model_mapping = {
    "Logistic Regression": log_reg,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Naive Bayes": nb_model  # Changed from SVM to Naive Bayes
}
best_model = model_mapping[best_model_name]

# Save vectorizer and selector
with open("vectorizer_8.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("selector_8.pkl", "wb") as f:
    pickle.dump(selector, f)

# Save best model
with open("best_model_8.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Log final comparison results
with mlflow.start_run(run_name="Model Comparison"):
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_artifact("best_model_8.pkl")
    mlflow.log_artifact("vectorizer_8.pkl") 
    mlflow.log_artifact("selector_8.pkl")

print(f"Best Model: {best_model_name} (Test Accuracy: {best_accuracy})")