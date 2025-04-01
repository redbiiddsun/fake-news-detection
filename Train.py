import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# ✅ 1. โหลดข้อมูลที่ clean แล้ว
df = pd.read_csv('final_fake_news.csv')  # โหลดข้อมูลที่ clean แล้วจากขั้นตอนก่อนหน้านี้

# ✅ 2. เตรียมข้อมูล
X = df['text'].astype(str)
y = df['label']

# Train / Validation / Test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 64/16/20 split

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Fake News Detection")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Feature Selection using SelectKBest
selector = SelectKBest(score_func=chi2, k=1000)
X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
X_val_selected = selector.transform(X_val_tfidf)
X_test_selected = selector.transform(X_test_tfidf)

# ✅ 3. Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_selected, y_train)

# Predict and evaluate
y_pred_log = log_reg.predict(X_val_selected)
accuracy_log = accuracy_score(y_val, y_pred_log)

# ✅ 4. ใช้ MLflow สำหรับการบันทึกผลลัพธ์
with mlflow.start_run():
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_metric("accuracy", accuracy_log)
    mlflow.sklearn.log_model(log_reg, "Logistic_Regression_Model")

print(f"Logistic Regression Accuracy: {accuracy_log}")

# ✅ 5. Save model
with open("logistic_regression_model2.pkl", "wb") as f:
    pickle.dump(log_reg, f)

# ✅ 6. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_selected, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_val_selected)
accuracy_rf = accuracy_score(y_val, y_pred_rf)

# Log Random Forest to MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_metric("accuracy", accuracy_rf)
    mlflow.sklearn.log_model(rf_model, "Random_Forest_Model")

print(f"Random Forest Accuracy: {accuracy_rf}")

# ✅ 7. Save Random Forest model
with open("random_forest_model2.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# ✅ 8. Compare Models and Save Best Model
best_model = log_reg if accuracy_log > accuracy_rf else rf_model
best_accuracy = max(accuracy_log, accuracy_rf)

with open("best_model2.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"Best Model: {'Logistic Regression' if accuracy_log > accuracy_rf else 'Random Forest'} (Accuracy: {best_accuracy})")

