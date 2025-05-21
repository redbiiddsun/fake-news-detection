import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ✅ โหลด vectorizer, selector, และ XGBoost model
with open("vectorizer_13.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("selector_13.pkl", "rb") as f:
    selector = pickle.load(f)

with open("best_model_13.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ ตรวจสอบว่า model มี feature_importances_
if not hasattr(model, "feature_importances_"):
    raise ValueError("Model นี้ไม่มี `feature_importances_`. โปรดใช้ XGBoost หรือ Random Forest เท่านั้น.")

# ✅ โหลดข้อมูล
df = pd.read_csv("final_fake_news_v2.csv")
X = df['text'].astype(str)

# ✅ แปลงข้อความเป็น TF-IDF และเลือก feature
X_tfidf = vectorizer.transform(X)
X_selected = selector.transform(X_tfidf)

# ✅ ดึงชื่อฟีเจอร์ที่ถูกเลือกไว้
feature_names = np.array(vectorizer.get_feature_names_out())
selected_indices = selector.get_support(indices=True)
selected_feature_names = feature_names[selected_indices]

# ✅ ดึงค่าความสำคัญของแต่ละฟีเจอร์
importances = model.feature_importances_

# ✅ หาคำสำคัญ (ทั้ง top และ bottom) โดยไม่แยก fake/true เพราะ XGBoost เป็น tree-based
top_n = 15
top_idx = np.argsort(importances)[-top_n:]
bottom_idx = np.argsort(importances)[:top_n]

top_words = selected_feature_names[top_idx]
top_scores = importances[top_idx]

bottom_words = selected_feature_names[bottom_idx]
bottom_scores = importances[bottom_idx]

# ✅ ฟังก์ชันวาดกราฟ
def plot_feature_importance(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = sorted(zip(top_words, top_scores), key=lambda x: x[1])
    bottom_pairs = sorted(zip(bottom_words, bottom_scores), key=lambda x: x[1], reverse=True)

    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]

    fig = plt.figure(figsize=(12, 8))

    plt.subplot(121)
    plt.barh(y_pos, bottom_scores, align='center', alpha=0.7)
    plt.title('Least Important Words', fontsize=16)
    plt.yticks(y_pos, bottom_words, fontsize=12)
    plt.xlabel('Importance', fontsize=14)

    plt.subplot(122)
    plt.barh(y_pos, top_scores, align='center', alpha=0.7)
    plt.title('Most Important Words', fontsize=16)
    plt.yticks(y_pos, top_words, fontsize=12)
    plt.xlabel('Importance', fontsize=14)

    plt.suptitle(name, fontsize=18)
    plt.subplots_adjust(wspace=0.6)
    plt.show()

# ✅ เรียกใช้
plot_feature_importance(top_scores, top_words, bottom_scores, bottom_words, "Top Important Words (XGBoost)")
