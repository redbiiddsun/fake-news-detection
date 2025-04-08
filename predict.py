import pickle
import pandas as pd

# โหลดโมเดลจากไฟล์ .pkl
with open("best_model.pkl", "rb") as f:
    log_reg = pickle.load(f)

# โหลด vectorizer และ selector ที่ได้เรียนรู้จากข้อมูลฝึก (X_train)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

# สมมติว่า X_test เป็นข้อความที่ต้องการทำนาย
X_test_raw = ["Heavy rainfall caused flooding in several counties across the Midwest, prompting emergency evacuations and closure of major highways."]  # ข้อความตัวอย่าง

# ใช้ TF-IDF vectorizer ที่ได้เรียนรู้จากข้อมูลฝึก (X_train)
X_test_tfidf = vectorizer.transform(X_test_raw)  # แปลงข้อความให้เป็น TF-IDF vector

# ใช้ SelectKBest ที่ได้เรียนรู้จากข้อมูลฝึก (X_train_selected)
X_test_selected = selector.transform(X_test_tfidf)  # เลือกฟีเจอร์ที่ดีที่สุด

# ตอนนี้ X_test_selected คือข้อมูลที่แปลงและเลือกฟีเจอร์แล้ว ซึ่งคุณสามารถใช้ในการทำนาย
y_pred = log_reg.predict(X_test_selected)

# แสดงผลลัพธ์การทำนาย
print("Predictions:", y_pred)

