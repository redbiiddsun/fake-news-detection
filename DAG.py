from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import mlflow

# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 11),  
    'retries': 1,
}

with DAG(
    'fake_news_data_cleaning',
    default_args=default_args,
    description='DAG for cleaning and feature engineering of Fake News dataset',
    schedule_interval='@daily',
    catchup=False,
) as dag:
    
    def clean_text(text):
        text = re.sub(r'[â€™œ€™]', '', text)  
        text = re.sub(r'[^a-zA-Z0-9.,!?\s]', '', text)  
        return text.strip()
    
    def load_data():
        true_df = pd.read_csv('/home/santitham/airflow/dags/Fake_New_Detection/True.csv')
        fake_df = pd.read_csv('/home/santitham/airflow/dags/Fake_New_Detection/Fake.csv')
        
        true_df['label'] = 1
        fake_df['label'] = 0
    
        df = pd.concat([true_df, fake_df], ignore_index=True)
        df.to_csv('/home/santitham/airflow/dags/Fake_New_Detection/merged_fake_news.csv', index=False)
    
    def clean_data():
        df = pd.read_csv('/home/santitham/airflow/dags/Fake_New_Detection/merged_fake_news.csv')
        df = df.drop_duplicates()
        df = df.fillna('')
        df.to_csv('/home/santitham/airflow/dags/Fake_New_Detection/cleaned_fake_news.csv', index=False)
    
    def preprocess_data():
        df = pd.read_csv('/home/santitham/airflow/dags/Fake_New_Detection/cleaned_fake_news.csv')
        df['title'] = df['title'].astype(str).apply(clean_text)
        df['text'] = df['text'].astype(str).apply(clean_text)
        df['subject'] = df['subject'].astype(str).apply(clean_text)
        
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        
        df.to_csv('/home/santitham/airflow/dags/Fake_New_Detection/final_fake_news.csv', index=False)
    
    def eda_analysis():
        df = pd.read_csv('/home/santitham/airflow/dags/Fake_New_Detection/final_fake_news.csv')
        
        # Check missing values and duplicates
        missing_values = df.isnull().sum()
        duplicates = df.duplicated().sum()
        print(f'Missing Values:\n{missing_values}')
        print(f'Duplicates: {duplicates}')
        
        # Analyze text length distribution
        df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(10, 5))
        sns.histplot(df['text_length'], bins=50, kde=True)
        plt.title('Text Length Distribution')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.savefig('/home/santitham/airflow/dags/Fake_New_Detection/text_length_distribution.png')
        
        # Visualize subject category distribution
        plt.figure(figsize=(10, 5))
        sns.countplot(y=df['subject'], order=df['subject'].value_counts().index)
        plt.title('Subject Category Distribution')
        plt.xlabel('Count')
        plt.ylabel('Subject')
        plt.savefig('/home/santitham/airflow/dags/Fake_New_Detection/subject_distribution.png')
        
        # Generate word cloud for fake and real news
        fake_text = ' '.join(df[df['label'] == 0]['text'])
        real_text = ' '.join(df[df['label'] == 1]['text'])
        
        fake_wordcloud = WordCloud(width=800, height=400).generate(fake_text)
        real_wordcloud = WordCloud(width=800, height=400).generate(real_text)
        
        fake_wordcloud.to_file('/home/santitham/airflow/dags/Fake_New_Detection/fake_wordcloud.png')
        real_wordcloud.to_file('/home/santitham/airflow/dags/Fake_New_Detection/real_wordcloud.png')
        
        # Perform TF-IDF Analysis
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(df['text'].astype(str))
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        tfidf_df.to_csv('/home/santitham/airflow/dags/Fake_New_Detection/tfidf_features.csv', index=False)
        
    def prepare_training_data():
            
        df = pd.read_csv('/home/santitham/airflow/dags/Fake_New_Detection/final_fake_news.csv')
    
        # ใช้เฉพาะข้อความจาก "text"
        X = df['text'].astype(str)
        y = df['label']

        # แบ่งข้อมูล train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ใช้ TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # บันทึกข้อมูล
        with open('/home/santitham/airflow/dags/Fake_New_Detection/train_test_data.pkl', 'wb') as f:
            pickle.dump((X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer), f)

    def train_logistic_regression():
    # โหลด Train/Test Data
        with open('/home/santitham/airflow/dags/Fake_New_Detection/train_test_data.pkl', 'rb') as f:
            X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = pickle.load(f)

        # เทรน Logistic Regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)

        # ทดสอบโมเดล
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)

        # ใช้ MLflow ติดตามผลลัพธ์
        with mlflow.start_run():
            mlflow.log_param("model", "Logistic Regression")
            mlflow.log_param("max_iter", 1000)
            mlflow.log_metric("accuracy", acc)

        # บันทึกโมเดล
        with open('/home/santitham/airflow/dags/Fake_New_Detection/logistic_model.pkl', 'wb') as f:
            pickle.dump((model, vectorizer), f)

    def install_missing_dependencies():
        import os
        os.system("pip install wordcloud")

    start_task = DummyOperator(
        task_id='start'
    )

    install_dependencies = PythonOperator(
        task_id='install_dependencies',
        python_callable=install_missing_dependencies,
    )
    
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )
    
    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
    )
    
    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )
    
    eda_task = PythonOperator(
        task_id='eda_analysis',
        python_callable=eda_analysis,
    )
    prepare_training_data_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    )

    train_logistic_regression_task = PythonOperator(
        task_id='train_logistic_regression',
        python_callable=train_logistic_regression,
    )
    
    end_task = DummyOperator(
        task_id='end'
    )
    
    start_task >> install_dependencies >> load_data_task >> clean_data_task >> preprocess_data_task >> eda_task >> prepare_training_data_task >> train_logistic_regression_task >> end_task