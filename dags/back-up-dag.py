import os
os.system("pip install wordcloud")
os.system("pip install mlflow")

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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

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
    schedule_interval='0 */2 * * *',
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
    
        # Logging before changes
        initial_shape = df.shape
        duplicate_count = df.duplicated().sum()
        missing_values = df.isnull().sum().sum()
    
        # Apply cleaning steps
        df = df.drop_duplicates()
        df = df.fillna('')
    
        # Logging after changes
        final_shape = df.shape
    
        log_message = (
            f"DATA CLEANING LOG:\n"
            f"Initial shape: {initial_shape}, Final shape: {final_shape}\n"
            f"Removed duplicates: {duplicate_count}\n"
            f"Filled missing values: {missing_values}\n"
        )
    
        print(log_message)
        
        # Save log
        with open('/home/santitham/airflow/dags/Fake_New_Detection/data_cleaning_log.txt', 'a') as log_file:
            log_file.write(f"{datetime.now()} - {log_message}\n")
    
        # Versioning
        versioned_filename = f"/home/santitham/airflow/dags/Fake_New_Detection/cleaned_fake_news_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        df.to_csv(versioned_filename, index=False)
    
        # Save latest version without timestamp for DAG continuity
        df.to_csv('/home/santitham/airflow/dags/Fake_New_Detection/cleaned_fake_news.csv', index=False)
    
    def preprocess_data():
        df = pd.read_csv('/home/santitham/airflow/dags/Fake_New_Detection/cleaned_fake_news.csv')
    
        # Logging before changes
        initial_shape = df.shape
    
        df['title'] = df['title'].astype(str).apply(clean_text)
        df['text'] = df['text'].astype(str).apply(clean_text)
        df['subject'] = df['subject'].astype(str).apply(clean_text)
    
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
    
        # Logging after changes
        final_shape = df.shape
    
        log_message = (
            f"DATA PREPROCESSING LOG:\n"
            f"Initial shape: {initial_shape}, Final shape: {final_shape}\n"
            f"Removed 'date' column if present.\n"
        )
    
        print(log_message)
    
        # Save log
        with open('/home/santitham/airflow/dags/Fake_New_Detection/data_preprocessing_log.txt', 'a') as log_file:
            log_file.write(f"{datetime.now()} - {log_message}\n")
    
        # Versioning
        versioned_filename = f"/home/santitham/airflow/dags/Fake_New_Detection/final_fake_news_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        df.to_csv(versioned_filename, index=False)
    
        # Save latest version without timestamp for DAG continuity
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
        fake_text = ' '.join(df[df['label'] == 0]['text'].dropna().astype(str))
        real_text = ' '.join(df[df['label'] == 1]['text'].dropna().astype(str))
        
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
    
        X = df['text'].astype(str)
        y = df['label']
    
        # Train / Validation / Test split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 64/16/20 split
    
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        X_test_tfidf = vectorizer.transform(X_test)
    
        selector = SelectKBest(score_func=chi2, k=1000)
        X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
        X_val_selected = selector.transform(X_val_tfidf)
        X_test_selected = selector.transform(X_test_tfidf)
    
        os.makedirs('/home/santitham/airflow/dags/Fake_New_Detection/data/train', exist_ok=True)
        os.makedirs('/home/santitham/airflow/dags/Fake_New_Detection/data/val', exist_ok=True)
        os.makedirs('/home/santitham/airflow/dags/Fake_New_Detection/data/test', exist_ok=True)
    
        with open('/home/santitham/airflow/dags/Fake_New_Detection/data/train/train.pkl', 'wb') as f:
            pickle.dump((X_train_selected, y_train), f)
    
        with open('/home/santitham/airflow/dags/Fake_New_Detection/data/val/val.pkl', 'wb') as f:
            pickle.dump((X_val_selected, y_val), f)
    
        with open('/home/santitham/airflow/dags/Fake_New_Detection/data/test/test.pkl', 'wb') as f:
            pickle.dump((X_test_selected, y_test), f)
    
        # Save vectorizer and selector
        with open('/home/santitham/airflow/dags/Fake_New_Detection/vectorizer_selector.pkl', 'wb') as f:
            pickle.dump((vectorizer, selector), f)
 

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

    
    end_task = DummyOperator(
        task_id='end'
    )
    
    start_task >> install_dependencies >> load_data_task >> clean_data_task >> preprocess_data_task >> eda_task >> prepare_training_data_task  >> end_task