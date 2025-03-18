from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime
import pandas as pd
import re

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
    
    
    start_task = DummyOperator(
        task_id='start'
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

    end_task = DummyOperator(
        task_id='end'
    )

    start_task >> load_data_task >> clean_data_task >> preprocess_data_task >> end_task