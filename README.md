# 📰 Fake News Detection

A machine learning project aimed at detecting fake news articles using natural language processing (NLP) techniques.

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 📖 Overview

This project classifies news articles as either **real** or **fake** based on their content. The system uses NLP preprocessing and machine learning models to identify deceptive news articles and help reduce misinformation.

Link to Deploy each service
- Flask Service: https://app.paperlesstransform.online/
- MLflow: https://mlflow.paperlesstransform.online/
- Minio: https://minio.paperlesstransform.online/

## ✨ Features

- Text preprocessing (tokenization, stopword removal, lemmatization)
- Feature extraction using TF-IDF or word embeddings
- Multiple machine learning models (e.g., Logistic Regression, Naive Bayes, XGBoot)
- Model evaluation metrics: Accuracy, Precision, Recall, F1-Score
- Optional MLflow integration for experiment tracking

## 📚 Dataset

- **Name**: [final_fake_news_v2.csv]
- **Source**: [This is an external link to dataset](http://minio.paperlesstransform.online/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL21sb3BzL2ZpbmFsX2Zha2VfbmV3c192Mi5jc3Y_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD0yUTBKS1o4Vk1VNTZJSjczVDZZNSUyRjIwMjUwNTI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyOVQwODAxNTZaJlgtQW16LUV4cGlyZXM9NDMxOTkmWC1BbXotU2VjdXJpdHktVG9rZW49ZXlKaGJHY2lPaUpJVXpVeE1pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmhZMk5sYzNOTFpYa2lPaUl5VVRCS1MxbzRWazFWTlRaSlNqY3pWRFpaTlNJc0ltVjRjQ0k2TVRjME9EVTBPRGt3T0N3aWNHRnlaVzUwSWpvaVVrOVBWRTVCVFVVaWZRLmY4cGhmQ19iWUNiN0hQM3VCY3I2R09KUDk5VUpmTmdDUFloaDliTV9uLW5wNFI0eGFmelFGTlU1bndybmN5b2pkSkQ4R0hyLTM5S2VSc0tWNUNnX3dnJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZ2ZXJzaW9uSWQ9bnVsbCZYLUFtei1TaWduYXR1cmU9NGJhMGQxNGEzMTIwYjVhYWU2ZTI3Njg5ZGZhZDdmNmJkNDJhMmVmMmRmNDMwZWNkMGJjYjkyYWE4ZDk4N2U3ZA)
- **Fields**: `title`, `text`, `label`
  - `label`: `1` = real, `0` = fake

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
    ```
2. Create Virtual environment
   ```bash
   python -m venv venv

   # Windows
    venv\Scripts\activate.bat

   # Mac OS / Linux
    source venv\bin\activate
    ```

4. Install required library
   ```bash
   pip install -r requirements.txt
    ```
4. Run a Flask Service
   ```bash
    python app/app.py
    ```

or using Docker 

1. Run a docker file 
   ```bash
   docker compose up -d 
    ```

## System Architecture

![Fake news detection System Architecture](https://github.com/redbiiddsun/fake-news-detection/blob/image/image/AWS%20(2025)%20horizontal%20framework%20(1).png?raw=true)

## CI/CD 
![CICD](https://github.com/redbiiddsun/fake-news-detection/blob/image/image/Gitlab%20parent-child%20pipeline.png?raw=true
)

