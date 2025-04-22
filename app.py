from flask import Flask, request, jsonify
import pickle
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import pandas as pd
import os
import mlflow.xgboost


mlflow.set_tracking_uri("http://127.0.0.1:5000") 

client = MlflowClient()

experiment = client.get_experiment("268375058792484634")  # Replace with your experiment ID
    
runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_accuracy DESC"],  # Sort by test_accuracy
        max_results=1  
)

best_run = runs[0]
best_run_id = best_run.info.run_id

# artifacts = client.list_artifacts(best_run_id)

# artifacts_path = artifacts[0].path

# # Local base path
# base_path = os.path.join("mlruns", experiment.experiment_id, best_run_id, "artifacts")

# # # For model logged with MLflow (directory)
# xgboost_model_path = os.path.join(base_path, artifacts_path)

# print(xgboost_model_path)

# model = mlflow.xgboost.load_model(xgboost_model_path)



model_uri = f"runs:/{best_run_id}/model"

model = mlflow.sklearn.load_model(model_uri)


app = Flask(__name__)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok"
    }

@app.route("/predict", methods=["GET"])
def predict():

    input_text = "The Trump administration has said it is freezing more than $2bn (£1.5bn) in federal funds for Harvard University, hours after the elite college rejected a list of demands from the White House. The White House sent a list of demands to Harvard last week which it said were designed to fight antisemitism on campus. They included changes to hiring, admissions and teaching. Since Donald Trump was re-elected, his government has tried to reshape elite universities by threatening to withhold federal funds, mostly spent on research. Harvard became the first major US university to reject the administration's demands on Monday, accusing the White House of trying to control its community."

    input_df = pd.DataFrame({"text": [input_text]})

    prediction = model.predict(input_df)

    return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)