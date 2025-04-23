from pprint import pprint
from flask import Flask, jsonify, request
import pickle
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


# โหลด vectorizer และ selector
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

mlflow.set_tracking_uri("http://127.0.0.1:5000") 

client = MlflowClient()

# experiment = client.get_experiment("268375058792484634")  # Replace with your experiment ID

# runs = client.search_runs(
#         experiment_ids=[experiment.experiment_id],
#         order_by=["metrics.test_accuracy DESC"],  # Sort by test_accuracy
#         max_results=1  
# )

# best_run = runs[0]
# best_run_id = best_run.info.run_id
# artifacts_path = client.list_artifacts(best_run_id)


def predict_mlflow(text):

    logged_model = 'runs:/3a4e5743e779467d82e656c738fa56cb/XGBoost_Model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    if  not isinstance(text, list):
        text = [text]

    X_test_tfidf = vectorizer.transform(text)  
    X_test_selected = selector.transform(X_test_tfidf)

    predictions = loaded_model.predict(X_test_selected)

    return predictions

app = Flask(__name__)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok"
    }

@app.route("/predict", methods=["POST"])
def predict():

    try:
        body = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid or missing JSON data"}), 400

    if "input" not in body:
        return jsonify({"error": "'features' key is missing from the request"}), 400 

    result = predict_mlflow(body["input"]).tolist()

    return {"status" : "success", "result": result}



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)