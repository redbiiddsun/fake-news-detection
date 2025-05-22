from pprint import pprint
from flask import Flask, jsonify, request
import pickle
import mlflow
from mlflow.tracking import MlflowClient


with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/selector.pkl", "rb") as f:
    selector = pickle.load(f)

mlflow.set_tracking_uri("https://mlflow.paperlesstransform.online") 

client = MlflowClient()

def predict_mlflow(text):

    logged_model = 'runs:/8c35e495fe024ad988f8bc9494271d4c/XGBoost_Model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Convert a raw text to a list
    if not isinstance(text, list):
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