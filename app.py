from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Import your model function from ml_algo.py (adjust the import path if needed)
import sys
import os
# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_algo import predict_quality  # Import the prediction function

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

@app.route("/")
def home():
    return "Groundwater Quality Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expecting JSON in the format: { "features": [value1, value2, ...] }
        data = request.json
        features = data.get("features", None)
        if features is None:
            return jsonify({"error": "No features provided"}), 400
        
        # Ensure features is a list of numbers
        features = list(map(float, features))
        result = predict_quality(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)


