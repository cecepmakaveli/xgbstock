# server.py
from flask import Flask, request, jsonify
import pandas as pd
from model import LinearModel

app = Flask(__name__)
ml_model = LinearModel()

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        df = pd.read_csv(file)
        df.to_csv("uploaded_data.csv", index=False)
        return jsonify({"message": "File uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    try:
        X, y = ml_model.load_data()
        test_metrics = ml_model.train(X, y)

        return jsonify({
            "message": "Model trained and saved",
            "test_metrics": test_metrics
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Support both single and multiple records
        if isinstance(data, dict):  # Single record
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):  # Multiple records
            input_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format"}), 400

        prediction = ml_model.predict(input_df)
        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)
