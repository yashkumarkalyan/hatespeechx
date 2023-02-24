from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
# Load the trained model and vectorizer
model = pickle.load(open("nunu.pkl", "rb"))
vectorizer = pickle.load(open("chunu.pkl", "rb"))
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the API request
    text = "Hkko there killer"

    # Convert the text input to a vector using the trained vectorizer
    vector = vectorizer.transform([text])

    # Use the trained model to make a prediction
    prediction = model.predict(vector)[0]

    # Return the prediction as a JSON response
    response = {'prediction': prediction}
    return jsonify(response)
if __name__ == '__main__':
    app.run(port=5000, debug=True)