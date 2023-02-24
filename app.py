from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
import joblib
# Load the trained model and vectorizer
model = joblib.load(open("clf.pkl", "rb"))
vectorizer = joblib.load(open("cv.pkl", "rb"))
app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def predict():
    # Get the text input from the API request
    text = "Bitch Fucker MOtherfucker racist"


    # Convert the text input to a vector using the trained vectorizer
    vector = vectorizer.transform([text]).toarray()

    # Use the trained model to make a prediction
    prediction = model.predict(vector)
    print(prediction[0])
    # Return the prediction as a JSON response
    # response = {'prediction': prediction}
    return ("Answer is " + str(prediction[0]))
if __name__ == '__main__':
    app.run(port=5000, debug=True)