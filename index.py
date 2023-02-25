from flask import Flask, request,render_template
import pandas as pd
import pickle
import numpy as np

import joblib
# Load the trained model and vectorizer
model = joblib.load(open("clf.pkl", "rb"))
vectorizer = joblib.load(open("cv.pkl", "rb"))
app = Flask(__name__)

@app.route("/")
def home():
    return render_template ("index.html")
@app.route('/predict',methods=['GET','POST'])
def predict():
    # Get the text input from the API request
    #text = "Bitch Fucker MOtherfucker racist"
    text = str(request.form["text"])


    # Convert the text input to a vector using the trained vectorizer
    vector = vectorizer.transform([text]).toarray()

    # Use the trained model to make a prediction
    prediction = model.predict(vector)
    print(prediction[0])
    # Return the prediction as a JSON response
    # response = {'prediction': prediction}
    return render_template("predict.html",msg = "Answer is " + str(prediction[0]))
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)