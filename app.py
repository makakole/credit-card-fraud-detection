import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import json

app = Flask(__name__)
model = pickle.load(open('fraud_detection_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # x = request.form.values("transaction")
    # print(x)
    features = json.loads(request.form["transaction"])
    print(features)
    df = pd.DataFrame(features["data"]).T
    print(df)
    predictions = model.predict(df)

    return render_template(
        "index.html",
        prediction_value=predictions[0]
        )


if __name__ == "__main__":
    app.run(debug=True)