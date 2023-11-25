import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

## Loading the model 
regmodel = pickle.load(open('regModel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    newData = np.array(list(data.values())).reshape(1, -1)
    print(newData)
    newData  = scaler.transform(newData)
    output = regmodel.predict(newData)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)
