from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained Random Forest model
'''RF_pkl_filename = 'model.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'rb')
model = pickle.load(RF_Model_pkl)'''
RF_Model_pkl = pickle.load(open('model.pkl','rb'))
#RF_Model_pkl.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        n = float(request.form['nitrogen'])
        p = float(request.form['phosphorus'])
        k = float(request.form['potassium'])
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
        prediction = RF_Model_pkl.predict(data)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')