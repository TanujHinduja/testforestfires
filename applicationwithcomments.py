import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pf
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)   # Creates a Flask application. __name__ tells Flask where your code is, so it can find files like HTML templates
app = application       # giving other name to the application

## import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))   # load the trained ridge regression model for predictions
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))   # load the standard scaler for scaling the input data


@app.route("/")     #This is a decorator that tells Flask to run the index() function when someone visits the main page ("/") of your website
def index():
    return render_template('index.html')    # Loads and returns the HTML file index.html from your templates folder to show to the user

@app.route('/predictdata', methods=['GET', 'POST']) # Creates a new page at /predictdata that accepts both GET and POST requests. GET is used to show the form and POST is used when the user submits the form
def predict_datapoint():
    if request.method=="POST":  #Checks if the user submitted the form.
        Temperature = float(request.form.get('Temperature'))    # Gets the value the user entered in the form for each field (like Temperature, RH, etc.)
        RH = float(request.form.get('RH'))  
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]) #Scales the user’s data using the same scaling as the training data
        result = ridge_model.predict(new_data_scaled)   # Uses your trained model to make a prediction based on the scaled data.

        return render_template('home.html', results= result[0])     #Shows the result to the user on the home.html page.

    else:
        return render_template('home.html')

if __name__=="__main__":    # This runs the Flask application when you run this script directly
    app.run(host="0.0.0.0")