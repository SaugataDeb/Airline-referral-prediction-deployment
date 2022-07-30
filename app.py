from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

mle = pickle.load(open('mle_deployment_review_model.pkl','rb'))
tfidf_vect = pickle.load(open('tfidf_airline.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        Customer_review = request.form['Customer_review']
        data = [Customer_review]
        vect = tfidf_vect.transform(data).toarray()
        my_prediction = mle.predict(vect)
    return render_template('index.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)