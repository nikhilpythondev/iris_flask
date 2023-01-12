import sklearn
from flask import Flask,render_template,redirect
from flask import request
import numpy as np
import pickle
model = pickle.load(open('iris_lgr.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    sepal_lenght = float(request.form.get("sepal_lenght"))
    sepal_width = float(request.form.get("sepal_width"))
    petal_lenght = float(request.form.get("petal_lenght"))
    petal_width = float(request.form.get("petal_width"))
    pred_ = model.predict([[sepal_lenght,sepal_width,petal_lenght,petal_width]])[0]
    if pred_ == 0:
        pred_ = 'setosa'
    elif pred_ == 1:
        pred_ = 'versicolor'
    else:
        pred_ = 'virginica'

    return render_template('index.html',pred=pred_)

@app.route('/about')
def about():
    return "Hello, About!"

@app.route('/contactus')
def contact_us():
    return render_template('contact_us.html')

if __name__ == ('__main__'):
    app.run(debug=True)
