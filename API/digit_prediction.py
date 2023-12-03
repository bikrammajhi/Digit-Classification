from flask import Flask,request
from sklearn import svm
from joblib import dump, load
import numpy as np

app = Flask(__name__)

@app.route("/predict",methods=["POST"])
def PREDICT():
    JSON_OUT = request.get_json()
    IMG1 = JSON_OUT["input1"]
    IMG1 = []
    for i in JSON_OUT["input1"]:    
        IMG1.append(float(i))
    model = load("models/svm_gamma:0.001_C:1.joblib")
    IMG1 = np.array(IMG1).reshape(-1,64)
    return str(model.predict(IMG1)[0])

@app.route("/compare",methods=["POST"])
def COMPARE():
    JSON_OUT = request.get_json()
    IMG1  = []
    img_2 = []
    for i in JSON_OUT["input1"]:    
        IMG1.append(float(i))
    for i in JSON_OUT["input2"]:    
        img_2.append(float(i))
    model = load("models/svm_gamma:0.001_C:1.joblib")
    return  str(model.predict(np.array(IMG1).reshape(-1,64)) == model.predict(np.array(img_2).reshape(-1,64)))
