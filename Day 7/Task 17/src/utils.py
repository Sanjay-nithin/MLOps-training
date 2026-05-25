import pickle as pkl
import numpy as np
import json
import pandas as pd

__model = None 

def load_utils():
    global __model 
    with open("/home/sanjay/Documents/MLOps training/Day 7/Task 17/model/burnout.pkl", 'rb')  as file:
        __model = pkl.load(file)


def predict_burnout(data: dict):
    print(data)
    df = pd.DataFrame([data])
    res = __model.predict(df)
    return res