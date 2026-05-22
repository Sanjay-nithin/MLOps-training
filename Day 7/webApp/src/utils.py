import pickle as pkl
import numpy as np
import json

__model = None 

def load_utils():
    global __model 
    with open('../model/burnout.pkl', 'rb')  as file:
        __model = pkl.load(file)

