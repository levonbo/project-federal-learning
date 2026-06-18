import config
import models
import torch 
import copy
from medmnist import Evaluator

def fedavg(models, weights):
    global_model = copy.deepcopy(models[0])
    for key in global_model:
        global_model[key] = sum(w * m[key] for m, w in zip(models, weights))
    return global_model