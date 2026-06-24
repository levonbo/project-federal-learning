import torch.nn as nn
import torch.optim as optim
import json
from types import SimpleNamespace
from medmnist import INFO
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime
import csv
from pathlib import Path

#* Load json data
def _load() -> SimpleNamespace:
    with open("param.json", "r") as file:
        data = json.load(file)
    return json.loads(
        json.dumps(data),
        object_hook=lambda d: SimpleNamespace(**d)
    )

param= _load()

def get_info(data_flag): 
    #* Load info of medmnist dataset
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    n_train_samples = info['n_samples']['train']
    return info, task, n_channels, n_classes,n_train_samples

#* Total amount of parameters in used 
def get_n_total_params(model):
    return sum(p.numel() for p in model.parameters())

#* optimizer
def get_optimizer(optimizer,model,lr):
    if optimizer.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    elif optimizer.lower() == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, momentum=0.9)
    elif optimizer.lower() == "adadelta":
        return optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6)
    elif optimizer.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


#*criterion
def get_criterion(data_flag):
    if INFO[data_flag]['task'] == "multi-label, binary-class":
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()

