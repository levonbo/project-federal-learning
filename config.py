from medmnist import INFO
import random
import numpy as np 
import torch
import torch.nn as nn
from torch import optim


param = {
    "data_flag": "breastmnist",
    "model_name": "cnn05",
    "optimizer": "adam",
    "num_clients": 3,
    "rounds": 25,
    "lr": 0.0003,
    "num_epoch": 3,
    "batch_size": 32,
    "size": 28,
    "record_tensorboard": True,
    "data_augmentation": False,
    "non_iid": False,
    "patience": 3,
    "delta": 0.001
}

def get_info(data_flag): 
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    n_train_samples = info['n_samples']['train']
    return info, task, n_channels, n_classes,n_train_samples

#* Total amount of parameters in use
def get_n_total_params(model):
    return sum(p.numel() for p in model.parameters())

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_criterion(info):
    if info['task'] == "multi-label, binary-class":
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()

def get_optimizer(optimizer_name,model,lr):
    if optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif optimizer_name.lower() == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, alpha=0.97)
    elif optimizer_name.lower() == "adadelta":
        return optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else: 
        raise ValueError(f"This optimizer is not known")
    
def print_intro(dataset, model, run_id, num_clients, distribution):
    print("=" * 50)
    print(f" Run UUID             : {run_id}")
    print(f" Clients              : {num_clients}")
    print(f" Sample Distribution  : {distribution}")
    print(f" Dataset              : {dataset}")
    print(f" Model                : {model}")
    print("=" * 50)