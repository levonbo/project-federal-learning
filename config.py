import torch.nn as nn
import torch.optim as optim
import json
from types import SimpleNamespace
from medmnist import INFO
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime
import csv
from pathlib import Path

#* Parameters
#num_clients = 3 
#rounds = 3 
model_name = "basiccnn"
data_flag ="organamnist"
num_epoch =  3
batch_size = 16
lr = 2.6e-3
size = 28

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