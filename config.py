import torch.nn as nn
import torch.optim as optim
import json
from types import SimpleNamespace
from models import BasicCNN, SqueezeNet, SmallCNN
from medmnist import INFO
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime
import csv
from pathlib import Path

#* Load json data
def _load() -> SimpleNamespace:
    with open("config.json", "r") as file:
        data = json.load(file)
    return json.loads(
        json.dumps(data),
        object_hook=lambda d: SimpleNamespace(**d)
    )

config = _load()


#* Load info of medmnist dataset
info = INFO[config.data_flag]

task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
n_train_samples = info['n_samples']['train']

#* Load Model
if config.model_name.lower()=="basiccnn":
    model = BasicCNN(in_channels=n_channels, num_classes=n_classes)
elif config.model_name.lower()=="squeezenet":
    model = SqueezeNet()
elif config.model_name.lower()=="smallcnn": 
    model = SmallCNN(in_channels=n_channels, num_classes=n_classes)
else:
    raise Exception("Sorry, this model is not known") 

#* Total amount of parameters in used model
n_total_params = sum(p.numel() for p in model.parameters())

#* optimizer
optimizer = getattr(optim, config.optimizer)(model.parameters(), lr=config.lr)

#* Define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

#*Tensorboard 
run_name = f"{config.data_flag}__{config.model_name}__{config.NUM_EPOCHS}__{date.today()}__{datetime.now().strftime("%H")}–{datetime.now().strftime("%M")}–{datetime.now().strftime("%S")}"
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text("Config", f"Optimizer: {config.optimizer} | Dataset: {config.data_flag} | Epochs: {config.NUM_EPOCHS} | Batch Size: {config.BATCH_SIZE} | lr: {config.lr} | Model: {config.model_name} | Image size: {config.size} ")
    
#* Write csv

CSV_PATH = "results.csv"

def save_result(params, train_samples, accuracy):
    file = Path(CSV_PATH)
    write_header = not file.exists()  # Header nur beim ersten Mal
    
    with open(file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["params", "train_samples", "accuracy"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "params": params,
            "train_samples": train_samples,
            "accuracy": accuracy
        })

