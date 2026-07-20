import optuna
import torch.optim as optim
import torch
import torch.nn as nn
import models
#import config
import dataset
import numpy as np
from medmnist import INFO,Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def objective(trial, data_flag):
    optimizer_name = trial.suggest_categorical("Optimizer", ["Adam"])
    learning_rate = trial.suggest_float('lr', 0.0001, 0.01, log=True)
    #momentum = trial.suggest_float('momentum', 0.5,0.99, log=False)
    #alpha = trial.suggest_float('alpha', 0.90,0.99, log=False)
    BATCH_SIZE = trial.suggest_categorical('Batch Size', [16, 32, 64,128])

    acc = 0
    #* Load info of medmnist dataset
    info = INFO[data_flag]
    task = info['task']
    model = models.get_model("basiccnn", data_flag).to(device)

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    train_loader, validation_loader, _ = dataset.get_loader(data_flag, "basiccnn", BATCH_SIZE, True, 28)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    for _ in range(5):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.view(-1).long()
                loss = criterion(outputs, targets)
                outputs = outputs.softmax(dim=-1)

            loss.backward()
            optimizer.step()
        val_loss = 0
        val_samples = 0.0
        model.eval()
        y_score = torch.empty(0,device=device)
        y_true = torch.empty(0,device=device)
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs,targets = inputs.to(device),targets.to(device)
                outputs = model(inputs)
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    val_loss += criterion(outputs, targets).item() * inputs.size(0)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.view(-1).long()
                    val_loss += criterion(outputs, targets).item() * inputs.size(0)
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().reshape(len(targets), 1)

                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
                val_samples += inputs.size(0)
            y_true = y_true.detach().cpu().numpy()
            y_score = y_score.detach().cpu().numpy()

            evaluator = Evaluator(data_flag, "val")
            metrics = evaluator.evaluate(y_score)
            auc, acc = metrics
            val_loss /= val_samples
    return acc

mnist_datasets = ["organamnist"]
best_params = {}
for data_flag in mnist_datasets: 
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, data_flag), n_trials=40) # type: ignore
    print("Best Hyperparameters:", study.best_params)
    best_params[data_flag] = study.best_params
print(best_params)