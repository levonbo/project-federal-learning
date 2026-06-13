import optuna
import torch.optim as optim
import torch
import torch.nn as nn
import models
import config
import dataset
import numpy as np
from medmnist import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def objective(trial):
    optimizer_name = trial.suggest_categorical("Optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    BATCH_SIZE = trial.suggest_categorical('Batch Size', [16, 32, 64, 128])

    mnist_datasets = ["retinamnist", "breastmnist","dermamnist"]
    vall_loss_all = []
    for data_flag in mnist_datasets:
        _, task, _, _,_ = config.get_info(data_flag)
        model = models.get_model(config.param.model_name, data_flag).to(device)

        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

        train_loader, validation_loader, _ = dataset.get_loader(data_flag, config.param.model_name, BATCH_SIZE, config.param.download, config.param.size)

        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        for _ in range(2):
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
            y_score = torch.tensor([])
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

                    y_score = torch.cat((y_score, outputs.cpu()), 0)
                    val_samples += inputs.size(0)
                val_loss /= val_samples
        vall_loss_all.append(val_loss) #type: ignore
    return np.mean(vall_loss_all)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2) # type: ignore
print("Best Hyperparameters:", study.best_params)