import optuna
import torch.optim as optim
import torch
import torch.nn as nn
from models import BasicCNN, SqueezeNet, SmallCNN
from dataset import train_loader
from config import config, n_channels, n_classes, task


def objective(trial) -> float:
    #* Load Model
    if config.model_name.lower()=="basiccnn":
        model = BasicCNN(in_channels=n_channels, num_classes=n_classes)
    elif config.model_name.lower()=="squeezenet":
        model = SqueezeNet()
    elif config.model_name.lower()=="smallcnn": 
        model = SmallCNN(in_channels=n_channels, num_classes=n_classes)
    else:
        raise Exception("Sorry, this model is not known") 

    

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    #batch_size = trial.suggest_int('bs', 4, 16, step=4)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    Aveg_loss = 0

    model.train()
    for epoch in range(30):
        running_loss = 0.0
        num_images = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.view(-1).long()
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            num_images += inputs.size(0)
            Aveg_loss = round(running_loss / num_images,2)
        
    return round(Aveg_loss,2)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
print("Best Hyperparameters:", study.best_params)