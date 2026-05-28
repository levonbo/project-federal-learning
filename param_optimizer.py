import optuna
import torch.optim as optim
import torch
import torch.nn as nn
import models
import config
import dataset


def objective(trial) -> float:
    data_flag = "retinamnist"
    _, task, _, _,_ = config.get_info(data_flag)
    model = models.get_model(config.param.model_name, data_flag)

    optimizer_name = trial.suggest_categorical("Optimizer", ["Adam", "RMSprop", "SGD", "RMSprop","Adadelta"])
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    BATCH_SIZE = trial.suggest_categorical('Batch Size', [16, 32, 64, 128])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    train_loader, _, _ = dataset.get_loader(data_flag, config.param.model_name, BATCH_SIZE, config.param.download, config.param.size)    

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    Aveg_loss = 0

    model.train()
    for epoch in range(3):
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
study.optimize(objective, n_trials=2)
print("Best Hyperparameters:", study.best_params)