import optuna
import torch.optim as optim
import torch
import torch.nn as nn
import models
import config
import dataset
import numpy as np
from medmnist import Evaluator

def objective(trial):
    optimizer_name = trial.suggest_categorical("Optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    BATCH_SIZE = trial.suggest_categorical('Batch Size', [16, 32, 64, 128])

    mnist_datasets = ["pathmnist", "chestmnist","dermamnist", "octmnist", "pneumoniamnist", "retinamnist", "breastmnist", "bloodmnist", "tissuemnist","organamnist", "organcmnist","organsmnist"]
    dataset_scores = []
    for data_flag in mnist_datasets:
        _, task, _, _,_ = config.get_info(data_flag)
        model = models.get_model(config.param.model_name, data_flag)

        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

        train_loader, validation_loader, _ = dataset.get_loader(data_flag, config.param.model_name, BATCH_SIZE, config.param.download, config.param.size)    

        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        for _ in range(2):
            model.train()
            for inputs, targets in train_loader:
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
            model.eval()
            y_score = torch.tensor([])
            with torch.no_grad():
                for inputs, targets in validation_loader:
                    outputs = model(inputs)
                    if task == 'multi-label, binary-class':
                        outputs = outputs.softmax(dim=-1)
                    else:
                        outputs = outputs.softmax(dim=-1)

                    y_score = torch.cat((y_score, outputs), 0)

            evaluator = Evaluator(data_flag, "val")
            metrics = evaluator.evaluate(y_score)
            auc, acc = metrics
            score = 0.7 * auc + 0.3 * acc
        dataset_scores.append(score)
    return np.mean(dataset_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2) # type: ignore
print("Best Hyperparameters:", study.best_params)