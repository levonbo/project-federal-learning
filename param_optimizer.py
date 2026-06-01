import optuna
import torch.optim as optim
import torch
import torch.nn as nn
import models
import config
import dataset


datasets = ["pneumoniamnist","breastmnist","tissuemnist","octmnist","organamnist","organcmnist","organamnist"]

def make_objective(data_flag):
    def objective(trial) -> float:
        optimizer_name = trial.suggest_categorical("Optimizer", ["Adam", "RMSprop", "SGD", "Adadelta"])
        learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('Batch Size', [16, 32, 64, 128])

        _, task, _, _, _ = config.get_info(data_flag)
        model = models.get_model(config.param.model_name, data_flag)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
        train_loader, _, _ = dataset.get_loader(
            data_flag, config.param.model_name, batch_size,
            config.param.download, config.param.size
        )

        criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()

        model.train()
        total_loss, total_epochs = 0.0, 0
        for epoch in range(30):
            running_loss, num_images = 0.0, 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                else:
                    targets = targets.view(-1).long()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                num_images += inputs.size(0)
            total_loss += running_loss / num_images
            total_epochs += 1

        return round(total_loss / total_epochs, 2)
    return objective


# Run one study per dataset
for data_flag in datasets:
    study = optuna.create_study(direction='minimize')
    study.optimize(make_objective(data_flag), n_trials=20)
    print(f"Best Hyperparameters for {data_flag}: {study.best_params}")