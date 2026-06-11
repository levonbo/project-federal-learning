import torch
import copy
from torch import optim


class Client:
    def __init__(self, client_id, train_loader, global_model):
        self.client_id = client_id
        self.train_loader = train_loader
        self.model = copy.deepcopy(global_model)

    def local_training(self, train_loader, task, criterion):
        optimizer = optim.Adadelta(self.model.parameters(), lr=1.0, rho=0.9, eps=1e-6)

        running_loss = 0.0
        num_images = 0
        self.model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = self.model(inputs)

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

        avg_loss = running_loss / num_images
        print(f"Train: Avg loss: {avg_loss:.6f}")
        return self.model.state_dict()
        

