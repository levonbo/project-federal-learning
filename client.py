import torch
import copy
from torch import optim
from medmnist import Evaluator




def train_local(model, dataloader,criterion, task, num_epoch, lr=0.01):
    model = copy.deepcopy(model) #Remember to deep copy
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    running_loss = 0.0
    num_images = 0
    for _ in range(num_epoch):
        for inputs, targets in dataloader:
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
        avg_loss = running_loss / num_images
        print(f"Train: Avg loss: {avg_loss:.6f}")
    return model.state_dict()

def validate_model(model, val_loader, task, criterion,data_flag):
    val_loss = 0.0
    val_samples = 0
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
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

            val_samples+=inputs.size(0)
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().numpy()
        y_true = y_true.numpy()

        evaluator = Evaluator(data_flag, 'val')
        metrics = evaluator.evaluate(y_score)
        auc, acc = metrics
        val_loss /= val_samples
        print(f"Val -> Avg loss: {val_loss:.3f} - AUC: {auc:.3f} - Accuracy: {acc:.3f} ")
        return val_loss, auc,acc