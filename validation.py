import torch 
from medmnist import Evaluator

###*   Validation 

def validate_model(model, val_loader, task, criterion,data_flag):
    running_loss = 0.0
    num_images = 0
    Aveg_loss = 0
    y_score = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.view(-1).long()
                loss = criterion(outputs, targets)
                outputs = outputs.softmax(dim=-1)
            
            running_loss += loss.item() * inputs.size(0)
            num_images += inputs.size(0)
            y_score = torch.cat((y_score, outputs), 0)
        y_score = y_score.detach().numpy()
        Aveg_loss = running_loss / num_images
        evaluator = Evaluator(data_flag, "val")
        metrics = evaluator.evaluate(y_score)
        auc, acc = metrics
        print(f"Val -> Avg loss: {Aveg_loss:.3f} - AUC: {auc:.3f} - Accuracy: {acc:.3f} ")
        return Aveg_loss, auc,acc