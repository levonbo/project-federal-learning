import torch 
from tqdm import tqdm

###*   Validation 

def validate_model(model, val_loader, task, criterion):
    running_loss = 0.0
    num_images = 0
    Aveg_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.view(-1).long()
                loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            num_images += inputs.size(0)
            Aveg_loss = running_loss / num_images
        print(f"Evaluation: Avg loss: {Aveg_loss:.6f}")
        return Aveg_loss