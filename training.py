import torch 
from tqdm import tqdm

###*    TRAINING 

def train_model(model, train_loader, optimizer, task, criterion):
    running_loss = 0.0
    num_images = 0
    Aveg_loss = 0
    model.train()
    for inputs, targets in tqdm(train_loader):
        # forward + backward + optimize
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
        Aveg_loss = running_loss / num_images
    print(f"Train: Avg loss: {Aveg_loss:.6f}")
    return Aveg_loss