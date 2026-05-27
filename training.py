import torch
from config import config, model, optimizer, task, criterion, n_total_params, writer 
from tqdm import tqdm
from dataset import train_loader

###*    TRAINING 

def training():
    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch",epoch+1)
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
        print(f"Avg loss: {Aveg_loss:.6f}")
        writer.add_scalar("Loss/train", Aveg_loss, epoch)
