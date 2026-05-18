from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from models import BasicCNN, SqueezeNet, SmallCNN
from config import data_flag, NUM_EPOCHS,lr, model_name
from dataset import train_loader, test_loader, train_loader_at_eval, info
from medmnist import Evaluator

task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])


if model_name=="BasicCNN":
    model = BasicCNN(in_channels=n_channels, num_classes=n_classes)
if model_name=="Squeeze":
    model = SqueezeNet()
if model_name=="SmallCNN":
    model = SmallCNN(in_channels=n_channels, num_classes=n_classes)
else:
    raise Exception("Sorry, this model is not known") 
    
# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

###*    TRAINING 
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    print("Epoch",epoch+1)
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    
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

# for p in model.parameters(): #returns models weights
#    print(p)

amount_total_params = sum(p.numel() for p in model.parameters())

###*    Evaluation

def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().view(-1, 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
    
        print('%s  AUC: %.3f  accuracy:%.3f' % (split, *metrics))
        #print(metrics)        
print('==> Evaluating ...')
test('train')
test('test')
print("Using", amount_total_params, "parameters")
print("\n")


