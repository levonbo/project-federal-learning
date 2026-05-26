from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from models import BasicCNN, SqueezeNet, SmallCNN
from dataset import train_loader, test_loader, train_loader_at_eval, info
from medmnist import Evaluator
import json
import csv 
import os 
from datetime import date, datetime
from torch.utils.tensorboard import SummaryWriter

#* CSV 
file_name = "results.csv"


#* Config from JSON 
with open("config.json", "r") as file:
    config = json.load(file)




data_flag = config["data_flag"]
NUM_EPOCHS = config["NUM_EPOCHS"]
lr = config["lr"]
model_name = (config["model_name"]).lower()
BATCH_SIZE = config["BATCH_SIZE"]
size = config["size"]

task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

#*Tensorboard 
run_name = f"{data_flag}__{model_name}__{date.today()}__{datetime.now().strftime("%H")}–{datetime.now().strftime("%M")}–{datetime.now().strftime("%S")}"
writer = SummaryWriter(f"runs/{run_name}")

if model_name=="basiccnn":
    model = BasicCNN(in_channels=n_channels, num_classes=n_classes)
elif model_name=="squeezenet":
    model = SqueezeNet()
elif model_name=="smallcnn": 
    model = SmallCNN(in_channels=n_channels, num_classes=n_classes)
else:
    raise Exception("Sorry, this model is not known") 

amount_total_params = sum(p.numel() for p in model.parameters())

# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

###*    TRAINING 
def training():
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
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
        writer.add_scalar("Heat Map", amount_total_params, len(train_loader))
    # for p in model.parameters(): #returns models weights
    #    print(p)

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
        auc, accuracy = metrics
        print('%s  AUC: %.3f  accuracy:%.3f' % (split, *metrics))
        writer.add_scalar("Accuracy/ Parameters", accuracy, amount_total_params )
        """if split=="test":
            auc, accuracy = metrics
            new_data = [data_flag, NUM_EPOCHS, BATCH_SIZE, model_name, size, amount_total_params, "%.3f" % round(auc, 2), "%.3f" % round(accuracy, 2), date.today()],
            file_exists = os.path.isfile(file_name)

            with open(file_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                if not file_exists:
                    writer.writerow(['Name of database', 'Epochs', 'Batch Size', 'Model name', 'size', 'parameters','AUC','Accuracy', 'Date'])

                writer.writerows(new_data)
            print("Results have been saved!... ")"""




def main():
    training()
    print('==> Evaluating ...')
    test('train')
    test('test')
    #print("Using", amount_total_params, "parameters")

if __name__ == "__main__":
    main()