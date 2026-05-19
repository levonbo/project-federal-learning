import medmnist
from medmnist import INFO, Evaluator
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import json 

#* Config from JSON 
with open("config.json", "r") as file:
    data = json.load(file)

data_flag = data["data_flag"]
download = data["download"]
BATCH_SIZE = data["BATCH_SIZE"]
size = data["size"]
model_name = data["model_name"]


info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
print(info['python_class'], "using a", model_name)
###*    load the data from medMNIST and encapsulate into dataloader form 

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
print("Loading MedMNIST Data ...")
###* Dataset 
#train_dataset = DataClass(split='train', transform=data_transform, download=download, size=size) #28
train_dataset = DataClass(split='train', transform=data_transform, download=download, size=size, mmap_mode='r') #224
test_dataset = DataClass(split='test', transform=data_transform, download=download, size=size)

###* Dataloader 
train_loader = torchdata.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = torchdata.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = torchdata.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)