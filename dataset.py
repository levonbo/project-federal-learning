import medmnist
from medmnist import INFO, Evaluator
import torchvision.transforms as transforms
import torch.utils.data as data
from config import data_flag, download, BATCH_SIZE, size, model_name

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
train_dataset = DataClass(split='train', transform=data_transform, download=download, size = size)
test_dataset = DataClass(split='test', transform=data_transform, download=download, size = size)

###* Dataloader 
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)