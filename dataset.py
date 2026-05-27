import medmnist
import torchvision.transforms as transforms
import torch.utils.data as torchdata
from config import config,info

DataClass = getattr(medmnist, info['python_class'])
print(info['python_class'], "using a", config.model_name)
###*    load the data from medMNIST and encapsulate into dataloader form 

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
print("Loading MedMNIST Data ...")
###* Dataset 
train_dataset = DataClass(split='train', transform=data_transform, download=config.download, size=config.size, mmap_mode='r') #224
test_dataset = DataClass(split='test', transform=data_transform, download=config.download, size=config.size)

###* Dataloader 
train_loader = torchdata.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
train_loader_at_eval = torchdata.DataLoader(dataset=train_dataset, batch_size=2*config.BATCH_SIZE, shuffle=False)
test_loader = torchdata.DataLoader(dataset=test_dataset, batch_size=2*config.BATCH_SIZE, shuffle=False)