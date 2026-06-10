import medmnist
import torchvision.transforms as transforms
import torch.utils.data as torchdata
from medmnist import INFO
from torch.utils.data import random_split, DataLoader
import config

def get_client_loader(data_flag, size, num_clients, batch_size):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    ###* Dataset 
    train_dataset = DataClass(split='train', transform=data_transform, download=True, size=size, mmap_mode='r') #224

    # Split dataset
    partition_size = len(train_dataset) // num_clients
    

    lengths = [partition_size] * num_clients
    lengths[-1] += len(train_dataset) - sum(lengths)

    client_datasets = random_split(train_dataset, lengths)

    # Create one DataLoader per client
    client_loaders = []

    for ds in client_datasets:
        loader = DataLoader(ds,batch_size=batch_size,shuffle=True)
        client_loaders.append(loader)
    return client_loaders

def get_val_loader(data_flag, size, batch_size):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    val_dataset = DataClass(split='val', transform=data_transform, download=True, size=size)
    val_loader = torchdata.DataLoader(dataset=val_dataset, batch_size=2*batch_size, shuffle=False)
    return val_loader