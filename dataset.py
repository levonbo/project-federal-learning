import medmnist
import torchvision.transforms as transforms
import torch.utils.data as torchdata
from medmnist import INFO
from torch.utils.data import random_split, DataLoader
import config

def get_client_loader(data_flag, size, num_clients, batch_size, data_augmentation=False):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    data_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(28, padding=12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    print("Loading MedMNIST Data ...")
    ###* Dataset 
    if data_augmentation:
        train_dataset_original = DataClass(split='train', transform=data_transform, download=True, size=28, mmap_mode='r') #224
        train_dataset_augmented = DataClass(split='train', transform=data_augmented, download=True, size=28, mmap_mode='r') #224
        train_dataset = torchdata.ConcatDataset([train_dataset_original, train_dataset_augmented])
    else:
        train_dataset = DataClass(split='train', transform=data_transform, download=True, size=28, mmap_mode='r') #224
    ###* Dataset 
    train_dataset = DataClass(split='train', transform=data_transform, download=True, size=28, mmap_mode='r') #224

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

def get_val_loader(data_flag, size, num_clients, batch_size):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    val_dataset = DataClass(split='val', transform=data_transform, download=True, size=size)

    partition_size = len(val_dataset) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] += len(val_dataset) - sum(lengths)

    client_val_datasets = random_split(val_dataset, lengths)
    client_val_loaders = []

    for ds in client_val_datasets:
        loader = DataLoader(ds,batch_size=batch_size,shuffle=False)
        client_val_loaders.append(loader)
    return client_val_loaders

def get_test_loader(data_flag, size, num_clients, batch_size):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    test_dataset = DataClass(split='test', transform=data_transform, download=True, size=size)

    partition_size = len(test_dataset) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] += len(test_dataset) - sum(lengths)

    client_test_datasets = random_split(test_dataset, lengths)
    client_test_loaders = []

    for ds in client_test_datasets:
        loader = DataLoader(ds,batch_size=batch_size,shuffle=False)
        client_test_loaders.append(loader)
    return client_test_loaders