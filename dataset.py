import medmnist
import torchvision.transforms as transforms
import torch.utils.data as torchdata
from medmnist import INFO

def get_loader(data_flag, model_name, BATCH_SIZE, download, size, data_augmentation=True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    print(info['python_class'], "using a", model_name)
    ###*    load the data from medMNIST and encapsulate into dataloader form 

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    data_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    print("Loading MedMNIST Data ...")
    ###* Dataset 
    if data_augmentation:
        train_dataset_original = DataClass(split='train', transform=data_transform, download=download, size=size, mmap_mode='r') #224
        train_dataset_augmented = DataClass(split='train', transform=data_augmented, download=download, size=size, mmap_mode='r') #224
        train_dataset = torchdata.ConcatDataset([train_dataset_original, train_dataset_augmented])
    else:
        train_dataset = DataClass(split='train', transform=data_transform, download=download, size=size, mmap_mode='r') #224


    test_dataset = DataClass(split='test', transform=data_transform, download=download, size=size)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, size=size)

    ###* Dataloader 
    train_loader = torchdata.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
    val_loader = torchdata.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = torchdata.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    return train_loader,val_loader, test_loader