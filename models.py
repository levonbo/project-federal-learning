import torch.nn as nn 
import torch.nn.functional as F
import math
import torch
import config
from torchvision.models import resnet18

#* StandardCNN -> 800.000 parameters
class StandardCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(StandardCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.GroupNorm(8,32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size=3),
            nn.GroupNorm(8,32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3),
            nn.GroupNorm(8,64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.GroupNorm(8,128),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#* CNN400 -> 400.000 parameters
class CNN400(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN400, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.GroupNorm(8, 16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.GroupNorm(8, 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.GroupNorm(8, 32),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3),
            nn.GroupNorm(8, 48),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(48, 80, kernel_size=3, padding=1),
            nn.GroupNorm(8, 80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(320 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN200(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN200, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.GroupNorm(8, 16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.GroupNorm(8, 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3),
            nn.GroupNorm(8, 24),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3),
            nn.GroupNorm(8, 32),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(1024, 160),
            nn.ReLU(),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

import torch.nn as nn


class CNN100(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN100, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3),
            nn.GroupNorm(4, 8),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.GroupNorm(4, 16),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.GroupNorm(8, 32),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(768, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

import torch.nn as nn


class CNN50(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN50, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3),
            nn.GroupNorm(2, 4),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3),
            nn.GroupNorm(2, 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3),
            nn.GroupNorm(4, 16),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3),
            nn.GroupNorm(4, 24),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(512, 72),
            nn.ReLU(),
            nn.Linear(72, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

import torch.nn as nn


class CNN10(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN10, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=3),
            nn.GroupNorm(1, 2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3),
            nn.GroupNorm(1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3),
            nn.GroupNorm(2, 4),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(4, 12, kernel_size=3, padding=1),
            nn.GroupNorm(4, 12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(300, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

import torch.nn as nn


class CNN5(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=3),
            nn.GroupNorm(1, 2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3),
            nn.GroupNorm(1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(288, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN4(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3),
            nn.GroupNorm(2, 4),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.GroupNorm(4, 16),
            nn.ReLU())

        self.pool = nn.AvgPool2d(kernel_size=6, stride=6) 

        self.fc = nn.Sequential(
            nn.Linear(16 * 2 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
#* 1k parameters
class CNN1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3),
            nn.GroupNorm(1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.GroupNorm(1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(72, 12),
            nn.ReLU(),
            nn.Linear(12, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN05(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN05, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3),
            nn.GroupNorm(1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4))  # aggressive, param-free downsampling
        self.fc = nn.Sequential(
            nn.Linear(36, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#* BasicCNN -> 230.000 parameters
class BasicCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BasicCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.GroupNorm(4,16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.GroupNorm(4,16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.GroupNorm(8,64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.GroupNorm(8,64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8,64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#* Small CNN 
class SmallCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SmallCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
#* Mini CNN 
class MiniCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MiniCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(model_name, data_flag):
    _,_, n_channels, n_classes,_ = config.get_info(data_flag)
    if model_name.lower()=="basiccnn":
        return BasicCNN(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="minicnn":
        return MiniCNN(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="smallcnn": 
        return SmallCNN(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="standardcnn":
        return StandardCNN(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="cnn400":
        return CNN400(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="cnn200":
        return CNN200(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="cnn100":
        return CNN100(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="cnn50":
        return CNN50(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="cnn10":
        return CNN10(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="cnn5":
        return CNN5(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="cnn1":
        return CNN1(in_channels=n_channels, num_classes=n_classes)
    elif model_name.lower()=="cnn05":
        return CNN05(in_channels=n_channels, num_classes=n_classes)
    else:
        raise Exception("Sorry, this model is not known")