import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
from CloudL9_dataset import CloudL9
import random
import logging
import time


## Data Handlers
def get_handler(name):
    if name == 'MNIST':
        return DataHandler3
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name ==  "CloudL9":
        return CloudL9_DataHandler
    elif name == "CloudL9_128":
        return CloudL9_DataHandler
    elif name == "CloudS2_128":
        return CloudS2_DataHandler
    elif name == "CloudL8_128":
        return CloudL8_DataHandler
    elif name ==  "CloudS2":
        return CloudS2_DataHandler
    elif name == "CloudSEN12_128":
        return CloudSEN12_DataHandler
    elif name == "LandCoverAI_128":
        return LandCoverAI_DataHandler
    else:
        return DataHandler4

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler5(Dataset):
    def __init__(self, sub_dataset, transform=None):
        self.sub_dataset = sub_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y, index = self.sub_dataset[index] 
        return x, y, index

    def __len__(self):
        return len(self.sub_dataset)
    
# Modified
class CloudL9_DataHandler(Dataset):
    def __init__(self, sub_dataset,  transform=None):
        self.sub_dataset = sub_dataset # Load on-the-fly
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.sub_dataset[index]
        x = x.squeeze(0)  
        return x, y, index
    
    def __len__(self):
        return len(self.sub_dataset)

class CloudS2_DataHandler(Dataset):
    def __init__(self, sub_dataset,  transform=None):
        self.sub_dataset = sub_dataset # Load on-the-fly
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.sub_dataset[index]
        x = x.squeeze(0)  
        return x, y, index
    
    def __len__(self):
        return len(self.sub_dataset)


class CloudL8_DataHandler(Dataset):
    def __init__(self, sub_dataset,  transform=None):
        self.sub_dataset = sub_dataset # Load on-the-fly
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.sub_dataset[index]
        x = x.squeeze(0)  
        return x, y, index
    
    def __len__(self):
        return len(self.sub_dataset)
    
class CloudSEN12_DataHandler(Dataset):
    def __init__(self, sub_dataset,  transform=None):
        self.sub_dataset = sub_dataset # Load on-the-fly
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.sub_dataset[index]
        x = x.squeeze(0)  
        return x, y, index
    
    def __len__(self):
        return len(self.sub_dataset)

class LandCoverAI_DataHandler(Dataset):
    def __init__(self, sub_dataset,  transform=None):
        self.sub_dataset = sub_dataset # Load on-the-fly
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.sub_dataset[index]
        x = x.squeeze(0)  
        return x, y, index
    
    def __len__(self):
        return len(self.sub_dataset)